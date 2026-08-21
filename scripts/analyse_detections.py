"""Per-year truck density grids, in a GeoPackage of their own.

`package_detections.py` ships the detections as the model produced them.
This ships the one derived product worth shipping beside them, and keeps the
two files apart on purpose: one is evidence, the other is a summary of it.
Everything here rebuilds from the detections in a minute, so the analysis
file is disposable in a way the detections are not.

Two layers per flight year. ``truck_grid_<year>_<size>m`` is a square grid over
the swept region carrying the number of trucks whose centre falls in each
cell. Cells are uniform, so the count *is* the density -- nothing needs
normalising, and a graduated style on ``trucks`` is the whole map. One layer
per year rather than one layer with per-year columns, because that is how the
years are grouped in QGIS: each grid belongs beside its own orthophoto and
its own detections, and styling a layer on a column for the wrong year is a
mistake nobody notices.

Beside each grid, ``truck_points_<year>_z<zoom>`` gives the same trucks as
centroid points. QGIS's heatmap renderer and the Processing KDE algorithm
both take **points only**, and detections are oriented footprint polygons, so
this layer is the prerequisite for any density surface. It carries the full
attribute set, so a heatmap can be filtered (one district, one street type)
or weighted (``length_m``, to weigh a semi-trailer above a box truck) without
going back to the detections.

Cells with no trucks are dropped. Most of the city has none, and an empty
cell says "no truck here that day", which the map already shows by there
being nothing drawn.

``--min-confidence`` (default 0.77, round 4's operating point) is applied
before counting, and recorded on every layer, because a count presented as a
census must name its threshold: the 2025 city truck total is 3,729 at 0.25
against 2,450 at 0.77.

The grid cell ids are stable across years -- the same ``cell`` value is the
same 250 m of ground in every layer. Joining the years on ``cell`` therefore
reconstructs anything cross-year, including how many years a cell held a
truck at all, without this script needing to take a view on it.

Finer questions -- trucks by district, by street, on residential streets --
are attribute queries against the detections themselves and want no layer of
their own; see the notes shipped with the data.

    uv run --extra detect python scripts/analyse_detections.py

Imagery attribution: (c) Helsingin kaupunki, Kaupunkimittauspalvelut.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from rekka_ai.detect.chunks import grid_cells
from rekka_ai.geo import GRID
from rekka_ai.imagery.aoi import load_region

if TYPE_CHECKING:
    # The `detect` extra, imported lazily in main() like the rest of the
    # project. `from __future__ import annotations` keeps this type-only.
    import geopandas

#: Flight years to grid, newest first -- the order a QGIS layer tree reads.
DEFAULT_YEARS = (2025, 2024, 2023)

#: Round 4's eval operating point, and `detect`'s own default. A census
#: threshold: high enough that a count can be quoted without a caveat.
DEFAULT_MIN_CONFIDENCE = 0.77

#: Cell edge. 250 m is sized against what the grid is for: fine enough to
#: separate one yard from the next, coarse enough that a truck parked 30 m
#: from where one stood last year lands in the same cell rather than reading
#: as two unrelated ones.
DEFAULT_GRID_SIZE_M = 250.0

#: The class this grid is about. The project detects four; this is a freight
#: product.
TRUCK = "truck"

app = typer.Typer(add_completion=False)


def _layer_for(year: int, zoom: int) -> str:
    """The detection layer a year lives in, as `package_detections` names it."""
    return f"helsinki_{year}_z{zoom}"


def _read_year(detections: Path, year: int, zoom: int) -> geopandas.GeoDataFrame:
    import geopandas

    frame = geopandas.read_file(detections, layer=_layer_for(year, zoom))
    if frame.crs is None or frame.crs.to_string() != GRID:
        raise typer.BadParameter(
            f"{_layer_for(year, zoom)} is in {frame.crs}, expected {GRID}"
        )
    return frame


@app.command()
def main(
    detections: Annotated[
        Path, typer.Option(help="GeoPackage of plain detections to summarise.")
    ] = Path("data/detections/rekka-ai-detections.gpkg"),
    out: Annotated[
        Path, typer.Option(help="GeoPackage of grid layers to write.")
    ] = Path("data/detections/rekka-ai-truck-grids.gpkg"),
    region: Annotated[Path, typer.Option(help="Swept region the grid covers.")] = Path(
        "data/production/helsinki-region.gpkg"
    ),
    year: Annotated[
        list[int] | None, typer.Option(help="Flight year; repeat for several.")
    ] = None,
    zoom: Annotated[int, typer.Option(help="Zoom the sweep ran at.")] = 16,
    min_confidence: Annotated[
        float, typer.Option(help="Confidence floor applied before counting.")
    ] = DEFAULT_MIN_CONFIDENCE,
    grid_size: Annotated[
        float, typer.Option(help="Cell edge in metres.")
    ] = DEFAULT_GRID_SIZE_M,
    force: Annotated[bool, typer.Option(help="Replace an existing output.")] = False,
) -> None:
    """Build one truck density grid per flight year."""
    try:
        import geopandas
        import pyogrio
    except ImportError as exc:  # optional extra, same as the rest of the project
        raise typer.BadParameter(
            "this script needs geopandas: uv sync --extra detect"
        ) from exc

    if not detections.exists():
        raise typer.BadParameter(
            f"no detections at {detections}; run scripts/package_detections.py first"
        )
    if out.exists():
        if not force:
            raise typer.BadParameter(
                f"{out} exists; pass --force to replace it. Appending would "
                "leave grids from an older run beside the new ones."
            )
        # Replaced, not appended to: `mode="a"` on a stale file keeps whatever
        # layers it already had, including years no longer being gridded.
        out.unlink()

    years = list(year) if year else list(DEFAULT_YEARS)
    available = {str(n) for n, _ in pyogrio.list_layers(detections)}
    missing = [
        _layer_for(y, zoom) for y in years if _layer_for(y, zoom) not in available
    ]
    if missing:
        raise typer.BadParameter(f"{detections} has no layer(s): {', '.join(missing)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(
        f"gridding {detections} at conf >= {min_confidence}, "
        f"{grid_size:g} m cells -> {out}"
    )

    swept = load_region(str(region))
    cells = grid_cells(swept, grid_size)
    typer.echo(f"  {len(cells):,} cells cover the region")

    for index, y in enumerate(years):
        frame = _read_year(detections, y, zoom)
        trucks = frame[
            (frame["confidence"] >= min_confidence) & (frame["label"] == TRUCK)
        ]
        # Detections are footprint polygons; the centroid of an oriented
        # rectangle is its centre, which is the point `within_region` and the
        # enrichment joins already used to decide where a detection is. So a
        # truck lands in exactly one cell, and in the same one they chose.
        points = trucks.copy()
        points["geometry"] = trucks.geometry.centroid
        points = points.reset_index(drop=True)
        points["min_confidence"] = min_confidence

        grid = geopandas.GeoDataFrame(
            {"cell": [c.name for c in cells]},
            geometry=[c.polygon() for c in cells],
            crs=GRID,
        )
        joined = geopandas.sjoin(
            points[["geometry"]], grid[["cell", "geometry"]], predicate="within"
        )
        counts = joined.groupby("cell").size()
        grid["trucks"] = grid["cell"].map(counts).fillna(0).astype(int)
        grid = grid[grid["trucks"] > 0].reset_index(drop=True)
        grid["year"] = y
        grid["min_confidence"] = min_confidence

        name = f"truck_grid_{y}_{grid_size:g}m"
        # The first write of the run creates the file; every later one
        # appends. Grids go first so a year reads grid-then-points.
        grid.to_file(out, layer=name, driver="GPKG", mode="w" if index == 0 else "a")
        typer.echo(
            f"  {name}: {len(grid):,} cells, {int(grid['trucks'].sum()):,} trucks, "
            f"busiest {int(grid['trucks'].max())}"
        )

        points_name = f"truck_points_{y}_z{zoom}"
        points.to_file(out, layer=points_name, driver="GPKG", mode="a")
        typer.echo(f"  {points_name}: {len(points):,} points")

    # Read back rather than trusting the writes: a GPKG layer that failed to
    # append is not an exception, it is a layer that is not there.
    written = {str(n) for n, _ in pyogrio.list_layers(out)}
    for y in years:
        for name in (f"truck_grid_{y}_{grid_size:g}m", f"truck_points_{y}_z{zoom}"):
            if name not in written:
                raise typer.BadParameter(f"{name} did not land in {out}")

    size_mb = out.stat().st_size / 1e6
    typer.echo(f"done: {len(written)} layer(s) -> {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    app()
