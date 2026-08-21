"""Package the enriched city-wide detections as one multi-layer GeoPackage.

The sweep writes one FlatGeobuf per flight year, which is the right shape for
the pipeline -- a year is swept, merged and enriched on its own -- and the
wrong shape for handing to anyone else. Three files with a naming convention
are three chances to open the wrong one, or the un-enriched one. A GeoPackage
holds all three as named layers in a single file that QGIS opens with the
layers already listed, so *which years exist* is visible instead of implied.

Only the enriched files go in. The un-enriched merges carry the same
geometry and no attributes the enriched ones lack, so shipping both would
only invite the question of which is authoritative. The layers are named for
what they are -- ``helsinki_<year>_z<zoom>`` -- and drop the ``_enriched``
suffix, since inside this file there is nothing else they could be.

This is a repackaging step and nothing more: no geometry is touched, no
attribute is recomputed, and the row counts are asserted to match the
sources. Everything it reads is reproducible from ``detect`` and ``enrich``,
so the output is safe to delete and rebuild -- but it still refuses to
overwrite without ``--force``, because a stale GeoPackage that gained a layer
instead of being replaced is the failure worth preventing.

    uv run --extra detect python scripts/package_detections.py

Imagery attribution: (c) Helsingin kaupunki, Kaupunkimittauspalvelut.
Enrichment attribution: City of Helsinki open data service.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from rekka_ai.geo import GRID

if TYPE_CHECKING:
    # geopandas is the `detect` extra, imported lazily inside main() so the
    # base install still runs `--help`. `from __future__ import annotations`
    # makes this import type-only -- it never executes at runtime.
    import geopandas

#: Flight years to package, newest first: a QGIS layer list is read top down,
#: and the current year is the one most runs are about.
DEFAULT_YEARS = (2025, 2024, 2023)

#: What `enrich` adds. Checked rather than assumed: an un-enriched file has
#: the same geometry and the same name shape, so the only thing that tells
#: the two apart is whether these columns are there.
ENRICHED_COLUMNS = (
    "district",
    "postal_code",
    "street",
    "street_type",
    "context",
    "street_part",
)

#: Classes the round-4 model emits, in the order a summary should read them.
#: Fixed rather than taken from the data, so a year that happens to hold no
#: buses still lines up column-wise with the years that do.
CLASSES = ("truck", "bus", "van", "car")

app = typer.Typer(add_completion=False)


def layer_name(year: int, zoom: int) -> str:
    """The layer a year's detections live in, inside the GeoPackage."""
    return f"helsinki_{year}_z{zoom}"


def source_path(root: Path, year: int, zoom: int) -> Path:
    """The enriched sweep output for a year, as `enrich` names it."""
    return root / f"helsinki_{year}_z{zoom}_enriched.fgb"


def _check(frame: geopandas.GeoDataFrame, path: Path) -> None:
    """Refuse a source that is not what the layer claims to be.

    Three ways this goes wrong quietly, so all three are errors: an empty
    file (a sweep that produced nothing, or a truncated write), the wrong CRS
    (a GeoPackage mixing EPSG:3879 with anything else would reproject on read
    and misplace every box by the ~1.74 deg the project's CRS note warns
    about), and a file that was never enriched.
    """
    if frame.empty:
        raise typer.BadParameter(f"{path} holds no detections")
    if frame.crs is None or frame.crs.to_string() != GRID:
        raise typer.BadParameter(
            f"{path} is in {frame.crs}, expected {GRID}; "
            "the sweep writes the grid CRS and this script does not reproject"
        )
    missing = [c for c in ENRICHED_COLUMNS if c not in frame.columns]
    if missing:
        raise typer.BadParameter(
            f"{path} is missing {', '.join(missing)} -- this looks like the "
            f"un-enriched merge. Run `rekka-ai enrich` on it first."
        )


def _summary(frame: geopandas.GeoDataFrame) -> str:
    """One line of per-class counts, for the operator reading the log."""
    counts = frame["label"].value_counts()
    seen = ", ".join(f"{c} {int(counts.get(c, 0)):,}" for c in CLASSES)
    other = sorted(set(counts.index) - set(CLASSES))
    if other:
        seen += ", " + ", ".join(f"{c} {int(counts[c]):,}" for c in other)
    return seen


@app.command()
def main(
    out: Annotated[Path, typer.Option(help="GeoPackage to write.")] = Path(
        "data/detections/rekka-ai-detections.gpkg"
    ),
    detections_dir: Annotated[
        Path, typer.Option(help="Directory holding the enriched sweep outputs.")
    ] = Path("data/detections"),
    year: Annotated[
        list[int] | None,
        typer.Option(help="Flight year to include; repeat for several."),
    ] = None,
    zoom: Annotated[int, typer.Option(help="Tile grid zoom the sweep ran at.")] = 16,
    force: Annotated[
        bool, typer.Option(help="Replace an existing GeoPackage.")
    ] = False,
) -> None:
    """Collect the enriched per-year detections into one GeoPackage."""
    try:
        import geopandas
        import pyogrio
    except ImportError as exc:  # optional extra, same as the rest of the project
        raise typer.BadParameter(
            "this script needs geopandas: uv sync --extra detect"
        ) from exc

    years = list(year) if year else list(DEFAULT_YEARS)

    # Every source is resolved before anything is written: a missing year
    # should not leave a half-built GeoPackage that looks complete.
    sources = {y: source_path(detections_dir, y, zoom) for y in years}
    absent = [str(p) for p in sources.values() if not p.exists()]
    if absent:
        raise typer.BadParameter(
            "no enriched detections at " + ", ".join(absent) + "\n"
            "Sweep and enrich the year first:\n"
            "  rekka-ai detect --checkpoint-dir ... --out <year>.fgb\n"
            "  rekka-ai enrich --detections <year>.fgb --out <year>_enriched.fgb"
        )

    if out.exists():
        if not force:
            raise typer.BadParameter(
                f"{out} exists; pass --force to replace it. Appending to it "
                "would leave layers from an older sweep sitting beside the "
                "new ones, indistinguishable once the log has scrolled away."
            )
        # Replaced, not appended to: `mode="a"` on a stale file keeps whatever
        # layers it already had, including years no longer being packaged.
        out.unlink()

    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"packaging {len(years)} year(s) at z{zoom} -> {out}")

    written: dict[str, int] = {}
    for index, y in enumerate(years):
        path = sources[y]
        frame = geopandas.read_file(path)
        _check(frame, path)
        name = layer_name(y, zoom)
        # First layer creates the file, the rest append; pyogrio has no
        # "create or append" that is also safe on a file we just removed.
        frame.to_file(out, layer=name, driver="GPKG", mode="w" if index == 0 else "a")
        written[name] = len(frame)
        typer.echo(f"  {name}: {len(frame):,} detections ({_summary(frame)})")

    # Read the file back rather than trusting the writes: a GPKG layer that
    # failed to append is not an exception, it is a layer that is not there.
    layers = {str(n): str(g) for n, g in pyogrio.list_layers(out)}
    for name, count in written.items():
        if name not in layers:
            raise typer.BadParameter(f"{name} did not land in {out}")
        stored = len(geopandas.read_file(out, layer=name))
        if stored != count:
            raise typer.BadParameter(
                f"{name}: wrote {count:,} detections but {out} holds {stored:,}"
            )

    size_mb = out.stat().st_size / 1e6
    typer.echo(
        f"done: {len(written)} layer(s), {sum(written.values()):,} detections "
        f"-> {out} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    app()
