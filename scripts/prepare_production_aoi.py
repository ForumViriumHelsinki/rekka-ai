"""Build the city-wide detection region from Helsinki's open WFS.

A production sweep wants "everywhere in Helsinki a truck could be", not the
city's bounding box. Two WFS layers give that:

- ``avoindata:Maavesi_kaupunginosat`` -- city districts split into land and
  water parts (*maa* + *vesi*), carrying ``tyyppi`` ("Maa-alue" / water) and
  ``nimi_fi``.
- ``avoindata:Liikennevaylat`` -- the traffic-way network, ~111k LineStrings
  citywide.

The noise this script exists to remove is **islands and skerries**, and the
important detail is that they are not separate features: Santahamina arrives
as one MultiPolygon of 25 parts, Lauttasaari 30 (checked 2026-08-14). A
feature-level filter would keep every skerry attached to a district that
happens to have roads. So the districts are *exploded to single polygons
first*, and each part is tested on its own for road coverage. A part with no
traffic way in it is ground no truck can reach.

Both layers are cached under ``--wfs-cache`` by ``enrich.fetch_layer``, so a
re-run costs nothing and works offline; ``--refresh-wfs`` re-fetches.

Output CRS is EPSG:3879 throughout -- the WFS serves it natively and it is
the project's grid CRS, so no transform happens here at all.

    uv run --extra detect python scripts/prepare_production_aoi.py

Attribution: contains data from the City of Helsinki open data service
(Helsingin kaupunki, Kaupunkimittauspalvelut).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from rekka_ai.enrich import fetch_layer
from rekka_ai.geo import GRID, crs_member

if TYPE_CHECKING:
    # geopandas is the `detect` extra, imported lazily inside main() so the
    # base install still runs `--help`. Annotations are strings here thanks to
    # `from __future__ import annotations`, so this import is never executed at
    # runtime -- it exists only so a type checker (and an editor) can resolve
    # GeoDataFrame in the signature below.
    import geopandas

#: Districts split into land and water parts. Not enrich.py's
#: ``Kaupunginosajako``: that one is the plain district division, with no
#: land/water split, so it carries the sea inside the coastal districts.
LAYER_LAND_WATER_DISTRICTS = "avoindata:Maavesi_kaupunginosat"

#: ``tyyppi`` on the districts layer. Land parts are what a sweep wants; the
#: water parts hold no targets and cost tiles. Measured 2026-08-14: the layer
#: as served is *already* land-only -- all 60 features are "Maa-alue" and no
#: water feature comes back at all, despite the layer's *maavesi* name. So
#: ``--land-only`` is a guard against that changing, not the filter that does
#: the work; the road test below is what removes the islands.
LAND_TYPE = "Maa-alue"

#: The road-coverage test layer. Deliberately NOT imported from enrich.py:
#: enrich's ``LAYER_STREETS`` was ``avoindata:Liikennevaylat`` until
#: 2026-08-20 and is now ``YLRE_Katualue_alue``, and this script followed it
#: silently. The region shipped for the 2025 production sweep (built
#: 2026-08-14) was made with Liikennevaylat, and re-runs must reproduce
#: that -- the full ~111k LineString network is also the right test here:
#: it counts a part reachable if any track, path or service road crosses it,
#: where street-area polygons only cover named streets.
LAYER_ROADS = "avoindata:Liikennevaylat"

app = typer.Typer(add_completion=False)


def _write(frame: geopandas.GeoDataFrame, out: Path) -> None:
    """Write the region, keeping its CRS readable by whatever opens it next.

    ``.gpkg``/``.fgb`` carry their own CRS, and ``imagery.aoi.load_region``
    lets that win over ``--crs``. GeoJSON does not: ``_load_region_geojson``
    ignores the ``crs`` member and trusts the ``--crs`` argument (default
    WGS84), so a GeoJSON region MUST be passed with ``--crs EPSG:3879`` or
    every coordinate is read as degrees. The member is still written for
    QGIS's sake, but the self-describing formats are the safe default.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".geojson", ".json"}:
        data = json.loads(frame.to_json())
        data["crs"] = crs_member()
        out.write_text(json.dumps(data))
    else:
        frame.to_file(out)


@app.command()
def main(
    out: Annotated[
        Path, typer.Option(help="Region file to write (.gpkg, .fgb or .geojson).")
    ] = Path("data/production/helsinki-region.gpkg"),
    wfs_cache: Annotated[
        Path, typer.Option(help="Directory for cached WFS layers.")
    ] = Path("data/wfs"),
    refresh_wfs: Annotated[bool, typer.Option(help="Re-fetch the WFS layers.")] = False,
    min_roads: Annotated[
        int,
        typer.Option(
            help="Keep a polygon part only if at least this many traffic ways "
            "intersect it. 1 drops roadless skerries; raising it also drops "
            "water parts that only a bridge crosses."
        ),
    ] = 1,
    land_only: Annotated[
        bool, typer.Option(help=f"Keep only parts with tyyppi == {LAND_TYPE!r}.")
    ] = True,
    min_area_m2: Annotated[
        float, typer.Option(help="Drop polygon parts smaller than this.")
    ] = 0.0,
) -> None:
    """District polygons that actually carry a road, as one detection region."""
    try:
        import geopandas
    except ImportError as exc:  # optional extra, same as the rest of the project
        raise typer.BadParameter(
            "this script needs geopandas: uv sync --extra detect"
        ) from exc

    districts = geopandas.GeoDataFrame.from_features(
        fetch_layer(
            LAYER_LAND_WATER_DISTRICTS, cache_root=wfs_cache, refresh=refresh_wfs
        )["features"],
        crs=GRID,
    )
    roads = geopandas.GeoDataFrame.from_features(
        fetch_layer(LAYER_ROADS, cache_root=wfs_cache, refresh=refresh_wfs)["features"],
        crs=GRID,
    )
    typer.echo(f"districts: {len(districts)} features, roads: {len(roads)} features")

    if land_only and "tyyppi" in districts.columns:
        before = len(districts)
        districts = districts[districts["tyyppi"] == LAND_TYPE]
        typer.echo(f"  land-only: {before} -> {len(districts)} features")

    # Islands live *inside* a district's MultiPolygon, so the road test only
    # bites after exploding to single polygons.
    parts = districts.explode(index_parts=False, ignore_index=True)
    typer.echo(f"  exploded to {len(parts)} polygon parts")

    if min_area_m2 > 0:
        before = len(parts)
        parts = parts[parts.geometry.area >= min_area_m2]
        typer.echo(f"  area >= {min_area_m2:g} m2: {before} -> {len(parts)} parts")

    # One row per (part, road) pair; the count per part is what we filter on.
    joined = geopandas.sjoin(
        parts[["geometry"]], roads[["geometry"]], predicate="intersects", how="inner"
    )
    counts = joined.groupby(level=0).size()

    kept_index = counts[counts >= min_roads].index
    kept = parts.loc[parts.index.intersection(kept_index)].copy()
    kept["road_segments"] = counts.reindex(kept.index).astype(int)

    dropped = len(parts) - len(kept)
    area_kept = kept.geometry.area.sum() / 1e6
    area_all = parts.geometry.area.sum() / 1e6
    typer.echo(
        f"  road filter (>= {min_roads}): kept {len(kept)} parts, dropped {dropped}"
    )
    typer.echo(
        f"  area: {area_kept:.1f} km2 of {area_all:.1f} km2 "
        f"({100 * area_kept / area_all:.1f}% kept)"
    )

    _write(kept, out)
    typer.echo(f"done: {len(kept)} polygons -> {out}")

    hint = "" if out.suffix.lower() not in {".geojson", ".json"} else " --crs EPSG:3879"
    typer.echo(
        "\nsweep it with:\n"
        f"  uv run --extra detect rekka-ai detect --aoi {out}{hint} \\\n"
        "      --weights runs/train/<run>/weights/best.pt \\\n"
        "      --out data/detections/helsinki.gpkg"
    )


if __name__ == "__main__":
    app()
