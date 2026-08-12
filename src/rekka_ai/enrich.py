"""Contextual enrichment of detections: administrative and parking attributes.

Takes the oriented-box detections written by `detect` (GeoJSON/FlatGeobuf/
GeoPackage, EPSG:3879 -- see detect/detections.py) and adds attributes that
come from *other* datasets rather than from the detection geometry itself:
city district, postal code, street, and whether the footprint falls on a
mapped parking area. Detection-derived numbers (length_m, width_m,
heading_deg) are not touched here -- they are already baked in by
`detections.write`, recomputed from geometry as the project's rule requires.

Source: Helsinki's open WFS service,
https://kartta.hel.fi/ws/geoserver/avoindata/wfs. Layers are cached to
`data/wfs/<layer>.geojson` (gitignored, like `data/osm/`) so a re-run and the
test suite do not depend on the network; `--refresh-wfs` re-fetches.

Layer names AND attribute schema below are verified against real sample
GetFeature responses (checked 2026-08-12, one feature per layer):

- avoindata:Kaupunginosajako (Polygon) -- district name is `nimi_fi`
  (e.g. "MEILAHTI"). `nimi_se` is the Swedish name, `tunnus` a numeric code.
- avoindata:Postinumeroalue (Polygon) -- `nimi_fi` is USELESS here (always
  "HELSINKI" on every feature seen); the actual postal code is `tunnus`
  (e.g. "00440").
- avoindata:Liikennevaylat (LineString, ~111k features citywide) -- name is
  `nimi`, but it is frequently null (seen null on a motorway ramp segment).
  Because this is line geometry, a detection can never be "within" a street;
  the join below uses nearest-line-within-distance instead of point-in-
  polygon.
- avoindata:Pysakointipaikat_alue (MultiPolygon) -- no attribute is needed
  for a plain on/off "parked" flag, just a geometric intersects test.
  `luokka_nimi` (e.g. "Pysäköinti sallittu pysäköintikieltoajan
  ulkopuolella") is available if a finer-grained parking-rule attribute is
  ever wanted instead of a boolean.

CRS note: every layer's `DefaultCRS` in GetCapabilities is EPSG:3879 -- the
same grid CRS detections are stored in (EPSG:3067 is only listed as an
`OtherCRS` option), and the sample responses above came back in EPSG:3879
when requested with `srsName`. So this module does *no* coordinate
transform of its own -- geo.py's rule that "transforms happen only in
geo.py" is moot here because there is no transform to make.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx  # base dependency, same client imagery/wmts.py and osm.py use

from rekka_ai.geo import GRID, GRID_CRS_NAME

#: Helsinki's open WFS endpoint. Access constraints are NONE per its
#: capabilities document (checked 2026-08), same as the WMTS imagery.
WFS_URL = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"

LAYER_DISTRICTS = "avoindata:Kaupunginosajako"
DISTRICT_NAME_COLUMN = "nimi_fi"

LAYER_POSTAL_AREAS = "avoindata:Postinumeroalue"
POSTAL_CODE_COLUMN = "tunnus"  # NOT nimi_fi -- see module docstring

LAYER_STREETS = "avoindata:Liikennevaylat"
STREET_NAME_COLUMN = "nimi"  # frequently null
#: How far (metres) a detection's centre may be from the nearest street
#: segment and still be attributed to it. Placeholder -- not tuned against
#: real detections; a semi-trailer truck is up to ~2.6 m wide and typically
#: parks or drives within a few metres of the carriageway edge, so this
#: should comfortably cover on-street cases without reaching across to an
#: unrelated parallel street. Confirm with Otto once real detection data is
#: available to check against.
STREET_MAX_DISTANCE_M = 30.0

#: Otto also flagged avoindata:Pysakoinnin_maksuvyohykkeet_alue (paid parking
#: zones) and avoindata:Asukas_ja_yrityspysakointivyohykkeet_alue (resident /
#: business permit zones) as available on the same WFS if "parked" ever
#: needs to distinguish zone types rather than just on/off a parking area.
LAYER_PARKING = "avoindata:Pysakointipaikat_alue"

CACHE_DIR = Path("data/wfs")


def _cache_path(layer: str, cache_root: Path) -> Path:
    return cache_root / f"{layer.split(':')[-1]}.geojson"


def fetch_layer(
    layer: str,
    *,
    cache_root: Path = CACHE_DIR,
    refresh: bool = False,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """One WFS layer as a GeoJSON FeatureCollection, cached on disk.

    Mirrors osm.py's cache-under-data/ pattern: the pipeline should be
    re-runnable and testable offline, and administrative boundaries change
    rarely enough that re-fetching every run buys nothing. Liikennevaylat
    alone is ~111k features citywide (no bbox filter here) -- fine for a
    one-time cached fetch, but worth knowing before adding a `--refresh-wfs`
    habit to a tight loop. ``cache_root`` is a parameter rather than always
    ``CACHE_DIR``, matching mine's separate ``--osm-cache`` next to its tile
    ``--cache``.

    ``client`` / ``transport`` exist so tests can inject a mock WFS without
    touching the network -- the same pattern as ``fetch_industrial`` and
    ``TileFetcher``.
    """
    cache_path = _cache_path(layer, cache_root)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())

    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=60.0, transport=transport)
    try:
        response = client.get(
            WFS_URL,
            params={
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeName": layer,
                "outputFormat": "application/json",
                "srsName": GRID_CRS_NAME,  # ask for EPSG:3879 directly -- no reprojection needed
            },
        )
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            client.close()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write, matching osm.py's fetch_industrial: a crash or Ctrl-C
    # mid-write must not leave a truncated .geojson that the next run then
    # trusts as a complete cache.
    temporary = cache_path.with_suffix(".geojson.part")
    temporary.write_text(json.dumps(data))
    temporary.replace(cache_path)
    return data


def _layer_frame(
    layer: str,
    *,
    cache_root: Path,
    refresh: bool,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
):
    """A cached WFS layer, already in the project's grid CRS (see module docstring).

    ``crs=GRID`` here, not ``GRID_CRS_NAME``: per geo.py, the URN form's
    easting-first override is a GDAL GeoJSON-driver behaviour tied to the
    ``crs`` member of a file's header (``crs_member()``), not something
    pyproj applies when a string is handed to a ``crs=`` keyword directly.
    Passing the URN here would just resolve to plain EPSG:3879 anyway, but
    ``GRID`` is what detections.py itself uses for this exact purpose --
    matching it keeps this frame's ``.crs`` identical to ``detections.crs``,
    which is what lets sjoin/sjoin_nearest below skip a CRS-mismatch check.
    """
    import geopandas  # optional extra, lazy like detect/sweep.py and detections.write

    geojson = fetch_layer(
        layer,
        cache_root=cache_root,
        refresh=refresh,
        client=client,
        transport=transport,
    )
    return geopandas.GeoDataFrame.from_features(geojson["features"], crs=GRID)


def enrich(
    input_path: Path,
    output_path: Path,
    *,
    cache_root: Path = CACHE_DIR,
    refresh_wfs: bool = False,
    street_max_distance_m: float = STREET_MAX_DISTANCE_M,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Read detections, add district/postal_code/street/parked, write the result.

    Geometry and detection-derived measurements (length_m, width_m,
    heading_deg, confidence, label) are untouched -- this only adds
    properties sourced from the WFS layers above.

    ``client`` / ``transport`` are forwarded to ``fetch_layer`` -- only
    relevant with ``refresh_wfs=True``, or a cold cache; tests can otherwise
    ignore them and pre-populate ``cache_root`` instead.
    """
    import geopandas

    detections = geopandas.read_file(input_path)
    if detections.crs is None:
        raise ValueError(f"{input_path} has no CRS; expected {GRID_CRS_NAME}")

    layer_kwargs = {
        "cache_root": cache_root,
        "refresh": refresh_wfs,
        "client": client,
        "transport": transport,
    }
    districts = _layer_frame(LAYER_DISTRICTS, **layer_kwargs)
    postal_areas = _layer_frame(LAYER_POSTAL_AREAS, **layer_kwargs)
    streets = _layer_frame(LAYER_STREETS, **layer_kwargs)
    parking = _layer_frame(LAYER_PARKING, **layer_kwargs)

    result = detections
    result = _attach_area_attribute(result, districts, DISTRICT_NAME_COLUMN, "district")
    result = _attach_area_attribute(
        result, postal_areas, POSTAL_CODE_COLUMN, "postal_code"
    )
    result = _attach_nearest_street(
        result, streets, max_distance_m=street_max_distance_m
    )
    result["parked"] = _intersects_any(result, parking)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    driver = "FlatGeobuf" if output_path.suffix.lower() == ".fgb" else "GPKG"
    result.to_file(output_path, driver=driver)


def _attach_area_attribute(detections, areas, source_column: str, dest_column: str):
    """Left-join `areas[source_column]` onto `detections` as `dest_column`.

    A detection's centre decides which area it belongs to -- matching the
    rule detections.within_region already applies to AOI membership, rather
    than inventing a different convention for this join. Both districts and
    postal areas are polygons that tile the city without gaps, so exactly
    one area should match each centre (or none, right at the coastline).
    """
    import geopandas

    named = areas[[source_column, "geometry"]].rename(
        columns={source_column: dest_column}
    )
    centres = detections.assign(geometry=detections.geometry.centroid)
    joined = geopandas.sjoin(centres, named, how="left", predicate="within")
    # A centre exactly on a shared boundary can match >1 polygon; sjoin then
    # duplicates that row. Keep the first match -- rare edge case, not worth
    # a tie-break rule.
    joined = joined[~joined.index.duplicated(keep="first")]
    return detections.assign(
        **{dest_column: joined[dest_column].reindex(detections.index)}
    )


def _attach_nearest_street(detections, streets, *, max_distance_m: float):
    """Left-join the nearest street's name, within ``max_distance_m``.

    Streets are LineStrings, so "within" never matches -- this is a nearest-
    neighbour join on detection centres instead, capped at a distance so a
    detection nowhere near any digitized street gets `street = None` rather
    than being pinned to whatever segment happens to be least-far-away.
    """
    import geopandas

    centres = detections.assign(geometry=detections.geometry.centroid)
    joined = geopandas.sjoin_nearest(
        centres,
        streets[[STREET_NAME_COLUMN, "geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="_street_distance_m",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    return detections.assign(
        street=joined[STREET_NAME_COLUMN].reindex(detections.index)
    )


def _intersects_any(detections, areas):
    """True where a detection's polygon overlaps any feature of `areas`."""
    import geopandas

    joined = geopandas.sjoin(
        detections, areas[["geometry"]], how="left", predicate="intersects"
    )
    # A detection matching >1 parking polygon appears more than once after
    # sjoin; collapse back to one boolean per original row.
    hit = ~joined["index_right"].isna()
    return (
        hit.groupby(level=0)
        .any()
        .reindex(detections.index, fill_value=False)
        .to_numpy()
    )
