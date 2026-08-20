"""Contextual enrichment of detections: administrative and parking attributes.

Takes the oriented-box detections written by `detect` (GeoJSON/FlatGeobuf/
GeoPackage, EPSG:3879 -- see detect/detections.py) and adds attributes that
come from *other* datasets rather than from the detection geometry itself:
city district, postal code, street (name and register type), and a location
`context` (parking / street / other, with `street_part` refining the street
case). Detection-derived numbers (length_m, width_m, heading_deg) are not
touched here -- they are already baked in by `detections.write`, recomputed
from geometry as the project's rule requires.

Why `context` and not a `parked` boolean: "parked" is a behaviour, and one
orthophoto cannot tell a parked vehicle from a moving or queued one. What
the data *can* say is where the vehicle is, so that is what the attributes
say. The boolean this replaced (2026-08-20) was also structurally wrong:
its only source, Pysakointipaikat_alue, is a registry of *regulated*
parking, so every unregulated kerbside bay and every courtyard lot read
`parked = false` -- measured on the 2025 city sweep, 73% of the 8,051 cars
sitting inside digitized on-street bays carried it.

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
- avoindata:YLRE_Katualue_alue (Polygon, 7,768 features) -- street name is
  `kadun_nimi`, filled on 100% of them, and the register's purpose
  `kayttotarkoitus` comes along as `street_type`. This replaced
  avoindata:Liikennevaylat on 2026-08-14: that layer's `nimi` is null on
  every one of its 110,981 features, so the street join produced `None` for
  every detection while fetching 78 MB to do it. Polygons also let a
  detection be genuinely *inside* a street, which a centreline never allows.
- avoindata:Pysakointipaikat_alue (MultiPolygon, 8,752 features) -- a
  registry of *regulated* parking (paid, disc, permit: `luokka_nimi` is
  filled accordingly), not of everywhere a vehicle may legally stand.
  Feeds `context = "parking"`. `luokka_nimi` (e.g. "Pysäköinti sallittu
  pysäköintikieltoajan ulkopuolella") is available if a finer-grained
  parking-rule attribute is ever wanted.
- avoindata:YLRE_Katu_ja_viherosat_ajorata_alue (Polygon, 49,392 features,
  checked 2026-08-20) -- partitions each street right-of-way into parts by
  `alatyyppi`: carriageway sections, on-street parking bays, driveways
  (tonttiliittyma) and kerb build-outs. Feeds `street_part`. The raw fetch
  is 96 MB, so the cache keeps only `alatyyppi` + geometry (see
  ``fetch_layer``'s ``keep``).

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

#: Street *areas* -- the right-of-way as polygons, one per named street.
#: NOT ``avoindata:Liikennevaylat``, which was used here until 2026-08-14:
#: that layer's ``nimi`` is null on **every** one of its 110,981 features
#: (0% fill measured over a 20,000-feature sample), so the join it fed could
#: never produce a value and silently wrote `street = None` for every
#: detection. It also cost the most of any layer here -- 78 MB fetched and
#: spatially indexed for nothing. ``YLRE_Katualue_alue`` is 7,768 polygons
#: with ``kadun_nimi`` filled on 100% of them.
LAYER_STREETS = "avoindata:YLRE_Katualue_alue"
STREET_NAME_COLUMN = "kadun_nimi"
#: The street-area layer is not only streets: `kayttotarkoitus` also marks
#: squares (Katuaukio, 150), market places (Tori, 151), pedestrian/cycle
#: areas (Kevyt liikenne, 1,847) and whole public parking areas
#: (Pysäköintialue, 255), measured 2026-08-20. Carried through verbatim as
#: `street_type` so those cases read as themselves instead of flattening
#: into `context = "street"` -- an airport apron (Malmin lentoaseman aukio)
#: is a Katuaukio, and the attribute should say so.
STREET_TYPE_COLUMN = "kayttotarkoitus"
#: How far (metres) a detection's centre may be from the nearest street area
#: and still be attributed to it. Was 30 m, sized for the old *centreline*
#: layer where a kerbside vehicle sits half a carriageway from the line. A
#: street polygon already contains the carriageway, so the same 30 m would
#: pin yard and car-park vehicles onto whatever street runs past. Measured
#: over the 125,882-detection city sweep (2026-08-14): 30.9% of detections
#: fall *inside* a street area, 32.9% within 2 m, 42.6% within 5 m and 78.2%
#: within 30 m. 2 m keeps the attribute meaning "on this street", allowing
#: only for kerb overhang and digitising slack; the 10 points between 2 m and
#: 5 m are off-street parking that would be mislabelled.
STREET_MAX_DISTANCE_M = 2.0

#: Otto also flagged avoindata:Pysakoinnin_maksuvyohykkeet_alue (paid parking
#: zones) and avoindata:Asukas_ja_yrityspysakointivyohykkeet_alue (resident /
#: business permit zones) as available on the same WFS. Both are zone-scale
#: polygons (2 and 16 features citywide, checked 2026-08-20) that cover the
#: carriageway too, so they cannot say anything about a single vehicle --
#: useful only as a per-area summary, not as a per-detection attribute.
LAYER_PARKING = "avoindata:Pysakointipaikat_alue"

#: Street *parts* -- the ajorata layer partitions each street right-of-way
#: into carriageway sections, on-street parking bays, driveways and kerb
#: build-outs (49,392 polygons, 96 MB raw, checked 2026-08-20). This is the
#: layer that catches the on-street parked vehicles Pysakointipaikat_alue
#: misses. The cache trims it to `alatyyppi` + geometry (fetch_layer's
#: ``keep``), a few MB instead of 96.
LAYER_STREET_PARTS = "avoindata:YLRE_Katu_ja_viherosat_ajorata_alue"
#: `street_part` values are the city's own YLRE terms, verbatim -- there is
#: no official English or Swedish variant in the WFS, and a translation
#: risks semantic drift (Tonttiliittymä is a plot *connection*, the strip
#: where a property meets the street, not a private driveway). Gloss for
#: readers: Ajorata = carriageway, Pysakointialue = on-street parking bay,
#: Tonttiliittymä = plot connection, Koroke = kerb build-out. "(Silta)"
#: marks the bridge-deck variant of each.
STREET_PART_COLUMN = "alatyyppi"

#: `context` values, in priority order. "parking" means *in a designated
#: parking place*, from whichever register says so: the regulated-parking
#: polygons (Pysakointipaikat_alue), a digitized on-street bay
#: (street-part alatyyppi Pysäköintialue), or a whole parking field
#: registered as a street area (kayttotarkoitus Pysäköintialue). All three
#: are needed -- at Mäntymäenkenttä (2026-08-20) the regulated polygons
#: cover only the paid sections, so 51 of 115 detections in one bay-covered
#: field read `street` until the other two sources counted. "other" covers
#: yards, plots and courtyards -- private residential lots are in no city
#: registry, so nothing finer is available for them.
CONTEXT_PARKING = "parking"
CONTEXT_STREET = "street"
CONTEXT_OTHER = "other"

#: The alatyyppi values that mark a digitized parking place. "(Silta)" is
#: the bridge-deck variant; prefix matching keeps any future variant in.
PARKING_PART_PREFIX = "Pysäköintialue"
#: The kayttotarkoitus marking a whole parking area registered as a street
#: area (255 of them citywide, measured 2026-08-20).
PARKING_STREET_TYPE = "Pysäköintialue"

# Deliberately NOT an attribute: distance-to-kerb as a parked/moving hint.
# Measured on the 2025 city sweep (8,000 on-carriageway cars, 2026-08-20):
# the median car centre sits 1.03 m from the nearest carriageway-part edge
# and 89.6% within 1.5 m, because the parts are narrow (per direction,
# around every build-out). The signal does not separate parked from moving.

CACHE_DIR = Path("data/wfs")


def _cache_path(layer: str, cache_root: Path) -> Path:
    return cache_root / f"{layer.split(':')[-1]}.geojson"


def fetch_layer(
    layer: str,
    *,
    cache_root: Path = CACHE_DIR,
    refresh: bool = False,
    keep: tuple[str, ...] | None = None,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """One WFS layer as a GeoJSON FeatureCollection, cached on disk.

    Mirrors osm.py's cache-under-data/ pattern: the pipeline should be
    re-runnable and testable offline, and administrative boundaries change
    rarely enough that re-fetching every run buys nothing. Most layers here
    are a few MB and fetch in seconds (measured 2026-08-14), so a cold cache
    is cheap -- that was not true of the 78 MB Liikennevaylat this module
    used to pull. ``cache_root`` is a parameter rather than always
    ``CACHE_DIR``, matching mine's separate ``--osm-cache`` next to its tile
    ``--cache``.

    ``keep`` trims every feature's properties to the named keys *before*
    the cache write, for layers whose payload is geometry plus one useful
    attribute -- the street-parts layer is 96 MB raw, of which the join
    below reads a single column. It applies only to what this call fetched;
    an already-cached full payload is returned as-is.

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

    if keep is not None:
        for feature in data["features"]:
            feature["properties"] = {
                key: value
                for key, value in feature["properties"].items()
                if key in keep
            }

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
    keep: tuple[str, ...] | None = None,
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
    import shapely.geometry

    geojson = fetch_layer(
        layer,
        cache_root=cache_root,
        refresh=refresh,
        keep=keep,
        client=client,
        transport=transport,
    )
    # from_features would do this, but it raises on the first bad feature,
    # and the source layers ship them: measured 2026-08-20, the street-area
    # layer carries 16 null geometries in 7,768 features and the
    # street-parts layer one malformed ring in 49,393. Neither may sink
    # the layer.
    properties, geometries = [], []
    for feature in geojson["features"]:
        raw = feature["geometry"]
        if raw is None:
            continue
        try:
            geometry = shapely.geometry.shape(raw)
        except TypeError, ValueError:  # malformed ring in the source data
            continue
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.is_empty:
            continue
        properties.append(feature["properties"])
        geometries.append(geometry)
    return geopandas.GeoDataFrame(properties, geometry=geometries, crs=GRID)


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
    """Read detections, add district/postal_code/street/context/street_part, write.

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

    layer_kwargs: dict[str, Any] = {
        "cache_root": cache_root,
        "refresh": refresh_wfs,
        "client": client,
        "transport": transport,
    }
    districts = _layer_frame(LAYER_DISTRICTS, **layer_kwargs)
    postal_areas = _layer_frame(LAYER_POSTAL_AREAS, **layer_kwargs)
    streets = _layer_frame(LAYER_STREETS, **layer_kwargs)
    parking = _layer_frame(LAYER_PARKING, **layer_kwargs)
    street_parts = _layer_frame(
        LAYER_STREET_PARTS, keep=(STREET_PART_COLUMN,), **layer_kwargs
    )

    result = detections
    result = _attach_area_attribute(result, districts, DISTRICT_NAME_COLUMN, "district")
    result = _attach_area_attribute(
        result, postal_areas, POSTAL_CODE_COLUMN, "postal_code"
    )
    result = _attach_nearest_street(
        result, streets, max_distance_m=street_max_distance_m
    )
    result = _attach_context(result, parking, streets, street_parts)

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
    """Left-join the nearest street area's name and type, within ``max_distance_m``.

    Street areas are polygons, so a detection on the carriageway matches at
    distance 0; the nearest-neighbour join (rather than plain ``within``)
    exists only to allow a small tolerance for kerb overhang and digitising
    slack. The distance cap is what keeps `street = None` meaningful: a
    vehicle in a yard or car park is not on a street, and saying so is more
    useful than pinning it to whatever street runs past.

    `street_type` is the register's own `kayttotarkoitus`, verbatim: the
    layer covers squares and pedestrian areas as well as streets, and the
    type is what tells those apart.
    """
    import geopandas

    centres = detections.assign(geometry=detections.geometry.centroid)
    joined = geopandas.sjoin_nearest(
        centres,
        streets[[STREET_NAME_COLUMN, STREET_TYPE_COLUMN, "geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="_street_distance_m",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    return detections.assign(
        street=joined[STREET_NAME_COLUMN].reindex(detections.index),
        street_type=joined[STREET_TYPE_COLUMN].reindex(detections.index),
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


def _centre_within_any(detections, areas):
    """True where a detection's centre falls inside any feature of `areas`.

    Same centre convention as _attach_area_attribute -- a vehicle parked
    half on the pavement belongs to the street its centre is on.
    """
    import geopandas

    centres = detections.assign(geometry=detections.geometry.centroid)
    joined = geopandas.sjoin(
        centres, areas[["geometry"]], how="left", predicate="within"
    )
    hit = ~joined["index_right"].isna()
    return (
        hit.groupby(level=0)
        .any()
        .reindex(detections.index, fill_value=False)
        .to_numpy()
    )


def _attach_context(detections, parking, streets, street_parts):
    """Add `context` and `street_part` columns.

    `context` says where the vehicle is, not what it is doing -- see the
    module docstring for why the `parked` boolean was retired, and the
    CONTEXT_PARKING comment for the three parking sources. `street_part`
    is filled independently of `context`: a detection in a regulated parking
    area that also sits on a digitized bay keeps both facts.
    """
    import geopandas

    centres = detections.assign(geometry=detections.geometry.centroid)
    joined = geopandas.sjoin(
        centres,
        street_parts[[STREET_PART_COLUMN, "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    street_part = joined[STREET_PART_COLUMN].reindex(detections.index)

    bay_parts = street_parts[
        street_parts[STREET_PART_COLUMN].str.startswith(PARKING_PART_PREFIX, na=False)
    ]
    parking_fields = streets[streets[STREET_TYPE_COLUMN] == PARKING_STREET_TYPE]
    parked = (
        _intersects_any(detections, parking)
        | _centre_within_any(detections, bay_parts)
        | _centre_within_any(detections, parking_fields)
    )
    on_street = _centre_within_any(detections, streets)
    context = [
        CONTEXT_PARKING if p else CONTEXT_STREET if s else CONTEXT_OTHER
        for p, s in zip(parked, on_street, strict=True)
    ]

    return detections.assign(context=context, street_part=street_part)
