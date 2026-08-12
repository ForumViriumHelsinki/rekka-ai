"""Helsinki WFS enrichment: caching, joins, and the enrich() pipeline."""

import json
from pathlib import Path

import geopandas
import httpx
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from rekka_ai.enrich import (
    DISTRICT_NAME_COLUMN,
    LAYER_DISTRICTS,
    LAYER_PARKING,
    LAYER_POSTAL_AREAS,
    LAYER_STREETS,
    POSTAL_CODE_COLUMN,
    STREET_NAME_COLUMN,
    _attach_area_attribute,
    _attach_nearest_street,
    _cache_path,
    _intersects_any,
    _layer_frame,
    enrich,
    fetch_layer,
)
from rekka_ai.geo import GRID


def _feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def _polygon_feature(properties: dict, ring: list[tuple[float, float]]) -> dict:
    # Lists, not tuples: fetch_layer's cache round-trips through json.dumps /
    # json.loads, which turns tuples into lists -- building the expected
    # payload with lists too keeps the equality checks below meaningful
    # instead of failing on a JSON-vs-Python-tuple technicality.
    closed = [list(point) for point in [*ring, ring[0]]]
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Polygon", "coordinates": [closed]},
    }


def _line_feature(properties: dict, coords: list[tuple[float, float]]) -> dict:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "LineString",
            "coordinates": [list(point) for point in coords],
        },
    }


class _Transport(httpx.BaseTransport):
    """Return a fixed WFS payload -- mirrors test_osm.py's _Transport for Overpass."""

    def __init__(self, payload: dict, *, status: int = 200):
        self.payload = payload
        self.status = status
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(self.status, request=request, json=self.payload)


#: A 100x100 m square. Districts and postal areas are unrelated real-world
#: tilings, but the join helpers below don't care which polygon layer they
#: get, so one square fixture stands in for both in these tests.
SQUARE = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
SQUARE_RING = list(SQUARE.exterior.coords)[:-1]


# ---------------------------------------------------------------- fetch_layer


def test_fetch_layer_writes_and_reuses_cache(tmp_path: Path) -> None:
    payload = _feature_collection(
        [_polygon_feature({DISTRICT_NAME_COLUMN: "MEILAHTI"}, SQUARE_RING)]
    )
    transport = _Transport(payload)
    first = fetch_layer(LAYER_DISTRICTS, cache_root=tmp_path, transport=transport)
    assert transport.calls == 1
    assert first == payload
    assert _cache_path(LAYER_DISTRICTS, tmp_path).exists()

    second = fetch_layer(LAYER_DISTRICTS, cache_root=tmp_path, transport=transport)
    assert transport.calls == 1  # cache hit, no second request
    assert second == payload


def test_fetch_layer_refresh_bypasses_cache(tmp_path: Path) -> None:
    transport = _Transport(_feature_collection([]))
    fetch_layer(LAYER_DISTRICTS, cache_root=tmp_path, transport=transport)
    fetch_layer(LAYER_DISTRICTS, cache_root=tmp_path, refresh=True, transport=transport)
    assert transport.calls == 2


def test_fetch_layer_write_is_atomic(tmp_path: Path) -> None:
    """No leftover .part file, and the cache holds valid JSON, after a fetch."""
    payload = _feature_collection([])
    fetch_layer(LAYER_DISTRICTS, cache_root=tmp_path, transport=_Transport(payload))
    path = _cache_path(LAYER_DISTRICTS, tmp_path)
    assert path.exists()
    assert not path.with_suffix(".geojson.part").exists()
    assert json.loads(path.read_text()) == payload


def test_fetch_layer_error_is_not_retried(tmp_path: Path) -> None:
    """fetch_layer has no retry loop (Helsinki's own geoserver WFS isn't
    Overpass's flaky public round-robin, so it doesn't get the same
    resilience as fetch_industrial) -- one failed request is one failed
    request, even for a transient-looking 503."""
    transport = _Transport({}, status=503)
    with pytest.raises(httpx.HTTPStatusError):
        fetch_layer(LAYER_DISTRICTS, cache_root=tmp_path, transport=transport)
    assert transport.calls == 1


# ---------------------------------------------------------------- _layer_frame


def test_layer_frame_has_the_project_grid_crs(tmp_path: Path) -> None:
    payload = _feature_collection(
        [_polygon_feature({DISTRICT_NAME_COLUMN: "MEILAHTI"}, SQUARE_RING)]
    )
    frame = _layer_frame(
        LAYER_DISTRICTS,
        cache_root=tmp_path,
        refresh=False,
        transport=_Transport(payload),
    )
    assert frame.crs == GRID
    assert frame.iloc[0][DISTRICT_NAME_COLUMN] == "MEILAHTI"


# ---------------------------------------------------------------- join helpers


def test_attach_area_attribute_matches_by_centroid_within_the_polygon() -> None:
    areas = geopandas.GeoDataFrame(
        [{DISTRICT_NAME_COLUMN: "MEILAHTI", "geometry": SQUARE}], crs=GRID
    )
    detections = geopandas.GeoDataFrame(
        [
            {"label": "truck", "geometry": Point(50, 50).buffer(1)},  # inside
            {"label": "truck", "geometry": Point(500, 500).buffer(1)},  # outside
        ],
        crs=GRID,
    )
    result = _attach_area_attribute(detections, areas, DISTRICT_NAME_COLUMN, "district")
    assert result["district"].iloc[0] == "MEILAHTI"
    assert pd.isna(result["district"].iloc[1])


def test_attach_nearest_street_respects_max_distance() -> None:
    streets = geopandas.GeoDataFrame(
        [
            {
                STREET_NAME_COLUMN: "Testikatu",
                "geometry": LineString([(0, 50), (100, 50)]),
            }
        ],
        crs=GRID,
    )
    detections = geopandas.GeoDataFrame(
        [
            {"label": "truck", "geometry": Point(50, 55).buffer(1)},  # 5 m away
            {"label": "truck", "geometry": Point(50, 500).buffer(1)},  # far away
        ],
        crs=GRID,
    )
    result = _attach_nearest_street(detections, streets, max_distance_m=30.0)
    assert result["street"].iloc[0] == "Testikatu"
    assert pd.isna(result["street"].iloc[1])


def test_intersects_any_flags_only_overlapping_detections() -> None:
    parking = geopandas.GeoDataFrame(
        [{"geometry": Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])}], crs=GRID
    )
    detections = geopandas.GeoDataFrame(
        [
            {"label": "truck", "geometry": Point(25, 50).buffer(1)},  # on the lot
            {"label": "truck", "geometry": Point(75, 50).buffer(1)},  # off the lot
        ],
        crs=GRID,
    )
    assert list(_intersects_any(detections, parking)) == [True, False]


# ---------------------------------------------------------------- enrich()


def test_enrich_end_to_end_with_a_prepopulated_cache(tmp_path: Path) -> None:
    """A full run against a pre-populated WFS cache -- no network, no
    trained model, nothing needed but the fixtures below and a tiny
    detections file. Only the columns enrich() adds are checked; the real
    property schema detections.write() produces (length_m, width_m, ...) is
    irrelevant to enrich() and untouched by it either way, so it is not
    reproduced here.

    The input file is written as GeoPackage, not GeoJSON: GDAL's GeoJSON
    driver defaults to RFC 7946 and silently reprojects non-WGS84 input to
    WGS84 on write, which would turn this fixture's EPSG:3879 metres into
    degrees -- exactly the round trip geo.py's crs_member() exists to avoid
    for this project's own files. GeoPackage has no such default.
    """
    cache_root = tmp_path / "wfs"
    cache_root.mkdir()
    fixtures = {
        LAYER_DISTRICTS: _feature_collection(
            [_polygon_feature({DISTRICT_NAME_COLUMN: "MEILAHTI"}, SQUARE_RING)]
        ),
        LAYER_POSTAL_AREAS: _feature_collection(
            [_polygon_feature({POSTAL_CODE_COLUMN: "00250"}, SQUARE_RING)]
        ),
        LAYER_STREETS: _feature_collection(
            [_line_feature({STREET_NAME_COLUMN: "Testikatu"}, [(0, 50), (100, 50)])]
        ),
        LAYER_PARKING: _feature_collection(
            [_polygon_feature({}, [(0, 0), (50, 0), (50, 100), (0, 100)])]
        ),
    }
    for layer, payload in fixtures.items():
        _cache_path(layer, cache_root).write_text(json.dumps(payload))

    input_path = tmp_path / "detections.gpkg"
    output_path = tmp_path / "enriched.gpkg"
    geopandas.GeoDataFrame(
        [
            # Inside the square, on the parking lot, 5 m from the street.
            {"label": "truck", "geometry": Point(25, 55).buffer(1)},
            # Nowhere near any of the four fixtures.
            {"label": "truck", "geometry": Point(500, 500).buffer(1)},
        ],
        crs=GRID,
    ).to_file(input_path)

    enrich(input_path, output_path, cache_root=cache_root)

    result = geopandas.read_file(output_path)
    assert result["district"].iloc[0] == "MEILAHTI"
    assert pd.isna(result["district"].iloc[1])
    assert result["postal_code"].iloc[0] == "00250"
    assert pd.isna(result["postal_code"].iloc[1])
    assert result["street"].iloc[0] == "Testikatu"
    assert pd.isna(result["street"].iloc[1])
    assert list(result["parked"]) == [True, False]
