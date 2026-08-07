"""Overpass industrial-geometry source: query, cache, and parsing."""

from pathlib import Path

import httpx
import pytest

from rekka_ai.osm import (
    MUNICIPALITIES,
    build_query,
    cache_key,
    cache_path,
    fetch_industrial,
    load_cached,
    query_hash,
    require_profile,
    require_supported_municipality,
    union_wgs84,
)


def _way(osm_id: int, coords: list[tuple[float, float]]) -> dict:
    # Overpass `out geom` closes rings by repeating the first node.
    nodes = [{"lat": lat, "lon": lon} for lon, lat in coords]
    if nodes[0] != nodes[-1]:
        nodes.append(nodes[0])
    return {
        "type": "way",
        "id": osm_id,
        "tags": {"landuse": "industrial"},
        "geometry": nodes,
    }


#: A small industrial rectangle inside Helsinki, in WGS84 lon/lat.
HELSINKI_INDUSTRIAL = [
    (24.95, 60.20),
    (24.96, 60.20),
    (24.96, 60.21),
    (24.95, 60.21),
]


class _Transport(httpx.BaseTransport):
    """Return a fixed Overpass payload, optionally failing a few times first."""

    def __init__(self, payload: dict, *, fail_times: int = 0, status: int = 200):
        self.payload = payload
        self.fail_times = fail_times
        self.status = status
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.fail_times:
            return httpx.Response(503, request=request, text="busy")
        return httpx.Response(self.status, request=request, json=self.payload)


def test_helsinki_query_pins_municipal_ref_091() -> None:
    query = build_query("Helsinki")
    assert 'area["ref"="091"]["admin_level"="8"]' in query
    assert '["landuse"="industrial"]' in query
    assert "out geom" in query
    assert query_hash(query) == query_hash(build_query("Helsinki"))


def test_espoo_and_vantaa_are_known_but_not_yet_supported() -> None:
    assert MUNICIPALITIES["Espoo"] == "049"
    assert MUNICIPALITIES["Vantaa"] == "092"
    with pytest.raises(ValueError, match="not supported yet"):
        require_supported_municipality("Espoo")
    with pytest.raises(ValueError, match="not supported yet"):
        build_query("Espoo")


def test_unknown_municipality_and_profile_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown municipality"):
        require_supported_municipality("Turku")
    with pytest.raises(ValueError, match="unknown profile"):
        require_profile("marina")


def test_cache_key_is_stable() -> None:
    assert cache_key("Helsinki") == "091_helsinki_industrial"


def test_fetch_writes_and_reuses_cache(tmp_path: Path) -> None:
    payload = {"elements": [_way(1, HELSINKI_INDUSTRIAL)]}
    transport = _Transport(payload)
    first = fetch_industrial("Helsinki", cache_root=tmp_path, transport=transport)
    assert transport.calls == 1
    assert first.path == cache_path(tmp_path, "Helsinki")
    assert first.path.exists()
    assert len(first.features) == 1
    assert first.attribution.startswith("© OpenStreetMap")

    second = fetch_industrial("Helsinki", cache_root=tmp_path, transport=transport)
    assert transport.calls == 1  # cache hit, no second request
    assert second.query_hash == first.query_hash
    assert second.features[0].osm_id == 1


def test_refresh_bypasses_cache(tmp_path: Path) -> None:
    transport = _Transport({"elements": [_way(1, HELSINKI_INDUSTRIAL)]})
    fetch_industrial("Helsinki", cache_root=tmp_path, transport=transport)
    fetch_industrial("Helsinki", cache_root=tmp_path, refresh=True, transport=transport)
    assert transport.calls == 2


def test_retries_transient_overpass_failures(tmp_path: Path) -> None:
    transport = _Transport({"elements": [_way(1, HELSINKI_INDUSTRIAL)]}, fail_times=2)
    cached = fetch_industrial("Helsinki", cache_root=tmp_path, transport=transport)
    assert transport.calls == 3
    assert len(cached.features) == 1


def test_malformed_geometry_is_skipped(tmp_path: Path) -> None:
    payload = {
        "elements": [
            _way(1, HELSINKI_INDUSTRIAL),
            {"type": "way", "id": 2, "tags": {"landuse": "industrial"}, "geometry": []},
            {
                "type": "way",
                "id": 3,
                "tags": {"landuse": "industrial"},
                "geometry": [{"lat": 60.2, "lon": 24.95}],  # too few nodes
            },
            {"type": "node", "id": 4, "lat": 60.2, "lon": 24.95},
        ]
    }
    cached = fetch_industrial(
        "Helsinki", cache_root=tmp_path, transport=_Transport(payload)
    )
    assert [f.osm_id for f in cached.features] == [1]


def test_load_cached_round_trips_metadata(tmp_path: Path) -> None:
    payload = {"elements": [_way(7, HELSINKI_INDUSTRIAL)]}
    written = fetch_industrial(
        "Helsinki", cache_root=tmp_path, transport=_Transport(payload)
    )
    reloaded = load_cached(tmp_path, "Helsinki")
    assert reloaded is not None
    assert reloaded.query_hash == written.query_hash
    assert reloaded.municipality == "Helsinki"
    assert reloaded.ref == "091"
    assert not union_wgs84(reloaded.features).is_empty


def test_client_4xx_is_not_retried(tmp_path: Path) -> None:
    transport = _Transport({"elements": []}, status=400)
    with pytest.raises(httpx.HTTPStatusError):
        fetch_industrial("Helsinki", cache_root=tmp_path, transport=transport)
    assert transport.calls == 1
