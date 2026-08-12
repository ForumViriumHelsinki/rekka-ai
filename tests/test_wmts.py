"""Fetching and caching. No test here touches the network."""

from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from rekka_ai.imagery.layers import ORTHO_LAYERS, layer_for_year
from rekka_ai.imagery.tiles import Tile
from rekka_ai.imagery.wmts import (
    QUEUE_DEPTH,
    TileFetcher,
    cache_path,
    tile_params,
)

TILE = Tile(zoom=16, col=32661, row=53612)

#: Stands in for a tile in every test: the fetcher validates the JPEG SOI
#: marker, so fake payloads have to carry it.
JPEG = b"\xff\xd8jpeg-bytes"


Handler = Callable[[httpx.Request], httpx.Response]


def _fetcher(tmp_path: Path, handler: Handler, *, workers: int = 1) -> TileFetcher:
    """A fetcher whose requests are served by ``handler`` instead of the network."""
    return TileFetcher(
        tmp_path, workers=workers, transport=httpx.MockTransport(handler)
    )


def test_tile_params_carry_row_and_col_separately() -> None:
    params = tile_params("Ortoilmakuva_2025_5cm", TILE)
    assert params["TILEMATRIX"] == "ETRS-GK25:16"
    assert params["TILEROW"] == "53612"
    assert params["TILECOL"] == "32661"
    assert params["LAYER"] == "Ortoilmakuva_2025_5cm"


def test_cache_path_layout(tmp_path: Path) -> None:
    path = cache_path(tmp_path, "Ortoilmakuva_2025_5cm", TILE)
    assert path == tmp_path / "Ortoilmakuva_2025_5cm" / "16" / "32661" / "53612.jpg"


def test_fetch_writes_tile_then_serves_it_from_cache(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=JPEG)

    with _fetcher(tmp_path, handler) as fetcher:
        first = fetcher.fetch("Ortoilmakuva_2025_5cm", TILE)
        second = fetcher.fetch("Ortoilmakuva_2025_5cm", TILE)

    assert first.from_cache is False
    assert second.from_cache is True
    assert first.path.read_bytes() == JPEG
    assert calls == 1


def test_fetch_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=JPEG)

    with _fetcher(tmp_path, handler) as fetcher:
        result = fetcher.fetch("Ortoilmakuva_2025_5cm", TILE)

    assert list(result.path.parent.iterdir()) == [result.path]


def test_fetch_retries_transient_failures(tmp_path: Path) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=JPEG)

    with _fetcher(tmp_path, handler) as fetcher:
        result = fetcher.fetch("Ortoilmakuva_2025_5cm", TILE)

    assert attempts == 3
    assert result.path.read_bytes() == JPEG


def test_fetch_does_not_retry_client_errors(tmp_path: Path) -> None:
    """A 404 means a bad layer name or an out-of-range tile; retrying cannot help."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404)

    with _fetcher(tmp_path, handler) as fetcher, pytest.raises(httpx.HTTPStatusError):
        fetcher.fetch("Ortoilmakuva_nonexistent", TILE)

    assert attempts == 1


XML_ERROR = (
    b"<?xml version='1.0'?><ows:ExceptionReport><ows:Exception/></ows:ExceptionReport>"
)


def test_fetch_retries_an_xml_exception_report(tmp_path: Path) -> None:
    """A 200 with an XML exception report is not a tile and must not be cached.

    GeoServer answers some failures this way (observed: an OutOfMemoryError
    under load), so the status code alone cannot be trusted; such a response
    is usually transient and goes through the same retry path as a 5xx.
    """
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(200, content=XML_ERROR)
        return httpx.Response(200, content=JPEG)

    with _fetcher(tmp_path, handler) as fetcher:
        result = fetcher.fetch("Ortoilmakuva_2025_5cm", TILE)

    assert attempts == 3
    assert result.path.read_bytes() == JPEG


def test_fetch_all_does_not_cache_an_xml_exception_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistently invalid tiles are recorded as failures, never written."""
    # One attempt, no backoff: the retry behaviour itself is covered above.
    monkeypatch.setattr("rekka_ai.imagery.wmts.MAX_ATTEMPTS", 1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=XML_ERROR)

    with _fetcher(tmp_path, handler) as fetcher:
        results = list(fetcher.fetch_all("Ortoilmakuva_2025_5cm", [TILE]))
        failures = list(fetcher.failures)

    assert results == []
    assert [tile for tile, _ in failures] == [TILE]
    assert not cache_path(tmp_path, "Ortoilmakuva_2025_5cm", TILE).exists()


def test_fetch_all_returns_one_result_per_tile(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=JPEG)

    tiles = [Tile(zoom=16, col=32661 + i, row=53612) for i in range(5)]
    with _fetcher(tmp_path, handler, workers=4) as fetcher:
        results = list(fetcher.fetch_all("Ortoilmakuva_2025_5cm", tiles))

    # Set, not sequence: results are yielded as they land, so a slow tile does
    # not hold up the ones behind it and the order is whatever the pool
    # produces. Every tile must still come back exactly once.
    assert sorted((r.tile.col, r.tile.row) for r in results) == sorted(
        (t.col, t.row) for t in tiles
    )
    assert all(r.path.exists() for r in results)


def test_fetch_all_does_not_drain_a_lazy_tile_iterator(tmp_path: Path) -> None:
    """The tile iterator is consumed as it is served, not up front.

    ``tiles_covering`` is a generator so a city-wide sweep never materialises;
    submitting every future at once would defeat that.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=JPEG)

    taken = 0

    def lazy() -> Iterator[Tile]:
        nonlocal taken
        for i in range(200):
            taken += 1
            yield Tile(zoom=16, col=32661 + i, row=53612)

    with _fetcher(tmp_path, handler, workers=2) as fetcher:
        stream = fetcher.fetch_all("Ortoilmakuva_2025_5cm", lazy())
        next(stream)
        # Bounded by workers * QUEUE_DEPTH, with room for the in-flight batch.
        assert taken < 200
        assert taken <= 2 * QUEUE_DEPTH + 2
        assert sum(1 for _ in stream) == 199


def test_fetch_all_records_failures_and_carries_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad tile must not kill a sweep: it is recorded, not raised."""
    # One attempt, no backoff: the retry behaviour itself is covered above.
    monkeypatch.setattr("rekka_ai.imagery.wmts.MAX_ATTEMPTS", 1)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["TILECOL"] == "32662":
            return httpx.Response(500)
        return httpx.Response(200, content=JPEG)

    tiles = [Tile(zoom=16, col=32661 + i, row=53612) for i in range(3)]
    with _fetcher(tmp_path, handler, workers=2) as fetcher:
        results = list(fetcher.fetch_all("Ortoilmakuva_2025_5cm", tiles))
        failures = list(fetcher.failures)

    assert sorted(r.tile.col for r in results) == [32661, 32663]
    assert [tile.col for tile, _ in failures] == [32662]


def test_layer_for_year() -> None:
    assert layer_for_year(2025) == "Ortoilmakuva_2025_5cm"
    assert layer_for_year(2017) == "Ortoilmakuva_2017_8cm"  # finest of two 2017 layers


def test_missing_flight_year_lists_alternatives() -> None:
    assert 2022 not in ORTHO_LAYERS  # no flight that year
    with pytest.raises(ValueError, match="available: 2014"):
        layer_for_year(2022)


def test_missing_flight_year_lists_alternatives_only_once() -> None:
    assert layer_for_year(2021) == "Ortoilmakuva_2021_5cm"  # finest of two 2021 layers
