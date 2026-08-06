"""Fetching and caching tiles from the Helsinki orthophoto WMTS.

Tiles for a past flight year never change, so the cache needs no invalidation
and a re-run costs nothing. That is what makes iterating on detection tolerable,
so caching is not an optimisation to add later -- it is the point of this
module.
"""

import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import httpx

from rekka_ai.imagery.tiles import Tile

ENDPOINT = "https://kartta.hel.fi/ws/geoserver/avoindata/gwc/service/wmts"
TILE_MATRIX_SET = "ETRS-GK25"
IMAGE_FORMAT = "image/jpeg"
SUFFIX = ".jpg"

#: Identifies us to Kaupunkimittauspalvelut, so a heavy sweep is attributable.
USER_AGENT = "rekka-ai/0.1 (+https://github.com/ForumViriumHelsinki/rekka-ai)"

#: The service publishes no rate limit and no access constraints. A modest
#: concurrency cap is the polite way to stay well inside whatever it is.
DEFAULT_WORKERS = 8
MAX_ATTEMPTS = 3

#: How many requests may be in flight per worker. Bounded so that a lazy tile
#: iterator over a city-wide area is consumed as it is served rather than
#: drained into memory up front.
QUEUE_DEPTH = 4

#: Longest backoff to honour from a ``Retry-After`` header, in seconds. A
#: service asking for more than this is asking us to stop, not to wait.
MAX_RETRY_AFTER = 30.0


class TileSource(Protocol):
    """Anything that can make tiles available on disk.

    Narrower than ``TileFetcher`` on purpose: a caller that only needs tiles to
    exist should not depend on the network, which keeps those callers testable.
    """

    def fetch_all(self, layer: str, tiles: Iterable[Tile]) -> Iterable[object]: ...


def ensure_cached(fetcher: TileSource, layer: str, tiles: Iterable[Tile]) -> None:
    """Make sure every tile is on disk, discarding the results.

    ``fetch_all`` is a generator, so it does nothing until consumed. Callers
    that want the cache populated rather than the results say so by name here.
    """
    for _ in fetcher.fetch_all(layer, tiles):
        pass


def tile_params(layer: str, tile: Tile) -> dict[str, str]:
    """WMTS GetTile query parameters for one tile."""
    return {
        "SERVICE": "WMTS",
        "VERSION": "1.0.0",
        "REQUEST": "GetTile",
        "LAYER": layer,
        "STYLE": "raster",
        "TILEMATRIXSET": TILE_MATRIX_SET,
        "TILEMATRIX": f"{TILE_MATRIX_SET}:{tile.zoom}",
        "TILEROW": str(tile.row),
        "TILECOL": str(tile.col),
        "FORMAT": IMAGE_FORMAT,
    }


def cache_path(root: Path, layer: str, tile: Tile) -> Path:
    """Where a tile lives on disk.

    Laid out ``layer/zoom/col/row.jpg`` so that a directory listing stays small
    and the tree is browsable by hand when a detection looks wrong.
    """
    return root / layer / str(tile.zoom) / str(tile.col) / f"{tile.row}{SUFFIX}"


@dataclass(frozen=True, slots=True)
class FetchResult:
    tile: Tile
    path: Path
    from_cache: bool


class TileFetcher:
    """Fetches tiles into an on-disk cache.

    Use as a context manager so the underlying connection pool is reused across
    a sweep and closed afterwards.
    """

    def __init__(
        self,
        cache_root: Path,
        *,
        workers: int = DEFAULT_WORKERS,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.workers = workers
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            limits=httpx.Limits(max_connections=workers),
            transport=transport,
        )
        # The pool lives as long as the client it protects: spawning one per
        # fetch_all call means one pool per detection window.
        self._pool = ThreadPoolExecutor(max_workers=workers)
        #: Tiles that failed after every retry, as ``(tile, cause)``. Collected
        #: rather than raised so one bad tile does not kill a whole sweep.
        self.failures: list[tuple[Tile, Exception]] = []
        # Appended to from pool threads. list.append is atomic under the GIL,
        # but that is not a guarantee on a free-threaded build, and this is a
        # 3.14 project.
        self._failures_lock = threading.Lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._client.close()

    def fetch(self, layer: str, tile: Tile) -> FetchResult:
        """Fetch one tile, returning immediately if it is already cached."""
        path = cache_path(self.cache_root, layer, tile)
        if path.exists():
            return FetchResult(tile=tile, path=path, from_cache=True)

        payload = self._get(layer, tile)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write via a temporary sibling so an interrupted run cannot leave a
        # truncated tile behind that later looks like a valid cache hit.
        tmp = path.with_suffix(f"{SUFFIX}.part")
        tmp.write_bytes(payload)
        tmp.replace(path)
        return FetchResult(tile=tile, path=path, from_cache=False)

    def fetch_all(self, layer: str, tiles: Iterable[Tile]) -> Iterator[FetchResult]:
        """Fetch many tiles concurrently, yielding results as they land.

        A tile that fails after every retry is recorded in ``failures`` and
        skipped, not raised: a long sweep should finish and report what it
        missed rather than die on one bad tile.

        Requests are submitted in a bounded window rather than all at once:
        ``tiles`` is a lazy iterator by design (see ``tiles_covering``), and
        draining it up front would materialise a city-wide sweep in memory and
        serialise the yields behind the first slow tile.
        """
        remaining = iter(tiles)
        pending: set[Future[FetchResult | None]] = set()
        limit = max(1, self.workers * QUEUE_DEPTH)
        while True:
            while len(pending) < limit:
                tile = next(remaining, None)
                if tile is None:
                    break
                pending.add(self._pool.submit(self._attempt, layer, tile))
            if not pending:
                return
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                if result is not None:
                    yield result

    def _attempt(self, layer: str, tile: Tile) -> FetchResult | None:
        try:
            return self.fetch(layer, tile)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            with self._failures_lock:
                self.failures.append((tile, exc))
            return None

    def _get(self, layer: str, tile: Tile) -> bytes:
        params = tile_params(layer, tile)
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.get(ENDPOINT, params=params)
                response.raise_for_status()
                return response.content
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                # A 4xx means the request itself is wrong -- retrying will not
                # fix it, and it usually points at a bad layer name or a tile
                # outside the layer's published limits. 429 is the exception:
                # it means "slow down", so failing the tile is the one response
                # that is certainly wrong.
                if status is not None and 400 <= status < 500 and status != 429:
                    raise
                last = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(_backoff(attempt, response))
        raise RuntimeError(
            f"failed to fetch {tile} of {layer} after {MAX_ATTEMPTS} attempts"
        ) from last


def _backoff(attempt: int, response: object | None) -> float:
    """Seconds to wait before the next attempt.

    Exponential, unless the service named a delay itself: a ``Retry-After``
    is the one number that is not a guess, so it wins when it is present and
    sane.
    """
    exponential = float(2**attempt)
    headers = getattr(response, "headers", None)
    raw = headers.get("Retry-After") if headers is not None else None
    if raw is None:
        return exponential
    try:
        # The HTTP-date form of Retry-After is legal but not worth parsing for
        # a tile server; fall back to the exponential delay when it is not a
        # plain number of seconds.
        requested = float(raw)
    except ValueError:
        return exponential
    return min(max(requested, exponential), MAX_RETRY_AFTER)
