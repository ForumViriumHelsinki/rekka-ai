"""Model-sized windows over the tile grid, and the pixel/ground mapping.

A 256 px tile is too small to detect on and cuts trucks in half, so tiles are
stitched into larger overlapping windows. Overlap is expressed in **metres**
rather than pixels: it has to exceed the longest object we expect to see, and
that is a fact about trucks, not about the zoom level.

The whole grid is one global pixel plane per zoom, related to EPSG:3879 by a
pure scale and translation:

    px = (E - ORIGIN_EASTING)  / resolution
    py = (ORIGIN_NORTHING - N) / resolution

Because that mapping has no rotation, an oriented box detected in pixels stays
a rectangle on the ground, and georeferencing a detection is exact arithmetic
rather than a warp.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from rekka_ai.imagery.tiles import (
    ORIGIN_EASTING,
    ORIGIN_NORTHING,
    TILE_SIZE,
    Bounds,
    Tile,
    resolution,
)
from rekka_ai.imagery.wmts import cache_path

#: Ultralytics OBB weights are trained at 1024 px; matching that avoids an
#: internal resize.
WINDOW_SIZE = 1024

#: Longer than any road-legal truck (a 25.25 m HCT rig is the Finnish maximum),
#: so every vehicle appears whole in at least one window.
OVERLAP_M = 30.0


def grid_to_pixel(easting: float, northing: float, zoom: int) -> tuple[float, float]:
    """EPSG:3879 to global pixel coordinates at ``zoom``."""
    res = resolution(zoom)
    return (easting - ORIGIN_EASTING) / res, (ORIGIN_NORTHING - northing) / res


def pixel_to_grid(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Global pixel coordinates at ``zoom`` to EPSG:3879."""
    res = resolution(zoom)
    return ORIGIN_EASTING + x * res, ORIGIN_NORTHING - y * res


@dataclass(frozen=True, slots=True)
class Window:
    """A square image window, positioned in the global pixel plane."""

    zoom: int
    x: int
    y: int
    size: int = WINDOW_SIZE

    def tiles(self) -> list[Tile]:
        """Every tile needed to fill this window."""
        first_col, last_col = self.x // TILE_SIZE, (self.x + self.size - 1) // TILE_SIZE
        first_row, last_row = self.y // TILE_SIZE, (self.y + self.size - 1) // TILE_SIZE
        return [
            Tile(zoom=self.zoom, col=col, row=row)
            for row in range(first_row, last_row + 1)
            for col in range(first_col, last_col + 1)
        ]

    def to_grid(self, x: float, y: float) -> tuple[float, float]:
        """Window-local pixel coordinates to EPSG:3879."""
        return pixel_to_grid(self.x + x, self.y + y, self.zoom)

    def bounds(self) -> Bounds:
        min_easting, max_northing = pixel_to_grid(self.x, self.y, self.zoom)
        max_easting, min_northing = pixel_to_grid(
            self.x + self.size, self.y + self.size, self.zoom
        )
        return Bounds(
            min_easting=min_easting,
            min_northing=min_northing,
            max_easting=max_easting,
            max_northing=max_northing,
        )


def windows_covering(
    bounds: Bounds,
    zoom: int,
    *,
    size: int = WINDOW_SIZE,
    overlap_m: float = OVERLAP_M,
) -> Iterator[Window]:
    """Overlapping windows covering ``bounds``, in row-major order."""
    overlap = math.ceil(overlap_m / resolution(zoom))
    if overlap >= size:
        raise ValueError(
            f"overlap of {overlap_m} m is {overlap} px, not less than {size}"
        )

    left, top = grid_to_pixel(bounds.min_easting, bounds.max_northing, zoom)
    right, bottom = grid_to_pixel(bounds.max_easting, bounds.min_northing, zoom)

    for y in _starts(math.floor(top), math.ceil(bottom), size, size - overlap):
        for x in _starts(math.floor(left), math.ceil(right), size, size - overlap):
            yield Window(zoom=zoom, x=x, y=y, size=size)


def _starts(low: int, high: int, size: int, step: int) -> list[int]:
    """Window origins covering ``low``..``high``, the last one flush to the end.

    Nudging the final window back to end exactly at ``high`` keeps coverage
    complete without emitting a mostly-empty extra window; it only increases
    overlap, which is harmless because duplicates are merged afterwards.
    """
    if high - low <= size:
        return [low]
    starts = list(range(low, high - size + 1, step))
    if starts[-1] + size < high:
        starts.append(high - size)
    return starts


#: Decoded tiles held between windows. Windows overlap by design, so
#: neighbouring ones share a column or row of tiles and would otherwise decode
#: the same JPEG several times; at the default overlap each tile is wanted by
#: about four windows. One window's worth of tiles is 25 (a 5x5 span at
#: 1024 px), so a few windows of history is a megabyte or two of pixels.
TILE_CACHE_SIZE = 128


@lru_cache(maxsize=TILE_CACHE_SIZE)
def _decoded_tile(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_window(window: Window, cache_root: Path, layer: str) -> Image.Image:
    """Stitch cached tiles into one window image.

    Tiles must already be cached; ``TileFetcher`` is cache-first, so the caller
    fetches the window's tiles beforehand and this stays pure disk I/O.
    """
    canvas = Image.new("RGB", (window.size, window.size))
    for tile in window.tiles():
        path = cache_path(cache_root, layer, tile)
        if not path.exists():
            raise FileNotFoundError(f"tile not cached: {path}")
        canvas.paste(
            _decoded_tile(path),
            (tile.col * TILE_SIZE - window.x, tile.row * TILE_SIZE - window.y),
        )
    return canvas
