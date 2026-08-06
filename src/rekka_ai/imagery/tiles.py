"""Tile geometry for the Helsinki ``ETRS-GK25`` WMTS grid.

Pure arithmetic, no I/O. Every value here was read from the service's
GetCapabilities document; see ``DESIGN.md`` for the provenance.

The grid is a plain power-of-two pyramid over EPSG:3879, anchored at a
top-left origin, so a tile index is exact arithmetic rather than a warp.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass

TILE_SIZE = 256
MAX_ZOOM = 17

#: Top-left corner of the grid, EPSG:3879. GetCapabilities reports this as
#: "northing easting" because EPSG:3879 declares (north, east) axis order.
ORIGIN_NORTHING = 8388608.0
ORIGIN_EASTING = 24451424.0

#: Ground resolution at zoom 0, metres per pixel. Halves at every level.
BASE_RESOLUTION = 8192.0


def resolution(zoom: int) -> float:
    """Ground resolution in metres per pixel at ``zoom``."""
    _check_zoom(zoom)
    return BASE_RESOLUTION / (1 << zoom)


def tile_span(zoom: int) -> float:
    """Ground width of one tile in metres at ``zoom``."""
    return resolution(zoom) * TILE_SIZE


def matrix_size(zoom: int) -> int:
    """Number of tiles per axis at ``zoom``."""
    _check_zoom(zoom)
    return 1 << zoom


def _check_zoom(zoom: int) -> None:
    if not 0 <= zoom <= MAX_ZOOM:
        raise ValueError(f"zoom {zoom} outside grid range 0..{MAX_ZOOM}")


@dataclass(frozen=True, slots=True)
class Bounds:
    """An axis-aligned extent in EPSG:3879 metres."""

    min_easting: float
    min_northing: float
    max_easting: float
    max_northing: float

    def __post_init__(self) -> None:
        corners = (
            self.min_easting,
            self.min_northing,
            self.max_easting,
            self.max_northing,
        )
        if not all(math.isfinite(v) for v in corners):
            # pyproj returns infinities rather than raising when a coordinate
            # falls outside the source CRS's domain, so this is what reading
            # projected metres as WGS84 degrees (or vice versa) looks like.
            raise ValueError(
                f"non-finite bounds {corners}; the coordinates were probably "
                "given in a different CRS than the one declared"
            )
        if self.min_easting > self.max_easting or self.min_northing > self.max_northing:
            raise ValueError(f"degenerate bounds: {self}")

    @property
    def width(self) -> float:
        return self.max_easting - self.min_easting

    @property
    def height(self) -> float:
        return self.max_northing - self.min_northing


@dataclass(frozen=True, slots=True)
class Tile:
    """One tile of the ``ETRS-GK25`` grid.

    ``row`` counts south from the grid origin and ``col`` counts east, matching
    the WMTS ``TILEROW`` / ``TILECOL`` request parameters.
    """

    zoom: int
    col: int
    row: int

    def __post_init__(self) -> None:
        limit = matrix_size(self.zoom)
        if not 0 <= self.col < limit or not 0 <= self.row < limit:
            raise ValueError(
                f"tile ({self.col}, {self.row}) outside {limit}x{limit} matrix at z{self.zoom}"
            )

    def bounds(self) -> Bounds:
        """Ground extent of this tile in EPSG:3879."""
        span = tile_span(self.zoom)
        min_easting = ORIGIN_EASTING + self.col * span
        max_northing = ORIGIN_NORTHING - self.row * span
        return Bounds(
            min_easting=min_easting,
            min_northing=max_northing - span,
            max_easting=min_easting + span,
            max_northing=max_northing,
        )


def tile_at(easting: float, northing: float, zoom: int) -> Tile:
    """The tile containing a projected coordinate."""
    span = tile_span(zoom)
    col = int((easting - ORIGIN_EASTING) // span)
    row = int((ORIGIN_NORTHING - northing) // span)
    return Tile(zoom=zoom, col=col, row=row)


def _tile_range(bounds: Bounds, zoom: int) -> tuple[int, int, int, int]:
    """Inclusive ``(first_col, last_col, first_row, last_row)`` covering ``bounds``.

    The extent's upper edges are treated as exclusive, so an AOI that lines up
    exactly with tile boundaries does not drag in a further row and column that
    it only touches. Zero-area extents still resolve to the single tile that
    contains them.
    """
    span = tile_span(zoom)
    first_col = math.floor((bounds.min_easting - ORIGIN_EASTING) / span)
    first_row = math.floor((ORIGIN_NORTHING - bounds.max_northing) / span)
    last_col = max(
        first_col, math.ceil((bounds.max_easting - ORIGIN_EASTING) / span) - 1
    )
    last_row = max(
        first_row, math.ceil((ORIGIN_NORTHING - bounds.min_northing) / span) - 1
    )
    return first_col, last_col, first_row, last_row


def tiles_covering(bounds: Bounds, zoom: int) -> Iterator[Tile]:
    """Every tile overlapping ``bounds``, in row-major order.

    Yields lazily: a city-wide AOI at z16 is a large number of tiles, and
    callers stream them into the fetcher rather than materialising a list.
    """
    first_col, last_col, first_row, last_row = _tile_range(bounds, zoom)
    for row in range(first_row, last_row + 1):
        for col in range(first_col, last_col + 1):
            yield Tile(zoom=zoom, col=col, row=row)


def count_tiles(bounds: Bounds, zoom: int) -> int:
    """How many tiles ``tiles_covering`` would yield, without building them."""
    first_col, last_col, first_row, last_row = _tile_range(bounds, zoom)
    return (last_col - first_col + 1) * (last_row - first_row + 1)
