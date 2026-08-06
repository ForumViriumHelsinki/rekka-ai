"""Coordinate reference systems and transforms.

Three CRSs, with distinct jobs, kept deliberately separate:

- **Input** is whatever the source uses. Finnish geodata is normally
  EPSG:3067 (ETRS-TM35FIN), the national standard.
- **EPSG:3879** (ETRS89 / GK25FIN) is used internally for all tile arithmetic,
  not by preference but because it is the CRS of the Helsinki tile grid.
- **WGS84** is used for output, and for bboxes a person typed by hand.

Only the middle one is forced on us, so nothing outside ``imagery`` should
assume it.
"""

from functools import cache

from pyproj import Transformer
from pyproj.exceptions import CRSError

#: The CRS of the ``ETRS-GK25`` tile grid. Internal to tile arithmetic.
GRID = "EPSG:3879"
#: ETRS-TM35FIN, the Finnish national standard and the usual CRS for AOI config.
TM35FIN = "EPSG:3067"
WGS84 = "EPSG:4326"


@cache
def transformer(src: str, dst: str) -> Transformer:
    """A cached ``src`` -> ``dst`` transformer, always (x, y) ordered.

    always_xy keeps every transform (x, y) = (easting, northing) regardless of
    the CRS's declared axis order. This matters here: EPSG:3067 declares
    (east, north) but EPSG:3879 declares (north, east), so without it one of
    the two comes back swapped -- silently, and at a plausible magnitude.
    """
    try:
        return Transformer.from_crs(src, dst, always_xy=True)
    except CRSError as exc:
        raise ValueError(f"unknown CRS {src!r} or {dst!r}: {exc}") from exc


def to_grid(x: float, y: float, src: str = WGS84) -> tuple[float, float]:
    """Coordinates in ``src`` to the tile grid's EPSG:3879 easting/northing."""
    return transformer(src, GRID).transform(x, y)


def to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """EPSG:3879 easting/northing to WGS84 lon/lat."""
    return transformer(GRID, WGS84).transform(easting, northing)
