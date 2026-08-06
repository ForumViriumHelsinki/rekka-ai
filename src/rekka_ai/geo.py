"""Coordinate reference systems and transforms.

Three CRSs, with distinct jobs, kept deliberately separate:

- **Input** is whatever the source uses. Finnish geodata is normally
  EPSG:3067 (ETRS-TM35FIN), the national standard.
- **EPSG:3879** (ETRS89 / GK25FIN) is the CRS of the Helsinki tile grid, so it
  is what all tile arithmetic and geometry maths uses -- and, since the whole
  pipeline already works in it, what the GeoJSON this project writes stores.
- **WGS84** is for bboxes a person typed by hand, and for handing geometry to
  something outside this project.

Storing EPSG:3879 costs a non-standard ``crs`` member (see ``crs_member``) and
buys the absence of a WGS84 round trip between the detector, the label files
and the labelling tool -- a round trip that rewrote every coordinate in a file
each time an operator changed one box's class.
"""

from functools import cache
from typing import Any

from pyproj import Transformer
from pyproj.exceptions import CRSError

#: The CRS of the ``ETRS-GK25`` tile grid, and of every GeoJSON written here.
GRID = "EPSG:3879"
#: ETRS-TM35FIN, the Finnish national standard and the usual CRS for AOI config.
TM35FIN = "EPSG:3067"
WGS84 = "EPSG:4326"

#: Decimal places kept for a stored coordinate: millimetres. Far finer than
#: 5 cm imagery, and short enough that Python's ``repr`` and JavaScript's
#: number formatting print the same digits -- which is what keeps a file
#: byte-identical when the CLI and the web tool take turns writing it.
COORD_DECIMALS = 3

#: How ``crs_member`` names the grid. GDAL resolves this to EPSG:3879 with
#: easting first, despite the CRS declaring (north, east) -- verified, because
#: getting it backwards would put every label 19 000 km away.
GRID_CRS_NAME = "urn:ogc:def:crs:EPSG::3879"


def crs_member() -> dict[str, Any]:
    """The ``crs`` member naming the grid, for the head of a FeatureCollection.

    RFC 7946 dropped ``crs`` and mandates WGS84; this project deviates, so the
    member is what tells a reader which is which. GDAL still honours it, which
    is what makes these files open correctly in QGIS -- and a file *without* it
    is by definition a WGS84 one, which is how ``labels.read`` catches a file
    written before the switch rather than silently measuring degrees as metres.
    """
    return {"type": "name", "properties": {"name": GRID_CRS_NAME}}


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
