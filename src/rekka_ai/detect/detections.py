"""Oriented detections on the ground, their deduplication, and GeoJSON output."""

import math
from dataclasses import dataclass
from typing import Any

from shapely import STRtree
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from rekka_ai.geo import COORD_DECIMALS, crs_member

#: Two boxes overlapping more than this are taken to be the same vehicle seen
#: from two windows.
DEFAULT_IOU = 0.4

Corners = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class Detection:
    """One oriented footprint, with corners in EPSG:3879 metres."""

    label: str
    confidence: float
    corners: Corners
    aoi: str = ""

    def polygon(self) -> Polygon:
        return Polygon(self.corners)

    @property
    def centre(self) -> tuple[float, float]:
        eastings = [e for e, _ in self.corners]
        northings = [n for _, n in self.corners]
        return sum(eastings) / len(eastings), sum(northings) / len(northings)

    @property
    def _sides(self) -> tuple[float, float]:
        """The two edge lengths at the first corner, longer first."""
        (x0, y0), (x1, y1), (x2, y2) = self.corners[0], self.corners[1], self.corners[2]
        first = math.dist((x0, y0), (x1, y1))
        second = math.dist((x1, y1), (x2, y2))
        return max(first, second), min(first, second)

    @property
    def length_m(self) -> float:
        return self._sides[0]

    @property
    def width_m(self) -> float:
        return self._sides[1]

    @property
    def heading_deg(self) -> float:
        """Compass bearing of the long axis, 0 = north, in [0, 180).

        A parked vehicle's axis has no direction -- nose-north and nose-south
        are the same orientation -- so the bearing is taken modulo 180 rather
        than reported as two different values for the same alignment.
        """
        (x0, y0), (x1, y1), (x2, y2) = self.corners[0], self.corners[1], self.corners[2]
        if math.dist((x0, y0), (x1, y1)) >= math.dist((x1, y1), (x2, y2)):
            d_east, d_north = x1 - x0, y1 - y0
        else:
            d_east, d_north = x2 - x1, y2 - y1
        return math.degrees(math.atan2(d_east, d_north)) % 180.0


def merge(
    detections: list[Detection], iou_threshold: float = DEFAULT_IOU
) -> list[Detection]:
    """Drop duplicates produced by overlapping windows, keeping the most confident.

    Windows overlap by design, so a vehicle near a seam is detected twice. This
    is global rather than per-window: comparing only within a window would
    leave exactly the duplicates that overlap creates.

    Candidates for each comparison come from an STRtree, so the cost grows
    with the number of actual neighbours rather than with every pair of
    detections -- which is what makes a city-wide sweep feasible.
    """
    if not detections:
        return []
    polygons = [d.polygon() for d in detections]
    tree = STRtree(polygons)
    kept: list[int] = []
    kept_set: set[int] = set()
    by_confidence = sorted(
        range(len(detections)), key=lambda i: detections[i].confidence, reverse=True
    )
    for index in by_confidence:
        detection = detections[index]
        if any(
            detections[other].label == detection.label
            and _iou(polygons[index], polygons[other]) > iou_threshold
            for other in tree.query(polygons[index])
            if other in kept_set
        ):
            continue
        kept.append(index)
        kept_set.add(index)
    return [detections[i] for i in kept]


def longer_than(detections: list[Detection], min_length_m: float) -> list[Detection]:
    """Keep only detections at least ``min_length_m`` long.

    DOTA's ``large vehicle`` is loose enough to fire on vans and estate cars,
    which at z16 are roughly half the candidates. The annotation guide already
    settles these with a 6 m gate, so applying it here means a human is not
    asked to reject the same cars over and over.

    A bobtail tractor unit can fall under the gate; that is a known limitation
    of the rule rather than of this filter.
    """
    return [d for d in detections if d.length_m >= min_length_m]


def within_region(detections: list[Detection], region: BaseGeometry) -> list[Detection]:
    """Detections whose centre falls inside a polygon region.

    Windows sweep the region's bounding box, so a postcode-area run also sees
    the neighbouring ground; the centre test keeps the output to what was
    asked for. A truck straddling the boundary belongs to whichever side its
    centre sits on — a rule, so it is decided once.
    """
    prepared = prep(region)
    return [d for d in detections if prepared.contains(Point(d.centre))]


def _iou(a: Polygon, b: Polygon) -> float:
    if not a.is_valid or not b.is_valid or not a.intersects(b):
        return 0.0
    intersection = a.intersection(b).area
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def to_geojson(detections: list[Detection], **properties: Any) -> dict[str, Any]:
    """An EPSG:3879 FeatureCollection, one oriented polygon per detection.

    Corners are already grid metres, so this writes them as they are: the
    candidates go on to become label files, and a WGS84 hop here would be a
    conversion out and a conversion back for nobody's benefit. The ``crs``
    member is what says so.

    Extra keyword arguments are copied onto every feature -- source layer and
    zoom in practice, so a file of candidates stays interpretable once it has
    been through an annotation tool and back.
    """
    return {
        "type": "FeatureCollection",
        "crs": crs_member(),
        "features": [_feature(d, properties) for d in detections],
    }


def _feature(detection: Detection, properties: dict[str, Any]) -> dict[str, Any]:
    ring = [[round(v, COORD_DECIMALS) for v in corner] for corner in detection.corners]
    ring.append(ring[0])  # GeoJSON rings must close
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "label": detection.label,
            "confidence": round(detection.confidence, 4),
            "length_m": round(detection.length_m, 2),
            "width_m": round(detection.width_m, 2),
            "heading_deg": round(detection.heading_deg, 1),
            "aoi": detection.aoi,
            **properties,
        },
    }
