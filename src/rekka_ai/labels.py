"""The labelling schema: candidates a human turns into training data.

Labels are the one artifact here that cannot be regenerated, so they live in
version control under ``labels/``, one GeoJSON file per AOI. Per-AOI files keep
diffs small, let the work be resumed area by area, and make progress countable
without a database.

Files are **WGS84**, per RFC 7946, so QGIS and every other tool reads them
without special handling. All geometry *maths* happens in EPSG:3879, where
metres are metres.

Measurements are always recomputed from geometry rather than trusted: an editor
moves a vertex, and a stored ``length_m`` becomes silently wrong the moment it
does.
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from rekka_ai.detect.detections import Detection
from rekka_ai.geo import to_grid

#: What a labelled object is. ``van`` is a real class rather than an exclusion:
#: the detector fires on vans, and labelling them explicitly is better than
#: leaving them as unlabelled background the model must guess about.
CLASSES = ("truck", "bus", "van")

#: ``candidate`` is the detector's guess, awaiting review. ``confirmed`` and
#: ``rejected`` are human verdicts on it; ``added`` is a human-drawn object the
#: detector missed. Rejects are kept, not deleted -- they are the hard negatives
#: the container areas failed to supply.
STATUSES = ("candidate", "confirmed", "rejected", "added")

UNLABELLED = ""


def measurements(ring: list[list[float]]) -> dict[str, float]:
    """Length, width and heading of a WGS84 ring, computed in EPSG:3879.

    Accepts the closing coordinate GeoJSON requires, and ignores it.
    """
    if len(ring) < 4:
        raise ValueError(f"expected at least 4 corners, got {len(ring)}")
    corners = [tuple(to_grid(lon, lat)) for lon, lat in ring[:4]]
    detection = Detection(label="", confidence=0.0, corners=tuple(corners))
    return {
        "length_m": round(detection.length_m, 2),
        "width_m": round(detection.width_m, 2),
        "heading_deg": round(detection.heading_deg, 1),
    }


def read(path: Path) -> dict[str, Any]:
    """One area's label file, as a GeoJSON FeatureCollection."""
    return json.loads(path.read_text())


def _normalize_numbers(value: Any) -> Any:
    """Integral floats become ints, so Python and JS writers agree on disk.

    json.dumps writes a float 175.0 as ``175.0``; JSON.stringify writes the
    same number as ``175``. Normalizing on write keeps the label files
    byte-stable across the Python CLI and the web labelling tool, so a review
    edit never reformats a file.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_normalize_numbers(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_numbers(v) for k, v in value.items()}
    return value


def write(path: Path, collection: dict[str, Any]) -> None:
    """Write a label file, creating its directory. Indented so diffs are readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_numbers(collection), indent=2) + "\n")


def summarise(collection: dict[str, Any]) -> dict[str, int]:
    """Counts by status and class, for progress reporting."""
    counts: Counter[str] = Counter()
    for feature in collection.get("features", []):
        properties = feature.get("properties", {})
        counts[properties.get("status", "candidate")] += 1
        klass = properties.get("class") or UNLABELLED
        if klass:
            counts[klass] += 1
    return dict(counts)


def reviewed(collection: dict[str, Any]) -> int:
    """How many features carry a human verdict."""
    return sum(
        1
        for f in collection.get("features", [])
        if f.get("properties", {}).get("status") != "candidate"
    )


def validate(collection: dict[str, Any]) -> list[str]:
    """Problems that would make a file unusable as training data."""
    problems = []
    for index, feature in enumerate(collection.get("features", [])):
        properties = feature.get("properties", {})
        status = properties.get("status")
        klass = properties.get("class", UNLABELLED)
        where = f"feature {index}"
        if status not in STATUSES:
            problems.append(f"{where}: status {status!r} not in {STATUSES}")
        if klass and klass not in CLASSES:
            problems.append(f"{where}: class {klass!r} not in {CLASSES}")
        if status in ("confirmed", "added") and not klass:
            problems.append(f"{where}: {status} but no class set")
        ring = _ring_of(feature)
        # A closed 4-corner ring has 5 positions; anything else is not an
        # oriented box and cannot become a YOLO-OBB label.
        if ring is None:
            problems.append(f"{where}: no polygon ring to check")
        elif len(ring) != 5:
            problems.append(f"{where}: ring has {len(ring)} positions, expected 5")
        elif not _is_rectangle(ring):
            problems.append(f"{where}: ring is not a rectangle")
    return problems


def _ring_of(feature: dict[str, Any]) -> list[list[float]] | None:
    """A feature's outer ring, or ``None`` if it has no usable one.

    Reported rather than raised: this is the function that exists to describe
    a broken label file, so a null geometry (legal per RFC 7946) or an empty
    coordinate list must become a line of output, not a traceback from the
    diagnostic itself.
    """
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        return None
    ring = coordinates[0]
    if not isinstance(ring, list):
        return None
    return ring


def _is_rectangle(ring: list[list[float]], tolerance_m: float = 0.5) -> bool:
    corners = [to_grid(lon, lat) for lon, lat in ring[:4]]
    # Opposite sides equal and diagonals equal is enough to pin a rectangle.
    sides = [math.dist(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    diagonals = [math.dist(corners[0], corners[2]), math.dist(corners[1], corners[3])]
    return (
        abs(sides[0] - sides[2]) < tolerance_m
        and abs(sides[1] - sides[3]) < tolerance_m
        and abs(diagonals[0] - diagonals[1]) < tolerance_m
    )
