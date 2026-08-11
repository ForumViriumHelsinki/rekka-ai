"""YOLO-OBB dataset export: reviewed labels become images and pixel labels.

Labels live in geographic coordinates so that every retiling decision stays
reversible; export is the one place pixel coordinates are born. Two rules make
the output honest:

- **Split by whole AOI, never at random.** The collection's ``split`` field
  decides which directory an area's windows land in, so near-duplicate
  adjacent windows cannot straddle train and validation (docs/DESIGN.md §2).
- **A box is written only into windows that fully contain it.** The 30 m
  window overlap exceeds the longest road-legal rig, so every vehicle is whole
  in at least one window; a box in the overlap lands in two, which skews
  instance counts mildly but never produces a clamped, distorted label.

``rejected`` features are not labels -- they are what the file's silence about
that ground means. Their windows still export, so a rejected container teaches
"background" rather than vanishing from the dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rekka_ai import labels
from rekka_ai.imagery.aoi import Aoi
from rekka_ai.imagery.windows import (
    OVERLAP_M,
    WINDOW_SIZE,
    Window,
    grid_to_pixel,
    load_window,
    windows_covering,
)
from rekka_ai.imagery.wmts import TileSource, ensure_cached

#: YOLO class indices, fixed by ``labels.CLASSES`` order: truck 0, bus 1, van 2, car 3.
CLASS_INDEX = {name: index for index, name in enumerate(labels.CLASSES)}

#: Statuses that are real, human-verified objects. Candidates are unreviewed
#: and rejected ones are non-objects; neither may become a label.
LABEL_STATUSES = frozenset({"confirmed", "added"})

#: JPEG re-encode quality for window images. The source tiles are JPEG too,
#: so this adds one more generation of artifacts; keep it light.
IMAGE_QUALITY = 95


def grid_corners(feature: dict[str, Any]) -> list[tuple[float, float]]:
    """A label feature's ring in EPSG:3879, closing coordinate dropped.

    Label files are stored in the grid CRS, so this reads corners rather than
    projecting them.
    """
    ring = feature["geometry"]["coordinates"][0]
    return [(easting, northing) for easting, northing in ring[:4]]


@dataclass(frozen=True, slots=True)
class LabelBox:
    """A label ready to write: its class index, and its ring in grid metres."""

    class_index: int
    corners: list[tuple[float, float]]


def label_boxes(features: list[dict[str, Any]]) -> list[LabelBox]:
    """The writable labels among ``features``, as grid-metre corners."""
    boxes = []
    for feature in features:
        properties = feature.get("properties", {})
        if properties.get("status") not in LABEL_STATUSES:
            continue
        klass = properties.get("class", labels.UNLABELLED)
        if klass not in CLASS_INDEX:
            continue
        boxes.append(
            LabelBox(class_index=CLASS_INDEX[klass], corners=grid_corners(feature))
        )
    return boxes


def window_label_lines(boxes: list[LabelBox], window: Window) -> list[str]:
    """YOLO-OBB lines for every box the window fully contains.

    A line is ``class x1 y1 x2 y2 x3 y3 x4 y4``, normalised to the window.
    Full containment is the dedup rule: a box in the overlap of two windows is
    written to both, but a box is never cut in half by one.
    """
    lines = []
    for box in boxes:
        corners = []
        for easting, northing in box.corners:
            px, py = grid_to_pixel(easting, northing, window.zoom)
            corners.append(
                ((px - window.x) / window.size, (py - window.y) / window.size)
            )
        if not all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in corners):
            continue
        coords = " ".join(f"{v:.6f}" for corner in corners for v in corner)
        lines.append(f"{box.class_index} {coords}")
    return lines


def export_area(
    aoi: Aoi,
    features: list[dict[str, Any]],
    *,
    layer: str,
    zoom: int,
    cache_root: Path,
    fetcher: TileSource,
    images_dir: Path,
    labels_dir: Path,
    window_size: int = WINDOW_SIZE,
    overlap_m: float = OVERLAP_M,
) -> tuple[int, int, list[str]]:
    """Write one area's windows and label files. Returns counts and skips."""
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    windows = boxes = 0
    skipped = []
    prepared = label_boxes(features)
    for window in windows_covering(
        aoi.bounds, zoom, size=window_size, overlap_m=overlap_m
    ):
        ensure_cached(fetcher, layer, window.tiles())
        try:
            image = load_window(window, cache_root, layer)
        except FileNotFoundError:
            # Same rule as bootstrap: a missing tile leaves a hole, skip the
            # window rather than kill the export.
            skipped.append(f"{aoi.name}_{window.x}_{window.y}")
            continue
        stem = f"{aoi.name}_{window.x}_{window.y}"
        image.save(images_dir / f"{stem}.jpg", quality=IMAGE_QUALITY)
        lines = window_label_lines(prepared, window)
        (labels_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n" if lines else ""
        )
        windows += 1
        boxes += len(lines)
    return windows, boxes, skipped


def dataset_yaml(root: Path, names: tuple[str, ...] = labels.CLASSES) -> str:
    """The Ultralytics dataset config, with fixed class ordering."""
    return yaml.safe_dump(
        {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": dict(enumerate(names)),
        },
        sort_keys=False,
    )
