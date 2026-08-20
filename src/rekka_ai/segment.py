"""Refine `detect`'s oriented boxes into vehicle outlines with SAM.

Input is exactly what `detect` writes -- oriented polygons in EPSG:3879 -- so
this is a step after detection, not a parallel path. Output is the same
features with the four-corner box replaced by the segmented outline, plus the
footprint area the outline implies.

**The shadow problem.** Prompted with a vehicle's box, SAM readily returns
"vehicle + its cast shadow": the shadow touches the vehicle and has a crisp
outer edge against the ground, so it looks like part of the same object.
Measured on 24 detections over one bright and one shadowed yard
(2026-08-15): in the shadowed yard 3 of 12 masks grew *past* their oriented
box, up to 1.10x its area, and every leaked region was darker than the
vehicle it hung off -- leak luminance 51-97 against 113-221 inside. The
bright yard showed none of it.

**The correction is clipping to the oriented box, and only that.** The
detector's box is a reliable bound and SAM's job here is to refine within
it, not to grow past it, so clipping removes the leak by construction.

**A luminance split was tried and removed, 2026-08-15.** The reasoning was
that shadow *inside* the box is darker than the body, so an Otsu threshold
on the mask's own luminance should separate them. It does -- but a vehicle
is not uniformly lit either, and the split cannot tell a cast shadow from a
truck's own dark cab under a pale cargo box, or a car's black glazing under
a white roof. Measured over two yards:

    split on    median fill 0.84    22% of vehicles under half their box
    split off   median fill 0.88    1-2% under half their box

So it cut more than a fifth of vehicles down to a fragment -- typically
keeping the pale box body and discarding cab and chassis -- to fix an
external leak that clipping already handles. The lesson worth keeping: the
justification came from 24 detections where the shadow was *outside* the
box, and was then generalised to shadow inside it, which is a different
problem with a different sign. Interior shadow now stays in the outline and
slightly inflates the footprint; that is the lesser error.

**Why this is a module, not a CLI command, 2026-08-20.** The outline was
meant to beat the detector's box as a measurement. It does not. Against
5,434 human-labelled vehicles matched at IoU >= 0.3 (median 0.78), the
detector's box already matches the labels to 0.11-0.36 m length MAE by
class, while the outline's minimum-area rectangle runs 0.22-0.84 m and is
biased 0.2-0.8 m *short* -- SAM decodes a 256x256 mask no matter the input
size, which quantizes the outline to 50 cm steps at z16 (vertex spacing
p10 = 0.50 m, measured on the 2025 city sweep) and eats both ends of long
vehicles. The box loses nothing that a coarse mask can recover. The module
stays in the tree because the window-slot batching and the shadow
measurements above may matter if a finer decode (per-detection crops, ~15x
the encode cost) ever becomes worth it -- but it is deliberately not wired
into the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rekka_ai.imagery.windows import WINDOW_SIZE, Window, grid_to_pixel

#: Meta's Segment Anything, via ultralytics. `sam_b` is the 358 MB base
#: model; `mobile_sam.pt` is ~40 MB and much faster if a city-scale run needs
#: it. Downloaded on first use to the working directory.
DEFAULT_SAM_WEIGHTS = "sam_b.pt"

#: Douglas-Peucker tolerance for the traced outline, in metres. A mask
#: contour has a vertex per boundary pixel; at z16 that is 12.5 cm of detail
#: on a vehicle whose real edges are straight.
SIMPLIFY_M = 0.15

#: How many box prompts to decode against one encoded window at a time.
#: Grouping by window fixes the *encoder* cost but leaves the decoder
#: unbounded, and prompt counts are heavily skewed: over the 2025 city sweep
#: the median window holds 10 detections, the 99th percentile 84, and the
#: worst a full harbour yard at 364. Decoding 364 at once asked for 1.4 GiB
#: and ran a 16 GB card out of memory 95% of the way through a city run
#: (2026-08-15). 32 keeps the peak flat regardless of how dense the ground is.
PROMPT_BATCH = 32

#: Below this many pixels a mask is noise rather than a vehicle outline, and
#: the detection keeps its original box instead.
MIN_MASK_PX = 40

#: An outline covering less of its box than this is not the vehicle, and the
#: detection keeps its box instead. A vehicle fills 0.79-0.85 of its oriented
#: box in the median (measured over two yards, 2026-08-15), because a box
#: circumscribes a rounded body. The failure this catches is SAM returning a
#: *part*: on a dark car the windscreen or roof highlight is the most
#: object-like contiguous region in the prompt, so the mask comes back as
#: that panel alone at 0.11-0.22 of the box. Falling back to the box is the
#: honest answer -- the detection is still real, only its outline is unknown.
MIN_FILL_RATIO = 0.35


def window_slot(easting: float, northing: float, zoom: int) -> tuple[int, int]:
    """Which fixed window a point belongs to.

    Detections are grouped by slot so SAM encodes each window **once** and
    then decodes many box prompts against it -- the encoder is the expensive
    half. Measured on the 2025 city sweep: 125,882 detections fall in 8,143
    slots, 15.5 detections each, so this is a 15x saving over one pass per
    detection.

    The grid is absolute (pixel coordinate // window size), not relative to
    the detections present, so the same ground always lands in the same slot.
    """
    px, py = grid_to_pixel(easting, northing, zoom)
    return int(px) // WINDOW_SIZE, int(py) // WINDOW_SIZE


def window_for_slot(slot: tuple[int, int], zoom: int) -> Window:
    col, row = slot
    return Window(zoom=zoom, x=col * WINDOW_SIZE, y=row * WINDOW_SIZE, size=WINDOW_SIZE)


def to_window_px(
    ring: list[tuple[float, float]], window: Window, zoom: int
) -> list[tuple[float, float]]:
    """A ring in grid metres to pixels local to ``window``."""
    out = []
    for easting, northing in ring:
        px, py = grid_to_pixel(easting, northing, zoom)
        out.append((px - window.x, py - window.y))
    return out


def polygon_mask(ring_px: list[tuple[float, float]], shape: tuple[int, int]):
    """A boolean raster of the oriented box, for clipping."""
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in ring_px], fill=255)
    return np.asarray(canvas) > 0


def refine(mask, grey, box_mask):
    """Clip a SAM mask to its oriented box.

    ``grey`` is unused and kept in the signature deliberately: a luminance
    split lived here until 2026-08-15 and was removed for cutting vehicles at
    their own light/dark boundaries (see the module docstring). Keeping the
    parameter documents that the image is available if a better-founded
    correction is ever added, and costs the caller nothing.
    """
    return mask & box_mask


def mask_to_ring(mask, window: Window, zoom: int, simplify_m: float = SIMPLIFY_M):
    """The mask's outer contour as a closed ring in grid metres.

    Returns None when the mask is too small to be an outline, so the caller
    can keep the original box rather than emit a speck.
    """
    import cv2
    import numpy as np
    from shapely.geometry import Polygon

    if mask.sum() < MIN_MASK_PX:
        return None
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    # float() rather than passing cv2.contourArea directly: its return type is
    # not comparable as far as the type checker is concerned.
    biggest = max(contours, key=lambda contour: float(cv2.contourArea(contour)))
    if len(biggest) < 3:
        return None
    points = [window.to_grid(float(p[0][0]), float(p[0][1])) for p in biggest]
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return None
    polygon = polygon.simplify(simplify_m, preserve_topology=True)
    return list(polygon.exterior.coords)


def group_by_window(
    features: list[dict[str, Any]], zoom: int
) -> dict[tuple[int, int], list[int]]:
    """Feature indices bucketed by the window that will be encoded for them."""
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, feature in enumerate(features):
        ring = feature["geometry"]["coordinates"][0]
        easting = sum(p[0] for p in ring[:-1]) / (len(ring) - 1)
        northing = sum(p[1] for p in ring[:-1]) / (len(ring) - 1)
        buckets.setdefault(window_slot(easting, northing, zoom), []).append(index)
    return buckets


def write(
    features: list[dict[str, Any]], path: Path, crs_member: dict[str, Any]
) -> None:
    """Write outlines; the suffix picks the format, mirroring detections.write."""
    import json

    suffix = path.suffix.lower()
    collection = {"type": "FeatureCollection", "crs": crs_member, "features": features}
    if suffix in {".geojson", ".json"}:
        path.write_text(json.dumps(collection))
        return
    if suffix not in {".fgb", ".gpkg"}:
        raise ValueError(
            f"unknown output format {suffix!r}; use .geojson, .fgb or .gpkg"
        )
    import geopandas

    from rekka_ai.geo import GRID

    frame = geopandas.GeoDataFrame.from_features(features, crs=GRID)
    frame.to_file(path, driver="FlatGeobuf" if suffix == ".fgb" else "GPKG")
