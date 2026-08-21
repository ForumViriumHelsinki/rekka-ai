"""SAM outline refinement: the pure parts, without torch or cv2.

`refine` is the piece that carries the project's shadow finding, so it is
tested here rather than left to a visual check.
"""

import numpy as np
import pytest

from rekka_ai.imagery.windows import WINDOW_SIZE
from rekka_ai.segment import (
    group_by_window,
    polygon_mask,
    refine,
    to_window_px,
    window_for_slot,
    window_slot,
)

GRID_E, GRID_N = 25_500_000.0, 6_670_000.0


def _ring(east: float, north: float, size: float = 10.0):
    return [
        (east, north),
        (east + size, north),
        (east + size, north + size / 3),
        (east, north + size / 3),
    ]


def test_window_slot_is_absolute_not_relative() -> None:
    """Slots come from absolute pixel coordinates, so the same ground always
    lands in the same window whatever else is being segmented alongside it."""
    a = window_slot(GRID_E, GRID_N, 16)
    b = window_slot(GRID_E + 0.5, GRID_N - 0.5, 16)
    assert a == b
    far = window_slot(GRID_E + 5000, GRID_N, 16)
    assert far != a


def test_window_for_slot_contains_its_own_points() -> None:
    slot = window_slot(GRID_E, GRID_N, 16)
    window = window_for_slot(slot, 16)
    x, y = to_window_px([(GRID_E, GRID_N)], window, 16)[0]
    assert 0 <= x < WINDOW_SIZE
    assert 0 <= y < WINDOW_SIZE


def test_group_by_window_buckets_neighbours_together() -> None:
    """The whole point of grouping: one SAM encode serves many prompts."""
    feats = [
        {
            "geometry": {
                "coordinates": [[*_ring(GRID_E + dx, GRID_N), (GRID_E + dx, GRID_N)]]
            }
        }
        for dx in (0.0, 3.0, 6.0)
    ]
    buckets = group_by_window(feats, 16)
    assert len(buckets) == 1
    assert sorted(next(iter(buckets.values()))) == [0, 1, 2]


def test_refine_clips_the_mask_to_the_oriented_box() -> None:
    """SAM growing past the box is the external shadow leak; clipping removes
    it by construction. This is the whole of refine() -- a luminance split
    was removed on 2026-08-15 for cutting vehicles at their own light/dark
    boundary, so a dark region inside the box must now survive."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True  # SAM's mask, larger than the box
    box = np.zeros((40, 40), dtype=bool)
    box[10:20, 10:20] = True  # the detector's box
    grey = np.full((40, 40), 200.0)  # uniform, so the dark-tail cut is inert
    out = refine(mask, grey, box)
    assert out.sum() == 100
    assert not (out & ~box).any()


def test_refine_keeps_dark_pixels_inside_the_box() -> None:
    """The removed split would have discarded this half. A pale cargo box
    over a dark cab is one vehicle, and the outline must cover both."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    box = mask.copy()
    grey = np.full((40, 40), 210.0)
    grey[20:30, 10:30] = 40.0
    out = refine(mask, grey, box)
    assert out.sum() == mask.sum()


def test_refine_on_an_empty_mask_is_empty_not_an_error() -> None:
    empty = np.zeros((10, 10), dtype=bool)
    grey = np.full((10, 10), 100.0)
    assert not refine(empty, grey, np.ones((10, 10), dtype=bool)).any()


def test_polygon_mask_rasterises_the_box() -> None:
    ring = [(2.0, 2.0), (8.0, 2.0), (8.0, 6.0), (2.0, 6.0)]
    mask = polygon_mask(ring, (10, 10))
    assert mask[3, 3]
    assert not mask[9, 9]


def test_polygon_mask_and_refine_agree_on_shape() -> None:
    """Guards the axis-order trap: a mask indexed [row, col] against a ring
    given as (x, y) is exactly where this would silently transpose."""
    ring = [(1.0, 1.0), (9.0, 1.0), (9.0, 3.0), (1.0, 3.0)]
    mask = polygon_mask(ring, (12, 10))
    assert mask.shape == (12, 10)
    assert mask[2, 5]  # inside: row 2 (y), col 5 (x)
    assert not mask[8, 5]  # outside in y


@pytest.mark.parametrize("zoom", [16, 17])
def test_to_window_px_is_inside_its_own_window(zoom: int) -> None:
    slot = window_slot(GRID_E, GRID_N, zoom)
    window = window_for_slot(slot, zoom)
    for x, y in to_window_px(_ring(GRID_E, GRID_N), window, zoom):
        assert -WINDOW_SIZE < x < 2 * WINDOW_SIZE
        assert -WINDOW_SIZE < y < 2 * WINDOW_SIZE
