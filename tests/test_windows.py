"""Windowing and the pixel/ground mapping."""

import itertools
import math
from pathlib import Path

import pytest
from PIL import Image

from rekka_ai.imagery.tiles import Bounds, Tile, resolution
from rekka_ai.imagery.windows import (
    TILE_SIZE,
    Window,
    grid_to_pixel,
    load_window,
    pixel_to_grid,
    windows_covering,
)
from rekka_ai.imagery.wmts import cache_path

ZOOM = 15
LAYER = "Ortoilmakuva_2025_5cm"


def test_pixel_and_grid_roundtrip() -> None:
    easting, northing = 25496580.4, 6673003.6
    x, y = grid_to_pixel(easting, northing, ZOOM)
    assert pixel_to_grid(x, y, ZOOM) == pytest.approx((easting, northing), abs=1e-6)


def test_a_tiles_pixel_origin_matches_its_index() -> None:
    tile = Tile(zoom=ZOOM, col=1000, row=2000)
    bounds = tile.bounds()
    x, y = grid_to_pixel(bounds.min_easting, bounds.max_northing, ZOOM)
    assert (round(x), round(y)) == (1000 * TILE_SIZE, 2000 * TILE_SIZE)


def test_one_pixel_is_one_resolution_step() -> None:
    e0, n0 = pixel_to_grid(0, 0, ZOOM)
    e1, n1 = pixel_to_grid(1, 1, ZOOM)
    assert e1 - e0 == pytest.approx(resolution(ZOOM))
    assert n0 - n1 == pytest.approx(resolution(ZOOM))  # y grows southward


def test_window_needs_the_tiles_it_covers() -> None:
    window = Window(zoom=ZOOM, x=512, y=768, size=512)
    assert window.tiles() == [
        Tile(zoom=ZOOM, col=2, row=3),
        Tile(zoom=ZOOM, col=3, row=3),
        Tile(zoom=ZOOM, col=2, row=4),
        Tile(zoom=ZOOM, col=3, row=4),
    ]


def test_window_straddling_tiles_pulls_in_the_neighbours() -> None:
    """An origin part-way into a tile needs one more tile per axis."""
    window = Window(zoom=ZOOM, x=100, y=100, size=256)
    assert {(t.col, t.row) for t in window.tiles()} == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_window_local_pixels_map_to_ground() -> None:
    window = Window(zoom=ZOOM, x=1024, y=2048, size=1024)
    assert window.to_grid(0, 0) == pytest.approx(pixel_to_grid(1024, 2048, ZOOM))
    corner = window.to_grid(1024, 1024)
    assert corner == pytest.approx(pixel_to_grid(2048, 3072, ZOOM))


def test_window_bounds_span_its_size_on_the_ground() -> None:
    window = Window(zoom=ZOOM, x=0, y=0, size=1024)
    bounds = window.bounds()
    assert bounds.width == pytest.approx(1024 * resolution(ZOOM))
    assert bounds.height == pytest.approx(1024 * resolution(ZOOM))


def _bounds_of_size(metres: float, zoom: int = ZOOM) -> Bounds:
    easting, northing = 25496000.0, 6673000.0
    return Bounds(
        min_easting=easting,
        min_northing=northing,
        max_easting=easting + metres,
        max_northing=northing + metres,
    )


def test_small_area_needs_one_window() -> None:
    windows = list(windows_covering(_bounds_of_size(50), ZOOM, size=1024))
    assert len(windows) == 1


def test_windows_overlap_by_at_least_the_requested_metres() -> None:
    windows = list(windows_covering(_bounds_of_size(600), ZOOM, size=512, overlap_m=30))
    xs = sorted({w.x for w in windows})
    assert len(xs) > 1
    overlap_px = 512 - (xs[1] - xs[0])
    assert overlap_px * resolution(ZOOM) >= 30


def test_windows_cover_the_whole_area() -> None:
    """Every corner of the AOI must fall inside some window."""
    bounds = _bounds_of_size(700)
    windows = list(windows_covering(bounds, ZOOM, size=512, overlap_m=30))
    corners = [
        (bounds.min_easting, bounds.min_northing),
        (bounds.max_easting, bounds.min_northing),
        (bounds.min_easting, bounds.max_northing),
        (bounds.max_easting, bounds.max_northing),
    ]
    for easting, northing in corners:
        covered = any(
            w.bounds().min_easting <= easting <= w.bounds().max_easting
            and w.bounds().min_northing <= northing <= w.bounds().max_northing
            for w in windows
        )
        assert covered, f"({easting}, {northing}) not covered"


def test_last_window_is_flush_with_the_far_edge() -> None:
    bounds = _bounds_of_size(700)
    windows = list(windows_covering(bounds, ZOOM, size=512, overlap_m=30))
    right = max(w.x + w.size for w in windows)
    edge_x, _ = grid_to_pixel(bounds.max_easting, bounds.max_northing, ZOOM)
    assert right >= edge_x


def test_no_truck_length_falls_between_windows() -> None:
    """Overlap must exceed the longest vehicle, or a truck can be cut by every window."""
    windows = list(
        windows_covering(_bounds_of_size(1000), ZOOM, size=512, overlap_m=30)
    )
    xs = sorted({w.x for w in windows})
    gaps = [b - a for a, b in itertools.pairwise(xs)]
    overlap_m = (512 - max(gaps)) * resolution(ZOOM)
    assert overlap_m >= 25.25  # longest Finnish HCT rig


def test_overlap_wider_than_the_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="not less than"):
        list(windows_covering(_bounds_of_size(500), ZOOM, size=64, overlap_m=1000))


def test_load_window_stitches_tiles_in_the_right_places(tmp_path: Path) -> None:
    """Each tile must land at its own offset; a transposed paste would swap them."""
    window = Window(zoom=ZOOM, x=0, y=0, size=512)
    colours = {
        (0, 0): (255, 0, 0),
        (1, 0): (0, 255, 0),
        (0, 1): (0, 0, 255),
        (1, 1): (255, 255, 0),
    }
    for (col, row), colour in colours.items():
        path = cache_path(tmp_path, LAYER, Tile(zoom=ZOOM, col=col, row=row))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (TILE_SIZE, TILE_SIZE), colour).save(path)

    image = load_window(window, tmp_path, LAYER)
    assert image.size == (512, 512)

    def nearest(point: tuple[int, int]) -> tuple[int, int]:
        # The cache holds JPEGs, so colours come back a shade off; match the
        # closest of the four rather than requiring exact bytes.
        pixel = image.getpixel(point)
        assert isinstance(pixel, tuple)
        return min(
            colours,
            key=lambda k: sum(
                (a - b) ** 2 for a, b in zip(colours[k], pixel, strict=True)
            ),
        )

    assert nearest((10, 10)) == (0, 0)
    assert nearest((300, 10)) == (1, 0)  # east of the origin
    assert nearest((10, 300)) == (0, 1)  # south of the origin
    assert nearest((300, 300)) == (1, 1)


def test_load_window_reports_a_missing_tile(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not cached"):
        load_window(Window(zoom=ZOOM, x=0, y=0, size=256), tmp_path, LAYER)


def test_a_window_pixel_offset_matches_the_ground_distance() -> None:
    """A 100 px step at z15 is 25 m; getting this wrong misplaces every detection."""
    window = Window(zoom=ZOOM, x=5000, y=7000, size=1024)
    e0, n0 = window.to_grid(0, 0)
    e1, n1 = window.to_grid(100, 100)
    assert math.dist((e0, n0), (e1, n1)) == pytest.approx(
        100 * resolution(ZOOM) * math.sqrt(2)
    )


def test_overlapping_windows_decode_a_shared_tile_once(tmp_path: Path) -> None:
    """Windows overlap by design, so the same JPEG is wanted several times."""
    from rekka_ai.imagery import windows as windows_module

    windows_module._decoded_tile.cache_clear()
    tile = Tile(zoom=ZOOM, col=0, row=0)
    path = cache_path(tmp_path, LAYER, tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), (10, 20, 30)).save(path)

    for _ in range(3):
        load_window(Window(zoom=ZOOM, x=0, y=0, size=256), tmp_path, LAYER)

    info = windows_module._decoded_tile.cache_info()
    assert info.misses == 1
    assert info.hits == 2
