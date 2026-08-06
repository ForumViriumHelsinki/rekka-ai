"""Export turns geographic labels into pixel labels exactly once, so the rules
that decide what becomes a label -- status, class, full containment -- all have
a checkable right answer.
"""

from pathlib import Path

import pytest
import yaml
from PIL import Image

from rekka_ai import labels
from rekka_ai.export import (
    dataset_yaml,
    export_area,
    label_boxes,
    window_label_lines,
)
from rekka_ai.geo import to_wgs84
from rekka_ai.imagery.aoi import Aoi
from rekka_ai.imagery.tiles import TILE_SIZE, Bounds
from rekka_ai.imagery.windows import Window, pixel_to_grid
from rekka_ai.imagery.wmts import cache_path

ZOOM = 16
LAYER = "Ortoilmakuva_2025_5cm"


def _feature(
    local_pixels: list[tuple[float, float]],
    window: Window,
    *,
    status: str = "confirmed",
    klass: str = "truck",
) -> dict:
    """A label feature covering window-local pixels, built via the ground."""
    ring = [list(to_wgs84(*pixel_to_grid(x, y, window.zoom))) for x, y in local_pixels]
    ring.append(ring[0])
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {"status": status, "class": klass},
    }


def _box(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


class NullFetcher:
    def fetch_all(self, layer, tiles):
        return []


def test_fully_contained_box_becomes_a_label() -> None:
    window = Window(zoom=ZOOM, x=0, y=0, size=256)
    feature = _feature(_box(64, 64, 64, 16), window)

    lines = window_label_lines(label_boxes([feature]), window)

    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"  # truck is class index 0
    coords = [float(v) for v in parts[1:]]
    assert coords == pytest.approx(
        [
            64 / 256,
            64 / 256,
            128 / 256,
            64 / 256,
            128 / 256,
            80 / 256,
            64 / 256,
            80 / 256,
        ],
        abs=1e-5,
    )


def test_class_index_follows_labels_classes() -> None:
    window = Window(zoom=ZOOM, x=0, y=0, size=256)
    features = [
        _feature(_box(10, 10, 20, 8), window, klass=name) for name in labels.CLASSES
    ]
    lines = window_label_lines(label_boxes(features), window)
    assert [int(line.split()[0]) for line in lines] == [0, 1, 2]


def test_straddling_box_is_dropped_not_clamped() -> None:
    window = Window(zoom=ZOOM, x=0, y=0, size=256)
    feature = _feature(_box(240, 10, 32, 16), window)  # crosses the east edge
    assert window_label_lines(label_boxes([feature]), window) == []


def test_only_human_verdicts_become_labels() -> None:
    window = Window(zoom=ZOOM, x=0, y=0, size=256)
    features = [
        _feature(_box(10, 10, 20, 8), window, status="candidate"),
        _feature(_box(10, 40, 20, 8), window, status="rejected", klass=""),
        _feature(_box(10, 70, 20, 8), window, status="added", klass="bus"),
        _feature(_box(10, 100, 20, 8), window, status="confirmed", klass=""),
    ]
    lines = window_label_lines(label_boxes(features), window)
    assert len(lines) == 1
    assert lines[0].startswith("1 ")  # only the added bus


def test_overlap_writes_the_box_to_every_window_that_fully_holds_it() -> None:
    """Duplication in the overlap is accepted; cutting a box in half is not."""
    west = Window(zoom=ZOOM, x=0, y=0, size=256)
    east = Window(zoom=ZOOM, x=128, y=0, size=256)
    shared = _feature(_box(150, 10, 50, 16), west)  # whole in both windows
    assert len(window_label_lines(label_boxes([shared]), west)) == 1
    assert len(window_label_lines(label_boxes([shared]), east)) == 1
    west_only = _feature(_box(100, 10, 40, 16), west)  # straddles east's edge
    assert len(window_label_lines(label_boxes([west_only]), west)) == 1
    assert window_label_lines(label_boxes([west_only]), east) == []


def test_export_area_writes_images_and_labels(tmp_path: Path) -> None:
    """One small window end to end: image stitched, label file beside it."""
    window = Window(zoom=ZOOM, x=0, y=0, size=TILE_SIZE)
    tile = window.tiles()[0]
    path = cache_path(tmp_path, LAYER, tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (TILE_SIZE, TILE_SIZE), (10, 20, 30)).save(path)

    min_e, max_n = pixel_to_grid(0, 0, ZOOM)
    max_e, min_n = pixel_to_grid(TILE_SIZE, TILE_SIZE, ZOOM)
    area = Aoi(
        name="tiny",
        bounds=Bounds(
            min_easting=min_e, min_northing=min_n, max_easting=max_e, max_northing=max_n
        ),
    )
    images_dir, labels_dir = tmp_path / "images", tmp_path / "labels"

    windows, boxes, skipped = export_area(
        area,
        [_feature(_box(64, 64, 64, 16), window)],
        layer=LAYER,
        zoom=ZOOM,
        cache_root=tmp_path,
        fetcher=NullFetcher(),
        images_dir=images_dir,
        labels_dir=labels_dir,
        window_size=TILE_SIZE,
    )

    assert (windows, boxes, skipped) == (1, 1, [])
    assert (images_dir / "tiny_0_0.jpg").exists()
    assert (labels_dir / "tiny_0_0.txt").read_text().startswith("0 ")


def test_export_area_reports_windows_it_could_not_fill(tmp_path: Path) -> None:
    min_e, max_n = pixel_to_grid(0, 0, ZOOM)
    max_e, min_n = pixel_to_grid(TILE_SIZE, TILE_SIZE, ZOOM)
    area = Aoi(
        name="tiny",
        bounds=Bounds(
            min_easting=min_e, min_northing=min_n, max_easting=max_e, max_northing=max_n
        ),
    )

    windows, boxes, skipped = export_area(
        area,
        [],
        layer=LAYER,
        zoom=ZOOM,
        cache_root=tmp_path,  # nothing cached
        fetcher=NullFetcher(),
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        window_size=TILE_SIZE,
    )

    assert (windows, boxes) == (0, 0)
    assert skipped == ["tiny_0_0"]


def test_dataset_yaml_round_trips(tmp_path: Path) -> None:
    config = yaml.safe_load(dataset_yaml(tmp_path))
    assert config["path"] == str(tmp_path.resolve())
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["names"] == {0: "truck", 1: "bus", 2: "van"}
