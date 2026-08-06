"""Detection geometry, seam deduplication, and GeoJSON output.

No test here loads a model: the detector is a protocol, so the parts with a
checkable right answer are exercised without torch.
"""

import math
from collections.abc import Iterable
from pathlib import Path

import pytest

from rekka_ai.detect.detections import (
    Detection,
    longer_than,
    merge,
    to_geojson,
    within_region,
)
from rekka_ai.detect.sweep import LARGE_VEHICLE, RawDetection, sweep
from rekka_ai.imagery.aoi import Aoi, load_aois
from rekka_ai.imagery.tiles import Bounds, Tile
from rekka_ai.imagery.windows import Window

ZOOM = 15
# A 16 m by 2.5 m box aligned east-west, somewhere in Helsinki.
EAST_WEST = (
    (25496000.0, 6673000.0),
    (25496016.0, 6673000.0),
    (25496016.0, 6673002.5),
    (25496000.0, 6673002.5),
)


def _box(
    easting: float,
    northing: float,
    length: float = 16.0,
    width: float = 2.5,
    angle: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """A rectangle centred on a point, rotated ``angle`` degrees from east."""
    theta = math.radians(angle)
    corners = []
    for dx, dy in (
        (-length / 2, -width / 2),
        (length / 2, -width / 2),
        (length / 2, width / 2),
        (-length / 2, width / 2),
    ):
        corners.append(
            (
                easting + dx * math.cos(theta) - dy * math.sin(theta),
                northing + dx * math.sin(theta) + dy * math.cos(theta),
            )
        )
    return tuple(corners)


def _detection(
    corners: tuple[tuple[float, float], ...], confidence: float = 0.9
) -> Detection:
    return Detection(label=LARGE_VEHICLE, confidence=confidence, corners=corners)


def test_length_and_width_are_the_longer_and_shorter_side() -> None:
    detection = _detection(EAST_WEST)
    assert detection.length_m == pytest.approx(16.0)
    assert detection.width_m == pytest.approx(2.5)


def test_length_is_orientation_independent() -> None:
    for angle in (0, 30, 45, 90, 137):
        detection = _detection(_box(25496000.0, 6673000.0, angle=angle))
        assert detection.length_m == pytest.approx(16.0)
        assert detection.width_m == pytest.approx(2.5)


def test_centre_is_the_middle_of_the_box() -> None:
    detection = _detection(_box(25496000.0, 6673000.0, angle=42))
    assert detection.centre == pytest.approx((25496000.0, 6673000.0))


@pytest.mark.parametrize(
    ("angle", "bearing"),
    [
        (0, 90.0),  # long axis points east -> bearing 90
        (90, 0.0),  # long axis points north -> bearing 0
        (45, 45.0),
        (135, 135.0),
    ],
)
def test_heading_is_a_compass_bearing(angle: float, bearing: float) -> None:
    detection = _detection(_box(25496000.0, 6673000.0, angle=angle))
    assert detection.heading_deg == pytest.approx(bearing, abs=0.01)


def test_heading_is_modulo_180() -> None:
    """A parked vehicle's axis has no direction; nose-north equals nose-south."""
    north = _detection(_box(25496000.0, 6673000.0, angle=90)).heading_deg
    south = _detection(_box(25496000.0, 6673000.0, angle=270)).heading_deg
    assert north == pytest.approx(south, abs=0.01)
    assert 0 <= north < 180


def test_merge_drops_the_same_vehicle_seen_twice() -> None:
    """Two windows overlapping a truck report it twice, slightly offset."""
    first = _detection(_box(25496000.0, 6673000.0), confidence=0.9)
    second = _detection(_box(25496000.4, 6673000.1), confidence=0.7)
    merged = merge([first, second])
    assert len(merged) == 1
    assert merged[0].confidence == 0.9  # the more confident survives


def test_merge_keeps_vehicles_parked_nose_to_tail() -> None:
    """Adjacent trucks must not collapse into one just because they are close."""
    first = _detection(_box(25496000.0, 6673000.0))
    second = _detection(_box(25496017.0, 6673000.0))  # one metre gap
    assert len(merge([first, second])) == 2


def test_merge_keeps_different_labels_apart() -> None:
    corners = _box(25496000.0, 6673000.0)
    a = Detection(label="large vehicle", confidence=0.9, corners=corners)
    b = Detection(label="ship", confidence=0.8, corners=corners)
    assert len(merge([a, b])) == 2


def test_merge_of_nothing_is_nothing() -> None:
    assert merge([]) == []


def test_geojson_is_wgs84_with_a_closed_ring() -> None:
    collection = to_geojson([_detection(EAST_WEST)], source_layer="L", zoom=ZOOM)
    assert collection["type"] == "FeatureCollection"
    ring = collection["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1], "GeoJSON rings must close"
    assert len(ring) == 5
    lon, lat = ring[0]
    assert 24 < lon < 26 and 59 < lat < 61  # Helsinki, in degrees


def test_geojson_carries_measurements_and_provenance() -> None:
    collection = to_geojson([_detection(EAST_WEST)], source_layer="L", zoom=ZOOM)
    props = collection["features"][0]["properties"]
    assert props["label"] == LARGE_VEHICLE
    assert props["length_m"] == pytest.approx(16.0)
    assert props["source_layer"] == "L"
    assert props["zoom"] == ZOOM


class FakeDetector:
    """Reports one box at a fixed window-local position, in every window."""

    def __init__(self, corners: tuple[tuple[float, float], ...]) -> None:
        self.corners = corners
        self.calls = 0

    def detect(self, image: object) -> list[RawDetection]:
        self.calls += 1
        return [RawDetection(label=LARGE_VEHICLE, confidence=0.8, corners=self.corners)]


def test_bootstrap_georeferences_into_the_aoi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detection at window pixel (x, y) must land at that spot on the ground."""
    area = load_aois("aois/helsinki.yaml", name="tattariharjuntie")[0]

    # 64 px box at the window origin; no imagery needed, so stub the stitching.
    monkeypatch.setattr("rekka_ai.detect.sweep.load_window", lambda *a, **k: object())
    detector = FakeDetector(((0.0, 0.0), (64.0, 0.0), (64.0, 10.0), (0.0, 10.0)))

    class NullFetcher:
        def fetch_all(self, layer: str, tiles: Iterable[Tile]) -> Iterable[object]:
            return []

    found = sweep(
        area,
        detector=detector,
        layer="L",
        zoom=ZOOM,
        cache_root=tmp_path,
        fetcher=NullFetcher(),
    )

    assert detector.calls > 0
    assert found, "expected candidates"
    for detection in found:
        easting, northing = detection.centre
        # Every detection sits within the AOI, generously bounded by the window
        # grid which can extend a little past the AOI edge.
        assert area.bounds.min_easting - 300 < easting < area.bounds.max_easting + 300
        assert (
            area.bounds.min_northing - 300 < northing < area.bounds.max_northing + 300
        )
        assert detection.aoi == "tattariharjuntie"
        # 64 px at z15 is 16 m.
        assert detection.length_m == pytest.approx(16.0, abs=0.01)


def test_bootstrap_merges_duplicates_across_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same window-local box in overlapping windows is different ground...

    ...so it must NOT be merged. This guards against a merge so aggressive it
    collapses genuinely distinct vehicles.
    """
    area = Aoi(
        name="tiny",
        bounds=Bounds(
            min_easting=25496000.0,
            min_northing=6673000.0,
            max_easting=25496400.0,
            max_northing=6673400.0,
        ),
    )
    monkeypatch.setattr("rekka_ai.detect.sweep.load_window", lambda *a, **k: object())
    detector = FakeDetector(((0.0, 0.0), (64.0, 0.0), (64.0, 10.0), (0.0, 10.0)))

    class NullFetcher:
        def fetch_all(self, layer: str, tiles: Iterable[Tile]) -> Iterable[object]:
            return []

    found = sweep(
        area,
        detector=detector,
        layer="L",
        zoom=ZOOM,
        cache_root=tmp_path,
        fetcher=NullFetcher(),
        window_size=256,
        overlap_m=30,
    )
    assert len(found) == detector.calls, (
        "distinct ground positions must survive merging"
    )


def test_bootstrap_skips_windows_whose_tiles_never_landed(tmp_path: Path) -> None:
    """A tile the fetcher gave up on leaves a hole: skip the window, don't die."""
    area = Aoi(
        name="tiny",
        bounds=Bounds(
            min_easting=25496000.0,
            min_northing=6673000.0,
            max_easting=25496100.0,
            max_northing=6673100.0,
        ),
    )

    class ExplodingDetector:
        def detect(self, image: object) -> list[RawDetection]:
            raise AssertionError("no window should reach the detector")

    class NullFetcher:
        def fetch_all(self, layer: str, tiles: Iterable[Tile]) -> Iterable[object]:
            return []

    # Nothing is cached, so load_window raises FileNotFoundError for real.
    found = sweep(
        area,
        detector=ExplodingDetector(),
        layer="L",
        zoom=ZOOM,
        cache_root=tmp_path,
        fetcher=NullFetcher(),
    )
    assert found == []


def test_window_to_grid_matches_a_hand_computed_position() -> None:
    """Pin the georeferencing arithmetic against a value computed independently."""
    window = Window(zoom=ZOOM, x=1000, y=2000, size=1024)
    # z15 resolution is 0.25 m/px; grid origin is (24451424, 8388608).
    expected_easting = 24451424.0 + (1000 + 40) * 0.25
    expected_northing = 8388608.0 - (2000 + 80) * 0.25
    assert window.to_grid(40, 80) == pytest.approx(
        (expected_easting, expected_northing)
    )


def test_length_gate_drops_cars_and_vans() -> None:
    """DOTA's 'large vehicle' fires on vans; the 6 m gate is what removes them."""
    car = _detection(_box(25496000.0, 6673000.0, length=4.5, width=1.8))
    van = _detection(_box(25496100.0, 6673000.0, length=5.6, width=2.0))
    lorry = _detection(_box(25496200.0, 6673000.0, length=10.0, width=2.5))
    kept = longer_than([car, van, lorry], 6.0)
    assert [round(d.length_m) for d in kept] == [10]


def test_length_gate_is_inclusive_at_the_boundary() -> None:
    exactly = _detection(_box(25496000.0, 6673000.0, length=6.0))
    assert longer_than([exactly], 6.0) == [exactly]


def test_length_gate_uses_the_long_side_whatever_the_rotation() -> None:
    for angle in (0, 37, 90, 128):
        rotated = _detection(_box(25496000.0, 6673000.0, length=8.0, angle=angle))
        assert longer_than([rotated], 6.0), f"dropped at {angle} degrees"


def test_within_region_keeps_only_detections_centred_inside() -> None:
    from shapely.geometry import Polygon

    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    inside = _detection(_box(40.0, 40.0, length=10.0))
    outside = _detection(_box(200.0, 40.0, length=10.0))
    straddling = _detection(_box(105.0, 40.0, length=20.0))  # centre at x=105

    kept = within_region([inside, outside, straddling], region)

    assert kept == [inside]
