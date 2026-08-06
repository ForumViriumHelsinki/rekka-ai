"""The labelling schema: what the web tool writes and training reads."""

import json
from pathlib import Path

import pytest

from rekka_ai import labels
from rekka_ai.geo import to_wgs84


def _ring(
    easting: float, northing: float, length: float, width: float
) -> list[list[float]]:
    """A closed WGS84 ring for an axis-aligned box, as the tool would write it."""
    corners = [
        (easting, northing),
        (easting + length, northing),
        (easting + length, northing + width),
        (easting, northing + width),
    ]
    ring = [list(to_wgs84(e, n)) for e, n in corners]
    return [*ring, ring[0]]


def _feature(status: str = "candidate", klass: str = "", **geometry: object) -> dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                geometry.get("ring") or _ring(25496000.0, 6673000.0, 16.0, 3.0)
            ],
        },
        "properties": {"status": status, "class": klass},
    }


def test_measurements_are_computed_in_metres_not_degrees() -> None:
    """The file is WGS84; measuring it as degrees would give ~0.0001."""
    m = labels.measurements(_ring(25496000.0, 6673000.0, 16.0, 3.0))
    assert m["length_m"] == pytest.approx(16.0, abs=0.01)
    assert m["width_m"] == pytest.approx(3.0, abs=0.01)


def test_measurements_ignore_the_closing_coordinate() -> None:
    ring = _ring(25496000.0, 6673000.0, 16.0, 3.0)
    assert labels.measurements(ring) == labels.measurements(ring[:4])


def test_confirmed_without_a_class_is_a_problem() -> None:
    problems = labels.validate({"features": [_feature(status="confirmed")]})
    assert any("no class set" in p for p in problems)


def test_unknown_class_is_a_problem() -> None:
    problems = labels.validate(
        {"features": [_feature(status="confirmed", klass="lorry")]}
    )
    assert any("not in" in p for p in problems)


def test_non_rectangle_is_a_problem() -> None:
    """A freehand polygon cannot become a YOLO-OBB label."""
    blob = [
        [24.93, 60.16],
        [24.94, 60.16],
        [24.945, 60.18],
        [24.93, 60.175],
        [24.93, 60.16],
    ]
    problems = labels.validate(
        {"features": [_feature(status="added", klass="truck", ring=blob)]}
    )
    assert any("not a rectangle" in p for p in problems)


def test_a_proper_rectangle_passes() -> None:
    assert (
        labels.validate({"features": [_feature(status="confirmed", klass="truck")]})
        == []
    )


def test_rejects_need_no_class() -> None:
    assert labels.validate({"features": [_feature(status="rejected")]}) == []


def test_reviewed_counts_only_verdicts() -> None:
    collection = {
        "features": [
            _feature(status="candidate"),
            _feature(status="confirmed", klass="truck"),
            _feature(status="rejected"),
            _feature(status="added", klass="van"),
        ]
    }
    assert labels.reviewed(collection) == 3
    counts = labels.summarise(collection)
    assert counts["truck"] == 1 and counts["van"] == 1 and counts["candidate"] == 1


def test_staged_files_are_valid(tmp_path: Path) -> None:
    """Everything stage() writes must pass validation as an unreviewed file."""
    for path in sorted(Path("labels").glob("*.geojson")):
        collection = json.loads(path.read_text())
        assert labels.validate(collection) == [], path.name


def test_validate_reports_broken_geometry_instead_of_raising() -> None:
    """The diagnostic must survive the files it exists to diagnose.

    A null geometry is legal GeoJSON and an empty coordinate list is a
    plausible result of a bad edit; both used to raise out of validate(),
    which turned a reportable problem into a traceback.
    """
    for geometry in (None, {}, {"type": "Polygon", "coordinates": []}):
        collection = {
            "features": [
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {"status": "confirmed", "class": "truck"},
                }
            ]
        }
        problems = labels.validate(collection)
        assert problems == ["feature 0: no polygon ring to check"], geometry


def test_write_normalizes_integral_floats_for_js_round_trip(tmp_path: Path) -> None:
    """json.dumps writes 175.0 as "175.0", JSON.stringify as "175". Normalizing
    keeps the file byte-stable across the Python CLI and the web tool."""
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[]]},
                "properties": {"heading_deg": 175.0, "length_m": 12.5},
            }
        ],
    }
    path = tmp_path / "area.geojson"
    labels.write(path, collection)
    text = path.read_text()
    assert '"heading_deg": 175' in text
    assert "175.0" not in text
    assert '"length_m": 12.5' in text
