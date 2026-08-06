"""The labelling schema: what the web tool writes and training reads."""

import json
from pathlib import Path

import pytest

from rekka_ai import labels
from rekka_ai.geo import crs_member


def _ring(
    easting: float, northing: float, length: float, width: float
) -> list[list[float]]:
    """A closed EPSG:3879 ring for an axis-aligned box, as the tool writes it."""
    ring = [
        [easting, northing],
        [easting + length, northing],
        [easting + length, northing + width],
        [easting, northing + width],
    ]
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


def test_measurements_are_computed_in_metres() -> None:
    """The file is EPSG:3879, so a stored coordinate is already a metre."""
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
        [25496000.0, 6673000.0],
        [25496016.0, 6673000.0],
        [25496022.0, 6673011.0],
        [25496000.0, 6673008.0],
        [25496000.0, 6673000.0],
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
    """Everything stage() writes must pass validation as an unreviewed file.

    Read through `labels.read`, so a real file that lost its `crs` member --
    or never had one -- fails here rather than at export time.
    """
    for path in sorted(Path("labels").glob("*.geojson")):
        assert labels.validate(labels.read(path)) == [], path.name


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


def test_write_stamps_the_grid_crs(tmp_path: Path) -> None:
    """A file without the member is a WGS84 file by definition, so the writer
    puts it there rather than trusting every caller to remember."""
    path = tmp_path / "area.geojson"
    labels.write(path, {"type": "FeatureCollection", "features": [_feature()]})
    document = json.loads(path.read_text())
    assert document["crs"] == crs_member()
    # Ahead of the features, matching what the web tool writes: the two take
    # turns on these files and key order is a diff like any other.
    assert list(document) == ["type", "crs", "features"]


def test_write_rounds_coordinates_to_the_shared_precision(tmp_path: Path) -> None:
    """Both writers round to the same place, or a save reformats the file."""
    ring = [[25496000.123456789, 6673000.987654321]] * 5
    path = tmp_path / "area.geojson"
    labels.write(path, {"features": [_feature(ring=ring)]})
    written = labels.read(path)["features"][0]["geometry"]["coordinates"][0]
    assert written[0] == [25496000.123, 6673000.988]


def test_write_then_read_then_write_is_byte_stable(tmp_path: Path) -> None:
    """The property the whole CRS choice exists for: a file that goes through
    the tools untouched comes back out identical, so a one-box edit is a
    one-box diff."""
    path = tmp_path / "area.geojson"
    labels.write(path, {"features": [_feature(status="confirmed", klass="truck")]})
    first = path.read_text()
    labels.write(path, labels.read(path))
    assert path.read_text() == first


def test_read_refuses_a_file_from_before_the_crs_switch(tmp_path: Path) -> None:
    """Degrees read as metres make every box a few centimetres across: the
    rectangle check still passes and the dataset silently exports empty."""
    path = tmp_path / "old.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [_feature(ring=[[24.93, 60.17]] * 5)],
            }
        )
    )
    with pytest.raises(ValueError, match="EPSG::3879"):
        labels.read(path)


def _bounds():
    from rekka_ai.imagery.tiles import Bounds

    return Bounds(25495900.0, 6672900.0, 25496200.0, 6673200.0)


def test_a_box_inside_its_area_is_not_displaced() -> None:
    collection = {"features": [_feature(status="confirmed", klass="truck")]}
    assert labels.displaced(collection, _bounds()) == []


def test_a_dragged_box_is_reported_with_how_far_it_went() -> None:
    """The silent failure: export writes a label only into windows that
    contain it, and windows only cover the area, so this one exports as
    nothing while `validate` still calls it perfectly good."""
    ring = _ring(25501000.0, 6678000.0, 16.0, 3.0)
    collection = {"features": [_feature(status="confirmed", klass="truck", ring=ring)]}
    assert labels.validate(collection) == []
    problems = labels.displaced(collection, _bounds())
    assert len(problems) == 1
    assert "4808 m outside" in problems[0]


def test_displaced_leaves_broken_geometry_to_validate() -> None:
    """One voice per problem: a null geometry is `validate`'s to report."""
    feature = {"type": "Feature", "geometry": None, "properties": {}}
    assert labels.displaced({"features": [feature]}, _bounds()) == []
