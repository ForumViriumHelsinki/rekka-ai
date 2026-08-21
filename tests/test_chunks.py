"""Cell splitting and manifest bookkeeping for resumable sweeps."""

import json
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, box

from rekka_ai.detect.chunks import (
    DONE,
    FAILED,
    GridCell,
    append,
    cell_path,
    grid_cells,
    load_cells,
    read_manifest,
    run_key,
    to_detections,
)
from rekka_ai.detect.detections import Detection, to_geojson


def test_grid_cells_skip_ground_outside_the_region() -> None:
    """The saving over sweeping a bounding box is entirely these cells."""
    # Two squares on a diagonal: the bbox is 4 cells, only 2 hold region.
    region = MultiPolygon([box(0, 0, 900, 900), box(1000, 1000, 1900, 1900)])
    cells = grid_cells(region, 1000.0)
    assert [c.name for c in cells] == ["e0n0", "e1n1"]


def test_grid_cell_names_come_from_absolute_coordinates() -> None:
    """A cell keeps its name when the region around it changes.

    Names indexed from the region's own corner would renumber every cell the
    moment the region file was clipped or extended, and the manifest from the
    previous run would then silently describe different ground.
    """
    small = grid_cells(box(25_500_000, 6_670_000, 25_500_900, 6_670_900), 1000.0)
    wider = grid_cells(box(25_499_000, 6_669_000, 25_500_900, 6_670_900), 1000.0)
    assert small[0].name in {c.name for c in wider}
    assert small[0].name == "e25500n6670"


def test_grid_cells_are_disjoint() -> None:
    cells = grid_cells(box(0, 0, 2900, 2900), 1000.0)
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            assert a.polygon().intersection(b.polygon()).area == 0.0


def test_grid_cell_bounds_match_its_polygon() -> None:
    cell = GridCell(col=3, row=4, size=500.0)
    b = cell.bounds
    assert (b.min_easting, b.min_northing) == (1500.0, 2000.0)
    assert (b.max_easting, b.max_northing) == (2000.0, 2500.0)
    assert cell.polygon().bounds == (1500.0, 2000.0, 2000.0, 2500.0)


def test_grid_cells_rejects_a_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        grid_cells(box(0, 0, 10, 10), 0.0)


def test_run_key_is_order_independent_and_path_safe() -> None:
    """Two spellings of the same run must compare equal, or a resume would
    refuse itself over key order or a Path/str difference."""
    a = run_key(zoom=16, weights=Path("runs/best.pt"), confidence=0.15)
    b = run_key(confidence=0.15, weights="runs/best.pt", zoom=16)
    assert a == b


def test_manifest_keeps_the_last_status_for_a_cell(tmp_path: Path) -> None:
    """running -> done is the normal path; the resume must see 'done'."""
    append(tmp_path, {"type": "run", "zoom": 16})
    append(tmp_path, {"cell": "e1n1", "status": "running"})
    append(tmp_path, {"cell": "e1n1", "status": DONE, "detections": 7})
    append(tmp_path, {"cell": "e2n2", "status": FAILED, "error": "boom"})

    header, cells = read_manifest(tmp_path)
    assert header is not None and header["zoom"] == 16
    assert cells["e1n1"]["status"] == DONE
    assert cells["e1n1"]["detections"] == 7
    assert cells["e2n2"]["status"] == FAILED


def test_manifest_survives_a_truncated_final_line(tmp_path: Path) -> None:
    """A kill mid-write leaves half a line; that cell must read as unfinished
    rather than taking the whole manifest down with it."""
    append(tmp_path, {"type": "run", "zoom": 16})
    append(tmp_path, {"cell": "e1n1", "status": DONE})
    with (tmp_path / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"cell": "e2n2", "sta')

    header, cells = read_manifest(tmp_path)
    assert header is not None
    assert cells["e1n1"]["status"] == DONE
    assert "e2n2" not in cells


def test_missing_manifest_is_not_an_error(tmp_path: Path) -> None:
    assert read_manifest(tmp_path) == (None, {})


def _detection(east: float, north: float) -> Detection:
    return Detection(
        label="truck",
        confidence=0.9,
        corners=(
            (east, north),
            (east + 10, north),
            (east + 10, north + 3),
            (east, north + 3),
        ),
        aoi="e1n1",
    )


def test_cell_files_round_trip_back_into_detections(tmp_path: Path) -> None:
    """The merge rebuilds Detections from what `write` produced, so the two
    have to agree about the closing ring coordinate and the properties."""
    original = [_detection(25_500_000, 6_670_000), _detection(25_500_100, 6_670_050)]
    path = cell_path(tmp_path, "e1n1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_geojson(original, source_layer="test", zoom=16)))

    restored = load_cells(tmp_path, ["e1n1"])
    assert len(restored) == 2
    for before, after in zip(original, restored, strict=True):
        assert after.label == before.label
        assert after.confidence == pytest.approx(before.confidence)
        assert len(after.corners) == 4
        assert after.centre == pytest.approx(before.centre)
        assert after.length_m == pytest.approx(before.length_m)


def test_load_cells_ignores_a_cell_that_was_never_written(tmp_path: Path) -> None:
    assert load_cells(tmp_path, ["e9n9"]) == []


def test_to_detections_skips_non_polygon_features() -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1, 2]},
                "properties": {"label": "truck"},
            },
            {"type": "Feature", "geometry": None, "properties": {}},
        ],
    }
    assert to_detections(geojson) == []
