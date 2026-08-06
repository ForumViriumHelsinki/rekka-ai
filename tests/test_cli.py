import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rekka_ai import labels
from rekka_ai.cli import app

runner = CliRunner()

COLLECTION = """
crs: "EPSG:3067"
aois:
  - name: vuosaari_harbour
    bbox: [399000, 6675000, 399500, 6675500]
    role: positive
  - name: kivikko
    bbox: [392000, 6683000, 392500, 6683500]
    role: hard-negative
    split: validation
"""


@pytest.fixture
def collection(tmp_path: Path) -> str:
    path = tmp_path / "helsinki.yaml"
    path.write_text(COLLECTION)
    return str(path)


def test_fetch_dry_run_reports_tile_count() -> None:
    result = runner.invoke(
        app, ["fetch", "--aoi", "24.93,60.16,24.95,60.18", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Ortoilmakuva_2025_5cm z16" in result.stdout
    assert "12.50 cm/px" in result.stdout
    assert "tiles covering" in result.stdout


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _flat(output: str) -> str:
    """Collapse rich's error box so wrapped messages can be matched.

    Strip ANSI escapes first: typer forces its console into terminal mode
    whenever GITHUB_ACTIONS (or FORCE_COLOR) is set, regardless of whether
    it writes to a tty, so the box is coloured on CI but not locally. The
    escapes ride on the box borders and land between the matched words.
    """
    plain = _ANSI.sub("", output).replace("\u2502", " ")
    return " ".join(plain.split())


def _total_tiles(output: str) -> int:
    line = next(x for x in output.splitlines() if "total:" in x)
    return int(line.split("total:")[1].split()[0])


def test_fetch_dry_run_scales_with_zoom() -> None:
    def tile_count(zoom: str) -> int:
        result = runner.invoke(
            app,
            ["fetch", "--aoi", "24.93,60.16,24.95,60.18", "--zoom", zoom, "--dry-run"],
        )
        assert result.exit_code == 0
        return _total_tiles(result.stdout)

    # A level finer is four times the tiles, give or take edge alignment.
    assert tile_count("16") > 3 * tile_count("15")


def test_fetch_rejects_year_without_a_flight() -> None:
    result = runner.invoke(
        app,
        ["fetch", "--aoi", "24.93,60.16,24.95,60.18", "--year", "2022", "--dry-run"],
    )
    assert result.exit_code != 0
    assert "no orthophoto layer for 2022" in _flat(result.output)


def test_fetch_rejects_malformed_aoi() -> None:
    result = runner.invoke(app, ["fetch", "--aoi", "not-a-bbox", "--dry-run"])
    assert result.exit_code != 0


def test_fetch_covers_every_aoi_in_a_collection(collection: str) -> None:
    result = runner.invoke(app, ["fetch", "--aoi", collection, "--dry-run"])
    assert result.exit_code == 0
    assert "vuosaari_harbour:" in result.stdout
    assert "kivikko:" in result.stdout
    assert _total_tiles(result.stdout) > 0


def test_fetch_name_selects_one_aoi(collection: str) -> None:
    everything = runner.invoke(app, ["fetch", "--aoi", collection, "--dry-run"])
    one = runner.invoke(
        app, ["fetch", "--aoi", collection, "--name", "kivikko", "--dry-run"]
    )
    assert one.exit_code == 0
    assert "vuosaari_harbour:" not in one.stdout
    assert _total_tiles(one.stdout) < _total_tiles(everything.stdout)


def test_fetch_rejects_unknown_aoi_name(collection: str) -> None:
    result = runner.invoke(
        app, ["fetch", "--aoi", collection, "--name", "pasila", "--dry-run"]
    )
    assert result.exit_code != 0
    assert "available: vuosaari_harbour, kivikko" in _flat(result.output)


def test_aois_lists_the_collection(collection: str) -> None:
    result = runner.invoke(app, ["aois", "--aoi", collection])
    assert result.exit_code == 0
    assert "vuosaari_harbour" in result.stdout
    assert "hard-negative" in result.stdout
    assert "validation" in result.stdout
    assert "tiles at z16" in result.stdout


def test_aois_reports_overlaps_and_role_conflicts(tmp_path: Path) -> None:
    path = tmp_path / "overlapping.yaml"
    path.write_text(
        'crs: "EPSG:3067"\naois:\n'
        "  - {name: keep, bbox: [399000, 6675000, 399300, 6675300], role: positive}\n"
        "  - {name: drop, bbox: [399200, 6675200, 399500, 6675500], role: hard-negative}\n"
    )
    result = runner.invoke(app, ["aois", "--aoi", str(path)])
    assert result.exit_code == 0
    assert "keep x drop" in result.stdout
    assert "role conflict: positive vs hard-negative" in result.stdout


def test_aois_warns_when_validation_lacks_a_hard_negative(tmp_path: Path) -> None:
    path = tmp_path / "no-neg.yaml"
    path.write_text(
        'crs: "EPSG:3067"\naois:\n'
        "  - {name: a, bbox: [399000, 6675000, 399300, 6675300], role: hard-negative}\n"
        "  - {name: b, bbox: [402000, 6678000, 402300, 6678300], split: validation}\n"
    )
    result = runner.invoke(app, ["aois", "--aoi", str(path)])
    assert "no hard-negative area in validation" in result.stdout


def test_bootstrap_rejects_an_unknown_role(collection: str) -> None:
    result = runner.invoke(
        app,
        ["bootstrap", "--aoi", collection, "--role", "postive", "--out", "/dev/null"],
    )
    assert result.exit_code != 0
    assert "expected one of hard-negative, positive, sparse" in _flat(result.output)


def test_bootstrap_reports_when_no_area_has_the_role(tmp_path: Path) -> None:
    path = tmp_path / "only-positive.yaml"
    path.write_text(
        'crs: "EPSG:3067"\naois:\n'
        "  - {name: a, bbox: [399000, 6675000, 399300, 6675300], role: positive}\n"
    )
    result = runner.invoke(
        app, ["bootstrap", "--aoi", str(path), "--role", "sparse", "--out", "/dev/null"]
    )
    assert result.exit_code != 0
    assert "no areas with role 'sparse'" in _flat(result.output)


# --- Guard rails: the paths that protect labels and fail fast ----------------
# None of these need torch: they all exit before a detector is constructed.


def _feature(aoi: str, **properties: object) -> dict:
    ring = [[24.9, 60.1], [24.91, 60.1], [24.91, 60.11], [24.9, 60.11], [24.9, 60.1]]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        # status/class default to what `stage` writes for an unreviewed candidate.
        "properties": {
            "aoi": aoi,
            "length_m": 8.0,
            "confidence": 0.5,
            "status": "candidate",
            "class": "",
            **properties,
        },
    }


def _write_geojson(path: Path, *features: dict) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": list(features)})
    )


def _stage(labels_dir: Path, candidates: Path, *extra: str):
    return runner.invoke(
        app,
        ["stage", "--candidates", str(candidates), "--labels", str(labels_dir), *extra],
    )


def test_stage_splits_candidates_per_area(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.geojson"
    _write_geojson(
        candidates, _feature("kivikko"), _feature("kivikko"), _feature("vuosaari")
    )
    labels_dir = tmp_path / "labels"

    result = _stage(labels_dir, candidates)

    assert result.exit_code == 0
    assert "staged 2 file(s), skipped 0" in result.stdout
    assert len(labels.read(labels_dir / "kivikko.geojson")["features"]) == 2
    assert len(labels.read(labels_dir / "vuosaari.geojson")["features"]) == 1


def test_stage_marks_unreviewed_candidates(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.geojson"
    _write_geojson(candidates, _feature("kivikko"))

    assert _stage(tmp_path / "labels", candidates).exit_code == 0

    properties = labels.read(tmp_path / "labels" / "kivikko.geojson")["features"][0][
        "properties"
    ]
    assert properties["status"] == "candidate"
    assert properties["class"] == ""


def test_stage_refuses_to_overwrite_reviewed_work(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_geojson(
        labels_dir / "kivikko.geojson",
        _feature("kivikko", status="confirmed", **{"class": "truck"}),
    )
    candidates = tmp_path / "candidates.geojson"
    _write_geojson(candidates, _feature("kivikko"), _feature("kivikko"))

    result = _stage(labels_dir, candidates)

    assert result.exit_code == 0
    assert "kivikko: exists with 1 reviewed, skipping" in result.stdout
    assert "staged 0 file(s), skipped 1" in result.stdout
    assert "re-run with --force" in result.stdout
    # The reviewed feature is untouched.
    assert labels.reviewed(labels.read(labels_dir / "kivikko.geojson")) == 1


def test_stage_force_replaces_a_file(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_geojson(
        labels_dir / "kivikko.geojson",
        _feature("kivikko", status="confirmed", **{"class": "truck"}),
    )
    candidates = tmp_path / "candidates.geojson"
    _write_geojson(candidates, _feature("kivikko"), _feature("kivikko"))

    result = _stage(labels_dir, candidates, "--force")

    assert result.exit_code == 0
    assert "staged 1 file(s), skipped 0" in result.stdout
    assert len(labels.read(labels_dir / "kivikko.geojson")["features"]) == 2


def test_progress_requires_label_files(tmp_path: Path) -> None:
    result = runner.invoke(app, ["progress", "--labels", str(tmp_path)])
    assert result.exit_code != 0
    assert "no label files" in _flat(result.output)


def test_progress_reports_counts(tmp_path: Path) -> None:
    _write_geojson(
        tmp_path / "kivikko.geojson",
        _feature("kivikko", status="confirmed", **{"class": "truck"}),
        _feature("kivikko", status="rejected"),
        _feature("kivikko"),  # still a candidate
    )
    result = runner.invoke(app, ["progress", "--labels", str(tmp_path)])
    assert result.exit_code == 0
    assert "reviewed 2/3" in result.stdout
    assert "truck=1" in result.stdout


def test_progress_reports_schema_problems(tmp_path: Path) -> None:
    _write_geojson(
        tmp_path / "kivikko.geojson",
        _feature("kivikko", status="confirmed"),  # no class set
    )
    result = runner.invoke(app, ["progress", "--labels", str(tmp_path)])
    assert result.exit_code == 0
    assert "confirmed but no class set" in result.stdout


def test_export_blocks_a_positive_area_without_labels(
    collection: str, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, ["export", "--aoi", collection, "--labels", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "vuosaari_harbour: no label file" in result.stderr
    # kivikko is hard-negative: it legitimately has no label file.
    assert "kivikko" not in result.stderr


def test_export_blocks_unreviewed_candidates(collection: str, tmp_path: Path) -> None:
    _write_geojson(tmp_path / "vuosaari_harbour.geojson", _feature("vuosaari_harbour"))
    result = runner.invoke(
        app, ["export", "--aoi", collection, "--labels", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "1 unreviewed candidate(s)" in result.stderr


def test_export_rejects_an_unknown_role(collection: str, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["export", "--aoi", collection, "--role", "postive", "--labels", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "expected one of hard-negative, positive, sparse" in _flat(result.output)


def test_train_requires_an_exported_dataset(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--data", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
    assert "no dataset at" in _flat(result.output)


def test_eval_requires_weights(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "--weights", str(tmp_path / "nope.pt")])
    assert result.exit_code != 0
    assert "no weights at" in _flat(result.output)


def test_eval_requires_an_exported_dataset(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.touch()
    result = runner.invoke(
        app, ["eval", "--weights", str(weights), "--data", str(tmp_path / "nope.yaml")]
    )
    assert result.exit_code != 0
    assert "no dataset at" in _flat(result.output)


def test_detect_requires_weights(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "detect",
            "--aoi",
            "24.9,60.1,24.95,60.15",
            "--weights",
            str(tmp_path / "nope.pt"),
            "--out",
            str(tmp_path / "out.geojson"),
        ],
    )
    assert result.exit_code != 0
    assert "no weights at" in _flat(result.output)
