"""The tracking store defaults to a SQLite file in the repo, overridable by env."""

from pathlib import Path

import pytest

from rekka_ai import track


def test_default_uri_is_sqlite_file_at_repo_root() -> None:
    assert track.tracking_uri() == f"sqlite:///{Path.cwd() / 'runs' / 'mlflow.db'}"


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracker.internal:5000")
    assert track.tracking_uri() == "http://tracker.internal:5000"


def test_setup_creates_experiment_with_repo_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow", reason="train extra not installed")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(track, "ARTIFACTS_DIR", tmp_path / "mlartifacts")

    experiment_id = track.setup()

    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == track.EXPERIMENT
    assert experiment.artifact_location == (tmp_path / "mlartifacts").as_uri()
    # Idempotent: a second call selects the same experiment, not a duplicate.
    assert track.setup() == experiment_id
