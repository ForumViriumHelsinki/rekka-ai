"""Training is Ultralytics plus MLflow; what this module owns is pointing both
at the same store, and refusing to start without a dataset.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rekka_ai import track
from rekka_ai.cli import app
from rekka_ai.train import configure_tracking

runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    return monkeypatch


def test_configure_tracking_defaults_to_the_repo_store() -> None:
    configure_tracking()
    import os

    assert os.environ["MLFLOW_TRACKING_URI"] == track.tracking_uri()
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == track.EXPERIMENT


def test_configure_tracking_respects_an_explicit_choice() -> None:
    import os

    os.environ["MLFLOW_EXPERIMENT_NAME"] = "someone-elses"
    configure_tracking()
    assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "someone-elses"


def test_train_requires_a_dataset(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--data", str(tmp_path / "missing.yaml")])
    assert result.exit_code != 0
    assert "rekka-ai export" in result.output
