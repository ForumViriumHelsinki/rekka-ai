"""MLflow experiment tracking, backed by a SQLite file in the repo.

The store is a single ``runs/mlflow.db``, with model artifacts under
``runs/mlartifacts/`` — ``runs/`` is already gitignored, like every other run
output. Experiment history is valuable but not version-controllable, and a
file-based store means the UI runs with no server:

    uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db

An explicit ``MLFLOW_TRACKING_URI`` environment variable always wins over the
file default, so pointing a run at a shared server later is a shell setting,
not a code change.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKING_DB = REPO_ROOT / "runs" / "mlflow.db"
ARTIFACTS_DIR = REPO_ROOT / "runs" / "mlartifacts"
EXPERIMENT = "rekka-ai"


def tracking_uri() -> str:
    """Where runs are recorded: the env override, else the repo's SQLite file."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{TRACKING_DB}")


def setup() -> str:
    """Point MLflow at the repo store and select the experiment.

    Returns the experiment id. Called by ``rekka-ai train`` before Ultralytics
    starts; Ultralytics' own MLflow callback then logs params, per-epoch
    metrics, and the final weights into the same run automatically.
    """
    import mlflow

    TRACKING_DB.parent.mkdir(exist_ok=True)  # SQLite will not create directories
    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACTS_DIR.as_uri())
    return mlflow.set_experiment(EXPERIMENT).experiment_id
