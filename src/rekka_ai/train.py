"""Fine-tuning the OBB detector on an exported Helsinki dataset.

A thin wrapper on purpose: Ultralytics owns the training loop, MLflow owns
the record of it. This module's two jobs are to point both at the same place
before anything starts, and to give the run honest defaults — the dataset's
own 1024 px windows, the bootstrap weights as the starting point, and
autobatch so the GPU decides how much it can hold.

Training needs the ``train`` extra (``uv sync --extra train``), which pulls in
mlflow and ultralytics: CI and the fetch path stay usable without them.
"""

import os
from pathlib import Path
from typing import Any, cast

from rekka_ai import track
from rekka_ai.detect.sweep import DEFAULT_WEIGHTS
from rekka_ai.imagery.windows import WINDOW_SIZE

#: Long enough for a small dataset to converge, short enough to iterate.
DEFAULT_EPOCHS = 100
#: Ultralytics autobatch: size the batch to ~60% of GPU memory.
AUTOBATCH = -1


def configure_tracking() -> None:
    """Point MLflow — ours and Ultralytics' callback — at the repo store.

    Ultralytics' MLflow integration reads its destination from environment
    variables; ``setdefault`` so an explicit operator choice always wins.
    """
    os.environ.setdefault("MLFLOW_TRACKING_URI", track.tracking_uri())
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", track.EXPERIMENT)


def train(
    data: Path,
    *,
    weights: str = DEFAULT_WEIGHTS,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = AUTOBATCH,
    device: str | None = None,
    imgsz: int = WINDOW_SIZE,
    project: str = "runs/train",
    name: str | None = None,
) -> Path:
    """Fine-tune ``weights`` on an exported dataset. Returns the best weights.

    Ultralytics logs params, per-epoch metrics, and the final weights to the
    MLflow run automatically once tracking is configured.
    """
    configure_tracking()
    # Small GPUs (a laptop card sharing memory with the display) fragment
    # during the validation pass; this must be set before torch touches CUDA.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        track.setup()
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError(
            "train needs the 'train' extra: uv sync --extra train"
        ) from exc
    model = YOLO(weights)
    model.train(
        data=str(data),
        epochs=epochs,
        batch=batch,
        device=device,
        imgsz=imgsz,
        # Ultralytics joins `project` onto the user's global runs_dir setting,
        # which can point at another project's directory entirely. Resolve to
        # an absolute path so run output always lands inside this repo.
        project=str(Path(project).resolve()),
        name=name,
    )
    trainer = model.trainer
    assert trainer is not None  # set by model.train() above
    # Ultralytics sets trainer.best (path to best.pt); its MultiTrainer type
    # union does not declare it, so read it off a cast.
    return Path(str(cast(Any, trainer).best))
