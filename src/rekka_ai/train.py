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
#: Epochs without a new best fitness before training stops. Ultralytics
#: defaults to 100, which on a 100-epoch run never fires. Measured on round 1
#: (2026-08-11): fitness last improved at epoch 46 and the longest plateau
#: preceding a real improvement was 20 epochs, so 20 would have survived by a
#: single epoch and 10 would have stopped in that plateau and kept worse
#: weights. 30 clears the observed worst case with margin and still ends the
#: run 24 epochs early. It can only ever save time — `epochs` remains the cap.
DEFAULT_PATIENCE = 30
#: Ultralytics reads -1 as autobatch, which is *not* the default here.
#: Measured 2026-08-11 on a 16 GB card at imgsz 1024 with yolo11x-obb:
#: autobatch chose **1**, using only 3.7 GB, four times more optimizer steps
#: per epoch, and BatchNorm statistics taken from single images. Every round
#: so far ran at 4 only because it was passed by hand. 4 fills ~10 GB and is
#: what the recorded rounds used, so it is the default; pass -1 to let
#: Ultralytics guess, or a smaller number on a smaller card.
DEFAULT_BATCH = 4
#: Ultralytics seeds torch, numpy and its own dataloader shuffling from this,
#: with `deterministic=True`, so two runs at the same seed on the same data are
#: identical. Varying it is the only way to see the run-to-run spread — which is
#: the number that says whether a round-to-round difference means anything.
DEFAULT_SEED = 0


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
    patience: int = DEFAULT_PATIENCE,
    batch: int = DEFAULT_BATCH,
    seed: int = DEFAULT_SEED,
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
        patience=patience,
        batch=batch,
        seed=seed,
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
