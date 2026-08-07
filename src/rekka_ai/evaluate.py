"""Standalone evaluation of a trained detector.

Training logs validation metrics per epoch, but a weights file outlives its
run — this measures one directly, and adds what the training pass does not:
an operating point chosen from the PR curve, per-area count checks, the
hard-negative regression check, and a verdict against the gates from
docs/DESIGN.md §7.
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rekka_ai import labels, track
from rekka_ai.detect.sweep import Detector, YoloObb, sweep
from rekka_ai.imagery.aoi import Aoi
from rekka_ai.imagery.windows import WINDOW_SIZE
from rekka_ai.imagery.wmts import TileSource

#: The ship-decision gates (docs/DESIGN.md §7). Recall first: a missed truck costs
#: hand-labelling, a false box costs a glance.
GATE_RECALL = 0.90
GATE_PRECISION = 0.85
GATE_COUNT_ERROR = 0.10
#: The hard-negative regression check tolerates a couple of blips, not a habit.
GATE_NEGATIVE_DETECTIONS = 2


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """One class's validation numbers, as Ultralytics reports them."""

    precision: float
    recall: float
    map50: float
    map50_95: float


def ground_truth_counts(collection: dict[str, Any]) -> Counter[str]:
    """Class tallies of the human-verified features in one label file."""
    return Counter(
        f.get("properties", {}).get("class", labels.UNLABELLED)
        for f in collection.get("features", [])
        if f.get("properties", {}).get("status") in ("confirmed", "added")
    )


def pick_operating_point(
    px: list[float], p: list[float], r: list[float], *, min_recall: float = GATE_RECALL
) -> tuple[float, float, float]:
    """The operating confidence from a class's PR curve.

    The best precision among points still meeting the recall gate; if no point
    meets it, the F1 maximum instead — the honest fallback, reported as such
    by the caller since the gate then fails by construction.
    """
    meeting = [i for i in range(len(px)) if r[i] >= min_recall]
    if meeting:
        i = max(meeting, key=lambda j: p[j])
    else:
        i = max(range(len(px)), key=lambda j: _f1(p[j], r[j]))
    return px[i], p[i], r[i]


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def validation_metrics(
    weights: Path,
    data: Path,
    *,
    device: str | None = None,
    imgsz: int = WINDOW_SIZE,
    project: Path | str = "runs/eval",
    name: str | None = None,
) -> tuple[dict[str, ClassMetrics], tuple[list, list, list], list[str]]:
    """Ultralytics val over the validation split.

    Returns per-class metrics, the (px, p_curve, r_curve) PR curves, and the
    class names in index order. Needs the 'train' extra.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError(
            "eval needs the 'train' extra: uv sync --extra train"
        ) from exc
    results = YOLO(str(weights)).val(
        data=str(data),
        split="val",
        device=device,
        imgsz=imgsz,
        workers=0,
        # Like train: pin an absolute path so Ultralytics' global runs_dir
        # setting cannot scatter plots into another project's directory.
        project=str(Path(project).resolve()),
        name=name,
        verbose=False,
    )
    box = results.box
    names = [results.names[i] for i in sorted(results.names)]
    per_class = {
        name: ClassMetrics(
            precision=float(box.p[i]),
            recall=float(box.r[i]),
            map50=float(box.ap50[i]),
            map50_95=float(box.ap[i]),
        )
        for i, name in enumerate(names)
    }
    curves = (box.px.tolist(), box.p_curve.tolist(), box.r_curve.tolist())
    return per_class, curves, names


def load_detector(weights: Path, confidence: float) -> YoloObb:
    """The trained detector, keeping every class it knows.

    Built once and passed to ``count_detections`` for each area: constructing
    it loads a ~450 MB checkpoint onto the GPU, and an eval visits every
    validation and negative area in turn.
    """
    return YoloObb(str(weights), confidence=confidence, keep=frozenset(labels.CLASSES))


def count_detections(
    aoi: Aoi,
    *,
    detector: Detector,
    layer: str,
    zoom: int,
    cache_root: Path,
    fetcher: TileSource,
) -> Counter[str]:
    """Run a trained model over one area; class tallies of what it finds."""
    found = sweep(
        aoi,
        detector=detector,
        layer=layer,
        zoom=zoom,
        cache_root=cache_root,
        fetcher=fetcher,
        min_length_m=0.0,  # no gate: eval measures the model, not the filter
    )
    return Counter(d.label for d in found)


def log_eval_run(
    run_name: str, params: dict[str, Any], metrics: dict[str, float]
) -> None:
    """Record an evaluation as an MLflow run, beside the training runs."""
    import mlflow

    track.setup()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
