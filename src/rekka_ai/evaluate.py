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

from shapely import STRtree
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from rekka_ai import labels, track
from rekka_ai.detect.detections import Detection
from rekka_ai.detect.sweep import MIN_LENGTH_M, Detector, YoloObb, sweep
from rekka_ai.imagery.aoi import Aoi
from rekka_ai.imagery.windows import WINDOW_SIZE
from rekka_ai.imagery.wmts import TileSource

#: The ship-decision gates (docs/DESIGN.md §7). Recall first: a missed truck costs
#: hand-labelling, a false box costs a glance.
GATE_RECALL = 0.90
GATE_PRECISION = 0.85
GATE_COUNT_ERROR = 0.10
#: The hard-negative regression check tolerates a couple of blips, not a habit.
#: Counted over *unexplained* detections — see ``unexplained``.
GATE_NEGATIVE_DETECTIONS = 2
#: A detection this far onto a labelled vehicle is that vehicle, so a negative
#: area is not marked down for finding it. Loose on purpose: the question is
#: "did the model invent something", which a half-overlapping box answers no to.
NEGATIVE_MATCH_IOU = 0.3
#: Fewest ground-truth trucks an area needs before its count check may gate.
#: Derived from the tolerance rather than picked: at 10 trucks a 10% error is
#: exactly one box, and below that the gate decides on a fraction of a box —
#: it measures the imagery, not the model (docs/DESIGN.md §7). Skipping those
#: areas leaves no blind spot: false positives anywhere in the split still
#: land on the precision gate, which is measured over every validation window.
GATE_COUNT_MIN_TRUCKS = round(1 / GATE_COUNT_ERROR)
#: Below this, "operating point" stops meaning anything: a point near conf=0
#: keeps every raw proposal the detector makes. A curve whose recall never
#: clears the gate above this floor has picked "keep everything" rather than
#: a point anyone would deploy at — the recall gate then fails by
#: construction instead of silently reporting a pass at a useless confidence.
MIN_OPERATING_CONFIDENCE = 0.05


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


def count_gate(found: int, truth: int) -> tuple[bool | None, str]:
    """One area's truck-count verdict, or ``None`` when it may not gate.

    The product's question is "how many vehicles per site", so per-area count
    error is a gate rather than box geometry — but only where the area holds
    enough trucks for the tolerance to cover a whole box
    (``GATE_COUNT_MIN_TRUCKS``). Below that, including the no-trucks case, the
    count is reported and left out of the verdict: r1-kamppi failed its area
    gate on 8 trucks because one seam-smeared box is 12% of 8, which said
    nothing about the model (docs/DESIGN.md §7).
    """
    if truth < GATE_COUNT_MIN_TRUCKS:
        return None, (
            f"{found} vs {truth} trucks — not gated, under {GATE_COUNT_MIN_TRUCKS}"
        )
    error = (found - truth) / truth
    return abs(error) <= GATE_COUNT_ERROR, f"{found} vs {truth} trucks ({error:+.0%})"


def pick_operating_point(
    px: list[float],
    p: list[float],
    r: list[float],
    *,
    min_recall: float = GATE_RECALL,
    min_confidence: float = MIN_OPERATING_CONFIDENCE,
) -> tuple[float, float, float]:
    """The operating confidence from a class's PR curve.

    The best precision among points at or above ``min_confidence`` that still
    meet the recall gate; if none qualify, the F1 maximum among points at or
    above the floor instead — the honest fallback, reported as such by the
    caller since the gate then fails by construction. Points below the floor
    are never candidates, even for the fallback: a "point" near conf=0 keeps
    every raw proposal the detector makes, which is not an operating point at
    all.
    """
    candidates = [i for i in range(len(px)) if px[i] >= min_confidence]
    meeting = [i for i in candidates if r[i] >= min_recall]
    if meeting:
        i = max(meeting, key=lambda j: p[j])
    else:
        i = max(candidates, key=lambda j: _f1(p[j], r[j]))
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


def sweep_area(
    aoi: Aoi,
    *,
    detector: Detector,
    layer: str,
    zoom: int,
    cache_root: Path,
    fetcher: TileSource,
    min_length_m: float = MIN_LENGTH_M,
) -> list[Detection]:
    """What a trained model would report over one area.

    The length floor is on by default, unlike the per-class metrics: the two
    measurement layers ask different questions (docs/DESIGN.md §7). Standard
    metrics judge the *model*, so they see every raw proposal. The operational
    gates — per-area counts and the negative check — judge what a city sweep
    would *emit*, and a sweep runs ``detect``, which drops everything under
    ``MIN_LENGTH_M`` as measured noise. Without the floor the gates counted
    boxes the pipeline would never produce: on round 2, sixteen of the twenty
    detections failing the negative check were 2-4 m slivers of parked cars
    (docs/rounds.md, 2026-08-11).
    """
    return sweep(
        aoi,
        detector=detector,
        layer=layer,
        zoom=zoom,
        cache_root=cache_root,
        fetcher=fetcher,
        min_length_m=min_length_m,
    )


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
    return Counter(
        d.label
        for d in sweep_area(
            aoi,
            detector=detector,
            layer=layer,
            zoom=zoom,
            cache_root=cache_root,
            fetcher=fetcher,
        )
    )


def unexplained(
    found: list[Detection],
    collection: dict[str, Any],
    *,
    iou_threshold: float = NEGATIVE_MATCH_IOU,
) -> list[Detection]:
    """Detections in a negative area that no labelled vehicle accounts for.

    The regression check asks "has fine-tuning started pulling lookalikes in",
    and the honest way to count that is to forgive every detection sitting on
    a vehicle the area really holds. A negative area is negative about
    *targets*, not empty: `r1-puotinharju` holds 93 cars and 5 vans,
    `r1-marjaniemi` two vans. Counting raw detections marked a model down by
    an order of magnitude for being right about them (docs/rounds.md,
    2026-08-11).

    Class is deliberately ignored in the match. The question is whether the
    model invented a vehicle, not whether it named it correctly — naming is
    what the per-class metrics are for.
    """
    truth = [
        shape(f["geometry"])
        for f in collection.get("features", [])
        if f.get("properties", {}).get("status") in ("confirmed", "added")
    ]
    if not truth:
        return list(found)
    tree = STRtree(truth)
    out = []
    for detection in found:
        box = detection.polygon()
        if not any(_iou(box, truth[j]) > iou_threshold for j in tree.query(box)):
            out.append(detection)
    return out


def _iou(a: BaseGeometry, b: BaseGeometry) -> float:
    if not a.intersects(b):
        return 0.0
    intersection = a.intersection(b).area
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def log_eval_run(
    run_name: str, params: dict[str, Any], metrics: dict[str, float]
) -> None:
    """Record an evaluation as an MLflow run, beside the training runs."""
    import mlflow

    track.setup()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
