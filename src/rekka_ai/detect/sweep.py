"""Running an oriented-box detector across an area.

One sweep: window the AOI, detect in each window, georeference the boxes, and
merge duplicates across the seams. Three callers use it for different ends --
``bootstrap`` pre-annotates with zero-shot weights so the first labelling round
is correction rather than drawing, ``detect`` produces results with trained
weights, and ``eval`` counts what a model finds on known ground. The machinery
is the same in all three; only the weights and the framing differ.

The detector is reached through a narrow protocol so the pipeline around it --
windowing, georeferencing, and merging, all of which have a checkable right
answer -- is testable without torch or downloaded weights.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL.Image import Image

from rekka_ai.detect.detections import DEFAULT_IOU, Detection, longer_than, merge
from rekka_ai.imagery.aoi import Aoi
from rekka_ai.imagery.windows import (
    OVERLAP_M,
    WINDOW_SIZE,
    Window,
    load_window,
    windows_covering,
)
from rekka_ai.imagery.wmts import TileSource, ensure_cached

#: DOTAv1's vehicle classes. ``large vehicle`` is the one that maps onto trucks
#: and buses; ``small vehicle`` is cars, and including it floods the output.
LARGE_VEHICLE = "large vehicle"
DEFAULT_WEIGHTS = "yolo11x-obb.pt"
DEFAULT_CONFIDENCE = 0.25

#: The annotation guide's length gate: a truck is at least this long, which
#: separates it from the vans and cars "large vehicle" also fires on.
MIN_LENGTH_M = 6.0


@dataclass(frozen=True, slots=True)
class RawDetection:
    """A detection in window-local pixels, as the model reports it."""

    label: str
    confidence: float
    corners: tuple[tuple[float, float], ...]


class Detector(Protocol):
    """Anything that finds oriented boxes in an image."""

    def detect(self, image: Image) -> list[RawDetection]: ...


class YoloObb:
    """Ultralytics OBB weights, pretrained on DOTAv1.

    Imported lazily: ultralytics pulls in torch, which is an optional extra
    (``uv sync --extra detect``) so that ``fetch`` and ``aois`` stay usable —
    and CI stays fast — without a multi-gigabyte install.

    ``keep`` filters by class name: the zero-shot bootstrap passes DOTA's
    ``large vehicle``; a fine-tuned model passes ``None`` and keeps every
    class it was trained on.
    """

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        keep: frozenset[str] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                "bootstrap needs the 'detect' extra: uv sync --extra detect"
            ) from exc
        self._model = YOLO(weights)
        self.confidence = confidence
        self.keep = keep

    def detect(self, image: Image) -> list[RawDetection]:
        results = self._model.predict(
            image, conf=self.confidence, imgsz=image.width, verbose=False
        )
        found = []
        for result in results:
            # ultralytics types predict() as returning Results | Tensor; only
            # the Results branch happens for image input.
            obb = getattr(result, "obb", None)
            names = getattr(result, "names", {})
            if obb is None:
                continue
            for corners, cls, conf in zip(
                obb.xyxyxyxy.tolist(), obb.cls.tolist(), obb.conf.tolist(), strict=True
            ):
                label = names[int(cls)]
                if self.keep is None or label in self.keep:
                    found.append(
                        RawDetection(
                            label=label,
                            confidence=float(conf),
                            corners=tuple((float(x), float(y)) for x, y in corners),
                        )
                    )
        return found


def sweep(
    aoi: Aoi,
    *,
    detector: Detector,
    layer: str,
    zoom: int,
    cache_root: Path,
    fetcher: TileSource,
    window_size: int = WINDOW_SIZE,
    overlap_m: float = OVERLAP_M,
    iou_threshold: float = DEFAULT_IOU,
    min_length_m: float = MIN_LENGTH_M,
    on_skip: Callable[[str], None] | None = None,
) -> list[Detection]:
    """Detect over one AOI and return merged, georeferenced detections.

    ``on_skip`` is called for each window dropped for missing tiles. Reporting
    belongs to the caller: a library that writes to stdout cannot be silenced,
    and cannot be tested for what it says.
    """
    report = on_skip or _silent
    found: list[Detection] = []
    for index, window in enumerate(
        _windows(aoi, zoom, window_size, overlap_m), start=1
    ):
        # Cache-first, so this is free once the AOI has been fetched.
        ensure_cached(fetcher, layer, window.tiles())
        try:
            image = load_window(window, cache_root, layer)
        except FileNotFoundError as exc:
            # A tile the fetcher gave up on (recorded in its failures) leaves
            # a hole in this window. Skip it rather than kill the sweep; the
            # window's neighbours still cover most of its ground.
            report(f"  {aoi.name}: window {index} skipped ({exc})")
            continue
        found.extend(_georeference(detector.detect(image), window, aoi.name))
    return longer_than(merge(found, iou_threshold), min_length_m)


def _silent(message: str) -> None:
    """Default reporter: a sweep says nothing unless the caller asks."""


def _windows(aoi: Aoi, zoom: int, size: int, overlap_m: float) -> Iterator[Window]:
    return windows_covering(aoi.bounds, zoom, size=size, overlap_m=overlap_m)


def _georeference(
    raw: list[RawDetection], window: Window, aoi_name: str
) -> list[Detection]:
    return [
        Detection(
            label=r.label,
            confidence=r.confidence,
            corners=tuple(window.to_grid(x, y) for x, y in r.corners),
            aoi=aoi_name,
        )
        for r in raw
    ]
