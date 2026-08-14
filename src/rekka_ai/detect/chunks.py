"""Cell-by-cell bookkeeping that makes a city-scale sweep resumable.

A city-wide `detect` is hours of work against a polite tile fetcher, and the
plain path holds every detection in memory until the final write -- so a crash
at 95% loses the lot. This module splits a region into a grid of cells, each
swept and written on its own, with an append-only manifest recording what is
finished. A re-run picks up where the last one stopped.

Why a grid rather than one unit per region feature (measured 2026-08-14 on
the 108-part Helsinki land region):

    one global bbox (the unchunked path)   437,736 tiles
    one unit per region feature            409,071 tiles   (7% saved)
    1 km grid cells                        323,571 tiles  (26% saved)
    500 m grid cells                       302,175 tiles  (31% saved)
    250 m grid cells                       309,246 tiles  (worse again)
    region area / tile area (the floor)    203,445 tiles

Feature bboxes overlap heavily, so per-feature saves almost nothing, and the
features are far too uneven to checkpoint on: the largest is 16.3 km2 against
a 0.44 km2 median, and 40 of 108 are under 0.1 km2. A grid gives units of
even size *and* skips the sea. Below ~500 m the per-cell rounding up to whole
tiles starts to cost more than the sea it saves, which is why 250 m reverses
the gain.

Cells are disjoint, and `detections.within_region` assigns a detection to
whichever cell its *centre* falls in, so no detection is claimed twice. The
merge still runs the same global NMS the unchunked path runs, because window
overlap at a cell seam can produce a genuine duplicate pair whose centres
land either side of the line. `detect` reports how many that merge removes,
so a cell size that starts hiding vehicles shows up as a number rather than
as quietly missing trucks.

**Chunked output is not identical to the unchunked path, and cannot be.**
Measured on a 3.94 km2 Vuosaari region (2026-08-14, round-3 weights, conf
0.25): 1,924 detections chunked against 1,930 plain, and matching the two by
IoU > 0.5 leaves 28 only-plain and 22 only-chunked -- about 1.5% either way
for a net -0.3%. The disagreements are *not* concentrated at cell seams; they
run out to 400 m from one, and roughly half sit below conf 0.4. The cause is
window alignment: the plain path lays its 1024 px windows across the whole
region bbox, the chunked path lays them per cell, so a vehicle falls at a
different offset inside its window and a near-threshold detection flips. The
unchunked result is not ground truth here -- it is one tiling among many, and
changing --cell-size shifts the same way. Worth knowing before comparing two
sweeps that used different cell sizes and reading the delta as real change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from rekka_ai.detect.detections import Detection
from rekka_ai.imagery.tiles import Bounds

#: Grid cell edge in metres. 1 km is the knee of the table above: it keeps
#: 26% of the saving with 311 units on Helsinki, where 500 m needs 1,093 for
#: another 5 points.
DEFAULT_CELL_SIZE_M = 1000.0

MANIFEST_NAME = "manifest.jsonl"
CELLS_DIRNAME = "cells"

#: Statuses a cell can hold in the manifest. ``running`` is written *before*
#: the sweep starts, so a process killed mid-cell leaves the evidence behind
#: and the resume knows to redo it rather than trust a half-written file.
PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


@dataclass(frozen=True, slots=True)
class GridCell:
    """One unit of work: a square of ground, identified by its grid indices.

    Indices come from absolute EPSG:3879 coordinates divided by the cell
    size, not from the region's corner, so a cell's name does not change if
    the region file is later edited or clipped. That is what lets a manifest
    from an earlier run stay meaningful.
    """

    col: int
    row: int
    size: float

    @property
    def name(self) -> str:
        return f"e{self.col}n{self.row}"

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            min_easting=self.col * self.size,
            min_northing=self.row * self.size,
            max_easting=(self.col + 1) * self.size,
            max_northing=(self.row + 1) * self.size,
        )

    def polygon(self) -> BaseGeometry:
        b = self.bounds
        return box(b.min_easting, b.min_northing, b.max_easting, b.max_northing)


def grid_cells(region: BaseGeometry, size: float) -> list[GridCell]:
    """Every grid cell that actually touches ``region``, in a stable order.

    Cells that miss the region entirely are never created, which is where the
    saving over sweeping one bounding box comes from -- most of Helsinki's
    bbox is sea.
    """
    if size <= 0:
        raise ValueError(f"cell size must be positive, got {size}")
    min_e, min_n, max_e, max_n = region.bounds
    parts = list(region.geoms) if hasattr(region, "geoms") else [region]
    tree = STRtree(parts)

    out: list[GridCell] = []
    for col in range(math.floor(min_e / size), math.floor(max_e / size) + 1):
        for row in range(math.floor(min_n / size), math.floor(max_n / size) + 1):
            cell = GridCell(col=col, row=row, size=size)
            square = cell.polygon()
            # Positive shared *area*, not `intersects`: a cell that merely
            # touches the region along an edge or a corner holds none of it,
            # so it would sweep pure sea and then have every detection thrown
            # away by the centre test. The tree query is the cheap filter; the
            # area check is what decides.
            if any(
                parts[j].intersection(square).area > 0.0 for j in tree.query(square)
            ):
                out.append(cell)
    # Sorted so a run's order is reproducible and progress reads geographically.
    return sorted(out, key=lambda c: (c.col, c.row))


def run_key(**params: Any) -> dict[str, Any]:
    """The parameters a resume must not silently change.

    Two models' detections in one output file would be wrong in a way nothing
    downstream could see, so the manifest pins what produced it and a resume
    refuses a mismatch rather than appending to it.
    """
    return {
        k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(params.items())
    }


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def cell_path(root: Path, name: str) -> Path:
    """Where one cell's detections live.

    Always GeoJSON: these are internal to the run, and reading them back for
    the merge should not need geopandas when the final output might not.
    """
    return root / CELLS_DIRNAME / f"{name}.geojson"


def append(root: Path, record: dict[str, Any]) -> None:
    """Add one line to the manifest.

    Append-only on purpose: rewriting a state file is what loses it when the
    power goes, and the last line for a cell is its current status.
    """
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def read_manifest(
    root: Path,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """The run header and the latest record per cell.

    A truncated final line (killed mid-write) is skipped rather than fatal:
    the cell it described simply reads as unfinished, which is the safe way
    to be wrong here.
    """
    path = manifest_path(root)
    if not path.exists():
        return None, {}
    header: dict[str, Any] | None = None
    cells: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "run":
            header = record
        elif "cell" in record:
            cells[record["cell"]] = record
    return header, cells


def to_detections(geojson: dict[str, Any]) -> list[Detection]:
    """Rebuild Detection objects from a cell file written by ``detections.write``.

    The closing coordinate GeoJSON rings carry is dropped again -- Detection
    stores the four corners, and measurements are recomputed from them.
    """
    out: list[Detection] = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Polygon":
            continue
        ring = geometry["coordinates"][0]
        corners = [(float(x), float(y)) for x, y in ring[:-1]]
        if len(corners) != 4:
            continue
        properties = feature.get("properties", {})
        out.append(
            Detection(
                label=properties.get("label", ""),
                confidence=float(properties.get("confidence", 0.0)),
                corners=tuple(corners),  # type: ignore[arg-type]
                aoi=properties.get("aoi", ""),
            )
        )
    return out


def load_cells(root: Path, names: list[str]) -> list[Detection]:
    """Every detection written by the named cells."""
    found: list[Detection] = []
    for name in names:
        path = cell_path(root, name)
        if path.exists():
            found.extend(to_detections(json.loads(path.read_text(encoding="utf-8"))))
    return found
