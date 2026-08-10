"""Propose the next labelling AOIs from OSM industrial land and a trained model.

``mine`` is active learning, not hand-picking: industrial polygons from OSM
are gridded into 300 m cells, cells that overlap the existing collection are
dropped, a geographically dispersed pool is swept with the current weights,
and a stratified shortlist is written as a proposal YAML. The proposal never
touches ``aois/helsinki.yaml`` — a human reviews it and copies accepted
entries in by hand.

Signals are recorded raw rather than collapsed into an opaque score.
Confidence is not calibrated uncertainty; strata quotas keep the shortlist
diverse even when one signal dominates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import yaml
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from rekka_ai.detect.detections import Detection
from rekka_ai.geo import (
    COORD_DECIMALS,
    TM35FIN,
    WGS84,
    crs_member,
    to_grid,
    transformer,
)
from rekka_ai.imagery.aoi import Aoi
from rekka_ai.imagery.tiles import Bounds, count_tiles
from rekka_ai.osm import OsmCache, OsmFeature

#: Cell edge length in metres, matching the current collection's plot size.
CELL_SIZE_M = 300.0
#: Fraction of a cell that must intersect industrial landuse to be eligible.
MIN_INDUSTRIAL_COVERAGE = 0.15
#: How far a new cell must sit from an existing AOI, in metres (source CRS).
EXCLUSION_BUFFER_M = 50.0
#: Minimum centre-to-centre distance between selected proposals, in metres.
MIN_SEPARATION_M = 600.0
#: Cap on detections per cell so one dense yard cannot soak the budget.
MAX_CANDIDATES = 80
#: Length band where truck/van confusion lives (see docs/DESIGN.md §5).
SIZE_BOUNDARY_MIN_M = 4.0
SIZE_BOUNDARY_MAX_M = 8.0
#: Half-width of the "near operating confidence" band.
NEAR_THRESHOLD_HALF_WIDTH = 0.10

#: Stratum quotas as fractions of ``count``. Order matters for fill priority
#: when a stratum runs dry: near-threshold first, then size-boundary, dense,
#: quiet.
STRATUM_QUOTAS: tuple[tuple[str, float], ...] = (
    ("near_threshold", 0.40),
    ("size_boundary", 0.25),
    ("dense", 0.20),
    ("quiet", 0.15),
)

DEFAULT_COUNT = 12
DEFAULT_POOL_SIZE = 100
DEFAULT_SEED = 0


@dataclass(frozen=True, slots=True)
class Cell:
    """One candidate AOI cell in EPSG:3067, the collection's own CRS."""

    col: int
    row: int
    bbox: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y
    industrial_coverage: float
    municipality_ref: str
    #: Labelling round this cell was proposed in, e.g. ``"r2"``. Prefixed onto
    #: ``name`` so a round's AOIs and label files sort and group together —
    #: empty for callers that don't track rounds.
    round_label: str = ""

    @property
    def name(self) -> str:
        """Stable name from municipality code and grid indices, not rank."""
        stem = f"mine-{self.municipality_ref}-{self.col}-{self.row}"
        return f"{self.round_label}-{stem}" if self.round_label else stem

    @property
    def centre(self) -> tuple[float, float]:
        min_x, min_y, max_x, max_y = self.bbox
        return (min_x + max_x) / 2, (min_y + max_y) / 2

    @property
    def bounds_grid(self) -> Bounds:
        """The cell projected to EPSG:3879 for tiling and sweeping."""
        min_x, min_y, max_x, max_y = self.bbox
        corners = [
            to_grid(min_x, min_y, TM35FIN),
            to_grid(max_x, min_y, TM35FIN),
            to_grid(max_x, max_y, TM35FIN),
            to_grid(min_x, max_y, TM35FIN),
        ]
        eastings = [e for e, _ in corners]
        northings = [n for _, n in corners]
        return Bounds(
            min_easting=min(eastings),
            min_northing=min(northings),
            max_easting=max(eastings),
            max_northing=max(northings),
        )

    def as_aoi(self, *, notes: str = "") -> Aoi:
        return Aoi(
            name=self.name,
            bounds=self.bounds_grid,
            role="positive",
            split="train",
            notes=notes,
            source_bbox=self.bbox,
        )


@dataclass(frozen=True, slots=True)
class CellSignals:
    """Raw model signals for one cell — recorded, not reduced to a score."""

    n_detections: int
    n_near_threshold: int
    n_size_boundary: int
    mean_confidence: float
    max_confidence: float


@dataclass(frozen=True, slots=True)
class RankedCell:
    cell: Cell
    signals: CellSignals
    stratum: str
    notes: str = ""


def project_features_to_tm35fin(features: Iterable[OsmFeature]) -> BaseGeometry:
    """Dissolve OSM features into one multipolygon in EPSG:3067."""
    from shapely.ops import unary_union

    project = transformer(WGS84, TM35FIN)
    projected = [
        transform(lambda xs, ys: project.transform(xs, ys), f.geometry)
        for f in features
        if not f.geometry.is_empty
    ]
    if not projected:
        from shapely.geometry import MultiPolygon

        return MultiPolygon()
    unioned = unary_union(projected)
    if not unioned.is_valid:
        unioned = unioned.buffer(0)
    return unioned


def cells_covering(
    industrial: BaseGeometry,
    *,
    municipality_ref: str,
    cell_size_m: float = CELL_SIZE_M,
    min_coverage: float = MIN_INDUSTRIAL_COVERAGE,
    round_label: str = "",
) -> list[Cell]:
    """Axis-aligned cells that cover enough industrial landuse.

    The grid is anchored at the national origin (0, 0) in EPSG:3067, so the
    same industrial polygon always yields the same cell indices regardless of
    which other polygons are present.
    """
    if industrial.is_empty:
        return []
    min_x, min_y, max_x, max_y = industrial.bounds
    col0 = math.floor(min_x / cell_size_m)
    row0 = math.floor(min_y / cell_size_m)
    col1 = math.floor(max_x / cell_size_m)
    row1 = math.floor(max_y / cell_size_m)
    cell_area = cell_size_m * cell_size_m
    found: list[Cell] = []
    for col in range(col0, col1 + 1):
        for row in range(row0, row1 + 1):
            x0 = col * cell_size_m
            y0 = row * cell_size_m
            cell_poly = box(x0, y0, x0 + cell_size_m, y0 + cell_size_m)
            if not cell_poly.intersects(industrial):
                continue
            coverage = cell_poly.intersection(industrial).area / cell_area
            if coverage < min_coverage:
                continue
            found.append(
                Cell(
                    col=col,
                    row=row,
                    bbox=(x0, y0, x0 + cell_size_m, y0 + cell_size_m),
                    industrial_coverage=coverage,
                    municipality_ref=municipality_ref,
                    round_label=round_label,
                )
            )
    return found


def exclude_existing(
    cells: Sequence[Cell],
    existing: Sequence[Aoi],
    *,
    buffer_m: float = EXCLUSION_BUFFER_M,
) -> list[Cell]:
    """Drop cells that overlap or sit within ``buffer_m`` of an existing AOI.

    Comparison is in the collection's own CRS (EPSG:3067), matching
    ``imagery.aoi.overlaps``: reprojected envelopes invent false overlaps.
    """
    blockers = []
    for aoi in existing:
        if aoi.source_bbox is None:
            continue
        min_x, min_y, max_x, max_y = aoi.source_bbox
        blockers.append(box(min_x, min_y, max_x, max_y).buffer(buffer_m))
    if not blockers:
        return list(cells)
    from shapely.ops import unary_union

    blocked = unary_union(blockers)
    kept = []
    for cell in cells:
        cell_poly = box(*cell.bbox)
        if cell_poly.intersects(blocked):
            continue
        kept.append(cell)
    return kept


def disperse_pool(
    cells: Sequence[Cell],
    pool_size: int,
    *,
    seed: int = DEFAULT_SEED,
) -> list[Cell]:
    """Pick up to ``pool_size`` cells spread across the eligible set.

    Greedy farthest-point sampling from a deterministic start: the cell whose
    (col, row) sorts first after a seeded rotation of the list. Coverage
    breaks ties so denser industrial cells are preferred when distances match.
    """
    if pool_size <= 0 or not cells:
        return []
    ordered = sorted(cells, key=lambda c: (c.col, c.row))
    if seed:
        # Rotate rather than shuffle: a seed of 1 always starts one step later
        # than a seed of 0, and the sequence is fully reproducible.
        start = seed % len(ordered)
        ordered = ordered[start:] + ordered[:start]
    selected: list[Cell] = [ordered[0]]
    remaining = list(ordered[1:])
    while remaining and len(selected) < pool_size:
        best_index = 0
        best_distance = -1.0
        best_coverage = -1.0
        for index, cell in enumerate(remaining):
            distance = min(_centre_distance(cell, s) for s in selected)
            if distance > best_distance or (
                distance == best_distance and cell.industrial_coverage > best_coverage
            ):
                best_index = index
                best_distance = distance
                best_coverage = cell.industrial_coverage
        selected.append(remaining.pop(best_index))
    return selected


def cell_signals(
    detections: Sequence[Detection],
    *,
    operating_confidence: float,
    near_half_width: float = NEAR_THRESHOLD_HALF_WIDTH,
) -> CellSignals:
    """Aggregate raw model signals for the detections that fell in one cell."""
    if not detections:
        return CellSignals(
            n_detections=0,
            n_near_threshold=0,
            n_size_boundary=0,
            mean_confidence=0.0,
            max_confidence=0.0,
        )
    near_lo = max(0.0, operating_confidence - near_half_width)
    near_hi = operating_confidence + near_half_width
    n_near = sum(1 for d in detections if near_lo <= d.confidence <= near_hi)
    n_size = sum(
        1
        for d in detections
        if SIZE_BOUNDARY_MIN_M <= d.length_m <= SIZE_BOUNDARY_MAX_M
    )
    confidences = [d.confidence for d in detections]
    return CellSignals(
        n_detections=len(detections),
        n_near_threshold=n_near,
        n_size_boundary=n_size,
        mean_confidence=sum(confidences) / len(confidences),
        max_confidence=max(confidences),
    )


def assign_stratum(signals: CellSignals) -> str:
    """Primary stratum for a cell, used for quota fill and notes.

    Priority matches the learning goal: uncertain detections first, then the
    truck/van length band, then dense novel yards, then quiet industrial
    ground (misses or useful negatives).
    """
    if signals.n_detections == 0:
        return "quiet"
    if signals.n_near_threshold > 0:
        return "near_threshold"
    if signals.n_size_boundary > 0:
        return "size_boundary"
    if signals.n_detections >= 5:
        return "dense"
    return "quiet"


def select_proposals(
    ranked: Sequence[RankedCell],
    count: int = DEFAULT_COUNT,
    *,
    min_separation_m: float = MIN_SEPARATION_M,
    max_candidates: int = MAX_CANDIDATES,
) -> list[RankedCell]:
    """Fill stratum quotas with geographically separated cells.

    Within a stratum, cells are ordered by the stratum's own signal (more
    near-threshold detections, more size-boundary boxes, more detections for
    dense, higher industrial coverage for quiet), then by name for a stable
    tie-break. Empty strata spill their remaining slots to later strata so a
    shortlist of ``count`` is still produced when one signal is rare.
    """
    eligible = [r for r in ranked if r.signals.n_detections <= max_candidates]
    by_stratum: dict[str, list[RankedCell]] = {name: [] for name, _ in STRATUM_QUOTAS}
    for item in eligible:
        by_stratum.setdefault(item.stratum, []).append(item)
    for name, items in by_stratum.items():
        by_stratum[name] = sorted(
            items, key=lambda r: _stratum_sort_key(r), reverse=True
        )

    quotas = _slot_counts(count)
    selected: list[RankedCell] = []
    selected_cells: list[Cell] = []

    def try_take(candidates: list[RankedCell], slots: int) -> int:
        taken = 0
        for candidate in candidates:
            if taken >= slots:
                break
            if any(
                _centre_distance(candidate.cell, other) < min_separation_m
                for other in selected_cells
            ):
                continue
            if any(s.cell.name == candidate.cell.name for s in selected):
                continue
            selected.append(
                RankedCell(
                    cell=candidate.cell,
                    signals=candidate.signals,
                    stratum=candidate.stratum,
                    notes=_notes_for(candidate),
                )
            )
            selected_cells.append(candidate.cell)
            taken += 1
        return taken

    remaining_slots = 0
    for stratum_name, _fraction in STRATUM_QUOTAS:
        slots = quotas[stratum_name] + remaining_slots
        taken = try_take(by_stratum.get(stratum_name, []), slots)
        remaining_slots = slots - taken

    if len(selected) < count and remaining_slots:
        # Final spill: any remaining eligible cell, still respecting spacing.
        leftovers = [
            r
            for r in sorted(eligible, key=lambda r: _stratum_sort_key(r), reverse=True)
            if all(s.cell.name != r.cell.name for s in selected)
        ]
        try_take(leftovers, count - len(selected))

    return selected[:count]


def rank_cells(
    cells: Sequence[Cell],
    detections_by_name: dict[str, list[Detection]],
    *,
    operating_confidence: float,
) -> list[RankedCell]:
    """Attach signals and a primary stratum to every pool cell."""
    ranked = []
    for cell in cells:
        signals = cell_signals(
            detections_by_name.get(cell.name, []),
            operating_confidence=operating_confidence,
        )
        stratum = assign_stratum(signals)
        ranked.append(RankedCell(cell=cell, signals=signals, stratum=stratum))
    return ranked


def detections_in_cell(detections: Sequence[Detection], cell: Cell) -> list[Detection]:
    """Detections whose centre falls inside the cell's EPSG:3879 envelope."""
    bounds = cell.bounds_grid
    found = []
    for detection in detections:
        easting, northing = detection.centre
        if (
            bounds.min_easting <= easting <= bounds.max_easting
            and bounds.min_northing <= northing <= bounds.max_northing
        ):
            found.append(detection)
    return found


def proposal_yaml(proposals: Sequence[RankedCell]) -> str:
    """A collection-shaped YAML a human can copy entries from."""
    document = {
        "crs": TM35FIN,
        "aois": [
            {
                "name": item.cell.name,
                "bbox": [round(v, 3) for v in item.cell.bbox],
                "role": "positive",
                "split": "train",
                "notes": item.notes,
            }
            for item in proposals
        ],
    }
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)


def proposal_geojson(
    proposals: Sequence[RankedCell],
    *,
    osm: OsmCache,
    operating_confidence: float,
) -> dict[str, Any]:
    """EPSG:3879 FeatureCollection with the signals that justified each pick."""
    features = []
    for item in proposals:
        bounds = item.cell.bounds_grid
        ring = [
            [bounds.min_easting, bounds.min_northing],
            [bounds.max_easting, bounds.min_northing],
            [bounds.max_easting, bounds.max_northing],
            [bounds.min_easting, bounds.max_northing],
            [bounds.min_easting, bounds.min_northing],
        ]
        ring = [[round(v, COORD_DECIMALS) for v in corner] for corner in ring]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "name": item.cell.name,
                    "stratum": item.stratum,
                    "role": "positive",
                    "split": "train",
                    "bbox_3067": list(item.cell.bbox),
                    "industrial_coverage": round(item.cell.industrial_coverage, 3),
                    "n_detections": item.signals.n_detections,
                    "n_near_threshold": item.signals.n_near_threshold,
                    "n_size_boundary": item.signals.n_size_boundary,
                    "mean_confidence": round(item.signals.mean_confidence, 4),
                    "max_confidence": round(item.signals.max_confidence, 4),
                    "operating_confidence": operating_confidence,
                    "notes": item.notes,
                    "osm_query_hash": osm.query_hash,
                    "osm_fetched_at": osm.fetched_at,
                    "attribution": osm.attribution,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "crs": crs_member(),
        "features": features,
    }


def estimate_tiles(cells: Sequence[Cell], zoom: int) -> int:
    return sum(count_tiles(cell.bounds_grid, zoom) for cell in cells)


def _centre_distance(a: Cell, b: Cell) -> float:
    ax, ay = a.centre
    bx, by = b.centre
    return math.hypot(ax - bx, ay - by)


def _slot_counts(count: int) -> dict[str, int]:
    """Integer slots per stratum that sum to ``count``."""
    raw = [(name, fraction * count) for name, fraction in STRATUM_QUOTAS]
    floors = {name: math.floor(value) for name, value in raw}
    remaining = count - sum(floors.values())
    # Hand leftover slots to the strata with the largest fractional parts.
    order = sorted(raw, key=lambda item: item[1] - math.floor(item[1]), reverse=True)
    for name, _ in order:
        if remaining <= 0:
            break
        floors[name] += 1
        remaining -= 1
    return floors


def _stratum_sort_key(item: RankedCell) -> tuple[float, float, str]:
    signals = item.signals
    if item.stratum == "near_threshold":
        primary = float(signals.n_near_threshold)
    elif item.stratum == "size_boundary":
        primary = float(signals.n_size_boundary)
    elif item.stratum == "dense":
        primary = float(signals.n_detections)
    else:
        primary = item.cell.industrial_coverage
    return (primary, signals.mean_confidence, item.cell.name)


def _notes_for(item: RankedCell) -> str:
    reasons = {
        "near_threshold": (
            f"Model-mined: {item.signals.n_near_threshold} detection(s) near the "
            "operating confidence — inspect the whole cell for misses."
        ),
        "size_boundary": (
            f"Model-mined: {item.signals.n_size_boundary} detection(s) in the "
            "4–8 m truck/van band — classify carefully."
        ),
        "dense": (
            f"Model-mined: dense industrial cell with {item.signals.n_detections} "
            "detections — inspect for missed vehicles between the boxes."
        ),
        "quiet": (
            "Model-mined: industrial landuse where the model was quiet — look "
            "for misses, or keep as a hard-negative if the ground is empty."
        ),
    }
    return reasons.get(item.stratum, "Model-mined industrial cell.")
