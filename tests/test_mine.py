"""AOI mining: grid, exclusion, signals, and stratified selection."""

from pathlib import Path

from shapely.geometry import box

from rekka_ai.detect.detections import Detection
from rekka_ai.imagery.aoi import Aoi
from rekka_ai.imagery.tiles import Bounds
from rekka_ai.mine import (
    CELL_SIZE_M,
    Cell,
    CellSignals,
    RankedCell,
    assign_stratum,
    cell_signals,
    cells_covering,
    detections_in_cell,
    disperse_pool,
    exclude_existing,
    project_features_to_tm35fin,
    proposal_geojson,
    proposal_yaml,
    rank_cells,
    select_proposals,
)
from rekka_ai.osm import OsmCache, OsmFeature


def _cell(col: int, row: int, coverage: float = 0.5, ref: str = "091") -> Cell:
    x0 = col * CELL_SIZE_M
    y0 = row * CELL_SIZE_M
    return Cell(
        col=col,
        row=row,
        bbox=(x0, y0, x0 + CELL_SIZE_M, y0 + CELL_SIZE_M),
        industrial_coverage=coverage,
        municipality_ref=ref,
    )


def _detection(
    easting: float,
    northing: float,
    *,
    confidence: float = 0.5,
    length_m: float = 12.0,
    width_m: float = 2.5,
    label: str = "truck",
) -> Detection:
    # Axis-aligned box centred on (easting, northing).
    half_l = length_m / 2
    half_w = width_m / 2
    return Detection(
        label=label,
        confidence=confidence,
        corners=(
            (easting - half_l, northing - half_w),
            (easting + half_l, northing - half_w),
            (easting + half_l, northing + half_w),
            (easting - half_l, northing + half_w),
        ),
    )


def _aoi(name: str, bbox: tuple[float, float, float, float]) -> Aoi:
    return Aoi(
        name=name,
        bounds=Bounds(0, 0, 1, 1),  # unused by exclude_existing
        source_bbox=bbox,
    )


def test_project_features_to_tm35fin_lands_in_finland() -> None:
    """WGS84 industrial rings must become EPSG:3067 metres around Helsinki."""
    from shapely.geometry import Polygon

    feature = OsmFeature(
        osm_type="way",
        osm_id=1,
        geometry=Polygon(
            [
                (24.95, 60.20),
                (24.96, 60.20),
                (24.96, 60.21),
                (24.95, 60.21),
                (24.95, 60.20),
            ]
        ),
        tags={"landuse": "industrial"},
    )
    projected = project_features_to_tm35fin([feature])
    min_x, min_y, _max_x, _max_y = projected.bounds
    # ETRS-TM35FIN eastings/northings for Helsinki, not degrees and not GK25.
    assert 380_000 < min_x < 420_000
    assert 6_660_000 < min_y < 6_700_000
    assert projected.area > 0


def test_cells_covering_anchors_to_national_origin() -> None:
    # A 200 m square sitting across a 300 m cell boundary must produce the
    # same cell indices regardless of which other polygons exist.
    industrial = box(100.0, 100.0, 300.0, 300.0)
    cells = cells_covering(industrial, municipality_ref="091", min_coverage=0.05)
    assert cells
    assert all(c.name.startswith("mine-091-") for c in cells)
    # The same polygon again yields identical names.
    again = cells_covering(industrial, municipality_ref="091", min_coverage=0.05)
    assert [c.name for c in again] == [c.name for c in cells]


def test_low_industrial_coverage_is_rejected() -> None:
    # A thin strip covering far less than 15% of a 300 m cell.
    industrial = box(0.0, 0.0, 300.0, 5.0)
    assert cells_covering(industrial, municipality_ref="091") == []


def test_exclude_existing_uses_source_crs_buffer() -> None:
    cells = [_cell(0, 0), _cell(5, 5)]
    # Existing AOI overlaps cell (0,0) and sits >50 m from cell (5,5).
    existing = [_aoi("yard", (50.0, 50.0, 250.0, 250.0))]
    kept = exclude_existing(cells, existing)
    assert [c.name for c in kept] == ["mine-091-5-5"]


def test_disperse_pool_is_deterministic_and_spread() -> None:
    cells = [_cell(c, r) for c in range(10) for r in range(10)]
    first = disperse_pool(cells, 8, seed=0)
    second = disperse_pool(cells, 8, seed=0)
    assert [c.name for c in first] == [c.name for c in second]
    assert len(first) == 8
    # A different seed rotates the start and changes the set.
    other = disperse_pool(cells, 8, seed=1)
    assert [c.name for c in other] != [c.name for c in first]


def test_cell_signals_count_near_threshold_and_size_boundary() -> None:
    detections = [
        _detection(0, 0, confidence=0.17, length_m=6.0),
        _detection(1, 1, confidence=0.80, length_m=14.0),
        _detection(2, 2, confidence=0.12, length_m=5.0),
    ]
    signals = cell_signals(detections, operating_confidence=0.17)
    assert signals.n_detections == 3
    assert signals.n_near_threshold == 2  # 0.17 and 0.12 within ±0.10
    assert signals.n_size_boundary == 2  # 6 m and 5 m


def test_assign_stratum_priority() -> None:
    assert assign_stratum(CellSignals(0, 0, 0, 0.0, 0.0)) == "quiet"
    assert assign_stratum(CellSignals(3, 2, 1, 0.2, 0.3)) == "near_threshold"
    assert assign_stratum(CellSignals(3, 0, 2, 0.5, 0.6)) == "size_boundary"
    assert assign_stratum(CellSignals(8, 0, 0, 0.7, 0.9)) == "dense"
    assert assign_stratum(CellSignals(2, 0, 0, 0.4, 0.5)) == "quiet"


def test_select_proposals_fills_quotas_with_spacing() -> None:
    # Build one strong cell per stratum, far apart, plus a near duplicate.
    cells = {
        "near": _cell(0, 0),
        "near_dup": _cell(0, 1),  # ~300 m away — inside 600 m separation
        "size": _cell(10, 0),
        "dense": _cell(20, 0),
        "quiet": _cell(30, 0),
    }
    ranked = [
        RankedCell(
            cells["near"],
            CellSignals(4, 4, 0, 0.2, 0.25),
            "near_threshold",
        ),
        RankedCell(
            cells["near_dup"],
            CellSignals(3, 3, 0, 0.2, 0.22),
            "near_threshold",
        ),
        RankedCell(
            cells["size"],
            CellSignals(3, 0, 3, 0.4, 0.5),
            "size_boundary",
        ),
        RankedCell(
            cells["dense"],
            CellSignals(12, 0, 0, 0.6, 0.8),
            "dense",
        ),
        RankedCell(
            cells["quiet"],
            CellSignals(0, 0, 0, 0.0, 0.0),
            "quiet",
        ),
    ]
    selected = select_proposals(ranked, count=4)
    names = {s.cell.name for s in selected}
    assert cells["near"].name in names
    assert cells["near_dup"].name not in names  # excluded by separation
    assert cells["size"].name in names
    assert cells["dense"].name in names
    assert cells["quiet"].name in names
    assert all(s.notes for s in selected)


def test_empty_stratum_spills_to_later_slots() -> None:
    # Only quiet cells available: still fill the requested count.
    ranked = [
        RankedCell(_cell(i * 5, 0), CellSignals(0, 0, 0, 0.0, 0.0), "quiet")
        for i in range(6)
    ]
    selected = select_proposals(ranked, count=4)
    assert len(selected) == 4
    assert {s.stratum for s in selected} == {"quiet"}


def test_max_candidates_filters_overloaded_cells() -> None:
    overloaded = RankedCell(
        _cell(0, 0), CellSignals(200, 50, 10, 0.3, 0.9), "near_threshold"
    )
    normal = RankedCell(_cell(10, 0), CellSignals(5, 4, 0, 0.2, 0.25), "near_threshold")
    selected = select_proposals([overloaded, normal], count=1)
    assert [s.cell.name for s in selected] == [normal.cell.name]


def test_detections_in_cell_use_centre() -> None:
    cell = _cell(0, 0)
    bounds = cell.bounds_grid
    inside = _detection(
        (bounds.min_easting + bounds.max_easting) / 2,
        (bounds.min_northing + bounds.max_northing) / 2,
    )
    outside = _detection(bounds.max_easting + 500, bounds.max_northing + 500)
    assert detections_in_cell([inside, outside], cell) == [inside]


def test_rank_and_proposal_outputs_are_collection_shaped() -> None:
    cell = _cell(1, 2, coverage=0.4)
    detections = {
        cell.name: [
            _detection(0, 0, confidence=0.18, length_m=6.0),
            _detection(1, 1, confidence=0.19, length_m=7.0),
        ]
    }
    # Place detections inside the cell's grid bounds so signals are non-zero.
    bounds = cell.bounds_grid
    detections[cell.name] = [
        _detection(
            (bounds.min_easting + bounds.max_easting) / 2,
            (bounds.min_northing + bounds.max_northing) / 2,
            confidence=0.18,
            length_m=6.0,
        )
    ]
    ranked = rank_cells([cell], detections, operating_confidence=0.17)
    assert ranked[0].stratum == "near_threshold"
    proposals = select_proposals(ranked, count=1)
    document = proposal_yaml(proposals)
    assert "EPSG:3067" in document
    assert proposals[0].cell.name in document
    assert "split: train" in document
    assert "role: positive" in document

    osm = OsmCache(
        municipality="Helsinki",
        ref="091",
        profile="industrial",
        query="q",
        fetched_at="2026-08-07T00:00:00+00:00",
        query_hash="abc",
        path=Path("x"),
        features=(),
    )
    geojson = proposal_geojson(proposals, osm=osm, operating_confidence=0.17)
    assert geojson["type"] == "FeatureCollection"
    assert geojson["crs"]["properties"]["name"].endswith("3879")
    props = geojson["features"][0]["properties"]
    assert props["stratum"] == "near_threshold"
    assert props["osm_query_hash"] == "abc"
    assert props["attribution"].startswith("© OpenStreetMap")
