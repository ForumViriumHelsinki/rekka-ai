"""The gates and the operating-point rule are the ship decision; they get a
checkable right answer without touching torch."""

from rekka_ai.evaluate import (
    GATE_COUNT_MIN_TRUCKS,
    count_gate,
    ground_truth_counts,
    pick_operating_point,
)


def _collection(*statuses_and_classes: tuple[str, str]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[]]},
                "properties": {"status": status, "class": klass},
            }
            for status, klass in statuses_and_classes
        ],
    }


def test_ground_truth_counts_only_human_verdicts() -> None:
    collection = _collection(
        ("confirmed", "truck"),
        ("added", "truck"),
        ("candidate", "truck"),  # unreviewed: not ground truth
        ("rejected", ""),  # a verdict of "not an object"
        ("confirmed", "van"),
    )
    counts = ground_truth_counts(collection)
    assert counts["truck"] == 2
    assert counts["van"] == 1
    assert sum(counts.values()) == 3


def test_count_gate_holds_a_truck_dense_area_to_ten_percent() -> None:
    assert count_gate(100, 100)[0] is True
    assert count_gate(109, 100)[0] is True  # exactly at the tolerance
    assert count_gate(111, 100)[0] is False
    assert count_gate(88, 100)[0] is False  # under-counting fails too


def test_count_gate_does_not_gate_an_area_too_small_to_measure() -> None:
    """r1-kamppi's failure mode: 10% of 8 trucks is 0.8 of a box, so the area
    passed or failed on one smeared vehicle. Reported, never gated — and the
    zero case goes the same way, since a bus yard holding no trucks cannot
    say anything about truck counting either.
    """
    for truth in range(GATE_COUNT_MIN_TRUCKS):
        verdict, detail = count_gate(truth + 3, truth)
        assert verdict is None
        assert "not gated" in detail
    # One truck more and the tolerance covers a whole box, so it gates again.
    assert count_gate(11, GATE_COUNT_MIN_TRUCKS)[0] is True


def test_operating_point_prefers_precision_inside_the_recall_gate() -> None:
    px = [0.1, 0.5, 0.9]
    p = [0.6, 0.9, 0.95]
    r = [0.95, 0.92, 0.6]  # the last point falls below the 0.90 gate
    conf, precision, recall = pick_operating_point(px, p, r, min_recall=0.90)
    assert (conf, precision, recall) == (0.5, 0.9, 0.92)


def test_operating_point_falls_back_to_f1_when_the_gate_is_unreachable() -> None:
    px = [0.1, 0.5, 0.9]
    p = [0.7, 0.8, 0.99]
    r = [0.8, 0.85, 0.5]  # nothing reaches 0.90 recall
    conf, precision, recall = pick_operating_point(px, p, r, min_recall=0.90)
    # F1: 0.747, 0.823, 0.662 -- the middle point wins despite lower precision.
    assert (conf, precision, recall) == (0.5, 0.8, 0.85)


def test_operating_point_floor_rejects_a_conf_zero_pick() -> None:
    """Round 3: recall peaked at conf=0 with no plateau above it, so the
    naive pick was conf=0 -- keep every raw proposal. The floor forces a
    real operating point instead, even though that means failing the gate.
    """
    px = [0.0, 0.02, 0.5, 0.9]
    p = [0.4, 0.6, 0.9, 0.95]
    r = [0.90, 0.83, 0.6, 0.4]  # only the floored-out conf=0 clears the gate
    conf, precision, recall = pick_operating_point(px, p, r, min_recall=0.90)
    assert conf >= 0.05
    # Fallback F1 among points at/above the floor: F1(0.9,0.6)=0.72 beats
    # F1(0.95,0.4)=0.563, so the 0.5 point wins despite lower precision.
    assert (conf, precision, recall) == (0.5, 0.9, 0.6)


def test_operating_point_min_confidence_is_configurable() -> None:
    px = [0.0, 0.02, 0.5]
    p = [0.5, 0.9, 0.95]
    r = [0.95, 0.92, 0.6]
    conf, precision, recall = pick_operating_point(
        px, p, r, min_recall=0.90, min_confidence=0.01
    )
    assert (conf, precision, recall) == (0.02, 0.9, 0.92)
