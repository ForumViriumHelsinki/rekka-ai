"""The gates and the operating-point rule are the ship decision; they get a
checkable right answer without touching torch."""

from rekka_ai.evaluate import ground_truth_counts, pick_operating_point


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
