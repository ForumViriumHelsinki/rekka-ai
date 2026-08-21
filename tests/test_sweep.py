"""The detector-class guard that keeps `bootstrap` from failing silently.

No torch here: the check is a pure function on class names precisely so it
can be tested against the base install, like the rest of the suite.
"""

from rekka_ai.detect.sweep import LARGE_VEHICLE, SMALL_VEHICLE, emits_dota_vehicles


def test_dota_vehicle_classes_are_recognised() -> None:
    """The pretrained weights `bootstrap` exists for."""
    assert emits_dota_vehicles({LARGE_VEHICLE, SMALL_VEHICLE, "ship"})
    assert emits_dota_vehicles({SMALL_VEHICLE})


def test_fine_tuned_class_names_are_rejected() -> None:
    """The trap this guard exists for.

    Fine-tuned weights emit ``truck/bus/van/car`` and share no class name with
    DOTA, so `bootstrap`'s ``keep`` filter matches nothing: the sweep runs
    every window, finds boxes, discards all of them, and reports **0
    candidates** with a clean exit — indistinguishable from empty ground. It
    cost two debugging sessions (docs/rounds.md 2026-08-13, and again
    2026-08-14, where it also returned 0 over an area holding 55 labelled
    trucks) before the caller learned to refuse up front.
    """
    assert not emits_dota_vehicles({"truck", "bus", "van", "car"})


def test_no_classes_at_all_is_rejected() -> None:
    """Weights that report no names cannot satisfy the filter either, and the
    empty case must not read as 'fine' just because the intersection is empty
    on both sides."""
    assert not emits_dota_vehicles(set())
