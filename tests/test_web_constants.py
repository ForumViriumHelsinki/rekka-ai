"""The labelling tool's mirrors of Python constants must not drift.

``web/`` restates a handful of values Python owns -- the class list, the tile
grid, the flight layer -- because the browser cannot import them. DESIGN.md
says the two "cannot drift"; on its own, that is a comment between two
hardcoded copies. This is the mechanism that makes it true.

Drift here is quiet and expensive. A stale ``LATEST_LAYER`` stamps last year's
layer name into every box drawn against this year's imagery, and a wrong grid
constant puts labels on the wrong ground -- neither raises anything.
"""

import re
from pathlib import Path

import pytest

from rekka_ai import labels
from rekka_ai.cli import BOOTSTRAP_ZOOM
from rekka_ai.imagery import tiles
from rekka_ai.imagery.layers import LATEST_YEAR, layer_for_year

WEB = Path("web/src/lib")
SCHEMA = WEB / "schema.ts"
GRID = WEB / "grid.ts"

pytestmark = pytest.mark.skipif(
    not SCHEMA.exists(), reason="the web app is not checked out"
)


def _const(source: str, name: str) -> str:
    """The right-hand side of a top-level ``export const NAME = ...;``."""
    match = re.search(rf"^export const {name}(?::[^=]+)? = (.+?);$", source, re.M)
    if match is None:
        raise AssertionError(f"no `export const {name}` in the file")
    return match.group(1).strip()


def _number(source: str, name: str) -> float:
    return float(_const(source, name).rstrip(" as const").strip())


def test_class_list_matches() -> None:
    """Class order fixes the YOLO indices; a reorder silently relabels data."""
    listed = re.findall(r"'([a-z]+)'", _const(SCHEMA.read_text(), "CLASSES"))
    assert tuple(listed) == labels.CLASSES


def test_source_zoom_matches() -> None:
    assert _number(SCHEMA.read_text(), "SOURCE_ZOOM") == BOOTSTRAP_ZOOM


def test_latest_layer_matches() -> None:
    """The layer stamped into `source_layer` on every hand-drawn box."""
    declared = _const(SCHEMA.read_text(), "LATEST_LAYER").strip("'\"")
    assert declared == layer_for_year(LATEST_YEAR)


@pytest.mark.parametrize(
    ("web_name", "python_value"),
    [
        ("TILE_SIZE", tiles.TILE_SIZE),
        ("MAX_ZOOM", tiles.MAX_ZOOM),
        ("BASE_RESOLUTION", tiles.BASE_RESOLUTION),
    ],
)
def test_grid_scalars_match(web_name: str, python_value: float) -> None:
    assert _number(GRID.read_text(), web_name) == python_value


def test_grid_origin_matches() -> None:
    """Written [easting, northing] for OpenLayers, the reverse of the
    capabilities document's axis order -- so this compares them swapped."""
    easting, northing = re.findall(
        r"-?\d+(?:\.\d+)?", _const(GRID.read_text(), "ORIGIN")
    )
    assert float(easting) == tiles.ORIGIN_EASTING
    assert float(northing) == tiles.ORIGIN_NORTHING
