"""Tile grid arithmetic.

The expected tile indices below were confirmed against the live service: each
one was fetched and the returned imagery checked to be the named place. The z6
case additionally matches the TileMatrixLimits published in GetCapabilities,
which pins down the row/col convention independently of our own arithmetic.
"""

import pytest

from rekka_ai.geo import to_grid, to_wgs84
from rekka_ai.imagery.tiles import (
    MAX_ZOOM,
    Bounds,
    Tile,
    count_tiles,
    matrix_size,
    resolution,
    tile_at,
    tile_span,
    tiles_covering,
)

# (name, lon, lat, easting, northing)
HELSINKI_CENTRE = ("Helsinki centre", 24.9384, 60.1699, 25496580.4, 6673003.6)
VUOSAARI = ("Vuosaari", 25.1960, 60.2090, 25510867.7, 6677374.5)


def test_resolution_ladder() -> None:
    assert resolution(0) == 8192.0
    assert resolution(15) == 0.25
    assert resolution(16) == 0.125
    assert resolution(MAX_ZOOM) == 0.0625


def test_tile_span_and_matrix_size() -> None:
    assert tile_span(16) == 32.0  # 0.125 m/px * 256 px
    assert matrix_size(0) == 1
    assert matrix_size(MAX_ZOOM) == 131072


@pytest.mark.parametrize("zoom", [-1, MAX_ZOOM + 1])
def test_zoom_outside_grid_is_rejected(zoom: int) -> None:
    with pytest.raises(ValueError, match="outside grid range"):
        resolution(zoom)


def test_tile_outside_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        Tile(zoom=0, col=1, row=0)


@pytest.mark.parametrize(
    ("easting", "northing", "zoom", "col", "row"),
    [
        (HELSINKI_CENTRE[3], HELSINKI_CENTRE[4], 17, 65322, 107225),
        (HELSINKI_CENTRE[3], HELSINKI_CENTRE[4], 16, 32661, 53612),
        (VUOSAARI[3], VUOSAARI[4], 17, 66215, 106952),
        (VUOSAARI[3], VUOSAARI[4], 16, 33107, 53476),
        # Matches the layer's published TileMatrixLimits at z6 (cols 31-32,
        # rows 51-52), so the row/col convention is fixed by the service.
        (HELSINKI_CENTRE[3], HELSINKI_CENTRE[4], 6, 31, 52),
    ],
)
def test_tile_at_known_locations(
    easting: float, northing: float, zoom: int, col: int, row: int
) -> None:
    assert tile_at(easting, northing, zoom) == Tile(zoom=zoom, col=col, row=row)


@pytest.mark.parametrize("point", [HELSINKI_CENTRE, VUOSAARI])
def test_tile_bounds_contain_their_point(
    point: tuple[str, float, float, float, float],
) -> None:
    _, _, _, easting, northing = point
    bounds = tile_at(easting, northing, 16).bounds()
    assert bounds.min_easting <= easting < bounds.max_easting
    assert bounds.min_northing <= northing < bounds.max_northing
    assert bounds.width == bounds.height == tile_span(16)


def test_tile_bounds_roundtrip_through_tile_at() -> None:
    tile = Tile(zoom=16, col=32661, row=53612)
    bounds = tile.bounds()
    # The grid origin is top-left, so it is the north-west corner that belongs
    # to this tile; the southern edge belongs to the tile in the next row down.
    assert tile_at(bounds.min_easting, bounds.max_northing, 16) == tile
    assert tile_at(bounds.min_easting, bounds.min_northing, 16) == Tile(
        zoom=16, col=tile.col, row=tile.row + 1
    )


def test_adjacent_tiles_share_an_edge() -> None:
    left = Tile(zoom=16, col=32661, row=53612).bounds()
    right = Tile(zoom=16, col=32662, row=53612).bounds()
    below = Tile(zoom=16, col=32661, row=53613).bounds()
    assert left.max_easting == right.min_easting
    assert left.min_northing == below.max_northing


def test_tiles_covering_a_single_tile_extent() -> None:
    tile = Tile(zoom=16, col=32661, row=53612)
    bounds = tile.bounds()
    inset = Bounds(
        min_easting=bounds.min_easting + 1,
        min_northing=bounds.min_northing + 1,
        max_easting=bounds.max_easting - 1,
        max_northing=bounds.max_northing - 1,
    )
    assert list(tiles_covering(inset, 16)) == [tile]


def test_tile_aligned_extent_does_not_pull_in_neighbours() -> None:
    """An AOI exactly on tile boundaries touches the next row/col but needs none of it."""
    tile = Tile(zoom=16, col=32661, row=53612)
    assert list(tiles_covering(tile.bounds(), 16)) == [tile]


def test_zero_area_extent_resolves_to_its_containing_tile() -> None:
    easting, northing = HELSINKI_CENTRE[3], HELSINKI_CENTRE[4]
    point = Bounds(
        min_easting=easting,
        min_northing=northing,
        max_easting=easting,
        max_northing=northing,
    )
    assert list(tiles_covering(point, 16)) == [tile_at(easting, northing, 16)]


def test_tiles_covering_is_row_major_and_complete() -> None:
    origin = Tile(zoom=16, col=32661, row=53612).bounds()
    span = tile_span(16)
    bounds = Bounds(
        min_easting=origin.min_easting + 1,
        min_northing=origin.min_northing - 2 * span + 1,
        max_easting=origin.min_easting + 3 * span - 1,
        max_northing=origin.max_northing - 1,
    )
    tiles = list(tiles_covering(bounds, 16))
    assert len(tiles) == 9
    assert tiles[0] == Tile(zoom=16, col=32661, row=53612)
    assert tiles[1] == Tile(zoom=16, col=32662, row=53612)  # east before south
    assert tiles[-1] == Tile(zoom=16, col=32663, row=53614)
    assert len(set(tiles)) == 9


def test_count_tiles_agrees_with_tiles_covering() -> None:
    bounds = Bounds(
        min_easting=25496000.0,
        min_northing=6672000.0,
        max_easting=25498000.0,
        max_northing=6674000.0,
    )
    for zoom in (14, 15, 16):
        assert count_tiles(bounds, zoom) == len(list(tiles_covering(bounds, zoom)))


def test_finer_zoom_needs_four_times_the_tiles() -> None:
    bounds = Tile(zoom=14, col=8165, row=13403).bounds()
    assert count_tiles(bounds, 15) == 4 * count_tiles(bounds, 14)


def test_degenerate_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        Bounds(min_easting=10.0, min_northing=0.0, max_easting=0.0, max_northing=10.0)


@pytest.mark.parametrize("point", [HELSINKI_CENTRE, VUOSAARI])
def test_projection_matches_known_coordinates(
    point: tuple[str, float, float, float, float],
) -> None:
    _, lon, lat, easting, northing = point
    got_easting, got_northing = to_grid(lon, lat)
    assert got_easting == pytest.approx(easting, abs=0.1)
    assert got_northing == pytest.approx(northing, abs=0.1)


@pytest.mark.parametrize("point", [HELSINKI_CENTRE, VUOSAARI])
def test_projection_roundtrips(point: tuple[str, float, float, float, float]) -> None:
    _, lon, lat, _, _ = point
    got_lon, got_lat = to_wgs84(*to_grid(lon, lat))
    assert (got_lon, got_lat) == pytest.approx((lon, lat), abs=1e-9)


def test_projection_does_not_swap_axes() -> None:
    """EPSG:3879 declares (north, east); a swap here would be silent."""
    easting, northing = to_grid(*HELSINKI_CENTRE[1:3])
    assert easting > northing  # GK25 eastings carry a 25_000_000 zone prefix
