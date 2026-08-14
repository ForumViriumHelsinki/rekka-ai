"""AOI resolution from YAML collections, GeoJSON, and bboxes.

The EPSG:3067 fixtures below are the same ground positions as the WGS84 ones,
so a CRS being ignored or an axis order being swapped shows up as a mismatch
rather than as plausible-looking numbers somewhere else in Finland.
"""

import json
from pathlib import Path

import pytest
import yaml

from rekka_ai.geo import TM35FIN, WGS84, to_grid
from rekka_ai.imagery.aoi import load_aois, load_region, overlaps, region_bounds

# Vuosaari harbour, the same point in three CRSs.
VUOSAARI_WGS84 = (25.1960, 60.2090)
VUOSAARI_TM35FIN = (400020.8, 6676053.5)
VUOSAARI_GRID = (25510867.7, 6677374.5)

COLLECTION = """
crs: "EPSG:3067"
aois:
  - name: yard
    bbox: [399000, 6675000, 401000, 6677000]
    role: positive
    notes: Rows of parked trucks.
  - name: containers
    bbox: [402000, 6678000, 403000, 6679000]
    role: hard-negative
  - name: heldout
    bbox: [392000, 6683000, 394000, 6685000]
    split: validation
"""


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    path = tmp_path / "helsinki.yaml"
    path.write_text(COLLECTION)
    return path


def _write(tmp_path: Path, body: str) -> str:
    path = tmp_path / "c.yaml"
    path.write_text(body)
    return str(path)


def test_tm35fin_converts_to_the_grid_crs() -> None:
    assert to_grid(*VUOSAARI_TM35FIN, TM35FIN) == pytest.approx(VUOSAARI_GRID, abs=0.5)


def test_tm35fin_and_wgs84_agree_on_the_same_place() -> None:
    """EPSG:3067 declares (east, north) and EPSG:3879 (north, east); both must normalise."""
    assert to_grid(*VUOSAARI_TM35FIN, TM35FIN) == pytest.approx(
        to_grid(*VUOSAARI_WGS84, WGS84), abs=1.0
    )


def test_unknown_crs_is_reported_clearly() -> None:
    with pytest.raises(ValueError, match="unknown CRS"):
        to_grid(0.0, 0.0, "EPSG:999999")


def test_collection_preserves_order(collection: Path) -> None:
    assert [a.name for a in load_aois(str(collection))] == [
        "yard",
        "containers",
        "heldout",
    ]


def test_collection_reads_role_split_and_notes(collection: Path) -> None:
    yard, containers, heldout = load_aois(str(collection))
    assert (yard.role, yard.split) == ("positive", "train")
    assert yard.notes == "Rows of parked trucks."
    assert containers.role == "hard-negative"
    # Defaults: role positive, split train.
    assert (heldout.role, heldout.split) == ("positive", "validation")


def test_collection_bounds_are_projected_to_the_grid(collection: Path) -> None:
    yard = load_aois(str(collection), name="yard")[0]
    assert yard.bounds.min_easting < VUOSAARI_GRID[0] < yard.bounds.max_easting
    assert yard.bounds.min_northing < VUOSAARI_GRID[1] < yard.bounds.max_northing
    # A 2 km square in EPSG:3067 is not axis-aligned in EPSG:3879: the two
    # projections have different central meridians (27E and 25E), so the box
    # arrives rotated by ~1.74 degrees and its envelope grows to
    # 2000 * (cos + sin) ~= 2060 m. Covering slightly more ground than asked
    # for is correct; covering less would silently clip the AOI.
    assert yard.bounds.width == pytest.approx(2060, abs=5)


def test_name_selects_one_aoi(collection: Path) -> None:
    assert [a.name for a in load_aois(str(collection), name="heldout")] == ["heldout"]


def test_unknown_name_lists_what_is_available(collection: Path) -> None:
    with pytest.raises(ValueError, match="available: yard, containers, heldout"):
        load_aois(str(collection), name="pasila")


def test_collection_ignores_the_crs_option(collection: Path) -> None:
    """The file declares EPSG:3067; a stray --crs must not silently reinterpret it."""
    assert load_aois(str(collection), crs=WGS84) == load_aois(
        str(collection), crs=TM35FIN
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("aois:\n  - {name: a, bbox: [1, 2, 3, 4]}\n", "must declare a 'crs'"),
        ("crs: EPSG:3067\n", "non-empty 'aois' list"),
        ("crs: EPSG:3067\naois: []\n", "non-empty 'aois' list"),
        # A mapping of areas is the old shape; it must not be silently accepted.
        (
            "crs: EPSG:3067\naois:\n  a:\n    bbox: [1, 2, 3, 4]\n",
            "non-empty 'aois' list",
        ),
        ("crs: EPSG:3067\naois:\n  - bbox: [1, 2, 3, 4]\n", "missing its 'name'"),
        ("crs: EPSG:3067\naois:\n  - name: a\n", "needs a 'bbox'"),
        ("crs: EPSG:3067\naois:\n  - {name: a, bbox: [1, 2, 3]}\n", r"\[min_x, min_y"),
        ("crs: EPSG:3067\naois:\n  - {name: a, bbox: [w, x, y, z]}\n", "four numbers"),
        ("- not a mapping\n", "YAML mapping"),
        ("crs: EPSG:3067\naois:\n  - just a string\n", "must be a mapping"),
    ],
)
def test_malformed_collection_is_rejected(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_aois(_write(tmp_path, body))


def test_misspelled_role_is_an_error_not_a_new_category(tmp_path: Path) -> None:
    """'postive' must not quietly become a role nothing ever selects."""
    body = "crs: EPSG:3067\naois:\n  - {name: a, bbox: [1, 2, 3, 4], role: postive}\n"
    with pytest.raises(
        ValueError, match="expected one of hard-negative, positive, sparse"
    ):
        load_aois(_write(tmp_path, body))


def test_misspelled_split_is_an_error(tmp_path: Path) -> None:
    """A bad split would silently move ground between train and validation."""
    body = "crs: EPSG:3067\naois:\n  - {name: a, bbox: [1, 2, 3, 4], split: val}\n"
    with pytest.raises(ValueError, match="expected one of train, validation"):
        load_aois(_write(tmp_path, body))


def test_duplicate_names_are_rejected(tmp_path: Path) -> None:
    body = (
        "crs: EPSG:3067\naois:\n"
        "  - {name: a, bbox: [1, 2, 3, 4]}\n"
        "  - {name: a, bbox: [5, 6, 7, 8]}\n"
    )
    with pytest.raises(ValueError, match="duplicate AOI name 'a'"):
        load_aois(_write(tmp_path, body))


def test_missing_collection_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AOI file not found"):
        load_aois(str(tmp_path / "absent.yaml"))


def test_overlaps_are_found_with_their_area(tmp_path: Path) -> None:
    body = (
        "crs: EPSG:3067\naois:\n"
        "  - {name: a, bbox: [0, 0, 300, 300]}\n"
        "  - {name: b, bbox: [200, 250, 500, 550]}\n"
        "  - {name: far, bbox: [9000, 9000, 9300, 9300]}\n"
    )
    found = overlaps(load_aois(_write(tmp_path, body)))
    assert len(found) == 1
    first, second, area = found[0]
    assert {first.name, second.name} == {"a", "b"}
    assert area == pytest.approx(100 * 50)  # 200..300 by 250..300


def test_overlap_is_measured_in_the_source_crs(tmp_path: Path) -> None:
    """Reprojection rotates a box; its EPSG:3879 envelope would invent overlaps.

    A 300 m box rotates ~1.74 degrees into EPSG:3879, growing its envelope to
    300 * (cos + sin) ~= 309 m. These two areas sit 6 m apart -- inside that
    ~9 m of growth -- so comparing envelopes would report a false overlap.
    """
    body = (
        "crs: EPSG:3067\naois:\n"
        "  - {name: a, bbox: [399000, 6675000, 399300, 6675300]}\n"
        "  - {name: b, bbox: [399306, 6675000, 399606, 6675300]}\n"
    )
    aois = load_aois(_write(tmp_path, body))
    assert overlaps(aois) == []
    # ...whereas the projected envelopes really do intersect.
    assert aois[0].bounds.max_easting > aois[1].bounds.min_easting


def test_bboxes_have_no_source_bbox_so_are_not_compared() -> None:
    assert overlaps(load_aois("24.93,60.16,24.95,60.18")) == []


def test_bbox_defaults_to_wgs84() -> None:
    bounds = load_aois("25.19,60.20,25.20,60.21")[0].bounds
    assert bounds.min_easting < VUOSAARI_GRID[0] < bounds.max_easting


def test_bbox_honours_an_explicit_crs() -> None:
    bounds = load_aois("399000,6675000,401000,6677000", crs=TM35FIN)[0].bounds
    assert bounds.min_easting < VUOSAARI_GRID[0] < bounds.max_easting
    assert bounds.min_northing < VUOSAARI_GRID[1] < bounds.max_northing


def test_bbox_in_tm35fin_is_not_silently_read_as_wgs84() -> None:
    """Projected metres read as degrees must fail loudly, not produce nonsense.

    pyproj returns infinities rather than raising for out-of-domain input, so
    without an explicit check this becomes inf bounds and a nan-wide AOI that
    only surfaces much later.
    """
    with pytest.raises(ValueError, match="different CRS than the one declared"):
        load_aois("399000,6675000,401000,6677000", crs=WGS84)


def test_all_four_corners_are_projected_not_just_two() -> None:
    """Projecting only min/max would under-cover a rotated box by ~120 m here.

    That is roughly four z16 tiles missing from each edge of every AOI, so the
    four-corner projection is load-bearing rather than pedantry.
    """
    bounds = load_aois("399000,6675000,401000,6677000", crs=TM35FIN)[0].bounds
    diagonal = [
        to_grid(x, y, TM35FIN) for x, y in [(399000, 6675000), (401000, 6677000)]
    ]
    assert bounds.width > abs(diagonal[1][0] - diagonal[0][0]) + 100


@pytest.mark.parametrize("spec", ["24.93,60.16,24.95", "a,b,c,d"])
def test_malformed_bbox_is_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        load_aois(spec)


def test_name_is_rejected_for_a_bbox() -> None:
    with pytest.raises(ValueError, match="only to a YAML AOI collection"):
        load_aois("24.93,60.16,24.95,60.18", name="somewhere")


def test_geojson_takes_the_total_extent(tmp_path: Path) -> None:
    path = tmp_path / "area.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[24.93, 60.16], [24.95, 60.16], [24.95, 60.18], [24.93, 60.18]]
                    ],
                },
            }
        )
    )
    aois = load_aois(str(path))
    assert [a.name for a in aois] == ["area"]  # named after the file
    assert aois[0].bounds == load_aois("24.93,60.16,24.95,60.18")[0].bounds


def test_geojson_honours_an_explicit_crs(tmp_path: Path) -> None:
    """GeoJSON is nominally WGS84, but Finnish exports are often EPSG:3067."""
    path = tmp_path / "area.geojson"
    path.write_text(
        json.dumps({"type": "Point", "coordinates": list(VUOSAARI_TM35FIN)})
    )
    bounds = load_aois(str(path), crs=TM35FIN)[0].bounds
    assert (bounds.min_easting, bounds.min_northing) == pytest.approx(
        VUOSAARI_GRID, abs=0.5
    )


def test_real_collection_parses(tmp_path: Path) -> None:
    """The checked-in collection must stay loadable as the schema evolves."""
    aois = load_aois("aois/helsinki.yaml")
    # Counted from the file rather than pinned: adding an area is routine, and
    # a magic number here only ever fails for the wrong reason.
    declared = yaml.safe_load(Path("aois/helsinki.yaml").read_text())["aois"]
    assert len(aois) == len(declared)
    assert {a.role for a in aois} == {"positive", "sparse", "hard-negative"}
    # The split membership *is* pinned: it decides which numbers are honest,
    # so moving ground between train and validation should never be a quiet
    # diff. Update this list deliberately, never to make the test pass.
    #
    # Round 4 added `r4-mustavuori` and `r4-kapyla` (docs/rounds.md,
    # 2026-08-13): validation held only `positive` ground before them, so the
    # negative-area check had nothing held out and the `sparse` role went
    # unmeasured. They also make round 4's gate numbers *not* directly
    # comparable with rounds 1-3, which were scored on the first four alone.
    #
    # `r4-orakas` was here too, and moved to train on 2026-08-14 *after*
    # round 4's gates were measured against it — so a replay of round 4 will
    # not reproduce the recorded numbers. Its 10 shadowed trucks are the
    # densest such ground in the corpus and were worth more in training:
    # held-out `r1-jatkasaari` went from 4 missed trucks to 1 with it. The
    # cost is that jatkasaari's 16 trucks are now the whole shadow signal.
    assert [a.name for a in aois if a.split == "validation"] == [
        "r1-veturitie",
        "r1-pohjois-haaga",
        "r1-kaivoksela",
        "r1-jatkasaari",
        "r4-mustavuori",
        "r4-kapyla",
    ]


def test_real_collection_has_no_role_or_split_conflicts() -> None:
    """No ground may be both positive and negative, or both train and validation.

    r1-vuosaari-channel-road used to overlap the retired vuosaari hard-negative by 134x17 m; its
    north edge now stops exactly at that boundary. This guards against
    reintroducing a conflict.
    """
    conflicts = [
        (a.name, b.name)
        for a, b, _ in overlaps(load_aois("aois/helsinki.yaml"))
        if a.role != b.role or a.split != b.split
    ]
    assert conflicts == []


def test_load_region_unions_polygons_into_the_grid_crs(tmp_path: Path) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [24.93, 60.16],
                            [24.94, 60.16],
                            [24.94, 60.17],
                            [24.93, 60.17],
                            [24.93, 60.16],
                        ]
                    ],
                },
                "properties": {},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [24.95, 60.16],
                            [24.96, 60.16],
                            [24.96, 60.17],
                            [24.95, 60.17],
                            [24.95, 60.16],
                        ]
                    ],
                },
                "properties": {},
            },
        ],
    }
    path = tmp_path / "region.geojson"
    path.write_text(json.dumps(geojson))

    region = load_region(str(path))

    assert region.geom_type == "MultiPolygon"  # two disjoint parts stay disjoint
    bounds = region_bounds(region)
    # EPSG:3879 metres around Helsinki, not degrees and not swapped axes.
    assert 25_400_000 < bounds.min_easting < 25_600_000
    assert 6_670_000 < bounds.min_northing < 6_690_000


def test_load_region_rejects_a_file_without_polygons(tmp_path: Path) -> None:
    path = tmp_path / "empty.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    with pytest.raises(ValueError, match="no polygon features"):
        load_region(str(path))


def test_load_region_reads_crs_carrying_formats(tmp_path: Path) -> None:
    """FlatGeobuf/GeoPackage declare their own CRS; a TM35FIN polygon must
    land in the same grid spot as its WGS84 twin."""
    geopandas = pytest.importorskip("geopandas", reason="detect extra not installed")
    from shapely.geometry import Polygon

    # The yard AOI's EPSG:3067 bbox as a polygon.
    polygon = Polygon(
        [(391857, 6680141), (392157, 6680141), (392157, 6680441), (391857, 6680441)]
    )
    frame = geopandas.GeoDataFrame({"name": ["yard"]}, geometry=[polygon], crs=TM35FIN)
    fgb = tmp_path / "region.fgb"
    gpkg = tmp_path / "region.gpkg"
    frame.to_file(fgb, driver="FlatGeobuf")
    frame.to_file(gpkg, driver="GPKG")

    for path in (fgb, gpkg):
        bounds = region_bounds(load_region(str(path)))
        # EPSG:3879 metres near Helsinki, not TM35FIN metres or degrees.
        assert 25_490_000 < bounds.min_easting < 25_510_000
        assert 6_680_000 < bounds.min_northing < 6_683_000
