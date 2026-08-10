"""Resolving areas of interest into named, projected bounds.

An AOI collection is a YAML file that declares its CRS once and then lists
areas in order:

.. code-block:: yaml

    crs: "EPSG:3067"
    aois:
      - name: r1-tattariharjuntie
        bbox: [391857, 6680141, 392157, 6680441]
        role: positive
        split: validation        # optional, defaults to train
        notes: Transport yard, rows of parked trucks.

``crs`` is required rather than defaulted. Finnish geodata is usually
EPSG:3067, the tile grid is EPSG:3879, and hand-typed bboxes are WGS84 -- three
plausible readings of the same four numbers, differing by hundreds of
kilometres. A file that does not say which it means is a bug waiting to happen.

``role`` and ``split`` are validated against fixed sets so that a typo becomes
an error rather than a silently-invented category: a misspelled ``postive``
would quietly drop an area out of the positives, and a misspelled split would
quietly move ground between train and validation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from shapely import union_all
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from rekka_ai.geo import GRID, WGS84, to_grid, transformer
from rekka_ai.imagery.tiles import Bounds

YAML_SUFFIXES = {".yaml", ".yml"}
BBOX_LENGTH = 4

#: What an area contributes to training.
#: ``positive`` holds trucks to label; ``hard-negative`` holds convincing
#: lookalikes (containers read as trailers from above); ``sparse`` is ordinary
#: city with few trucks, which keeps the model from firing everywhere.
ROLES = frozenset({"positive", "hard-negative", "sparse"})
SPLITS = frozenset({"train", "validation"})
DEFAULT_ROLE = "positive"
DEFAULT_SPLIT = "train"


@dataclass(frozen=True, slots=True)
class Aoi:
    """A named area, projected to the tile grid's EPSG:3879."""

    name: str
    bounds: Bounds
    role: str = DEFAULT_ROLE
    split: str = DEFAULT_SPLIT
    notes: str = ""
    #: The bbox as written, in the collection's own CRS. Kept because overlap
    #: between areas must be measured where they are still true rectangles:
    #: reprojecting rotates them, and the axis-aligned envelope in EPSG:3879 is
    #: ~9 m larger per edge, enough to invent or hide a narrow overlap.
    source_bbox: tuple[float, float, float, float] | None = None

    @property
    def is_negative(self) -> bool:
        """Whether this area is expected to hold no targets.

        Both ``hard-negative`` and ``sparse`` are negatives for measurement:
        the regression check asks "did fine-tuning start pulling lookalikes
        in", and either role answers it.
        """
        return self.role != "positive"


def load_aois(spec: str, *, crs: str = WGS84, name: str | None = None) -> list[Aoi]:
    """Resolve an ``--aoi`` argument into one or more named areas.

    ``spec`` is a path to a YAML collection, a path to a GeoJSON file, or a
    bbox ``min_x,min_y,max_x,max_y``. ``crs`` applies to the latter two; a
    collection carries its own. ``name`` selects a single entry.
    """
    path = Path(spec)

    if path.suffix.lower() in YAML_SUFFIXES:
        if not path.exists():
            raise ValueError(f"AOI file not found: {path}")
        return _select(_load_collection(path), name, path)

    if name is not None:
        raise ValueError("--name applies only to a YAML AOI collection")

    if path.exists():
        return [Aoi(name=path.stem, bounds=_bounds_from_geojson(path, crs))]

    return [Aoi(name="aoi", bounds=_bounds_from_bbox(spec, crs))]


def _select(aois: list[Aoi], name: str | None, path: Path) -> list[Aoi]:
    if name is None:
        return aois
    for aoi in aois:
        if aoi.name == name:
            return [aoi]
    available = ", ".join(a.name for a in aois)
    raise ValueError(f"no AOI named {name!r} in {path}; available: {available}")


def _load_collection(path: Path) -> list[Aoi]:
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        # ValueError, not TypeError: this is a malformed config file, and the
        # CLI turns ValueError into a clean message rather than a traceback.
        raise ValueError(f"{path} must contain a YAML mapping")  # noqa: TRY004

    crs = document.get("crs")
    if not crs:
        raise ValueError(f"{path} must declare a 'crs', for example 'EPSG:3067'")

    entries = document.get("aois")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a non-empty 'aois' list")

    aois = [_parse_entry(entry, crs, path) for entry in entries]

    seen: set[str] = set()
    for aoi in aois:
        if aoi.name in seen:
            raise ValueError(f"duplicate AOI name {aoi.name!r} in {path}")
        seen.add(aoi.name)
    return aois


def _parse_entry(entry: Any, crs: str, path: Path) -> Aoi:
    if not isinstance(entry, dict):
        message = f"each entry in {path} must be a mapping with 'name' and 'bbox'"
        raise ValueError(message)  # noqa: TRY004

    name = entry.get("name")
    if not name:
        raise ValueError(f"an AOI in {path} is missing its 'name'")
    name = str(name)

    if "bbox" not in entry:
        raise ValueError(f"AOI {name!r} in {path} needs a 'bbox'")
    bbox = _as_bbox(entry["bbox"], name)

    role = str(entry.get("role", DEFAULT_ROLE))
    if role not in ROLES:
        raise ValueError(
            f"AOI {name!r} has role {role!r}; expected one of {_listed(ROLES)}"
        )

    split = str(entry.get("split", DEFAULT_SPLIT))
    if split not in SPLITS:
        raise ValueError(
            f"AOI {name!r} has split {split!r}; expected one of {_listed(SPLITS)}"
        )

    return Aoi(
        name=name,
        bounds=_project_bbox(bbox, crs),
        role=role,
        split=split,
        notes=str(entry.get("notes", "")),
        source_bbox=bbox,
    )


def _listed(values: frozenset[str]) -> str:
    return ", ".join(sorted(values))


def overlaps(aois: list[Aoi]) -> list[tuple[Aoi, Aoi, float]]:
    """Pairs of areas that share ground, with the overlapping area in m².

    Overlapping areas mean the same vehicles are annotated twice, and if the
    two disagree on ``role`` or ``split`` they mean the same ground is both a
    positive and a negative, or both trained on and validated against.

    Only areas from a collection are compared, since a bbox or GeoJSON AOI has
    no bbox recorded in its own CRS to compare in.
    """
    found = []
    measurable = [(a, a.source_bbox) for a in aois if a.source_bbox is not None]
    for i, (first, first_bbox) in enumerate(measurable):
        for second, second_bbox in measurable[i + 1 :]:
            area = _intersection_area(first_bbox, second_bbox)
            if area > 0:
                found.append((first, second, area))
    return found


def _intersection_area(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0 and height > 0 else 0.0


def _bounds_from_bbox(spec: str, crs: str) -> Bounds:
    parts = spec.split(",")
    if len(parts) != BBOX_LENGTH:
        raise ValueError(
            f"expected a bbox 'min_x,min_y,max_x,max_y' or a GeoJSON/YAML path, got {spec!r}"
        )
    return _project_bbox(_as_bbox(parts, "bbox"), crs)


def _as_bbox(values: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(values, list | tuple) or len(values) != BBOX_LENGTH:
        raise ValueError(f"bbox for {name!r} must be [min_x, min_y, max_x, max_y]")
    try:
        min_x, min_y, max_x, max_y = (float(v) for v in values)
    except TypeError, ValueError:
        raise ValueError(
            f"bbox for {name!r} must be four numbers, got {values!r}"
        ) from None
    return min_x, min_y, max_x, max_y


def _project_bbox(bbox: tuple[float, float, float, float], crs: str) -> Bounds:
    min_x, min_y, max_x, max_y = bbox
    return _project_extent(
        [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
        crs,
    )


def _bounds_from_geojson(path: Path, crs: str) -> Bounds:
    data = json.loads(path.read_text())
    points = list(_walk_coordinates(data))
    if not points:
        raise ValueError(f"no coordinates found in {path}")
    return _project_extent(points, crs)


def load_region(spec: str, crs: str = WGS84) -> BaseGeometry:
    """Polygon features from a geodata file, unioned, in EPSG:3879.

    Detection regions are real boundaries — postcode areas, districts — not
    just their bounding boxes: sweeping the bounds would cover ground the
    user never asked about, so detections are filtered against the union.

    GeoJSON is read directly (``crs`` applies, default WGS84 per RFC 7946).
    FlatGeobuf, GeoPackage, shapefile and friends go through geopandas — an
    optional dependency — and carry their own CRS, which wins over ``crs``.
    """
    path = Path(spec)
    if not path.exists():
        raise ValueError(f"no such file: {path}")
    if path.suffix.lower() in {".geojson", ".json"}:
        return _load_region_geojson(path, crs)
    try:
        import geopandas
    except ImportError as exc:
        raise ValueError(
            f"{path.suffix} regions need geopandas: uv sync --extra detect"
        ) from exc
    frame = geopandas.read_file(path)
    if frame.empty:
        raise ValueError(f"no polygon features found in {path}")
    if frame.crs is None:
        # A shapefile without a .prj is the one format that still needs the
        # caller to say what its coordinates mean.
        frame = frame.set_crs(crs)
    return union_all(list(frame.to_crs(GRID).geometry))


def _load_region_geojson(path: Path, crs: str) -> BaseGeometry:
    data = json.loads(path.read_text())
    if data.get("type") == "FeatureCollection":
        geometries = [f["geometry"] for f in data["features"] if f.get("geometry")]
    elif "coordinates" in data:
        geometries = [data]
    else:
        raise ValueError(f"no polygon features found in {path}")
    if not geometries:
        raise ValueError(f"no polygon features found in {path}")
    project = transformer(crs, GRID)
    return union_all(
        [
            transform(lambda xs, ys: project.transform(xs, ys), shape(g))
            for g in geometries
        ]
    )


def region_bounds(region: BaseGeometry) -> Bounds:
    min_easting, min_northing, max_easting, max_northing = region.bounds
    return Bounds(
        min_easting=min_easting,
        min_northing=min_northing,
        max_easting=max_easting,
        max_northing=max_northing,
    )


def _walk_coordinates(node: Any) -> Any:
    """Yield every (x, y) pair in a GeoJSON document.

    Bounds are all we need, so the geometry type does not matter -- and this
    avoids a geometry dependency for what is ultimately a min/max.
    """
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_coordinates(value)
    elif isinstance(node, list):
        if len(node) >= 2 and all(isinstance(v, int | float) for v in node[:2]):
            yield float(node[0]), float(node[1])
        else:
            for item in node:
                yield from _walk_coordinates(item)


def _project_extent(points: list[tuple[float, float]], crs: str) -> Bounds:
    # Project every corner rather than just the min/max pair: the transform is
    # not axis-aligned, so two corners can under-cover the true extent.
    projected = [to_grid(x, y, crs) for x, y in points]
    eastings = [e for e, _ in projected]
    northings = [n for _, n in projected]
    return Bounds(
        min_easting=min(eastings),
        min_northing=min(northings),
        max_easting=max(eastings),
        max_northing=max(northings),
    )
