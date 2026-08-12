"""OpenStreetMap industrial geometry via Overpass, cached on disk.

Mining proposes the next labelling batch from OSM industrial landuse rather
than hand-picking yards. The Overpass response is large and changes slowly,
so it is fetched once, written under ``data/osm/``, and reused until
``--refresh-osm`` asks for a new copy. Raw responses are gitignored: they are
regenerable and not part of the project's own data.

Helsinki only for now. The 2025 5 cm orthophoto WMTS covers Helsinki; Espoo
and Vantaa need a different imagery source (HSY) before their municipal
codes can be enabled here.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from rekka_ai.imagery.wmts import USER_AGENT

#: Public Overpass interpreter. DNS round-robin across the two public nodes.
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Municipal codes (kuntaliitto / OSM ``ref``) this module can query.
#: Espoo (049) and Vantaa (092) are listed but rejected until imagery covers
#: them — see ``require_supported_municipality``.
MUNICIPALITIES: dict[str, str] = {
    "Helsinki": "091",
    "Espoo": "049",
    "Vantaa": "092",
}

#: Municipalities whose industrial landuse can be mined against the current
#: Helsinki City Survey WMTS. Others raise until an imagery source exists.
SUPPORTED_MUNICIPALITIES = frozenset({"Helsinki"})

#: Profile name -> Overpass tag filters (AND within a filter, OR across them).
#: ``industrial`` is the first profile; the others exist for van mining:
#: parcel depots and fleets sit on commercial ground, and tradesmen's vans
#: cluster on construction sites — neither is reliably ``industrial`` in OSM.
PROFILES: dict[str, tuple[str, ...]] = {
    "industrial": ('["landuse"="industrial"]',),
    "commercial": ('["landuse"="commercial"]',),
    "construction": ('["landuse"="construction"]',),
    #: RV/motorhome-shaped ground: camp and caravan sites, and caravan
    #: dealers/storage. Rastila's train-side counterpart lives here.
    "camping": (
        '["tourism"="camp_site"]',
        '["tourism"="caravan_site"]',
        '["shop"="caravan"]',
    ),
}

ATTRIBUTION = "© OpenStreetMap contributors"
DEFAULT_CACHE = Path("data/osm")
DEFAULT_TIMEOUT = 180.0
MAX_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class OsmFeature:
    """One OSM polygon, still in WGS84 as Overpass returns it."""

    osm_type: str
    osm_id: int
    geometry: BaseGeometry
    tags: dict[str, str]


@dataclass(frozen=True, slots=True)
class OsmCache:
    """A cached Overpass response plus the metadata needed to audit it."""

    municipality: str
    ref: str
    profile: str
    query: str
    fetched_at: str
    query_hash: str
    path: Path
    features: tuple[OsmFeature, ...]

    @property
    def attribution(self) -> str:
        return ATTRIBUTION


def require_supported_municipality(name: str) -> str:
    """Return the municipal ``ref``, or raise if imagery cannot cover it yet."""
    if name not in MUNICIPALITIES:
        available = ", ".join(sorted(MUNICIPALITIES))
        raise ValueError(f"unknown municipality {name!r}; expected one of {available}")
    if name not in SUPPORTED_MUNICIPALITIES:
        supported = ", ".join(sorted(SUPPORTED_MUNICIPALITIES))
        raise ValueError(
            f"municipality {name!r} is not supported yet: the current imagery "
            f"covers {supported} only. Espoo/Vantaa need an HSY imagery source "
            "before mining can include them."
        )
    return MUNICIPALITIES[name]


def require_profile(name: str) -> tuple[str, ...]:
    """Return the tag filters for ``name``, or raise on an unknown profile."""
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown profile {name!r}; expected one of {available}")
    return PROFILES[name]


def build_query(municipality: str, profile: str = "industrial") -> str:
    """Overpass QL for the profile's landuse inside one Finnish municipality.

    Areas are selected by ``ref`` (kuntaliitto code) rather than by name, so a
    renamed place or a Swedish-language name tag cannot silently retarget the
    query. ``out geom`` returns full rings so shapely can build polygons
    without a second lookup.
    """
    ref = require_supported_municipality(municipality)
    filters = require_profile(profile)
    # One subquery per filter, unioned: Overpass has no OR inside a single
    # tag filter, and future profiles will add more than one tag pair.
    parts = "\n".join(
        f"  way{tag_filter}(area.a);\n  relation{tag_filter}(area.a);"
        for tag_filter in filters
    )
    return (
        f"[out:json][timeout:{int(DEFAULT_TIMEOUT)}];\n"
        f'area["ref"="{ref}"]["admin_level"="8"]->.a;\n'
        f"(\n{parts}\n);\n"
        "out geom;\n"
    )


def cache_key(municipality: str, profile: str = "industrial") -> str:
    """Stable filename stem for one municipality + profile pair."""
    ref = require_supported_municipality(municipality)
    require_profile(profile)
    return f"{ref}_{municipality.lower()}_{profile}"


def query_hash(query: str) -> str:
    """Short content hash of the query string, for the cache metadata."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def cache_path(
    cache_root: Path, municipality: str, profile: str = "industrial"
) -> Path:
    return cache_root / f"{cache_key(municipality, profile)}.json"


def load_cached(
    cache_root: Path, municipality: str, profile: str = "industrial"
) -> OsmCache | None:
    """Return a previously written response, or ``None`` if none exists."""
    path = cache_path(cache_root, municipality, profile)
    if not path.exists():
        return None
    document = json.loads(path.read_text())
    features = tuple(_parse_element(element) for element in document["elements"])
    features = tuple(f for f in features if f is not None)
    return OsmCache(
        municipality=document["municipality"],
        ref=document["ref"],
        profile=document["profile"],
        query=document["query"],
        fetched_at=document["fetched_at"],
        query_hash=document["query_hash"],
        path=path,
        features=features,
    )


def fetch_industrial(
    municipality: str = "Helsinki",
    *,
    profile: str = "industrial",
    cache_root: Path = DEFAULT_CACHE,
    refresh: bool = False,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
) -> OsmCache:
    """Return industrial polygons, from cache unless ``refresh`` is set.

    ``client`` / ``transport`` exist so tests can inject a mock Overpass
    without touching the network — the same pattern as ``TileFetcher``.
    """
    query = build_query(municipality, profile)
    path = cache_path(cache_root, municipality, profile)
    if not refresh and path.exists():
        cached = load_cached(cache_root, municipality, profile)
        assert cached is not None
        # A query change (new profile filter, timeout, …) must not reuse a
        # response that answered a different question.
        if cached.query_hash == query_hash(query):
            return cached

    owns_client = client is None
    if client is None:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
            transport=transport,
        )
    try:
        payload = _post(client, query)
    finally:
        if owns_client:
            client.close()

    elements = payload.get("elements", [])
    if not isinstance(elements, list):
        # ValueError, not TypeError: this is a malformed Overpass payload, and
        # the CLI turns ValueError into a clean message rather than a traceback.
        raise ValueError("Overpass response missing an 'elements' list")  # noqa: TRY004

    document = {
        "municipality": municipality,
        "ref": MUNICIPALITIES[municipality],
        "profile": profile,
        "query": query,
        "query_hash": query_hash(query),
        "fetched_at": datetime.now(UTC).isoformat(),
        "attribution": ATTRIBUTION,
        "elements": elements,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)
    cached = load_cached(cache_root, municipality, profile)
    assert cached is not None  # just written above
    return cached


def union_wgs84(features: Iterable[OsmFeature]) -> BaseGeometry:
    """Dissolve every feature into one multipolygon in WGS84."""
    geometries = [f.geometry for f in features if not f.geometry.is_empty]
    if not geometries:
        return MultiPolygon()
    return unary_union(geometries)


def _post(client: httpx.Client, query: str) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.post(OVERPASS_URL, data={"data": query})
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(  # noqa: TRY004
                    "Overpass response is not a JSON object"
                )
            return payload
        except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            # 4xx other than 429 means the query itself is wrong; retrying
            # will not help. 429 / 5xx / transport errors are transient.
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Overpass query failed after {MAX_ATTEMPTS} attempts") from last


def _parse_element(element: dict[str, Any]) -> OsmFeature | None:
    """One Overpass element to a shapely geometry, or ``None`` if unusable.

    Ways and relations arrive with ``geometry`` / ``members`` under
    ``out geom``. Incomplete rings and non-area geometries are skipped rather
    than raised: OSM is noisy, and one broken warehouse must not kill a
    city-wide mine.
    """
    osm_type = str(element.get("type", ""))
    osm_id = int(element.get("id", 0))
    tags = {str(k): str(v) for k, v in element.get("tags", {}).items()}
    geometry = _geometry_of(element)
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return None
    return OsmFeature(osm_type=osm_type, osm_id=osm_id, geometry=geometry, tags=tags)


def _geometry_of(element: dict[str, Any]) -> BaseGeometry | None:
    osm_type = element.get("type")
    if osm_type == "way":
        return _polygon_from_nodes(element.get("geometry") or [])
    if osm_type == "relation":
        return _relation_geometry(element)
    # A bare node cannot be industrial landuse for our purposes.
    return None


def _polygon_from_nodes(nodes: list[dict[str, Any]]) -> BaseGeometry | None:
    if len(nodes) < 4:
        return None
    try:
        coords = [(float(n["lon"]), float(n["lat"])) for n in nodes]
    except KeyError, TypeError, ValueError:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    if len(coords) < 4:
        return None
    polygon = Polygon(coords)
    return polygon if not polygon.is_empty else None


def _relation_geometry(element: dict[str, Any]) -> BaseGeometry | None:
    """Build a (multi)polygon from a multipolygon relation's outer members.

    Inner rings are ignored for mining: we only need ground that intersects
    industrial landuse, and a hole would at worst leave a cell slightly more
    covered than the true footprint — never less.
    """
    members = element.get("members") or []
    outers = []
    for member in members:
        if member.get("type") != "way" or member.get("role") not in ("outer", ""):
            continue
        polygon = _polygon_from_nodes(member.get("geometry") or [])
        if polygon is not None:
            outers.append(polygon)
    if not outers:
        # Some Overpass responses also expose a top-level geometry list for
        # simple relations; try that before giving up.
        return _polygon_from_nodes(element.get("geometry") or [])
    if len(outers) == 1:
        return outers[0]
    return MultiPolygon(outers)
