import json
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from rekka_ai import labels
from rekka_ai.detect.chunks import (
    DEFAULT_CELL_SIZE_M,
    DONE,
    FAILED,
    RUNNING,
    append,
    cell_path,
    grid_cells,
    load_cells,
    manifest_path,
    read_manifest,
    run_key,
)
from rekka_ai.detect.detections import Detection, merge, within_region, write
from rekka_ai.detect.sweep import (
    DEFAULT_CONFIDENCE,
    DEFAULT_WEIGHTS,
    LARGE_VEHICLE,
    MIN_LENGTH_M,
    SMALL_VEHICLE,
    YoloObb,
    emits_dota_vehicles,
    sweep,
)
from rekka_ai.enrich import (
    CACHE_DIR as DEFAULT_WFS_CACHE,
)
from rekka_ai.enrich import (
    STREET_MAX_DISTANCE_M,
)
from rekka_ai.enrich import (
    enrich as run_enrich,
)
from rekka_ai.evaluate import (
    GATE_PRECISION,
    GATE_RECALL,
    MIN_OPERATING_CONFIDENCE,
    NEGATIVE_DETECTIONS_WATCH,
    count_detections,
    count_gate,
    ground_truth_counts,
    load_detector,
    log_eval_run,
    pick_operating_point,
    sweep_area,
    unexplained,
    validation_metrics,
)
from rekka_ai.export import dataset_yaml, export_area
from rekka_ai.geo import WGS84
from rekka_ai.imagery.aoi import (
    ROLES,
    YAML_SUFFIXES,
    Aoi,
    load_aois,
    load_region,
    overlaps,
    region_bounds,
)
from rekka_ai.imagery.layers import LATEST_YEAR, layer_for_year
from rekka_ai.imagery.tiles import count_tiles, resolution, tiles_covering
from rekka_ai.imagery.wmts import (
    DEFAULT_WORKERS,
    MAX_ATTEMPTS,
    TileFetcher,
)
from rekka_ai.mine import (
    DEFAULT_COUNT,
    DEFAULT_POOL_SIZE,
    DEFAULT_SEED,
    cells_covering,
    detections_in_cell,
    disperse_pool,
    estimate_tiles,
    exclude_existing,
    project_features_to_tm35fin,
    proposal_geojson,
    proposal_yaml,
    rank_cells,
    select_proposals,
)
from rekka_ai.osm import (
    DEFAULT_CACHE as DEFAULT_OSM_CACHE,
)
from rekka_ai.osm import (
    PROFILES,
    fetch_industrial,
    require_profile,
    require_supported_municipality,
)
from rekka_ai.train import (
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
)
from rekka_ai.train import DEFAULT_SEED as DEFAULT_TRAIN_SEED
from rekka_ai.train import train as run_train

app = typer.Typer(help="Truck detection from Helsinki aerial orthophotos.")

#: z16 is 12.5 cm/px, where a semi-trailer is ~132 px. See docs/DESIGN.md.
DEFAULT_ZOOM = 16
#: Also z16. Matching DOTA's pretrain GSD at z15 was the obvious guess and it
#: is wrong: measured on the `r1-tattariharjuntie` area, z15 finds 7 candidates and z16 finds
#: 22 above the length gate. See docs/DESIGN.md section 4.
BOOTSTRAP_ZOOM = 16
DEFAULT_CACHE = Path("data/cache")
#: Version-controlled: labels are the one artifact that cannot be regenerated.
DEFAULT_LABELS = Path("labels")


def _progress(index: int, total: int, found: int) -> None:
    """Sweep progress for long runs: the first and last window, then every
    hundredth. A 300 m AOI has ~10 windows and stays quiet; a polygon region
    can have tens of thousands, and a multi-hour fetch with no output looks
    dead."""
    if index == 1 or index == total or index % 100 == 0:
        typer.echo(f"  window {index}/{total}, {found} detections")


def _elapsed(start: float) -> str:
    """Wall-clock for a done line: '42s', '12m 5s', '1h 3m'."""
    seconds = int(time.monotonic() - start)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


@app.callback()
def main() -> None:
    """Keep subcommand dispatch even with a single command registered."""


@app.command()
def fetch(
    aoi: Annotated[
        str,
        typer.Option(
            help="YAML AOI collection, GeoJSON file, or bbox 'min_x,min_y,max_x,max_y'."
        ),
    ],
    name: Annotated[
        str | None,
        typer.Option(
            help="Select one AOI from a YAML collection. Default: all of them."
        ),
    ] = None,
    crs: Annotated[
        str,
        typer.Option(
            help="CRS of a bbox or GeoJSON. A YAML collection declares its own."
        ),
    ] = WGS84,
    year: Annotated[
        int, typer.Option(help="Flight year of the orthophoto layer.")
    ] = LATEST_YEAR,
    zoom: Annotated[
        int, typer.Option(help="Tile grid zoom level (0-17).")
    ] = DEFAULT_ZOOM,
    cache: Annotated[Path, typer.Option(help="Tile cache directory.")] = DEFAULT_CACHE,
    workers: Annotated[
        int, typer.Option(help="Concurrent requests.")
    ] = DEFAULT_WORKERS,
    dry_run: Annotated[
        bool, typer.Option(help="Report the tile count and stop.")
    ] = False,
) -> None:
    """Populate the tile cache for one or more areas of interest.

    Separate from detection on purpose: this is the slow, network-bound part,
    and running it once lets detection iterate offline.
    """
    try:
        aois = load_aois(aoi, crs=crs, name=name)
        layer = layer_for_year(year)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    counts = {a.name: count_tiles(a.bounds, zoom) for a in aois}
    total = sum(counts.values())
    typer.echo(f"{layer} z{zoom} ({resolution(zoom) * 100:.2f} cm/px)")
    for area in aois:
        typer.echo(
            f"  {area.name}: {counts[area.name]} tiles covering "
            f"{area.bounds.width:.0f}x{area.bounds.height:.0f} m"
        )
    typer.echo(f"  total: {total} tiles")
    if dry_run:
        return

    fetched = cached = 0
    start = time.monotonic()
    with TileFetcher(cache, workers=workers) as fetcher:
        for area in aois:
            for result in fetcher.fetch_all(layer, tiles_covering(area.bounds, zoom)):
                if result.from_cache:
                    cached += 1
                else:
                    fetched += 1
                done = fetched + cached + len(fetcher.failures)
                if done % 100 == 0 or done == total:
                    typer.echo(
                        f"  {done}/{total} tiles ({fetched} fetched, {cached} cached)"
                    )
        failures = list(fetcher.failures)

    typer.echo(
        f"done: {fetched} fetched, {cached} already cached -> {cache} "
        f"in {_elapsed(start)}"
    )
    if failures:
        typer.echo(
            f"warning: {len(failures)} tile(s) failed after {MAX_ATTEMPTS} attempts; "
            "re-run to pick them up",
            err=True,
        )
        for tile, exc in failures[:5]:
            typer.echo(f"  {tile}: {exc}", err=True)
        raise typer.Exit(1)


@app.command()
def bootstrap(
    aoi: Annotated[
        str,
        typer.Option(
            help="YAML AOI collection, GeoJSON file, or bbox 'min_x,min_y,max_x,max_y'."
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output file for candidates: .geojson, .fgb or .gpkg."),
    ],
    name: Annotated[
        str | None, typer.Option(help="Select one AOI from a collection.")
    ] = None,
    role: Annotated[
        str | None,
        typer.Option(help=f"Only areas with this role ({', '.join(sorted(ROLES))})."),
    ] = None,
    crs: Annotated[str, typer.Option(help="CRS of a bbox or GeoJSON.")] = WGS84,
    year: Annotated[
        int, typer.Option(help="Flight year of the orthophoto layer.")
    ] = LATEST_YEAR,
    zoom: Annotated[
        int, typer.Option(help="Tile grid zoom level (0-17).")
    ] = BOOTSTRAP_ZOOM,
    weights: Annotated[
        str, typer.Option(help="Ultralytics OBB weights.")
    ] = DEFAULT_WEIGHTS,
    confidence: Annotated[
        float, typer.Option(help="Minimum detection confidence.")
    ] = DEFAULT_CONFIDENCE,
    min_length: Annotated[
        float, typer.Option(help="Drop candidates shorter than this, in metres.")
    ] = MIN_LENGTH_M,
    cache: Annotated[Path, typer.Option(help="Tile cache directory.")] = DEFAULT_CACHE,
    workers: Annotated[
        int, typer.Option(help="Concurrent requests.")
    ] = DEFAULT_WORKERS,
) -> None:
    """Pre-annotate an area with a pretrained detector, for a human to correct.

    The output is a starting point for labelling, not a result: correcting
    candidates is several times faster than drawing boxes on blank imagery.
    Needs the 'detect' extra (uv sync --extra detect).
    """
    try:
        areas = load_aois(aoi, crs=crs, name=name)
        layer = layer_for_year(year)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if role is not None:
        if role not in ROLES:
            raise typer.BadParameter(
                f"role {role!r}; expected one of {', '.join(sorted(ROLES))}"
            )
        areas = [a for a in areas if a.role == role]
        if not areas:
            raise typer.BadParameter(f"no areas with role {role!r}")

    typer.echo(
        f"{layer} z{zoom} ({resolution(zoom) * 100:.2f} cm/px), weights {weights}, "
        f"{len(areas)} area(s)"
    )
    detector = YoloObb(
        weights, confidence=confidence, keep=frozenset({LARGE_VEHICLE, SMALL_VEHICLE})
    )
    if not emits_dota_vehicles(detector.class_names):
        raise typer.BadParameter(
            f"{weights} emits {', '.join(sorted(detector.class_names)) or 'no classes'} "
            f"— not DOTA's {LARGE_VEHICLE!r}/{SMALL_VEHICLE!r}, which is all "
            "bootstrap keeps. It would sweep every tile and propose nothing. "
            "bootstrap is the cold-start command, for the pretrained weights "
            "only; propose from trained weights with "
            "`rekka-ai detect --weights <best.pt>`."
        )

    found: list[Detection] = []
    start = time.monotonic()
    with TileFetcher(cache, workers=workers) as fetcher:
        for area in areas:
            candidates = sweep(
                area,
                detector=detector,
                layer=layer,
                zoom=zoom,
                cache_root=cache,
                fetcher=fetcher,
                min_length_m=min_length,
                on_skip=typer.echo,
                on_progress=_progress,
            )
            typer.echo(f"  {area.name} ({area.role}): {len(candidates)} candidates")
            found.extend(candidates)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        write(found, out, source_layer=layer, zoom=zoom)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"done: {len(found)} candidates -> {out} in {_elapsed(start)}")


@app.command()
def enrich(
    detections: Annotated[
        Path,
        typer.Option(help="Detections GeoJSON/FlatGeobuf/GeoPackage from detect."),
    ],
    out: Annotated[
        Path,
        typer.Option(
            help="Output file for enriched detections: .geojson, .fgb or .gpkg."
        ),
    ],
    wfs_cache: Annotated[
        Path, typer.Option(help="Directory for cached WFS layers.")
    ] = DEFAULT_WFS_CACHE,
    refresh_wfs: Annotated[
        bool,
        typer.Option(
            help="Re-fetch WFS layers (districts, postal areas, streets, parking)."
        ),
    ] = False,
    street_max_distance: Annotated[
        float,
        typer.Option(
            help="How far (metres) a detection's centre may be from the nearest "
            "street and still be attributed to it."
        ),
    ] = STREET_MAX_DISTANCE_M,
) -> None:
    """Add district/postal_code/street/street_type/context/street_part.

    `street_type` is the street-area register's own purpose (Asuntokatu,
    Katuaukio, Tori, ...) -- the layer covers squares and pedestrian areas
    too, and the type is what tells them apart. `context` is
    parking/street/other and `street_part` refines the street
    case with the city's own YLRE terms (Ajorata/Pysakointialue/
    Tonttiliittymä/Koroke) -- where the vehicle *is*, not whether it is
    parked; one orthophoto cannot tell that. See rekka_ai.enrich's module
    docstring for the layers and the measurements.

    Reads detect's output and joins in attributes from *other* datasets --
    detection-derived numbers (length_m, width_m, heading_deg) are untouched.
    WFS layers are cached under --wfs-cache so a re-run does not depend on
    the network; --refresh-wfs re-fetches. Needs the 'detect' extra
    (uv sync --extra detect), for geopandas.
    """
    if not detections.exists():
        raise typer.BadParameter(f"no detections file at {detections}")

    start = time.monotonic()
    try:
        run_enrich(
            detections,
            out,
            cache_root=wfs_cache,
            refresh_wfs=refresh_wfs,
            street_max_distance_m=street_max_distance,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"done: enriched {detections} -> {out} in {_elapsed(start)}")


@app.command()
def stage(
    candidates: Annotated[Path, typer.Option(help="Candidate GeoJSON from bootstrap.")],
    labels_dir: Annotated[
        Path, typer.Option("--labels", help="Directory of per-AOI label files.")
    ] = DEFAULT_LABELS,
    force: Annotated[
        bool, typer.Option(help="Overwrite label files that already exist.")
    ] = False,
    aoi: Annotated[
        str | None,
        typer.Option(
            help="YAML AOI collection: also stage an empty label file for every "
            "named area that has no candidates (quiet mined cells)."
        ),
    ] = None,
) -> None:
    """Split bootstrap candidates into per-AOI label files, ready to review.

    Existing files are left alone unless --force: re-running the detector must
    never quietly discard hours of human correction.

    ``--aoi`` writes an empty FeatureCollection for collection entries that
    produced no candidates — mined quiet cells still need a reviewed file
    before export will accept them as background.
    """
    collection = labels.read(candidates)
    by_aoi: dict[str, list[dict[str, object]]] = {}
    for feature in collection["features"]:
        properties = dict(feature["properties"])
        name = str(properties.get("aoi") or "unknown")
        properties.setdefault("status", "candidate")
        properties.setdefault("class", labels.UNLABELLED)
        properties.pop("label", None)
        by_aoi.setdefault(name, []).append({**feature, "properties": properties})

    if aoi is not None:
        try:
            areas = load_aois(aoi)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        for area in areas:
            by_aoi.setdefault(area.name, [])

    written = skipped = 0
    for name, features in sorted(by_aoi.items()):
        path = labels_dir / f"{name}.geojson"
        if path.exists() and not force:
            existing = labels.read(path)
            typer.echo(
                f"  {name}: exists with {labels.reviewed(existing)} reviewed, skipping"
            )
            skipped += 1
            continue
        labels.write(path, {"type": "FeatureCollection", "features": features})
        typer.echo(f"  {name}: {len(features)} candidates -> {path}")
        written += 1

    typer.echo(f"staged {written} file(s), skipped {skipped} -> {labels_dir}")
    if skipped and not force:
        typer.echo("re-run with --force to replace the skipped files")


@app.command()
def mine(
    weights: Annotated[Path, typer.Option(help="Trained weights to score cells with.")],
    operating_confidence: Annotated[
        float,
        typer.Option(
            help="Operating confidence from eval; near-threshold cells are scored around it."
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            help="Proposal YAML to write (sibling .geojson report too). "
            "Default: data/mining/round<round>.yaml, or "
            "data/mining/proposal.yaml without --round."
        ),
    ] = None,
    round_: Annotated[
        int | None,
        typer.Option(
            "--round",
            help="Labelling round number; prefixes proposed names 'rN-mine-...' "
            "so a round's AOIs and label files sort and group together.",
        ),
    ] = None,
    existing: Annotated[
        Path,
        typer.Option(help="Current AOI collection to exclude (overlap + buffer)."),
    ] = Path("aois/helsinki.yaml"),
    municipality: Annotated[
        str, typer.Option(help="Finnish municipality to mine inside.")
    ] = "Helsinki",
    profile: Annotated[
        str,
        typer.Option(help=f"OSM landuse profile ({', '.join(sorted(PROFILES))})."),
    ] = "industrial",
    count: Annotated[
        int, typer.Option(help="How many AOIs to propose.")
    ] = DEFAULT_COUNT,
    pool_size: Annotated[
        int, typer.Option(help="How many dispersed cells to sweep before selecting.")
    ] = DEFAULT_POOL_SIZE,
    seed: Annotated[
        int, typer.Option(help="Deterministic seed for pool dispersion.")
    ] = DEFAULT_SEED,
    year: Annotated[
        int, typer.Option(help="Flight year of the orthophoto layer.")
    ] = LATEST_YEAR,
    zoom: Annotated[
        int, typer.Option(help="Tile grid zoom level (0-17).")
    ] = DEFAULT_ZOOM,
    min_length: Annotated[
        float, typer.Option(help="Drop detections shorter than this, in metres.")
    ] = MIN_LENGTH_M,
    confidence: Annotated[
        float | None,
        typer.Option(
            help="Detection floor for the sweep. Default: half the operating confidence."
        ),
    ] = None,
    cache: Annotated[Path, typer.Option(help="Tile cache directory.")] = DEFAULT_CACHE,
    osm_cache: Annotated[
        Path, typer.Option(help="Directory for cached Overpass responses.")
    ] = DEFAULT_OSM_CACHE,
    refresh_osm: Annotated[
        bool, typer.Option(help="Re-fetch industrial geometry from Overpass.")
    ] = False,
    workers: Annotated[
        int, typer.Option(help="Concurrent tile requests.")
    ] = DEFAULT_WORKERS,
    dry_run: Annotated[
        bool,
        typer.Option(
            help="Report eligible cells and tile estimate; skip imagery and inference."
        ),
    ] = False,
) -> None:
    """Propose the next labelling AOIs from OSM industrial land and a model.

    Writes a standalone proposal YAML and a sibling GeoJSON report — never
    edits the version-controlled collection. Review the report, copy accepted
    entries into aois/, then detect and stage as usual. Needs the 'detect'
    extra (uv sync --extra detect). Helsinki only: the current imagery does
    not cover Espoo or Vantaa.
    """
    if not weights.exists():
        raise typer.BadParameter(f"no weights at {weights}")
    if not (0.0 < operating_confidence < 1.0):
        raise typer.BadParameter("operating-confidence must be between 0 and 1")
    if count < 1:
        raise typer.BadParameter("count must be at least 1")
    if pool_size < count:
        raise typer.BadParameter("pool-size must be at least count")

    if out is None:
        out = Path(
            f"data/mining/round{round_}.yaml"
            if round_ is not None
            else "data/mining/proposal.yaml"
        )

    try:
        require_supported_municipality(municipality)
        require_profile(profile)
        layer = layer_for_year(year)
        existing_aois = load_aois(str(existing))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    detection_floor = (
        confidence if confidence is not None else max(0.05, operating_confidence / 2)
    )

    typer.echo(
        f"mining {municipality} ({profile}): excluding {len(existing_aois)} existing "
        f"AOIs, proposing {count} of {pool_size} pool cells"
    )
    osm = fetch_industrial(
        municipality,
        profile=profile,
        cache_root=osm_cache,
        refresh=refresh_osm,
    )
    typer.echo(
        f"  OSM: {len(osm.features)} features "
        f"(query {osm.query_hash}, fetched {osm.fetched_at})"
    )

    industrial = project_features_to_tm35fin(osm.features)
    if industrial.is_empty:
        raise typer.BadParameter(
            f"no usable {profile} geometry for {municipality}; "
            "try --refresh-osm or a different profile"
        )

    eligible = exclude_existing(
        cells_covering(
            industrial,
            municipality_ref=osm.ref,
            round_label=f"r{round_}" if round_ is not None else "",
        ),
        existing_aois,
    )
    if not eligible:
        raise typer.BadParameter(
            "no eligible cells after excluding the existing collection; "
            "widen the profile or shrink the exclusion buffer"
        )
    pool = disperse_pool(eligible, pool_size, seed=seed)
    tile_estimate = estimate_tiles(pool, zoom)
    typer.echo(
        f"  eligible cells: {len(eligible)}; pool: {len(pool)}; "
        f"~{tile_estimate} tiles at z{zoom}"
    )
    if dry_run:
        typer.echo("dry run: skipping imagery fetch and inference")
        return

    detector = YoloObb(str(weights), confidence=detection_floor)
    detections_by_name: dict[str, list[Detection]] = {}
    start = time.monotonic()
    with TileFetcher(cache, workers=workers) as fetcher:
        for index, cell in enumerate(pool, start=1):
            area = cell.as_aoi()
            found = sweep(
                area,
                detector=detector,
                layer=layer,
                zoom=zoom,
                cache_root=cache,
                fetcher=fetcher,
                min_length_m=min_length,
                on_skip=typer.echo,
            )
            # Centre-filter to the cell: windows overhang by the overlap, and
            # a detection in the overhang belongs to a neighbour (or nowhere).
            detections_by_name[cell.name] = detections_in_cell(found, cell)
            if index == len(pool) or index % 10 == 0:
                typer.echo(f"  swept {index}/{len(pool)} pool cells")

    ranked = rank_cells(
        pool, detections_by_name, operating_confidence=operating_confidence
    )
    proposals = select_proposals(ranked, count)
    if not proposals:
        typer.echo("no proposals could be selected from the pool", err=True)
        raise typer.Exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(proposal_yaml(proposals))
    report = out.with_suffix(".geojson")
    report.write_text(
        json.dumps(
            proposal_geojson(
                proposals, osm=osm, operating_confidence=operating_confidence
            ),
            indent=2,
        )
        + "\n"
    )
    by_stratum = Counter(p.stratum for p in proposals)
    typer.echo(
        f"done: {len(proposals)} proposals "
        f"({', '.join(f'{k}={v}' for k, v in sorted(by_stratum.items()))}) "
        f"-> {out} and {report} in {_elapsed(start)}"
    )
    typer.echo(
        "review the GeoJSON, copy accepted entries into the AOI collection, "
        "then detect + stage. This command never edits aois/ or labels/."
    )


@app.command()
def progress(
    labels_dir: Annotated[
        Path, typer.Option("--labels", help="Directory of per-AOI label files.")
    ] = DEFAULT_LABELS,
) -> None:
    """Report labelling progress, and any problems that block training."""
    files = sorted(labels_dir.glob("*.geojson"))
    if not files:
        raise typer.BadParameter(f"no label files in {labels_dir}")

    totals: Counter[str] = Counter()
    problems: list[str] = []
    rows: list[tuple[float, str, str, str, str]] = []
    for path in files:
        collection = labels.read(path)
        counts = labels.summarise(collection)
        total = len(collection["features"])
        done = labels.reviewed(collection)
        totals.update(counts)
        totals["total"] += total
        totals["reviewed"] += done
        bar = f"{done}/{total}"
        detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        # Candidates only: "added" boxes are hand-drawn misses (a recall
        # signal), not the model over-triggering, so they don't belong in a
        # rejection rate. -1 sorts files with nothing reviewed yet last.
        judged = counts.get("confirmed", 0) + counts.get("rejected", 0)
        rate = counts.get("rejected", 0) / judged if judged else -1.0
        reject = f"reject {rate:>4.0%}" if judged else "reject  n/a"
        rows.append((rate, path.stem, bar, reject, detail))
        problems.extend(f"{path.name}: {p}" for p in labels.validate(collection))

    for _, stem, bar, reject, detail in sorted(rows, reverse=True):
        typer.echo(f"{stem:14} {bar:>9}  {reject}  {detail}")

    typer.echo(f"\nreviewed {totals['reviewed']}/{totals['total']}")
    typer.echo("  " + " ".join(f"{k}={totals[k]}" for k in labels.CLASSES if totals[k]))
    judged_total = totals.get("confirmed", 0) + totals.get("rejected", 0)
    if judged_total:
        typer.echo(f"  reject rate {totals.get('rejected', 0) / judged_total:.0%}")
    if problems:
        typer.echo(f"\n{len(problems)} problem(s):")
        for problem in problems[:20]:
            typer.echo(f"  {problem}")
        if len(problems) > 20:
            typer.echo(f"  ... and {len(problems) - 20} more")


@app.command()
def export(
    aoi: Annotated[
        str,
        typer.Option(
            help="YAML AOI collection, GeoJSON file, or bbox 'min_x,min_y,max_x,max_y'."
        ),
    ],
    out: Annotated[Path, typer.Option(help="Dataset directory to write.")] = Path(
        "data/dataset"
    ),
    labels_dir: Annotated[
        Path, typer.Option("--labels", help="Directory of per-AOI label files.")
    ] = DEFAULT_LABELS,
    name: Annotated[
        str | None, typer.Option(help="Select one AOI from a collection.")
    ] = None,
    role: Annotated[
        str | None,
        typer.Option(help=f"Only areas with this role ({', '.join(sorted(ROLES))})."),
    ] = None,
    year: Annotated[
        int, typer.Option(help="Flight year of the orthophoto layer.")
    ] = LATEST_YEAR,
    zoom: Annotated[
        int, typer.Option(help="Tile grid zoom level (0-17).")
    ] = DEFAULT_ZOOM,
    cache: Annotated[Path, typer.Option(help="Tile cache directory.")] = DEFAULT_CACHE,
    workers: Annotated[
        int, typer.Option(help="Concurrent requests.")
    ] = DEFAULT_WORKERS,
) -> None:
    """Turn reviewed labels into a YOLO-OBB dataset, split by whole AOI.

    Refuses to run while any area has unreviewed candidates or schema
    problems: pixel labels are baked to a zoom and tiling, so errors baked
    with them are expensive to find later.
    """
    try:
        areas = load_aois(aoi, name=name)
        layer = layer_for_year(year)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if role is not None:
        if role not in ROLES:
            raise typer.BadParameter(
                f"role {role!r}; expected one of {', '.join(sorted(ROLES))}"
            )
        areas = [a for a in areas if a.role == role]
        if not areas:
            raise typer.BadParameter(f"no areas with role {role!r}")

    # Guards first: nothing is exported while a file could still teach the
    # model something wrong.
    features_by_name: dict[str, list[dict[str, object]]] = {}
    problems: list[str] = []
    for area in areas:
        path = labels_dir / f"{area.name}.geojson"
        if not path.exists():
            if area.role == "positive":
                problems.append(f"{area.name}: no label file (staged yet?)")
            features_by_name[area.name] = []
            continue
        collection = labels.read(path)
        unreviewed = len(collection["features"]) - labels.reviewed(collection)
        if unreviewed:
            problems.append(f"{area.name}: {unreviewed} unreviewed candidate(s)")
        problems.extend(f"{area.name}: {p}" for p in labels.validate(collection))
        problems.extend(
            f"{area.name}: {p}" for p in labels.displaced(collection, area.bounds)
        )
        features_by_name[area.name] = collection["features"]
    if problems:
        typer.echo(f"export blocked by {len(problems)} problem(s):", err=True)
        for problem in problems[:20]:
            typer.echo(f"  {problem}", err=True)
        if len(problems) > 20:
            typer.echo(f"  ... and {len(problems) - 20} more", err=True)
        raise typer.Exit(1)

    typer.echo(f"{layer} z{zoom}, {len(areas)} area(s) -> {out}")
    totals: Counter[str] = Counter()
    skipped: list[str] = []
    start = time.monotonic()
    with TileFetcher(cache, workers=workers) as fetcher:
        for area in areas:
            split_dir = "val" if area.split == "validation" else "train"
            windows, boxes, area_skipped = export_area(
                area,
                features_by_name[area.name],
                layer=layer,
                zoom=zoom,
                cache_root=cache,
                fetcher=fetcher,
                images_dir=out / "images" / split_dir,
                labels_dir=out / "labels" / split_dir,
            )
            totals[f"{split_dir}_windows"] += windows
            totals[f"{split_dir}_boxes"] += boxes
            skipped.extend(area_skipped)
            typer.echo(
                f"  {area.name} ({split_dir}): {windows} windows, {boxes} labels"
            )

    (out / "dataset.yaml").write_text(dataset_yaml(out))
    typer.echo(
        f"done: {totals['train_windows']} train + {totals['val_windows']} val windows, "
        f"{totals['train_boxes']} train + {totals['val_boxes']} val labels "
        f"-> {out / 'dataset.yaml'} in {_elapsed(start)}"
    )
    if skipped:
        typer.echo(
            f"warning: {len(skipped)} window(s) skipped for missing tiles; "
            "re-run fetch to pick them up",
            err=True,
        )


@app.command()
def train(
    data: Annotated[
        Path, typer.Option(help="dataset.yaml from rekka-ai export.")
    ] = Path("data/dataset/dataset.yaml"),
    weights: Annotated[
        str, typer.Option(help="Starting weights to fine-tune from.")
    ] = DEFAULT_WEIGHTS,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = DEFAULT_EPOCHS,
    patience: Annotated[
        int,
        typer.Option(
            help="Stop after this many epochs with no new best fitness. Only "
            "ever ends a run early; --epochs stays the cap."
        ),
    ] = DEFAULT_PATIENCE,
    batch: Annotated[
        int,
        typer.Option(
            help="Images per optimizer step. -1 asks Ultralytics to autobatch, "
            "which measured badly here — see train.DEFAULT_BATCH."
        ),
    ] = DEFAULT_BATCH,
    seed: Annotated[
        int,
        typer.Option(
            help="Training seed. Same seed and data reproduce a run exactly; "
            "vary it to measure the run-to-run spread."
        ),
    ] = DEFAULT_TRAIN_SEED,
    device: Annotated[
        str | None, typer.Option(help="Torch device, e.g. 0 or cpu. Default: auto.")
    ] = None,
    name: Annotated[
        str | None, typer.Option(help="Run name under runs/train/.")
    ] = None,
) -> None:
    """Fine-tune the detector on an exported dataset, tracked in MLflow.

    Needs the 'train' extra (uv sync --extra train). Metrics per epoch land in
    the MLflow run: uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
    """
    if not data.exists():
        raise typer.BadParameter(f"no dataset at {data}; run rekka-ai export first")
    start = time.monotonic()
    try:
        best = run_train(
            data,
            weights=weights,
            epochs=epochs,
            patience=patience,
            batch=batch,
            seed=seed,
            device=device,
            name=name,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"done: best weights -> {best} in {_elapsed(start)}")


@app.command("eval")
def evaluate_cmd(
    weights: Annotated[Path, typer.Option(help="Trained weights to evaluate.")],
    data: Annotated[
        Path, typer.Option(help="dataset.yaml from rekka-ai export.")
    ] = Path("data/dataset/dataset.yaml"),
    aoi: Annotated[
        str | None,
        typer.Option(help="AOI collection for per-area counts and the negative check."),
    ] = None,
    labels_dir: Annotated[
        Path, typer.Option("--labels", help="Directory of per-AOI label files.")
    ] = DEFAULT_LABELS,
    min_recall: Annotated[
        float, typer.Option(help="Truck recall gate at the operating point.")
    ] = GATE_RECALL,
    min_confidence: Annotated[
        float,
        typer.Option(
            help="Floor below which a PR-curve point is never picked as the "
            "operating confidence, even as a fallback: near conf=0 keeps every "
            "raw proposal, which is not a usable operating point."
        ),
    ] = MIN_OPERATING_CONFIDENCE,
    year: Annotated[
        int, typer.Option(help="Flight year of the orthophoto layer.")
    ] = LATEST_YEAR,
    zoom: Annotated[
        int, typer.Option(help="Tile grid zoom level (0-17).")
    ] = DEFAULT_ZOOM,
    cache: Annotated[Path, typer.Option(help="Tile cache directory.")] = DEFAULT_CACHE,
    workers: Annotated[
        int, typer.Option(help="Concurrent requests.")
    ] = DEFAULT_WORKERS,
    device: Annotated[
        str | None, typer.Option(help="Torch device, e.g. 0 or cpu. Default: auto.")
    ] = None,
    name: Annotated[
        str | None, typer.Option(help="MLflow run name. Default: eval-<training run>.")
    ] = None,
) -> None:
    """Evaluate trained weights against the validation split and the gates.

    Two layers: the standard per-class metrics, and the operational checks —
    a PR-chosen operating confidence, per-area truck counts, and the
    hard-negative regression check. Logged to MLflow as its own run.
    Needs the 'train' extra (uv sync --extra train).
    """
    if not weights.exists():
        raise typer.BadParameter(f"no weights at {weights}")
    if not data.exists():
        raise typer.BadParameter(f"no dataset at {data}; run rekka-ai export first")

    run_name = name or f"eval-{weights.parent.parent.name}"
    start = time.monotonic()
    try:
        per_class, (px, p_curve, r_curve), names = validation_metrics(
            weights, data, device=device, name=run_name
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"validation split of {data}:")
    for klass in names:
        m = per_class[klass]
        typer.echo(
            f"  {klass:6} P {m.precision:.3f}  R {m.recall:.3f}  "
            f"mAP50 {m.map50:.3f}  mAP50-95 {m.map50_95:.3f}"
        )

    if "truck" not in names:
        typer.echo(
            f"model has classes {names}, with no 'truck' to gate on; "
            "eval measures the truck class, so these weights cannot be judged",
            err=True,
        )
        raise typer.Exit(1)
    truck = names.index("truck")
    conf, op_p, op_r = pick_operating_point(
        px,
        p_curve[truck],
        r_curve[truck],
        min_recall=min_recall,
        min_confidence=min_confidence,
    )
    typer.echo(
        f"\noperating point (truck): conf {conf:.3f} -> P {op_p:.3f}, R {op_r:.3f}"
    )

    gates: list[tuple[str, bool, str]] = [
        ("truck recall", op_r >= min_recall, f"{op_r:.3f} >= {min_recall}"),
        ("truck precision", op_p >= GATE_PRECISION, f"{op_p:.3f} >= {GATE_PRECISION}"),
    ]
    #: Measured but never decisive — see NEGATIVE_DETECTIONS_WATCH.
    observations: list[tuple[str, str]] = []
    metrics: dict[str, float] = {
        f"{k}_{f}": v
        for k, m in per_class.items()
        for f, v in (
            ("precision", m.precision),
            ("recall", m.recall),
            ("map50", m.map50),
            ("map50_95", m.map50_95),
        )
    }
    metrics.update(
        {
            "operating_conf": conf,
            "operating_truck_precision": op_p,
            "operating_truck_recall": op_r,
        }
    )

    if aoi is not None:
        try:
            areas = load_aois(aoi)
            layer = layer_for_year(year)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        # One detector for every area: constructing it loads the checkpoint
        # onto the GPU, and this visits each validation and negative area.
        detector = load_detector(weights, conf)
        with TileFetcher(cache, workers=workers) as fetcher:
            # Per-area truck counts on the validation areas: the product's
            # question is "how many vehicles per site", so count error is a
            # gate, not box geometry.
            for area in [
                a for a in areas if a.split == "validation" and a.role == "positive"
            ]:
                path = labels_dir / f"{area.name}.geojson"
                truth = (
                    ground_truth_counts(labels.read(path))["truck"]
                    if path.exists()
                    else 0
                )
                found = count_detections(
                    area,
                    detector=detector,
                    layer=layer,
                    zoom=zoom,
                    cache_root=cache,
                    fetcher=fetcher,
                )
                if truth:
                    metrics[f"count_error_{area.name}"] = (
                        found["truck"] - truth
                    ) / truth
                ok, detail = count_gate(found["truck"], truth)
                if ok is None:
                    # Too few trucks to gate on, but the number still belongs
                    # in the report — a bus area's truck count is worth a
                    # glance even when it cannot carry a verdict.
                    typer.echo(f"  count {area.name}: {detail}")
                else:
                    gates.append((f"count {area.name}", ok, detail))
            # The regression check: training on truck shapes must not start
            # pulling containers in. Only detections the area's own labels
            # cannot account for count — a negative area is negative about
            # targets, not empty, and marking the model down for finding the
            # cars that really are there measured nothing (docs/rounds.md).
            negatives = 0
            for area in areas:
                if not area.is_negative:
                    continue
                found = sweep_area(
                    area,
                    detector=detector,
                    layer=layer,
                    zoom=zoom,
                    cache_root=cache,
                    fetcher=fetcher,
                )
                path = labels_dir / f"{area.name}.geojson"
                collection = (
                    labels.read(path)
                    if path.exists()
                    else {"type": "FeatureCollection", "features": []}
                )
                loose = unexplained(found, collection)
                negatives += len(loose)
                typer.echo(
                    f"  negative {area.name}: {len(loose)} unexplained "
                    f"of {len(found)} detection(s)"
                )
        # Reported, never gated: the same dataset at three seeds gave 1, 7 and
        # 3 against what used to be a threshold of 2 (docs/rounds.md).
        observations.append(
            (
                "negative areas",
                f"{negatives} unexplained detection(s)"
                + (
                    f" — above the watch level of {NEGATIVE_DETECTIONS_WATCH}, "
                    "worth a look"
                    if negatives > NEGATIVE_DETECTIONS_WATCH
                    else ""
                ),
            )
        )
        metrics["negative_detections"] = negatives

    typer.echo("\ngates:")
    for gate_name, passed, detail in gates:
        typer.echo(f"  {'PASS' if passed else 'FAIL'}  {gate_name}: {detail}")
    if observations:
        typer.echo("\nreported, not gated:")
        for label, detail in observations:
            typer.echo(f"  {label}: {detail}")

    log_eval_run(run_name, {"weights": str(weights), "data": str(data)}, metrics)
    typer.echo(f"\nlogged to MLflow as {run_name!r} in {_elapsed(start)}")


def _detect_chunked(
    *,
    aoi: str,
    out: Path,
    weights: Path,
    confidence: float,
    crs: str,
    layer: str,
    year: int,
    zoom: int,
    min_length: float,
    cache: Path,
    workers: int,
    root: Path,
    cell_size: float,
    merge_only: bool,
    retry_failed: bool,
) -> None:
    """Sweep a region cell by cell, resuming from a manifest under ``root``.

    Split out of ``detect`` rather than folded into it: the plain path holds
    everything in memory and writes once, which is right for a 300 m AOI and
    fatal for a city. Nothing here changes what a detection *is* -- the same
    sweep, the same filters, and the same global NMS at the end.
    """
    try:
        region = load_region(aoi, crs=crs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    cells = grid_cells(region, cell_size)
    key = run_key(
        weights=weights,
        confidence=confidence,
        zoom=zoom,
        year=year,
        min_length=min_length,
        cell_size=cell_size,
        region=aoi,
    )
    header, recorded = read_manifest(root)
    if header is None:
        append(root, {"type": "run", **key})
    else:
        stored = {k: v for k, v in header.items() if k != "type"}
        if stored != key:
            differing = sorted(k for k in key if stored.get(k) != key[k])
            raise typer.BadParameter(
                f"{manifest_path(root)} was written with different settings "
                f"({', '.join(differing)}). Detections from two models in one "
                "file would be wrong in a way nothing downstream can see — "
                "use a fresh --checkpoint-dir."
            )

    done = {n for n, r in recorded.items() if r.get("status") == DONE}
    failed = {n for n, r in recorded.items() if r.get("status") == FAILED}
    skip = done if retry_failed else done | failed
    todo = [c for c in cells if c.name not in skip]

    typer.echo(
        f"{layer} z{zoom}, weights {weights.name}, conf {confidence}, "
        f"cells {cell_size:g} m -> {root}"
    )
    typer.echo(
        f"  {len(cells)} cells cover the region; {len(done)} done, "
        f"{len(failed)} failed, {len(todo)} to sweep"
    )

    start = time.monotonic()
    if not merge_only and todo:
        detector = YoloObb(str(weights), confidence=confidence)
        with TileFetcher(cache, workers=workers) as fetcher:
            for index, cell in enumerate(todo, start=1):
                # Written before the work, so a process killed mid-cell leaves
                # the evidence and the resume redoes it instead of trusting a
                # half-written file.
                append(root, {"cell": cell.name, "status": RUNNING})
                cell_started = time.monotonic()
                # sweep reports one message per window dropped for missing
                # tiles; collecting them counts the holes in this cell.
                skipped: list[str] = []
                try:
                    candidates = sweep(
                        Aoi(name=cell.name, bounds=cell.bounds),
                        detector=detector,
                        layer=layer,
                        zoom=zoom,
                        cache_root=cache,
                        fetcher=fetcher,
                        min_length_m=min_length,
                        on_skip=skipped.append,
                    )
                    # Clip to the cell *and* the region: the cell keeps the
                    # units disjoint, the region keeps the sea out.
                    kept = within_region(
                        candidates, cell.polygon().intersection(region)
                    )
                    path = cell_path(root, cell.name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Atomic, like enrich.py's layer cache: a kill mid-write
                    # must not leave a truncated cell the resume then trusts.
                    # The suffix stays .geojson because `write` picks the
                    # format from it.
                    partial = path.with_name(f"{path.stem}.part.geojson")
                    write(kept, partial, source_layer=layer, zoom=zoom)
                    partial.replace(path)
                except Exception as exc:  # noqa: BLE001 - one cell must not end the run
                    append(
                        root,
                        {"cell": cell.name, "status": FAILED, "error": repr(exc)},
                    )
                    typer.echo(f"  [{index}/{len(todo)}] {cell.name} FAILED: {exc}")
                    continue

                append(
                    root,
                    {
                        "cell": cell.name,
                        "status": DONE,
                        "detections": len(kept),
                        "windows_skipped": len(skipped),
                        "seconds": round(time.monotonic() - cell_started, 1),
                    },
                )
                typer.echo(
                    f"  [{index}/{len(todo)}] {cell.name}: {len(kept)} detections "
                    f"({_elapsed(start)} elapsed)"
                )

    _, final = read_manifest(root)
    finished = [c.name for c in cells if final.get(c.name, {}).get("status") == DONE]
    found = load_cells(root, finished)

    # The cell test already makes units disjoint, but window overlap at a seam
    # can produce a genuine duplicate pair whose centres land either side of
    # the line. Run the same global NMS the unchunked path runs, and say how
    # much it caught — if this is ever large, the cell size is hiding vehicles.
    deduped = merge(found)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        write(deduped, out, source_layer=layer, zoom=zoom)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    still_pending = len(cells) - len(finished)
    typer.echo(
        f"done: {len(deduped)} detections from {len(finished)}/{len(cells)} cells "
        f"-> {out} in {_elapsed(start)}"
    )
    if len(found) != len(deduped):
        typer.echo(f"  seam duplicates removed at merge: {len(found) - len(deduped)}")
    if still_pending:
        typer.echo(
            f"warning: {still_pending} cell(s) unfinished; re-run the same "
            "command to continue",
            err=True,
        )


@app.command()
def detect(
    aoi: Annotated[
        str,
        typer.Option(
            help="Polygon region file (GeoJSON, FlatGeobuf, GeoPackage, "
            "shapefile — e.g. postcode areas), YAML AOI collection, or bbox "
            "'min_x,min_y,max_x,max_y'."
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output file for detections: .geojson, .fgb or .gpkg."),
    ],
    weights: Annotated[Path, typer.Option(help="Trained weights to detect with.")],
    confidence: Annotated[
        float,
        typer.Option(
            help="Minimum detection confidence. eval picks this from the PR "
            "curve; the default is a census threshold, so pass a lower one "
            "when staging candidates for review."
        ),
    ] = 0.77,  # eval-round4's operating point (0.772, 2026-08-16); re-check each round
    name: Annotated[
        str | None, typer.Option(help="Select one AOI from a collection.")
    ] = None,
    role: Annotated[
        str | None,
        typer.Option(help=f"Only areas with this role ({', '.join(sorted(ROLES))})."),
    ] = None,
    crs: Annotated[str, typer.Option(help="CRS of a GeoJSON region or bbox.")] = WGS84,
    year: Annotated[
        int, typer.Option(help="Flight year of the orthophoto layer.")
    ] = LATEST_YEAR,
    zoom: Annotated[
        int, typer.Option(help="Tile grid zoom level (0-17).")
    ] = DEFAULT_ZOOM,
    min_length: Annotated[
        float, typer.Option(help="Drop detections shorter than this, in metres.")
    ] = MIN_LENGTH_M,
    cache: Annotated[Path, typer.Option(help="Tile cache directory.")] = DEFAULT_CACHE,
    workers: Annotated[
        int, typer.Option(help="Concurrent requests.")
    ] = DEFAULT_WORKERS,
    checkpoint_dir: Annotated[
        Path | None,
        typer.Option(
            help="Sweep a polygon region cell by cell, recording progress here "
            "so a re-run resumes instead of starting over. For city-scale runs."
        ),
    ] = None,
    cell_size: Annotated[
        float, typer.Option(help="Cell edge in metres, with --checkpoint-dir.")
    ] = DEFAULT_CELL_SIZE_M,
    merge_only: Annotated[
        bool,
        typer.Option(
            help="Skip sweeping; just merge the cells already written to "
            "--checkpoint-dir into --out."
        ),
    ] = False,
    retry_failed: Annotated[
        bool, typer.Option(help="Also redo cells recorded as failed.")
    ] = False,
) -> None:
    """Detect vehicles with trained weights, over polygons or AOI areas.

    A polygon file (postcode areas, districts, a hand-drawn study area)
    becomes the detection region: windows sweep its bounds, and detections
    are filtered to those whose centre falls inside a polygon. GeoJSON is
    read directly; FlatGeobuf/GeoPackage/shapefile need geopandas (the
    'detect' extra) and carry their own CRS. Needs the 'detect' extra
    (uv sync --extra detect).
    """
    if not weights.exists() and not merge_only:
        raise typer.BadParameter(f"no weights at {weights}")
    try:
        layer = layer_for_year(year)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if checkpoint_dir is not None:
        if Path(aoi).suffix.lower() in YAML_SUFFIXES or not Path(aoi).exists():
            raise typer.BadParameter(
                "--checkpoint-dir needs a polygon region file; an AOI collection "
                "is already a list of small areas and needs no chunking"
            )
        _detect_chunked(
            aoi=aoi,
            out=out,
            weights=weights,
            confidence=confidence,
            crs=crs,
            layer=layer,
            year=year,
            zoom=zoom,
            min_length=min_length,
            cache=cache,
            workers=workers,
            root=checkpoint_dir,
            cell_size=cell_size,
            merge_only=merge_only,
            retry_failed=retry_failed,
        )
        return

    # Trained weights: keep every class the model was trained on.
    detector = YoloObb(str(weights), confidence=confidence)
    typer.echo(
        f"{layer} z{zoom}, weights {weights.name}, "
        f"conf {confidence}, min-length {min_length} m -> {out}"
    )
    found: list[Detection] = []
    start = time.monotonic()
    with TileFetcher(cache, workers=workers) as fetcher:
        # A file that is not a YAML collection is a polygon region: GeoJSON,
        # FlatGeobuf, GeoPackage, shapefile — load_region knows the formats.
        if Path(aoi).exists() and Path(aoi).suffix.lower() not in YAML_SUFFIXES:
            try:
                region = load_region(aoi, crs=crs)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            area = Aoi(name=name or Path(aoi).stem, bounds=region_bounds(region))
            found = within_region(
                sweep(
                    area,
                    detector=detector,
                    layer=layer,
                    zoom=zoom,
                    cache_root=cache,
                    fetcher=fetcher,
                    min_length_m=min_length,
                    on_skip=typer.echo,
                    on_progress=_progress,
                ),
                region,
            )
            typer.echo(f"  {area.name}: {len(found)} detections inside the region")
        else:
            try:
                areas = load_aois(aoi, crs=crs, name=name)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            if role is not None:
                if role not in ROLES:
                    raise typer.BadParameter(
                        f"role {role!r}; expected one of {', '.join(sorted(ROLES))}"
                    )
                areas = [a for a in areas if a.role == role]
                if not areas:
                    raise typer.BadParameter(f"no areas with role {role!r}")
            for area in areas:
                candidates = sweep(
                    area,
                    detector=detector,
                    layer=layer,
                    zoom=zoom,
                    cache_root=cache,
                    fetcher=fetcher,
                    min_length_m=min_length,
                    on_skip=typer.echo,
                    on_progress=_progress,
                )
                typer.echo(f"  {area.name} ({area.role}): {len(candidates)} detections")
                found.extend(candidates)

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        write(found, out, source_layer=layer, zoom=zoom)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"done: {len(found)} detections -> {out} in {_elapsed(start)}")


@app.command("aois")
def list_aois(
    aoi: Annotated[str, typer.Option(help="YAML AOI collection to inspect.")],
    zoom: Annotated[
        int, typer.Option(help="Tile grid zoom level (0-17).")
    ] = DEFAULT_ZOOM,
) -> None:
    """List a collection, and report overlaps between its areas."""
    try:
        areas: list[Aoi] = load_aois(aoi)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    for area in areas:
        typer.echo(
            f"{area.name:14} {area.role:13} {area.split:10} "
            f"{count_tiles(area.bounds, zoom):5} tiles at z{zoom}"
        )

    by_split = Counter(a.split for a in areas)
    by_role = Counter(a.role for a in areas)
    typer.echo(f"\n{len(areas)} areas: {dict(by_role)}, {dict(by_split)}")

    # Validation that cannot see a hard negative cannot measure the false
    # positives the hard negative exists to suppress.
    validation_roles = {a.role for a in areas if a.split == "validation"}
    missing = {r for r in by_role if r != "positive"} - validation_roles
    if missing:
        typer.echo(f"warning: no {', '.join(sorted(missing))} area in validation")

    shared = overlaps(areas)
    if not shared:
        return
    typer.echo("\noverlapping areas (same ground annotated twice):")
    for first, second, area_m2 in sorted(shared, key=lambda x: -x[2]):
        conflict = ""
        if first.role != second.role:
            conflict = f"  <- role conflict: {first.role} vs {second.role}"
        elif first.split != second.split:
            conflict = f"  <- split conflict: {first.split} vs {second.split}"
        typer.echo(f"  {first.name} x {second.name}: {area_m2:,.0f} m2{conflict}")
