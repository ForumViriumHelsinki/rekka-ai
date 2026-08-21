# AGENTS.md

Guidance for AI coding agents working in this repository. The reader is assumed
to know nothing about the project. `README.md` and `docs/DESIGN.md` are the
primary docs and are kept current; when they and this file disagree, trust
them.

**The rationale is in `docs/DESIGN.md`, and only there.** `README.md` is
commands and consequences — it leads with inference (running the shipped
model) and keeps the training loop below that — and it deliberately does not
repeat the reasoning. So: never answer a *why* from the README, read the
matching DESIGN section before changing anything the README merely asserts,
and do not migrate rationale back into it. `docs/model-card.md` describes the
weights currently shipped and where they fail; `docs/LABELLING.md` is the
labeller's card. `docs/rounds.md` is the dated log of correction rounds —
history, not rules.

## Project overview

**rekka-ai** detects trucks (and buses, vans) in Helsinki aerial orthophotos and
emits their locations as georeferenced vector data. It is a **model-training
loop**, not a straight-through pipeline: a pretrained oriented-bounding-box
(OBB) detector proposes candidates, a human corrects them in a web tool, the
corrected labels fine-tune the model, and the improved model proposes the next
batch on new ground.

```
fetch → bootstrap → stage → review (web/) → export → train → eval → detect → stage → …
                                         ↘ mine (propose next AOIs) ↗
```

Four rounds in, that loop has produced shipping weights, so there is a second
chain that only runs the model — the one `README.md` leads with:

```
prepare_production_aoi.py → detect (chunked, resumable) → enrich → package/analyse
```

Two parts, two licenses:

- **Python backend** (everything outside `web/`) — AGPL-3.0, because it builds
  on Ultralytics YOLO (AGPL-3.0).
- **Labelling web app** (`web/`) — MIT; a separate work that communicates with
  the backend only through data files.

## Tech stack and layout

### Python backend

- Python **≥ 3.14**, managed with [uv](https://docs.astral.sh/uv/) (`uv sync`,
  `uv run …`); hatchling build backend, `src/` layout, package `rekka_ai`.
- CLI entry point: `rekka-ai = rekka_ai.cli:app` (typer).
- Runtime deps: httpx, pillow, pyproj, pyyaml, shapely, typer.
- Optional extras, **imported lazily on purpose**: `detect` (ultralytics,
  geopandas — pulls in torch) and `train` (ultralytics, mlflow). The base
  install — and CI — has none of them, and that is intentional: `fetch` and
  `aois` must stay usable without a multi-gigabyte install. `pyproject.toml`
  has `[[tool.ty.overrides]]` ignoring `unresolved-import` in exactly the
  files that do lazy imports (`detect/sweep.py`, `detect/detections.py`,
  `enrich.py`, `evaluate.py`, `imagery/aoi.py`, `segment.py`, `track.py`,
  `train.py`, and the three `scripts/*.py`). If you add lazy
  imports of optional deps, extend that list — do not make the imports eager.
- Dev tools (uv dependency group `dev`): pytest, pytest-cov, ruff, ty.
- **MLflow.** Local tracking store is `runs/mlflow.db` (`track.py` is the
  single place that configures it; `MLFLOW_TRACKING_URI` overrides). Agents
  should use the **mlflow MCP server** to inspect it — list/search
  experiments and runs, compare runs, read metrics/params — rather than
  querying the sqlite file directly or shelling out to the `mlflow` CLI.

```
src/rekka_ai/
  cli.py               # thin typer layer; every command lives here
  geo.py               # CRS transforms — the ONE place they happen
  labels.py            # label schema, measurements (always recomputed), validation
  evaluate.py          # ship gates: recall ≥ 0.90, precision ≥ 0.85, counts; negatives reported
  export.py            # geographic labels -> YOLO-OBB pixel dataset (the only such place)
  train.py             # Ultralytics fine-tuning wrapper
  track.py             # MLflow configuration — single place; runs/mlflow.db, MLFLOW_TRACKING_URI overrides
  enrich.py            # WFS enrichment: district/postal/street(+type)/context/street_part on detections
  segment.py           # SAM outline refinement — measured worse than the OBB vs labels, kept as a library only
  osm.py               # Overpass industrial geometry, cached under data/osm/
  mine.py              # grid / exclude / stratum selection for the next AOIs
  imagery/
    tiles.py           # WMTS grid arithmetic, pure functions
    wmts.py            # fetch + on-disk tile cache (data/cache/<layer>/<zoom>/<col>/<row>.jpg)
    windows.py         # tiles -> ~1024 px model windows, pixel <-> ground
    aoi.py             # AOI collections (YAML/GeoJSON/bbox), region polygons, overlap detection
    layers.py          # flight year -> WMTS layer name (naming is irregular; explicit map)
  detect/
    sweep.py           # detector protocol, YOLO-OBB adapter, area sweep
    detections.py      # oriented-box geometry, seam merging (global NMS via STRtree), GeoJSON output

scripts/                 # production run, outside the CLI: one-off, not loop steps
  prepare_production_aoi.py  # WFS districts + roads -> the 208 km2 detection region
  package_detections.py      # enriched per-year sweeps -> one delivery GeoPackage
  analyse_detections.py      # truck density grids + point layers, per flight year
```

### Labelling web app (`web/`)

SvelteKit + OpenLayers + Tailwind 4 + proj4, TypeScript, bundled with Vite 8,
run/tested with [bun](https://bun.sh/). adapter-node build, but in practice it
is a **local tool**: no database, no auth, localhost only. Two API routes
(`GET`/`PUT /api/labels/[aoi]`, `GET /api/aois`) read and write `labels/` and
`aois/` in the repository directly; writes go to a temp sibling and rename so
a crash mid-save cannot truncate a label file.

`web/src/lib/` mirrors backend constants and geometry — **keep these in sync
when changing either side**: `grid.ts` ↔ `imagery/tiles.py` (tile grid),
`obb.ts` ↔ `labels.py` (oriented-box geometry), `projection.ts` (proj4 defs).
`tests/test_web_constants.py` asserts the Python/web constants do not drift.

### Data and configuration

- `aois/helsinki.yaml` — the AOI collection (40 areas as of 2026-08, 34
  train / 6 validation — the count grows each round; `rekka-ai aois` reports
  the current shape), in **EPSG:3067**, with `role` (`positive` |
  `hard-negative` | `sparse`) and `split` (`train` | `validation`) per area.
  `crs` is required and always wins over `--crs`. The per-area `notes` *are*
  the annotation guide — the web tool displays them beside the map.
- `labels/<aoi>.geojson` — one file per area, **EPSG:3879** declared via a
  `crs` member. Feature properties: `class` (`""`/`truck`/`bus`/`van`/`car`)
  and `status` (`candidate`/`confirmed`/`rejected`/`added`).
- **Gitignored and reproducible:** `data/` (tile cache, candidates, dataset,
  OSM Overpass responses, mining proposals), `runs/` (training runs, MLflow
  store), `models/`, `*.pt`. The bootstrap weights (`yolo11x-obb.pt` etc.) sit
  in the repo root but are ignored.
- **Version-controlled and irreplaceable:** `labels/` — human hours. See
  Safety below.

### Round-N AOI mining (`mine`)

After a trained round exists, do not hand-pick the next yards. `rekka-ai mine`
proposes new 300 m training AOIs from cached OpenStreetMap landuse geometry
and the current weights (near-threshold, truck/van-band, dense, and quiet
strata). `--profile` selects the OSM landuse profile (`industrial` default,
plus `commercial`, `construction`, `camping` — `PROFILES` in `osm.py`). It
writes a proposal YAML + GeoJSON report under `data/mining/` and **never**
edits `aois/` or `labels/`. Helsinki
only: the 2025 5 cm WMTS covers Helsinki (`ref=091`); Espoo/Vantaa wait on an
HSY imagery source. OSM responses live in `data/osm/` (gitignored);
`--refresh-osm` re-fetches. Attribution: © OpenStreetMap contributors.

## Build and test commands

Python (all via `uv run`):

```sh
uv sync                      # base install (CI parity: uv sync --locked)
uv sync --extra detect       # + torch/ultralytics, needed for bootstrap/detect
uv sync --extra train        # + mlflow, needed for train
uv run rekka-ai --help       # CLI: aois fetch bootstrap stage progress export train eval detect enrich mine
uv run ruff check            # lint (extend-select = I, UP, B)
uv run ruff format           # format (CI runs ruff format --check)
uv run ty check              # type check
uv run pytest                # offline, no GPU
```

Web (in `web/`, with bun):

```sh
bun install                  # CI: bun install --frozen-lockfile
bun run dev                  # dev server on http://localhost:3000
bun run test                 # vitest run — `bun run`, NOT bare `bun test`
                             # (bare bun test is bun's own runner, not vitest)
bun run check                # svelte-check
bun run lint                 # prettier --check
```

CI (`.github/workflows/ci.yml`, on PRs and pushes to main) runs **both**:
the Python four (ruff check, ruff format --check, ty check, pytest) and the
web three (bun run lint, check, test). Run all of them before pushing.

## Code style and conventions

- Follow the surrounding style. The codebase comments heavily, and comments
  carry **rationale and measured evidence**, not narration — preserve that
  register when editing; update comments/docstrings that describe behaviour
  you change.
- Python ≥ 3.14 syntax throughout (`X | None`, builtin generics). Ruff owns
  lint and format; `ty` owns types. All four Python checks must stay green.
- The `web/` tool has its own design constraints (docs/DESIGN.md §10): almost
  nothing animates on the hot path; one desaturated amber drives the chrome;
  class colours are map symbology only; Geist / Geist Mono typography.
  Prettier (with the Svelte plugin) owns formatting there.
- **Docs have three genres, kept apart.** `README.md` is operational: the
  command, what it produces, and the caveat that costs you if you ignore it —
  no rationale, no history. `docs/DESIGN.md` is evergreen: it
  describes the present, with the reasoning and the measured evidence —
  update it in the same change as the behaviour it documents. `docs/rounds.md` is the dated log: when a labelling, training,
  or eval round completes (or an area is retired, or a decision is made on
  record), *append* a dated entry there — do not add "Update after round N"
  blocks to DESIGN.md, and do not rewrite old log entries when the present
  moves on. Numbers that justify a rule stay in DESIGN.md with their date;
  the log gets the decision and its cost. **Log entries are short** — a
  decision, the numbers behind it, what it cost. Not a narrative of how it
  was reached, and never the labelling guidance, which belongs in the AOI
  `notes`.

## Testing strategy

- **Python:** pytest, `testpaths = ["tests"]`, plain `test_<module>.py` files
  mirroring the source modules. Tests are offline and deterministic — no
  network, no GPU, no ultralytics/mlflow imports (matching the base install).
  Known geometric traps (axis order, tile-origin corner, reprojection
  rotation, non-finite bounds) are encoded as tests; add the trap, then the
  code, when touching geometry.
- **Web:** vitest, `jsdom` environment (OpenLayers reaches for the DOM on
  import), tests colocated as `src/**/*.test.ts` (`obb.test.ts`,
  `labelStore.test.ts`, `undo.test.ts`, `mapStyles.test.ts`). They cover the
  save path, the geometry the labels are built from, and the symbology —
  regression coverage for the parts where the project can actually lose work.
- **Cross-stack:** `tests/test_web_constants.py` keeps mirrored
  Python/TypeScript constants from drifting.

## Domain rules that cost real correctness if violated

- **Three CRSs, kept separate.** Config/input: EPSG:3067. Tile arithmetic and
  stored geometry: EPSG:3879. Hand-typed bboxes: WGS84. EPSG:3067 declares
  (east, north) but EPSG:3879 declares (north, east) — every transform uses
  `always_xy=True`, and transforms happen only in `geo.py`. A box in one CRS
  is not axis-aligned in the other (~1.74° rotation), so project all four
  corners, never just min/max. pyproj returns infinities rather than raising
  outside a CRS's domain.
- **Labels live in geographic coordinates (EPSG:3879 GeoJSON polygons), not
  pixels.** Conversion to YOLO-OBB pixel labels happens once, at `export`
  time, in `export.py` — never store pixel labels.
- **Everything runs at z16** (12.5 cm/px; a semi-trailer is ~132 px). z15 was
  measured to miss two thirds of trucks; z17 adds little for 4× the tiles.
- **Train/val split by whole AOI** from the collection's `split` field —
  never random: adjacent windows are near-duplicates and a random split
  makes the metrics fiction.
- **Measurements (`length_m`, `width_m`, `heading_deg`) are always recomputed
  from geometry**, never trusted from a file.
- **Rejects are kept, not deleted** — rejected candidates are the project's
  hard negatives. `hard-negative`/`sparse` areas export as pure background;
  their label files (reviewed, like any other area) record the non-target
  vehicles the ground really holds, which is what the negative-area check
  forgives detections against. A missing file for these roles is tolerated.
- **The pipeline is deterministic**: same collection + same weights + same
  zoom → same candidates. The tile cache never invalidates (past flight years
  never change).

## Safety considerations

- **`labels/` is the only irreplaceable artifact.** Everything else
  regenerates; labels are human hours. `rekka-ai stage` refuses to overwrite
  an existing label file without `--force` — that refusal is the single most
  important safety property in the pipeline. Do not delete, bulk-rewrite, or
  "clean up" label files, and preserve its round-trip property: an untouched
  box must round-trip as identical bytes (both writers round to
  `geo.COORD_DECIMALS`).
- **`export` refuses to run dirty** (unreviewed candidates, schema problems,
  or a box whose centre drifted outside its area). Do not weaken these
  guards — pixel labels are baked to a zoom and tiling, so errors baked with
  them are expensive to find later.
- **Keep optional-extra imports lazy.** Eagerly importing ultralytics,
  geopandas, or mlflow from base code breaks the base install and CI.
- **Fetch etiquette.** The imagery is Helsinki City Survey Services WMTS
  (`https://kartta.hel.fi/ws/geoserver/avoindata/gwc/service/wmts`). Access
  constraints are `NONE`, but the fetcher caps concurrency, retries only
  transient failures, and sends an identifying `User-Agent` — preserve that
  behaviour. A tile that fails after retries is collected and reported at the
  end, not raised mid-sweep. Imagery attribution: © Helsingin kaupunki,
  Kaupunkimittauspalvelut.
- **Licensing.** Backend code must stay AGPL-3.0-compatible (Ultralytics);
  `web/` is MIT and communicates only through data files — do not link them
  more tightly.
