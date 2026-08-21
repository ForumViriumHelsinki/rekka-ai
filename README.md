# rekka-ai

Truck (and bus, van, car) detection from Helsinki aerial orthophotos, as
georeferenced oriented bounding boxes.

The model is finished. Round-4 z16 weights — published as `model/v4.0.0` —
pass the ship gates at truck recall 0.911, precision 0.945, and have swept the
whole city on three flight years: 208 km², 22 minutes, 125,882 detections on
the 2025 one. This page runs them.
[docs/model-card.md](docs/model-card.md) is what they do and where they fail;
[docs/DESIGN.md](docs/DESIGN.md) is why anything is the way it is.

```
prepare_production_aoi.py → detect → enrich → package_detections.py
```

## Install

```sh
git clone git@github.com:ForumViriumHelsinki/rekka-ai.git
cd rekka-ai
uv sync --extra detect          # torch + geopandas; needed for everything below
uv run rekka-ai --help

gh release download --pattern 'yolo11x-obb-fvh-*'   # latest weights, 118 MB
sha256sum -c SHA256SUMS
```

Add a tag (`gh release download model/v4.0.0 …`) to pin a specific model
rather than taking the newest.

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14+ (uv installs it), plus
weights from the [releases][rel] — 118 MB, never in git. Training your own
instead is docs/model-card.md §7; it leaves them at
`runs/train/round<N>/weights/best.pt`, which is what the training-loop
examples below use. **The zoom in the filename matters** — z16 and z17
weights are not interchangeable, and the mismatch degrades quietly rather
than failing.

A GPU is optional: the city sweep was 28.5 ms per window on an RTX 5070 Ti,
and CPU works.

## Run the detector

**1. Build the region.** Not the city's bounding box — everywhere in Helsinki
a truck could be, from two open WFS layers (docs/DESIGN.md §6):

```sh
uv run --extra detect python scripts/prepare_production_aoi.py
# -> data/production/helsinki-region.gpkg   208 km², EPSG:3879
```

**2. Detect.** Any polygon file, or a bbox, is a region; detections are kept
where the box *centre* falls inside it.

```sh
uv run rekka-ai detect --aoi my-study-area.gpkg \
    --weights yolo11x-obb-fvh-z16-v4.0.0.pt \
    --year 2025 --zoom 16 --out data/detections/study-area.fgb
```

City-scale runs add `--checkpoint-dir`, which sweeps cell by cell and resumes
where it stopped — one failed cell is recorded and stepped over rather than
ending the run:

```sh
uv run rekka-ai detect --aoi data/production/helsinki-region.gpkg \
    --weights yolo11x-obb-fvh-z16-v4.0.0.pt \
    --year 2025 --zoom 16 --confidence 0.25 \
    --checkpoint-dir data/detections/helsinki-2025 --cell-size 1000 \
    --out data/detections/helsinki_2025_z16.fgb
```

`--merge-only` rebuilds the output from cells already on disk;
`--retry-failed` redoes the failures. **Sweep low, threshold on read:**
`confidence` is stored per feature, so a higher cut is a filter afterwards
while a lower one costs another sweep.

Flight years are 2014–2021 and 2023–2025; 2023–2025 are the 5 cm ones the
model knows. Tiles cache under `data/cache/` and never expire — past imagery
does not change — so a second sweep of a year is 22 minutes rather than 47.

**3. Enrich.** Joins in district, postal code, street and location context
from the city's WFS. Detection-derived numbers are untouched.

```sh
uv run rekka-ai enrich --detections data/detections/helsinki_2025_z16.fgb \
    --out data/detections/helsinki_2025_z16_enriched.fgb
```

**4. Ship it.** One GeoPackage of named per-year layers, plus truck density
grids beside it:

```sh
uv run --extra detect python scripts/package_detections.py --year 2025
uv run --extra detect python scripts/analyse_detections.py --year 2025
# -> data/detections/rekka-ai-detections.gpkg, rekka-ai-truck-grids.gpkg
```

Both refuse to overwrite without `--force`. Only enriched files get packaged.

## Output

One oriented footprint per vehicle, EPSG:3879 — the tile grid's own CRS, so
nothing is reprojected on the way out. `.geojson` declares it with a `crs`
member; `.fgb` and `.gpkg` carry it natively and are what a real sweep should
write.

| field | | |
|---|---|---|
| `label` | model | `truck` / `bus` / `van` / `car` |
| `confidence` | model | see below before counting anything |
| `length_m` `width_m` `heading_deg` | geometry | recomputed from the box, never read back from a file |
| `aoi` | run | the AOI, or the sweep cell |
| `source_layer` `zoom` | run | `Ortoilmakuva_2025_5cm`, 16 — multi-year comparison needs no bookkeeping |
| `district` `postal_code` | `enrich` | |
| `street` `street_type` | `enrich` | name, and the street register's own purpose (Asuntokatu, Katuaukio, Tori …) |
| `context` | `enrich` | `parking` / `street` / `other` — where the vehicle *is*; one orthophoto cannot tell whether it is parked |
| `street_part` | `enrich` | Ajorata (carriageway), Pysäköintialue (bay), Tonttiliittymä (plot connection), Koroke (kerb build-out) |

### The one number to get right

**Quote counts at the operating confidence, 0.77.** On the 2025 city sweep the
same detections read as **3,729 trucks at conf 0.25 and 2,442 at 0.772** — a
third of the census riding on the threshold. At 0.25 only about half of
predicted trucks are trucks. Either report at the operating point or publish
the confidence distribution with the count.

Then read docs/model-card.md §6: shadowed trucks are missed, `van` is weak in
both directions, and this is Helsinki-only.

Imagery © Helsingin kaupunki, Kaupunkimittauspalvelut; enrichment from the
City of Helsinki open data service; mining data © OpenStreetMap contributors.

## The training loop

How the weights got made, and how the next round would run. A detector
proposes, a human corrects, the corrections fine-tune it:

```
detect → stage → review (web/) → export → train → eval → mine → detect → …
```

Areas live in `aois/helsinki.yaml` (EPSG:3067), each with a `role`
(`positive` / `hard-negative` / `sparse`) and a `split`. Their `notes` are the
annotation guide, shown beside the map.

```sh
uv run rekka-ai aois  --aoi aois/helsinki.yaml          # list, report overlaps
uv run rekka-ai fetch --aoi aois/helsinki.yaml --dry-run    # tile count, stop

# Propose on new ground, at a staging confidence well under the census one:
uv run rekka-ai detect --aoi aois/helsinki.yaml --name <area> \
    --weights runs/train/round4/weights/best.pt \
    --confidence 0.25 --out data/candidates/round5-<area>.geojson

uv run rekka-ai stage --candidates data/candidates/round5-<area>.geojson
uv run rekka-ai progress                    # review counts, and schema problems

cd web && bun install && bun run dev        # review at localhost:3000

uv run rekka-ai export --aoi aois/helsinki.yaml
uv sync --extra train
uv run rekka-ai train --name round5         # 16 GB card, ~30 min, batch 4
uv run rekka-ai eval  --weights runs/train/round5/weights/best.pt \
    --aoi aois/helsinki.yaml
uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
```

Four things here are not preferences:

- **`labels/` is the only irreplaceable artifact.** `stage` refuses to
  overwrite an existing label file without `--force`.
- **Staging detects at 0.25**, not the default. The census threshold discards
  exactly the near-misses the next round has to fix.
- **`export` splits by whole AOI**, never at random — adjacent windows are
  near-duplicates — and refuses to run on unreviewed candidates, schema
  problems, or a box that drifted outside its area.
- **Rejects are kept.** A rejected candidate exports as background in the
  window that held it; that is how the model learned containers are not
  trucks.

`mine` proposes the next areas from OSM landuse and the trained model instead
of hand-picking yards; `bootstrap` was the round-1 cold start and refuses
trained weights. What to label is [docs/LABELLING.md](docs/LABELLING.md),
what each round cost is [docs/rounds.md](docs/rounds.md), and the gates,
metrics and round replay are docs/DESIGN.md §7.

## Development

```sh
uv run ruff check && uv run ruff format && uv run ty check && uv run pytest

cd web
bun run test      # `bun run`, not bare `bun test` — that is bun's own runner
bun run check
bun run lint
```

CI runs the Python four and the web three on every PR and push to `main`.
Publishing trained weights is `docs/releasing.md` — tag `model/vN.0.0`, upload
the renamed `.pt`, publish.
Optional extras stay **lazily imported**: the base install and CI have neither
torch nor mlflow, so `fetch` and `aois` work without a multi-gigabyte install.

## License

- **Python backend** (everything outside `web/`) — [AGPL-3.0](LICENSE),
  because it builds on Ultralytics YOLO. That includes serving it over a
  network.
- **Labelling web app** (`web/`) — [MIT](web/LICENSE); a separate work that
  talks to the backend only through data files.

[rel]: https://github.com/ForumViriumHelsinki/rekka-ai/releases
