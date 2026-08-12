---
name: detect-sweep
description: Run or plan a `rekka-ai detect` sweep over a region (city-wide or AOI cells), with realistic timing, caching behaviour, failure modes, and how to compare sweep outputs
type: prompt
whenToUse: When the user asks to run detection over an area, estimate how long a sweep takes, compare detection outputs, or debug a sweep crash/slowdown
---

# Detection sweeps: operational knowledge

Measured on an RTX 5070 Ti, 2026-08-11/12, Helsinki WMTS. Treat these as
grounded expectations, not guesses.

## Command shape

```sh
uv run rekka-ai detect --aoi <region.fgb|collection.yaml> \
  --weights runs/train/<run>/weights/best.pt \
  --confidence 0.25 --out data/detections/<name>.geojson
```

- Match zoom to the weights: z17 weights need `--zoom 17`; default is z16.
- Output format follows the suffix: `.geojson` direct, `.fgb`/`.gpkg` via
  geopandas (lazy optional dep).
- `--time uv run …` or the CLI's own `done: … in Xh Ym` line both work for
  timing; progress prints as `window N/total, K detections` (K is pre-merge).

## Timing expectations

- Throughput: **~850–950 windows/min** when fetching fresh tiles,
  **~1,500–1,800/min** when tiles are already cached. GPU sits at ~30–40%
  fetch-bound — a faster GPU does not help a fresh sweep.
- The WMTS caps at **~85 tiles/s regardless of workers** (measured 8 vs 16:
  no difference). Keep the default 8; raising it gained nothing and the
  server once answered an OutOfMemoryError under load.
- Zoom cost: z17 produces **~8–9× the windows** of z16 over the same ground
  (fixed-metre overlap costs more at higher zoom). Reference: full OSM
  industrial Helsinki = 38k windows / 48 min at z16, 313.7k windows / 4h 39m
  at z17.

## Caching and restarts

- Tiles live in `data/cache/<layer>/<zoom>/<col>/<row>.jpg` and never
  invalidate. A restarted sweep **does not resume window progress** — it
  replays from window 1, but cached tiles make the replay GPU-bound, so the
  catch-up is ~2× faster than the first pass. Detections from the aborted
  run are lost.
- The fetcher validates the JPEG SOI marker: GeoServer can answer failures
  with an XML exception report and status 200. Such tiles are retried, then
  recorded in `failures`, and the affected windows are skipped and reported —
  never cached, never fatal. If a sweep reports skipped windows, refetching
  later fills the holes.

## Output semantics

- Detections are merged with **class-agnostic global NMS (IoU > 0.4)** before
  writing — cross-class duplicates are gone by construction. Known survivor:
  same-class seam duplicates at IoU 0.2–0.39 with near-coincident centres
  (~1% of detections; a centre-proximity rule is the planned fix — do NOT
  just lower the IoU threshold, depot rows of buses/trucks genuinely overlap
  at 0.3).
- `heading_deg` is the long axis **modulo 180°** by design — a rectangle has
  no front.
- Counts at conf 0.25 are recall-flavoured; per-class totals shift with zoom
  (z16 inflated trucks ~25% with misread vans; trust z17 class mix more).

## Comparing two sweeps

Load both GeoJSONs, `Counter` on `properties["label"]`, and report per-class
counts + totals + mean confidence. For duplicate audits: STRtree over the
polygons, flag pairs by IoU *and* by centre distance as a fraction of box
length (≤0.2 = same vehicle; ~0.3 = probably adjacent parked vehicles).
