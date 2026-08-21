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
  A **released** asset carries its zoom in the filename
  (`yolo11x-obb-fvh-z16-v4.0.0.pt`); a `runs/train/<run>/weights/best.pt`
  does not, so check the run it came from before sweeping with it.
- **`--confidence` is always explicit.** The default is 0.77 — round 4's
  operating point, a *census* threshold. Any sweep whose output will be
  staged, compared, or re-thresholded should pass 0.25 and cut higher on
  read; `confidence` is per-feature, so a higher cut is a filter over a file
  while a lower one is another full sweep.
- Output format follows the suffix: `.geojson` direct, `.fgb`/`.gpkg` via
  geopandas (lazy optional dep). City-scale output should be `.fgb`.
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
- City-wide reference (2025, 208.3 km², 314 cells / ~31.4k windows at z16):
  **22 min warm, 47 min cold**, 28.5 ms/window, zero failed cells.

## Caching and restarts

- Tiles live in `data/cache/<layer>/<zoom>/<col>/<row>.jpg` and never
  invalidate.
- **`--checkpoint-dir` makes a sweep resumable — use it for anything
  city-scale.** The region is split into `--cell-size` cells (1 km default),
  each cell's detections written to its own file, each cell's status appended
  to `manifest.jsonl`. A re-run redoes only what is not `done`; a failed cell
  is recorded and stepped over instead of ending the run (`--retry-failed`
  redoes those), and `--merge-only` rebuilds the output from cells already on
  disk. Recovery is "invoke it again", so the overnight wrapper is a retry
  loop around the same command — see `data/detections/run-2023-sweep.sh`.
- **Without `--checkpoint-dir` there is no resume**: a killed sweep replays
  from window 1 and its detections are lost. Cached tiles make the replay
  GPU-bound (~2× the first pass), but that is a re-run, not a resume.
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
- Counts at conf 0.25 are **recall-flavoured and must not be quoted as a
  census**: on the 2025 city sweep the same detections read 3,729 trucks at
  0.25 and 2,442 at 0.772, and at 0.25 only ~53% of predicted trucks are
  trucks (model card §5).
- Zoom: **z16 is the production model** — at round 4 z17 fails the gates for
  7.4× the training time, and z16 wins truck and van (model card §8). Earlier
  advice to trust the z17 class mix was a round-1/2 observation and no longer
  holds.

## Shipping a sweep

A sweep meant for delivery — city-wide, enriched, packaged — is the
**city-census** skill, not this one. This skill stops at the `.fgb`.

## Comparing two sweeps

Load both GeoJSONs, `Counter` on `properties["label"]`, and report per-class
counts + totals + mean confidence. For duplicate audits: STRtree over the
polygons, flag pairs by IoU *and* by centre distance as a fraction of box
length (≤0.2 = same vehicle; ~0.3 = probably adjacent parked vehicles).
