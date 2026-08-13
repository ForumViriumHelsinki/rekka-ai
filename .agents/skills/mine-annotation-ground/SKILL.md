---
name: mine-annotation-ground
description: Find and propose new labelling AOIs (annotation ground) — model-mined strata via `rekka-ai mine`, or class-targeted density mining from existing detection sweeps (e.g. more van training data)
type: prompt
whenToUse: When the user asks for new annotation/labelling areas, more training data for a class, where to label next, or candidate AOI proposals
---

# Finding new annotation ground

Two complementary routes. Both end at a **proposal** under `data/mining/`
(gitignored) — never write to `aois/` or `labels/`; a human reviews the
proposal and copies accepted entries into `aois/helsinki.yaml` by hand.

## Route A: built-in strata mining

`uv run rekka-ai mine` grids OSM `landuse=industrial` into 300 m cells,
excludes the existing collection, sweeps a dispersed pool with the current
weights, and writes a stratified shortlist (near-threshold / size-boundary /
dense / quiet) to `data/mining/`. Use this for general-purpose next-round
AOIs. `--profile` selects the OSM landuse profile, `--refresh-osm` re-fetches
Overpass responses (cached in `data/osm/`).

## Route B: class-targeted density mining from sweep detections

Better when the goal is "more of class X" (established 2026-08-12 for vans).
Existing sweep outputs are a class-density map of the whole city — let the
model do the prospecting.

1. **OSM profile.** Profiles live in `PROFILES` in `src/rekka_ai/osm.py`
   (industrial / commercial / construction / camping). Add one if the target
   geography is missing, then fetch (network, cached):

   ```python
   from pathlib import Path
   from rekka_ai.osm import fetch_industrial

   cache = fetch_industrial(
       "Helsinki", profile="commercial", cache_root=Path("data/osm"), refresh=True
   )
   ```

2. **Grid and exclude.** Reuse `rekka_ai.mine` — do not reimplement:

   ```python
   from rekka_ai.imagery.aoi import load_aois
   from rekka_ai.mine import cells_covering, exclude_existing, project_features_to_tm35fin

   union = project_features_to_tm35fin(cache.features)
   cells = exclude_existing(
       cells_covering(union, municipality_ref=cache.ref), load_aois("aois/helsinki.yaml")
   )
   ```

3. **Detections for the cells.** If an existing sweep covers the ground
   (e.g. the industrial z17 sweep), load its GeoJSON into
   `rekka_ai.detect.detections.Detection` objects. If not, write the cells as
   a collection YAML (`crs: ETRS89 / TM35FIN(E,N)`, one entry per cell — see
   `data/mining/vanmine-commercial-construction.yaml`) and sweep it:
   `uv run rekka-ai detect --aoi <cells.yaml> --weights <best.pt>
   --confidence 0.25 --out <out.geojson>`. **Match the zoom to the weights**
   (z16 weights → default zoom; z17 weights → `--zoom 17`). A 127-cell sweep
   is minutes once tiles are cached.

4. **Rank.** Count target-class detections per cell
   (`rekka_ai.mine.detections_in_cell`), sort by count, apply the 600 m
   separation rule (`MIN_SEPARATION_M`) for the shortlist. Counts from a
   sweep are for ranking only — class leakage at the sweep's zoom biases
   them; human review settles the truth.

5. **Write the proposal.** Collection-shaped YAML + EPSG:3879 GeoJSON report
   under `data/mining/` (e.g. `van-candidates-combined.yaml`). Notes per cell
   should state the mining source and raw counts so the reviewer knows why
   the cell was picked.

## Rules that keep this safe and useful

- Proposals never touch `aois/` or `labels/` — `mine` has the same contract.
- Label against the written morphology rule (cab + separate box → truck;
  one-piece body → van) — new volume into an un-audited boundary just scales
  the label noise.
- Before adding volume for a class, check the confusion matrix first: if the
  class boundary is the problem, consistency beats quantity.
