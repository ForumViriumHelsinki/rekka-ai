---
name: city-census
description: Run the production detection chain end to end for a flight year — WFS region, chunked resumable sweep, enrich, package, density grids — and the rules for reporting the counts it produces
type: prompt
whenToUse: When the user asks to sweep the whole city, produce or refresh the detections for a flight year, enrich/package/deliver detections, build the truck grids, or quote a city-wide vehicle count
---

# The production census run

The loop skills train a model; this one runs it over Helsinki and hands the
result to someone else. Commands are `README.md` §"Run the detector"; the
reasoning is `docs/DESIGN.md` §6. This is the operational knowledge around
both.

## The chain, and the naming contract

```
prepare_production_aoi.py → detect --checkpoint-dir → enrich → package → analyse
```

`package_detections.py` finds its inputs **by filename**, not by argument:
`data/detections/helsinki_<year>_z<zoom>_enriched.fgb`. Underscores, not
hyphens. Name the sweep output `helsinki_<year>_z<zoom>.fgb` and the enriched
output the same plus `_enriched`, or packaging fails with "no enriched
detections at …". Checkpoint directories use hyphens
(`data/detections/helsinki-<year>/`) and are unrelated to that contract.

## Per step

- **Region.** `uv run --extra detect python scripts/prepare_production_aoi.py`
  → `data/production/helsinki-region.gpkg`, 208.3 km², EPSG:3879, 314 cells
  at `--cell-size 1000`. Both WFS layers are cached under `data/wfs/`, so a
  rebuild is offline and cheap. Rebuild only if the WFS layers changed — a
  different region silently changes every count downstream.
- **Sweep.** Always `--checkpoint-dir`, always an explicit
  `--confidence 0.25`. Mechanics, timing and failure modes are the
  **detect-sweep** skill. Expect 22 min warm / 47 min cold per year at z16;
  2023 had no tile cache and ran ~45–50 min, 2024/2025 are cached.
- **Enrich.** `rekka-ai enrich --detections <fgb> --out <…>_enriched.fgb`.
  WFS layers cache under `data/wfs/`; the street-parts layer is 96 MB raw and
  the cache trims it to `alatyyppi` + geometry. `--street-max-distance`
  defaults to 2 m. Enrichment adds columns only — it never touches geometry,
  `length_m`, `width_m` or `heading_deg`.
- **Package.** `scripts/package_detections.py` — enriched files only, refuses
  a wrong-CRS or un-enriched source, refuses to overwrite without `--force`.
- **Analyse.** `scripts/analyse_detections.py` — 250 m truck grids + centroid
  points per year, `--min-confidence` 0.77 recorded on every layer.
  Disposable; rebuilds in a minute.

## Verifying a sweep before shipping it

- **Cells:** every cell `done` in `manifest.jsonl`, zero `failed`. The
  recorded runs had zero of both — a nonzero count is a real finding, not
  noise to retry away.
- **Seam duplicates:** the merge removed 17 of 125,882 (0.013%) on 2025. An
  order of magnitude more means the cells or the merge changed.
- **Class length medians** against the LABELLING.md bands: car 4.71 m, van
  5.49 m, bus 13.80 m, truck 7.99 m (2025). Truck sits at the low edge of the
  8.0–11.3 m band, as a city-wide census should against a corpus weighted
  toward industrial yards.
- **Scale:** 2025 = 125,882 detections at conf 0.25 — 108,780 car, 12,936
  van, 3,729 truck, 437 bus; 17.9 trucks/km².

## Reporting the counts — the part that goes wrong

**Never quote a conf-0.25 count as a census.** The same 2025 detections read
**3,729 trucks at 0.25 and 2,442 at 0.772**; at 0.25 only ~53% of predicted
trucks are trucks. Report at the operating confidence (0.77) or publish the
confidence distribution with the number, and always name the flight year.

Carry the model card's limitations with any count that leaves the repo:
shadowed trucks are under-counted, `van` is weak in both directions (some
predicted trucks are vans), Helsinki only. `docs/model-card.md` §5–6.

Attribution on anything shipped: imagery © Helsingin kaupunki,
Kaupunkimittauspalvelut; enrichment from the City of Helsinki open data
service.

## Do not

- Re-sweep to change the reported threshold — filter the existing file.
- Hand out the un-enriched merge, or a bare per-year `.fgb`, when a packaged
  GeoPackage exists; three similarly-named files is how the wrong one ships.
- Delete a checkpoint directory to "start clean" — it is the only record of
  which cells failed and why.
