---
name: new-aoi
description: Turn a mining proposal into a reviewable labelling area — aois/helsinki.yaml entry anatomy (rN- naming, EPSG:3067, role/split, notes), the overlap and proximity checks, then fetch → bootstrap → stage → progress
type: prompt
whenToUse: When the user has accepted mining proposals and wants to add them to the AOI collection, asks how to add a new labelling area, or needs the proposal-to-labelling handoff
---

# New AOI: proposal → reviewable area

`rekka-ai mine` (and class-targeted density mining — see the
**mine-annotation-ground** skill) deliberately stops at a proposal under
`data/mining/` (gitignored). Nothing writes `aois/` or `labels/`
automatically: a human reviews the proposal and accepted entries are copied
into `aois/helsinki.yaml`. This skill is that handoff.

## Anatomy of a collection entry

```yaml
crs: ETRS89 / TM35FIN(E,N)    # EPSG:3067 — the collection declares its CRS,
                              # and it always wins over --crs
areas:
  - name: r4-sompaasaari      # rN-place, see below
    bounds: [min_e, min_n, max_e, max_n]   # EPSG:3067 metres
    role: positive            # positive | hard-negative | sparse
    split: train              # train | validation
    notes: >                  # the labelling guide for THIS area — the web
      What is here and how to call it.   # tool renders it beside the map
```

- **Name: `rN-place`.** The round prefix is load-bearing — the ablation
  filters key on `name.split("-")[0]` in {r1..rN}, so a wrong prefix means
  the area silently joins the wrong round's training ground. Use a real
  place name, not a grid id; the notes and the log read better for it.
- **Split by whole AOI, never random.** Adjacent windows are near-duplicates
  and a random split makes the metrics fiction.
- **notes are the annotation guide**, displayed beside the map in the web
  tool. Generic labelling rules live in `docs/LABELLING.md`; notes hold what
  is *specific* to the ground (what the mining found, local confusers, how
  to call the edge cases). State why the cell was picked — the mining source
  and raw counts come from the proposal.
- **Roles:** `hard-negative`/`sparse` areas get no candidates and export as
  pure background. If validation gains a role it did not have, `aois` warns
  — validation that cannot see a hard negative cannot measure the false
  positives it exists to suppress.

## Proximity caveat (check before assigning split)

Near-duplicate ground across the train/validation boundary inflates
held-out reads. On record (docs/rounds.md): `r1-veturitie` sits 1,196 m from
`r1-ruskeasuo` (train) and `r3-olympiaterminaali` 1,567 m from
`r1-jatkasaari` (validation) — both accepted with the caveat logged. Check
distance from each new validation candidate to the nearest train area; if
you accept one anyway, append the caveat to `docs/rounds.md`.

## The sequence

```sh
# 1. Edit aois/helsinki.yaml (the human review step — copy from the proposal)

# 2. Sanity-check the collection: tile counts, role/split balance, and
#    overlaps (same ground annotated twice), with role/split conflicts flagged
uv run rekka-ai aois --aoi aois/helsinki.yaml

# 3. Warm the tile cache (tiles never invalidate; --dry-run first if unsure)
uv run rekka-ai fetch --aoi aois/helsinki.yaml

# 4. Propose candidates. The length floor defaults to 4.0 m (`MIN_LENGTH_M`)
#    because vans are a labelled class and live in the 5–6 m band — measured
#    over the 17 round-1 positive areas: 713 candidates at 4.0 m vs 559 at
#    6 m, so a quarter of the batch would otherwise never be seen. Name the
#    batch by round.
uv run rekka-ai bootstrap --aoi aois/helsinki.yaml --role positive \
    --out data/candidates/roundN.geojson
#    (or --name r4-sompaasaari for a single area)

# 5. Stage into labels/. --aoi also stages an EMPTY file for collection
#    areas with no candidates — a quiet mined cell still needs a reviewed
#    file before export accepts it as background. NEVER --force: stage
#    skipping an existing file is the pipeline's core safety property.
uv run rekka-ai stage --candidates data/candidates/roundN.geojson \
    --aoi aois/helsinki.yaml

# 6. Hand to the reviewer
uv run rekka-ai progress
```

Then review in the web tool (`cd web && bun run dev`), and when the area is
done: export → train → eval per the **replay-round** skill.

## Failure modes seen

- Forgetting `--aoi` on stage for quiet mined cells → export refuses the
  area later ("no label file"), which reads like a stage bug.
- `--force` on stage to "refresh" candidates → discards reviewed verdicts.
  If candidates must be regenerated for an unreviewed area, delete nothing;
  let stage skip reviewed files and only write new ones.
- Train holding a role validation lacks (e.g. hard-negatives only in
  train) → `aois` warns; take it seriously, it means a gate input is
  unmeasured.
