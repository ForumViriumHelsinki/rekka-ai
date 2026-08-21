---
name: new-aoi
description: Turn a mining proposal into a reviewable labelling area — aois/helsinki.yaml entry anatomy (rN- naming, EPSG:3067, role/split, notes), the overlap and proximity checks, then fetch → detect → stage → progress
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

# 4. Propose candidates with the CURRENT WEIGHTS — `detect`, never
#    `bootstrap` (see Failure modes). The length floor defaults to 4.0 m
#    (`MIN_LENGTH_M`) because vans are a labelled class and live in the 5–6 m
#    band — measured over the 17 round-1 positive areas: 713 candidates at
#    4.0 m vs 559 at 6 m, so a quarter of the batch would otherwise never be
#    seen. Name the batch by round.
uv run rekka-ai detect --aoi aois/helsinki.yaml --name rN-place \
    --weights runs/train/round<N-1>/weights/best.pt \
    --confidence 0.25 --out data/candidates/roundN-rN-place.geojson
#    0.25 is the staging confidence, always explicit: the default (0.77) is
#    the census operating point and discards exactly the near-misses the
#    next round has to fix.
#    (--role positive instead of --name to do every positive area at once)

# 5. Stage into labels/. NEVER --force: stage skipping an existing file is
#    the pipeline's core safety property.
uv run rekka-ai stage --candidates data/candidates/roundN-rN-place.geojson
#    Add --aoi aois/helsinki.yaml ONLY when the batch covers every area that
#    still needs a file — it writes an EMPTY label file for every collection
#    area absent from the candidates, which a quiet mined cell needs before
#    export accepts it as background, but which also BLOCKS a later per-area
#    stage (see Failure modes).

# 6. Hand to the reviewer
uv run rekka-ai progress
```

Then review in the web tool (`cd web && bun run dev`), and when the area is
done: export → train → eval per the **replay-round** skill.

## Failure modes seen

- **`bootstrap` with trained weights proposes NOTHING, silently.** It is a
  cold-start command only: `cli.py` hardcodes `keep={LARGE_VEHICLE,
  SMALL_VEHICLE}` — DOTA's class names — so a fine-tuned model emitting
  `truck/bus/van/car` matches none of them and the sweep reports `0
  candidates` with a clean exit. It reads exactly like empty ground. This has
  bitten twice (docs/rounds.md, 2026-08-13; again 2026-08-14 on
  `r5-paulig`, where it also produced 0 on a known-good area with 55 labelled
  trucks). **Use `detect --weights <best.pt>` for every round after the
  first.** Sanity check when a sweep returns 0: re-run on an area you know
  holds vehicles; if that is 0 too, the command is wrong, not the ground.
- **`stage --aoi` writes empty files that then block the real candidates.**
  With `--aoi`, stage writes an EMPTY label file for every collection area
  absent from the candidates file. Stage a second per-area batch afterwards
  and it is refused — "exists with 0 reviewed, skipping" — so those
  candidates never land, and the summary line calls it "skipped" rather than
  failed. Seen 2026-08-14 staging `r4-vuosaari-terminal` then
  `r4-vuosaari-ratapiha`. Stage per-area batches WITHOUT `--aoi`, or stage
  one combined candidates file covering every area at once.
- `--force` on stage to "refresh" candidates → discards reviewed verdicts.
  If candidates must be regenerated for an unreviewed area, delete nothing;
  let stage skip reviewed files and only write new ones. The one safe use is
  an empty file you just created yourself: verify 0 features and 0 reviewed
  first, and scope it by passing a single-area candidates file with no
  `--aoi`, so `--force` cannot reach any other label file.
- Forgetting `--aoi` on stage for quiet mined cells → export refuses the
  area later ("no label file"), which reads like a stage bug. This is the
  mirror of the trap above: `--aoi` is right for a whole-collection batch,
  wrong for a per-area one.
- Train holding a role validation lacks (e.g. hard-negatives only in
  train) → `aois` warns; take it seriously, it means a gate input is
  unmeasured.
