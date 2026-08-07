# Correction rounds — the log

`docs/DESIGN.md` holds the rules and the decisions; this file holds what
happened, in the order it happened. Entries are dated snapshots — they are
allowed to age, and are not updated when the present moves on. The design
doc points here wherever a decision rests on a measured round.

## The old collection (rounds 1–2)

### Bootstrap measurements, 2026-08-04

The measurements that fixed the design's load-bearing choices are quoted in
DESIGN.md §4–§5 (the zoom table, the first full bootstrap run, the
negative-area zeros, the yolo11x-vs-yolo26 comparison, the determinism
rebuild) and live there because they still justify the rules.

### Round 1 — the loop closes, precision fails

First fine-tune on corrected labels, old collection. Shape of the result:
**recall gate met (0.92), precision failed** on truck/van confusion at the
short end, and **11 detections across the negative areas** — the §4
prediction that fine-tuning on truck-shaped objects could start pulling
containers in, confirmed. Two consequences followed: the `vuosaari`
hard-negative area was retired (the round-2 sweep covered its ground, and
reviewed rejects are better hard negatives than an unlabelled area), and
round 2's correction effort was aimed at exactly that confusion.

### Round 2 — Vuosaari harbour, over-large

A 6 km² map-sheet sweep (`vuosaari-harbour`, 2×3 km) proved the loop closes
end to end — detect with round-1 weights, stage, correct. It produced 1,267
reviewed candidates: 243 trucks, 179 vans, and a 67% reject rate that made
it the best hard-negative set the project had. It also proved 6 km² is too
much area per unit of labelling attention, which is why the collection is
300 m plots now.

The area was retired when `labels/` was reset, and its 1,267 labels were
**deleted on purpose** as part of the fresh start — the restructured
collection chose re-review over reuse. Ground inside the old footprint is
reviewed again, like everything else.

### The restructure

The collection was renamed to real place names and restructured into small
plots; `rastila` (hard-negative, validation — van-fronted motorhomes, the
annotation guide's hardest confuser) and `marjaniemi` (hard-negative, train
— a marina of boat hulls on cradles) were added, filling the negative role
on both sides of the split for the first time. `labels/` was wiped: every
area started over as unreviewed candidates.

## The fresh start (current)

### Round 1 over the new collection — in progress

Bootstrap, stage, review over the 20 areas (15 train / 5 validation), in
small chunks rather than 6 km² sweeps. Labelling is underway; **no training
run has been made on proper labels yet**, so no gate numbers exist for the
new collection. The confirmed/rejected ratio per chunk is the signal for
whether more ground of that character is worth sweeping at all.
