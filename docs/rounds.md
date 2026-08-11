# Correction rounds — the log

`docs/DESIGN.md` holds the rules; this file holds what happened, in the order
it happened. Entries are dated snapshots: they may age, and are not rewritten
when the present moves on.

**Keep entries short.** A decision, the numbers that justify it, and its cost.
The reasoning that outlives the round belongs in DESIGN.md; the guidance a
labeller needs belongs in `aois/helsinki.yaml` notes. Neither belongs here.

## The old collection (rounds 1–2)

**Bootstrap measurements, 2026-08-04.** Quoted in DESIGN §4–§5 (zoom table,
first full bootstrap run, negative-area zeros, yolo11x-vs-yolo26, determinism
rebuild) because they still justify the rules.

**Round 1 — the loop closes, precision fails.** First fine-tune on corrected
labels: recall gate met (0.92), precision failed on truck/van confusion at the
short end, 11 detections across the negative areas. The `vuosaari`
hard-negative area was retired — reviewed rejects are better hard negatives
than an unlabelled area — and round 2 was aimed at that confusion.

**Round 2 — Vuosaari harbour, over-large.** A 6 km² sweep proved the loop
closes end to end: 1,267 reviewed candidates, 243 trucks, 179 vans, 67% reject
rate. It also proved 6 km² is too much ground per unit of labelling attention,
which is why the collection is 300 m plots now. Its labels were deleted on
purpose when `labels/` was reset.

**The restructure.** Areas renamed to real place names and cut to 300 m plots;
`rastila` and `marjaniemi` added, filling the negative role on both sides of
the split for the first time. `labels/` wiped — every area started over.

## The fresh start (current)

### Round 1 over the new collection — labelling done, 2026-08-11

3,092 candidates reviewed across 21 areas: 370 truck, 154 bus, 381 van, 1,961
car, 7% reject rate. No training run yet, so no gate numbers exist for this
collection.

### r1-kamppi moved to train, 2026-08-11

Downtown imagery is the collection's worst — shadow, building lean, mosaic
seams across a busy street — so its ground truth is guesswork. Validation as
it stood:

| area | trucks | buses | vans |
|---|---|---|---|
| r1-kaivoksela | 109 | 7 | 32 |
| r1-jatkasaari | 16 | 1 | 21 |
| r1-kamppi | 8 | 16 | 14 |
| r1-pohjois-haaga | 0 | 3 | 7 |

6% of validation's trucks against 59% of its buses — it barely touched the
gates that decide anything, while 10% count error over 8 trucks is 0.8 of a
box. **Moved to train.** Difficulty is not the disqualifier; untrustworthy
ground truth is. Generalised into DESIGN §7 ("an area too small to gate must
not gate").

### r1-veturitie added as the bus validation site, 2026-08-11

Pohjolan Liikenne's depot, Veturitie 25, Pohjois-Pasila (OSM
`industrial=depot`) — ~70 coaches on open concrete, the cleanest imagery in
the collection. **Held out rather than swapped for the already-labelled
r1-ruskeasuo**, on the principle that validation is where ground truth must be
trustworthy; the cost is a ~300-box labelling round.

Caveat on record (DESIGN §5): `r1-ruskeasuo` is 1,196 m away in train and
shares the yard's character, so held-out bus recall reads optimistically.
Helsinki has no bus ground of a different character, and `bus` never gates.

**Count-gate floor implemented the same day**, because this area needs it:
`evaluate.count_gate` skips any area below `GATE_COUNT_MIN_TRUCKS`
(= `1 / GATE_COUNT_ERROR` = 10, where the tolerance first covers a whole box).
The no-trucks case, previously gated at "zero detections", now goes the same
way — the split-wide precision gate already catches false positives.

### Area notes trimmed to annotation guides, 2026-08-11

`aois/helsinki.yaml` notes had drifted into carrying justification;
`r1-veturitie`'s ran to 1,372 characters and pushed the area tree out of the
sidebar. Rule now stated in that file: **notes say what to look for and how to
call it.** Four cut (veturitie, kamppi, marjaniemi, rastila); the web tool
gained a clamp so a note cannot eat the sidebar again.

Two justifications rescued from the cut notes:

- **r1-rastila** moved from validation to train so the model can learn from
  its ~60 RV instances directly, costing the project its one held-out
  RV-confusion measurement — a deliberate tradeoff. Review found 1 truck and
  7 vans real against 75 RVs rejected, which is why its role is `positive`.
- **r1-marjaniemi** exists because fine-tuning rebuilds the head on land-only
  imagery, discarding DOTA's `ship` knowledge entirely. Framed 100 m west of
  the basin to trade open water for the storage yard. Its note used to declare
  the square empty of trucks, buses and vans; review confirmed two vans, so
  that claim is retired.

### Labelling-round audit, 2026-08-11

Swept every file for overlap defects. Fixed: two cross-file class
disagreements and two omissions on ground shared by `r1-tattariharju`,
`r1-kivikko` and `r1-malmi-airport` (the same vehicles were labelled twice,
inconsistently); one double label in `r1-vuosaari-rahtarinkatu`; three
class/length outliers. Afterwards: 0 double labels, 0 cross-file
disagreements, export guards clean.

**Truck + full trailer decided and recorded in DESIGN §5:** box the truck
unit, the trailer is a negative whether coupled or not. Confirmed from the
labels — every truck in `r1-kaivoksela` measures 6.1–12.5 m, so no
combination had ever been boxed whole.

Still open: ~26 vehicles sit in two label files because three plots physically
overlap, and will export twice. Plot geometry, not labelling.

### Open: negative-role areas now hold labelled cars, 2026-08-11

The eval regression check counts *every* detection in a negative-role area and
gates at 2 (DESIGN §7). Written when negative meant empty; `car` is a class
now, and `r1-puotinharju` holds 93 cars and 5 vans, `r1-marjaniemi` 17 cars
and 2 vans. A model that detects them is right and fails the gate by an order
of magnitude.

Left as a decision, not a quiet patch: count only truck/bus/van, or count only
detections the area's own ground truth does not account for. The second is
more honest and more work. Nothing depends on it until the first eval.

### Reject audit, and two open cases decided, 2026-08-11

Swept all 229 rejects: 179 were a second box on an already-labelled vehicle,
15 were rastila RVs, the rest trailers, containers, skips and shadows. Ten
standalone rejects sat on unlabelled objects and were re-checked by eye — one
was a genuinely missed car (`r1-malmi-airport` #29), one a van
(`r1-kylasaari` #134); seven were RVs and one a rooftop structure.

Two known-hard cases from DESIGN §5 settled on those examples:

- **Motorhomes / RVs / caravans → reject.** "Avoid annotating either way" was
  not a verdict a label file can hold. They are also not confined to
  `r1-rastila`: white 6–8 m RVs turned up in four industrial car parks, each a
  plausible van at a glance.
- **Pickups → `car`**, not `truck` — they measure and behave like cars, and
  `truck` is the only gated class.
