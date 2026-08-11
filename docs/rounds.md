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

### Negative gate fixed: count only unexplained detections, 2026-08-11

**Was open, now closed.** The regression check counted *every* detection in a
negative-role area and gated at 2 — written when negative meant empty. It no
longer is: `r1-puotinharju` holds 93 confirmed cars and 5 vans,
`r1-marjaniemi` 17 cars and 2 vans, and round 1's eval reported **152**, an
order of magnitude past the gate, for a model that was right about them.

`evaluate.unexplained` now forgives any detection matching a labelled vehicle
(IoU > 0.3, class ignored — the question is whether the model invented a
vehicle, not whether it named it right). Rejected boxes forgive nothing: a
reject is the human saying "not a vehicle", so a detection on one is the
error being counted. Threshold stays at 2 — a gate gets tuned after it has
produced one honest measurement, not before.

`r2-vuosaari-harbour-road` became `role: hard-negative` in the same change:
53 candidates reviewed, no trucks kept, 2 vans and a car on the access road.
It is the best regression probe the project has for the round-1 failure. Its
three real vehicles stay labelled — they are what stops the gate counting
them as false alarms.

Known weakness, not fixed: all three negative areas are `split: train`, and
the gate does not filter by split, so it measures whether the model
hallucinates trucks on ground it trained on. A held-out negative area is
still the outstanding ask in §11.3.

### Superseded: negative-role areas now hold labelled cars, 2026-08-11

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

## Round 1 Training and Eval

```bash
# export and train 
uv run rekka-ai export --aoi aois/helsinki.yaml
uv run rekka-ai train --name round1 --batch 4
...
100 epochs completed in 0.286 hours.

# eval
uv run rekka-ai eval --weights runs/train/round1/weights/best.pt --aoi aois/helsinki.yaml
...
validation split of data/dataset/dataset.yaml:
  truck  P 0.897  R 0.830  mAP50 0.910  mAP50-95 0.822
  bus    P 0.825  R 0.962  mAP50 0.976  mAP50-95 0.931
  van    P 0.845  R 0.748  mAP50 0.855  mAP50-95 0.795
  car    P 0.933  R 0.953  mAP50 0.968  mAP50-95 0.895

operating point (truck): conf 0.058 -> P 0.757, R 0.908
  count r1-veturitie: 6 vs 1 trucks — not gated, under 10
  count r1-pohjois-haaga: 2 vs 0 trucks — not gated, under 10

gates:
  PASS  truck recall: 0.908 >= 0.9
  FAIL  truck precision: 0.757 >= 0.85
  FAIL  count r1-kaivoksela: 131 vs 106 trucks (+24%)
  FAIL  count r1-jatkasaari: 23 vs 14 trucks (+64%)
  FAIL  negative areas: 152 detections (<= 2)
```

### Round 1 trained and evaluated — precision fails again, 2026-08-11

Fine-tune from `yolo11x-obb.pt` on the 4-class dataset, 100 epochs. Gates:

| | | |
|---|---|---|
| truck recall ≥ 0.90 | 0.908 | pass, but only at conf **0.058** — just over the 0.05 floor |
| truck precision ≥ 0.85 | 0.757 | **fail** |
| count error ≤ 10% | +23.6% kaivoksela, +64% jatkasaari | **fail**, both over-counting |
| negative areas ≤ 2 | 152 | the broken gate above, not a model result |

**No operating point satisfies both gates.** At default confidence the model is
precision 0.897 / recall 0.830; pushed to 0.058 to clear recall, precision
collapses. The curve does not reach the target anywhere, so this is not
threshold tuning.

A `detect` pass over kaivoksela at the operating point says why. 124 truck
detections against 106 real, 103 matched — 97% recall in-area — and of the 21
extras, **12–15 sit on cab-less truck-shaped objects**: stored bodies, tipper
skips, trailers, containers, four of them in the equipment corner. Six more are
boxes labelled `van`. The equipment-corner errors are the model's *most
confident* (0.90, 0.85, 0.73), so raising the threshold does not help: dropping
everything under 0.30 removes half the false positives and costs recall that is
already scraping its gate. DESIGN §4 predicted exactly this — training on
truck-shaped objects pulls lookalikes in.

Two consequences for round 2:

- **Relabelling those bodies as rejects would change nothing.** `export` writes
  only confirmed/added, so a reject and an omission are the same background —
  the labour of rejecting a container buys exactly what leaving it alone
  buys. What the areas contribute is their **windows**: imagery in which
  containers and stored bodies appear as background. Training has had little
  of it — r1-hermanni, r1-vuosaari-rahtarinkatu and r1-vuosaari-channel-road,
  a few hundred boxes — and kaivoksela, where the failure shows, is
  validation ground the model never trained on. So mine for equipment yards,
  trailer parks and container ground; kaivoksela's 97% recall says more truck
  yards buy nothing.
- **The van/truck boundary at 7–8 m needs a rule**, the way car/van at 5–6 m
  got one. `van` recall is the weakest class at 0.748, and this is round 1 on
  the old collection repeating itself.

Training plateaued by epoch 10; best mAP50 0.937 at epoch 44 of 100. Use
`patience=30` next round — the last 55 epochs bought nothing.

## Round 2 Training and eval

```bash
uv run rekka-ai export --aoi aois/helsinki.yaml
uv run rekka-ai train --name round2 --batch 4
...
100 epochs completed in 0.324 hours.

uv run rekka-ai eval --weights runs/train/round2/weights/best.pt --aoi aois/helsinki.yaml
...
gates:
  PASS  truck recall: 0.909 >= 0.9
  PASS  truck precision: 0.867 >= 0.85
  PASS  count r1-kaivoksela: 114 vs 106 trucks (+8%)
  FAIL  count r1-jatkasaari: 19 vs 14 trucks (+36%)
  FAIL  negative areas: 20 unexplained detection(s) (<= 2)
...

```

### Round 2 — precision passes, 2026-08-11

Same recipe, three mined areas added (two Vuosaari container squares, one
Kivikko estate), 189 candidates of which 137 were rejects.

```
PASS  truck recall     0.909 >= 0.90     (round 1: 0.908)
PASS  truck precision  0.867 >= 0.85     (round 1: 0.757)
PASS  count kaivoksela 111 vs 106  +5%   (round 1: +23.6%)
FAIL  count jatkasaari  19 vs 14  +36%   (round 1: +64%)
FAIL  negative areas    3 unexplained    (round 1: 87, recounted)
```

**Precision is the result.** Round 1 only reached the recall gate by dropping
the operating point to 0.058 — keep-everything — where precision collapsed.
Round 2 clears the same recall at **conf 0.114**, twice as high, and holds
precision at 0.867. The curve moved, not the threshold.

The negative comparison is 87 → 3, but re-measured: round-1 weights were
re-run through the new counting to get a baseline, since the old 152 counted
raw detections. Caveat on the biggest single win — `r2-vuosaari-harbour-road`
went 55 → 0 and it is in round 2's *training* set, so that is not evidence of
generalisation. The held-out evidence is the precision and kaivoksela numbers.

Still failing: `r1-jatkasaari` at +36% (14 trucks, 19 found) — shipyard ground
of cranes, hulls and containers that round 2 did not cover. And the negative
check by one detection, all three in `r1-puotinharju`.

`van` recall is unchanged at 0.731 and remains the weakest class. The
van/truck boundary at 7–8 m still has no rule, unlike car/van at 5–6 m.

**Gates now sweep with the pipeline's length floor.** They stand in for a city
sweep, and a sweep runs `detect`, which drops everything under `MIN_LENGTH_M`.
Counting raw proposals had the negative check failing on 2–4 m slivers of
parked cars: 20 became 3 once the floor was applied, and kaivoksela's count
error +7.6% became +5%. Standard metrics still see the raw model.
