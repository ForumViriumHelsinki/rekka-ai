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

### Round 3 areas mined and added, 2026-08-11

Mined from the round-2 weights at its operating confidence (0.114). Picked
against what round 2 left failing rather than by rank:

- **r3-olympiaterminaali** — Eteläsatama ferry quay, trailers staged in rows.
  For the `r1-jatkasaari` count gate, still +36% in harbour ground round 2
  never covered.
- **r3-herttoniemi** — delivery yards off Sahaajankatu; 65 of its 68
  detections sit in the 4–8 m band. For `van` recall, 0.73 across both rounds.
- **r3-roihupelto-varikko** — metro and tram depots. Rolling stock is on the
  confuser list, both harbour areas have track running through them, and the
  model has never trained on any.

Skipped: more Vuosaari container ground (round 2 solved containers — that
square went to 0 unexplained), a motorway bridge with 2 detections, and a
marina that duplicates r1-marjaniemi.

**Caveat on record.** `r3-olympiaterminaali` sits 1,567 m from
`r1-jatkasaari`, which is validation, and shares its harbour character. So
jatkasaari's count gate improving in round 3 will partly be "we trained on its
neighbourhood" — the same kind of dependence as the ruskeasuo/veturitie bus
pairing. Taken knowingly: the alternative is never training on harbour ground,
which guarantees that gate keeps failing.

## Round 3 Train and Eval

```bash
uv run rekka-ai export --aoi aois/helsinki.yaml
uv run rekka-ai train --name round3 --batch 4
```

### Round 3 — four gates pass, 2026-08-11

```sh
uv run rekka-ai export --aoi aois/helsinki.yaml --out data/dataset-r3
uv run rekka-ai train  --data data/dataset-r3/dataset.yaml --name round3
uv run rekka-ai eval   --weights runs/train/round3/weights/best.pt \
    --data data/dataset-r3/dataset.yaml --aoi aois/helsinki.yaml --name eval-round3
```

| gate | round 1 | round 2 | round 3 |
|---|---|---|---|
| operating confidence | 0.058 | 0.114 | **0.474** |
| truck recall ≥ 0.90 | 0.908 | 0.909 | **0.915** |
| truck precision ≥ 0.85 | 0.757 | 0.867 | **0.950** |
| count kaivoksela ≤ 10% | +23.6% | +4.7% | **−0.9%** |
| count jatkasaari ≤ 10% | +64% | +36% | **−7.1%** |
| negative areas ≤ 2 | 87 (recounted) | 3 | 5 |

The operating point tripling is the story: round 1 could only reach the recall
gate at conf 0.058 — keep-everything — and round 3 clears it at 0.474 with
precision 0.950. Jatkasaari swung from +36% to −7%.

**The round was picked badly and worked anyway.** Of its three mined areas,
`r3-roihupelto-varikko` was a dud (a metro depot photographed mid-morning has
its stock out in service, so the yard is empty) and `r3-olympiaterminaali`
yielded no trucks. `r3-herttoniemi` carried it — chosen for the 4–8 m
van/truck band, which the jatkasaari diagnostic then showed *was* jatkasaari's
failure, not the shipyard confusers the other two were picked for.

### Ground-truth pass, and the client's truck definition, 2026-08-11

Round-3 weights are good enough to audit the labels: 82 detections across 27
areas matched nothing labelled. Staged as candidates and reviewed — net +6
truck, +7 van, +78 car, +34 rejects. Re-evaluated on the corrected data with
**no retraining**:

```sh
uv run rekka-ai export --aoi aois/helsinki.yaml --out data/dataset-r3b
uv run rekka-ai eval   --weights runs/train/round3/weights/best.pt \
    --data data/dataset-r3b/dataset.yaml --aoi aois/helsinki.yaml \
    --name eval-round3-fixed
```

```
PASS  truck recall     0.910 >= 0.90
PASS  truck precision  0.956 >= 0.85
PASS  count kaivoksela 105 vs 107 trucks  -2%
FAIL  count jatkasaari  13 vs  15 trucks -13%
PASS  negative areas    2 unexplained (<= 2)
```

Two gates moved in opposite directions, and both moves were label error, not
model change. The negative check **passed** because its two failing detections
in `r1-puotinharju` were real cars nobody had labelled. Jatkasaari **failed**
because review added two real trucks (7.7 m and 10.9 m) the model does not
find — it had been passing partly on a ground truth that was missing them.
The two remaining unexplained detections are container trailers in
`r2-vuosaari-harbour-road`, genuine false positives, sitting exactly at the
gate's tolerance.

**The truck/van rule was written down for the first time**, from the client's
spec: a truck — including the small ones — has a **cab that stops and a load
body that starts**, boxy at the rear, open or closed; a van is one continuous
shell. DESIGN §5 had said "length ≥ 6 m" for three rounds while 88% of what
was labelled 6.0–6.5 m was a van. Length is now a sanity check: it decides
below 6.5 m and above 8 m, and nothing in between, where 49 vans and 78 trucks
share the same lengths.

Also recorded: **check for a shadow before trying to read a smeared object.**
These are morning flights over 3–4 m vehicles, so anything real throws a hard
dark shape. `r1-vuosaari-rahtarinkatu` #211 was a 19.2 m streak the model
found at conf 0.78 with no shadow at all — a mark on the apron, not a rig.

### Ablation: what each round's ground actually bought, 2026-08-11

Every round's numbers were confounded by label corrections landing at the same
time, so all three configurations were re-trained on today's corrected labels,
against the identical four validation areas. Only the training ground varies.

```sh
for t in r1 r1r2 r1r2r3; do
  uv run rekka-ai export --aoi data/ablation/$t.yaml --out data/dataset-abl-$t &&
  uv run rekka-ai train  --data data/dataset-abl-$t/dataset.yaml --name abl-$t --batch 4 &&
  uv run rekka-ai eval   --weights runs/train/abl-$t/weights/best.pt \
      --data data/dataset-abl-$t/dataset.yaml --aoi data/ablation/$t.yaml \
      --name eval-abl-$t || break
done
```

| training ground | truck P | truck R | kaivoksela | jatkasaari | negatives | gates |
|---|---|---|---|---|---|---|
| r1 only (17 areas) | 0.773 | 0.910 | +20.6% | +26.7% | 4 | 1/5 |
| + r2 (20) | 0.822 | 0.904 | +15.0% | +33.3% | 3 | 1/5 |
| + r3 (23) | **0.884** | 0.904 | **+4.7%** | **+6.7%** | **1** | **5/5** |

`abl-r1r2r3` is the first model to pass every gate.

**What the mined ground bought is out of proportion to its instances.** r2 adds
52 labelled instances and r3 adds 125, nearly all cars — 8% more instances
between them — but they add **54 windows** of container yards, trailer rows and
car parks that export as background. The improvement tracks the windows, not
the instance count.

**Round 3 carried it, and it was the round judged a write-off at the time**
(two of its three areas were duds — see the round-3 entry). The area that
worked, `r3-herttoniemi`, was picked for the 4–8 m van/truck band, which the
jatkasaari diagnostic later showed *was* jatkasaari's failure.

**Caveat, and it limits what may be claimed.** These are single runs. The
existing `round3` weights — same ground, *worse* labels — score truck precision
0.956 against `abl-r1r2r3`'s 0.884. A 0.07 swing with no data cause means
run-to-run variation is larger than the r1→r1r2 step (+0.049), so individual
steps are not reliable evidence. The monotonic trend across four independent
measures is; a single number is not.

**Batch size was the trap.** The first attempt ran with the `AUTOBATCH = -1`
default and Ultralytics chose **batch 1** — 3.7 GB of a 16 GB card, four times
the optimizer steps, BatchNorm from single images, and not comparable to
rounds 1–3, which all ran at 4 because it was passed by hand. Killed and
restarted. `train.DEFAULT_BATCH` is now 4, and README records the 16 GB
expectation.

### Seed spread: which gates actually measure the model, 2026-08-11

`--seed` was added to `train` and the `abl-r1r2r3` dataset trained three times
at seeds 0, 1, 2 — identical data, hyperparameters and validation, only the
shuffle differs.

| metric | seed 0 | seed 1 | seed 2 | spread |
|---|---|---|---|---|
| truck recall | 0.904 | 0.905 | 0.904 | **0.001** |
| truck mAP50 | 0.953 | 0.952 | 0.962 | 0.010 |
| truck precision | 0.884 | 0.899 | 0.910 | 0.027 |
| van recall | 0.713 | 0.776 | 0.796 | 0.083 |
| count kaivoksela | +4.7% | +7.5% | +0.9% | 6.5 pts |
| count jatkasaari | +6.7% | +6.7% | +26.7% | **20 pts** |
| negative areas | 1 | 7 | 3 | **6** |
| operating confidence | 0.776 | 0.189 | 0.299 | 0.59 |

**Trustworthy at single-run resolution:** truck recall, truck mAP50, and truck
precision — whose 2.7-point spread is smaller than the ablation's steps
(+4.9 and +6.2), so **the ablation's precision trend survives**.

**Not trustworthy:** everything else.

- `r1-jatkasaari`'s count error moves **20 points on the seed alone**, against a
  gate of ±10%. Two seeds pass, one fails badly. The floor
  `GATE_COUNT_MIN_TRUCKS = 10` was derived from box arithmetic, not from noise:
  at 15 trucks a 10% tolerance is 1.5 boxes and the seed moves it by 3.
  `r1-kaivoksela` at 107 trucks has 10.7 boxes of tolerance against ~7 boxes of
  spread, so even the good case is marginal.
- The negative check reads 1, 7, 3 against a threshold of 2 — the fix work
  (unexplained counting, the length floor) made it *correct*, but it is still
  measuring a quantity noisier than its own threshold.
- `van` recall spreads 8.3 points, which is larger than every round-to-round
  change ever recorded for it.
- The operating confidence — which DESIGN §7 promises becomes `detect`'s
  confidence — lands anywhere from 0.19 to 0.78. The PR curve is flat near the
  recall gate, so the argmax is close to arbitrary.

**Earlier conclusions that do not survive**, corrected here rather than by
editing the entries that made them:

- "`abl-r1r2r3` is the first model to pass all five gates" is a **seed-0
  property**. Seeds 1 and 2 fail the negative check, and seed 2 also fails
  jatkasaari.
- Round 3 taking the negative check from 5 to 2, and round 2 from 87 to 3, are
  within seed noise. The *direction* is consistent across the ablation, the
  magnitudes are not evidence.
- `van` recall rising 0.731 → 0.778 was read as evidence against the
  "resolution-limited" theory (the z17 discussion). It is not evidence of
  anything; that question remains open.

**What this asks for.** Two of the five gates have noise exceeding their
thresholds, so they cannot decide a ship question as written. Either the
thresholds widen, the floor rises so only large areas gate, or those two become
reported numbers rather than gates. Not decided here — but no ship decision
should rest on them until it is.

### Gates rebuilt around the measured noise, 2026-08-11

Acting on the seed spread above.

- **Count-gate floor 10 → 60.** The old floor came from box arithmetic (at 10
  trucks a 10% error is one box). The new one comes from the measurement: noise
  runs at about `0.77*sqrt(n)` boxes against `0.1*n` of tolerance, and they
  cross at n = 60. Tolerance would reach twice the noise only at n = 240, which
  no area has. **`r1-kaivoksela` is now the only area that gates on count** —
  `r1-jatkasaari`, at 15 trucks, is reported instead.
- **The negative check is demoted to reported.** It counts the right thing, but
  1/7/3 across three seeds against a threshold of 2 is not a measurement. It
  prints under "reported, not gated" and says so above a watch level of 2.

Three gates remain: truck recall, truck precision, and the kaivoksela count.
`abl-r1r2r3` passes all three at every seed tried.

The cost is stated plainly: the ship decision now rests on **one area's count**
and two split-wide truck metrics. That is less coverage than it looks, and the
fix is a wider validation split (§11.3), not a lower bar.
