# Correction rounds — the log

`docs/DESIGN.md` holds the rules; this file holds what happened, in the order
it happened. Entries are dated snapshots. **Keep entries short.** A decision,
the numbers that justify it, and its cost. Labelling guidance belongs in the
AOI `notes`, not here.

## The old collection (before 2026-08-11)

Two rounds over hand-picked, oversized areas proved the loop closes — round 1
met the recall gate (0.92) and failed precision on truck/van confusion; round
2 swept 6 km² of Vuosaari harbour (1,267 reviewed candidates, 243 trucks, 67%
rejects) and proved 6 km² is too much ground per unit of labelling attention.
The collection was rebuilt as 300 m plots with real place names; `labels/`
was wiped and every area started over.

## The current collection

**Round-1 labelling, 2026-08-11.** 3,092 candidates reviewed across 21 areas:
370 truck, 154 bus, 381 van, 1,961 car, 7% rejects. Two validation decisions:
`r1-kamppi` moved to train (downtown imagery too degraded for trustworthy
ground truth — difficulty is not the disqualifier, untrustworthy ground truth
is), and `r1-veturitie` added as the held-out bus site. Caveats on record:
`r1-veturitie` sits 1,196 m from `r1-ruskeasuo` in train, and
`r3-olympiaterminaali` 1,567 m from `r1-jatkasaari` in validation — held-out
reads for those two characters are optimistic. Two labelling rules settled
from the client's spec: a truck has a **cab that stops and a load body that
starts**, a van is one continuous shell (length only decides below 6.5 m and
above 8 m); box the truck unit, the trailer is always a negative.

## Rounds 1–3: train, eval, conclusion

Each round fine-tunes `yolo11x-obb.pt` (batch 4, patience 30, seed 0) on the
cumulative training ground, gated against the **identical four validation
areas** on the same corrected labels — only the training ground varies, so
the rounds are comparable. Re-run 2026-08-11 under the rebuilt gates
(see below); runs `round1`–`round3` and `eval-round1`–`eval-round3` in
MLflow.

| | round 1 (17 areas) | round 2 (+3, 20) | round 3 (+3, 23) |
|---|---|---|---|
| truck recall ≥ 0.90 | **0.910** pass | **0.904** pass | **0.904** pass |
| truck precision ≥ 0.85 | 0.773 fail | 0.822 fail | **0.884** pass |
| count r1-kaivoksela ≤ 10% | +21% fail | +15% fail | **+5%** pass |
| operating confidence | 0.108 | 0.105 | **0.776** |
| (reported) unexplained in negative areas | 4 | 3 | 1 |

**Round 1 — precision fails on lookalikes.** A detect pass over the failing
area showed the model's *most confident* errors sit on cab-less truck-shaped
objects: containers, stored bodies, tipper skips, trailers. Relabelling those
as rejects would buy nothing — `export` writes only confirmed/added, so a
reject and an omission are the same background. What training lacked was
their *windows*. Conclusion: mine equipment yards, trailer parks and
container ground.

**Round 2 — the curve moves, not the threshold.** Three mined areas (two
Vuosaari container squares, one Kivikko estate) lifted precision 0.773 →
0.822 at the same recall. The mined ground added only 52 labelled instances
but 27 windows of container and trailer background — the improvement tracks
the windows, not the instance count.

**Round 3 — all gates pass, and the round was picked badly.** Of the three
mined areas, `r3-roihupelto-varikko` (metro depot, stock out in service) and
`r3-olympiaterminaali` (ferry quay, no trucks) were duds; `r3-herttoniemi`
carried the round — picked for the 4–8 m van/truck band, which was the actual
failure. The operating confidence tripling (0.105 → 0.776) is the story: the
PR curve moved far enough that the gates clear at keep-almost-nothing
confidence, not at keep-everything.

**Ground-truth pass.** Round-3 weights were good enough to audit the labels:
82 detections matched nothing labelled; review found them mostly real (+6
truck, +7 van, +78 car, +34 rejects). Two earlier "failures" were label
error, not model error — a model is only as gated as its ground truth.

## What a single run resolves (seed spread, 2026-08-11)

Round-3 ground trained three times at seeds 0/1/2, identical everything else:

- **Trustworthy at single-run resolution:** truck recall (spread 0.001),
  truck mAP50 (0.010), truck precision (0.027 — smaller than the round-to-round
  steps, so the table's precision trend survives).
- **Not trustworthy:** jatkasaari's count error (20 points of spread against
  a ±10% gate), the negative-area count (1/7/3 against a threshold of 2), van
  recall (8 points — larger than any recorded round-to-round change), and the
  operating confidence (0.19–0.78; the PR curve is flat near the recall gate,
  so the argmax is close to arbitrary).

**Gates rebuilt around this measurement:** the count-gate floor rose from 10
to 60 trucks (where 10% tolerance first exceeds the measured
`0.77*sqrt(n)` noise), so only `r1-kaivoksela` gates on count; the negative
check was demoted to reported. Three gates remain: truck recall, truck
precision, and the kaivoksela count. The cost, stated plainly: the ship
decision rests on one area's count and two split-wide truck metrics — the fix
is a wider validation split (DESIGN §11.3), not a lower bar.

## First test run on industial areas

```bash
time uv run rekka-ai detect --aoi data/osm/091_helsinki_industrial.fgb --weights runs/train/round3/weights/best.pt --confidence 0.25 --out data/detections/industrial-conf25.geojson 
```

## Round 4 begins, 2026-08-13

`r4-kruununhaka` (downtown, by Senate Square) added — hand-picked, intended
as validation's first `sparse` area ahead of a city-wide run. Round-3 `detect`
candidates over it (168 at conf 0.25, mostly parked cars, no false-positive
storm on shadowed ground) showed the model handles the character but misses
cars in deep shadow — and the reviewer cannot call those shapes confidently
either. Per the kamppi precedent (untrustworthy ground truth, not difficulty,
is the disqualifier) it could not be validation — but labelling proved the
ground readable, so it went to `split: train` as `positive`: keeping it
`sparse` would have exported its confirmed shadow cars as background and
taught the model to ignore them. Proximity caveat:
743 m from `r1-jatkasaari` (validation). Validation still has no `sparse` or
`hard-negative` area — the city-wide hold-out gap remains open, to be filled
with cleanly-imaged ground, not deep-shadow downtown.

Also on record: `bootstrap --weights <trained.pt>` silently proposes nothing
(trained class names never match its hardcoded DOTA `large vehicle` filter);
round-N candidate proposals are `detect → stage`, as the pipeline diagram
says. The bootstrap guard is unfixed.

**Van cells accepted, 2026-08-13.** Eight of the 19 distinct cells in
`data/mining/van-candidates-combined.yaml` (one exact duplicate dropped, one
cell per shared yard) joined train, aimed at the truck/van confusion:
`r4-tattarisuo`, `r4-tapulikaupunki`, `r4-konala`, `r4-laippatie`,
`r4-sepanmaki`, `r4-viikinranta`, `r4-hernesaari`, `r4-herttoniemi-n`.
Round-3 `detect` at conf 0.25 staged 2,554 candidates: 339 van, 197 truck,
7 bus, ~2,010 car — the mining's van counts reproduced almost exactly, so
the cells are the 4-8 m band ground the confusion needs. Proximity caveat:
`r4-hernesaari` sits 553 m from `r1-jatkasaari` (validation) — closer than
any previously accepted pair, and jatkasaari now has train ground on two
sides (`r4-kruununhaka` 743 m); its held-out read joins the optimistic-caveat
list. The cost: ~2,550 new candidates on top of kruununhaka's, the largest
review batch since round 1. Later the same day: `r4-sepanmaki` and
`r4-viikinranta` dropped before review — the reviewer judged their ground
repetitive of cells already labelled (mostly cars and vans); both label files
were untouched candidates, so nothing human was discarded. Six van cells
remain, ~2,050 candidates.

**Validation gains a negative, 2026-08-13.** `r4-mustavuori` (Mustavuori
forest, Vuosaari) added as `hard-negative`/`validation` — hand-picked from
imagery, 2.2 km from the nearest collection area (the round's first entry
with no proximity caveat). Pure canopy and rock outcrops: an easy negative
with no lookalikes, measuring "does the model fire on nothing" rather than
resistance to truck-shaped clutter, and the first held-out negative since
`r1-rastila` left validation — the negative-area check has held-out ground
again. Validation's missing `sparse` role remains open.

**Validation gains residential, 2026-08-13.** `r4-kapyla` (Käpylä apartment
rows) added as `sparse`/`validation` — hand-picked, 1,635 m from the nearest
train area (no caveat). Round-3 `detect` staged 105 candidates, mostly
courtyard cars; review is confirm-the-cars work, and the negative check
forgives only labelled vehicles. With it, validation holds all three roles
and the `aois` warning is clear. Still open: the second truck-dense
validation area (DESIGN §11.3).

**`r4-orakas` joins validation, 2026-08-13.** Small industrial estate in
Heikinlaakso housing — the ordinary small-yard character a city-wide sweep's
trucks mostly sit in, absent from validation until now. Found by ranking the
existing industrial/commercial/construction sweeps by truck detections:
**model-guided selection**, so its recall read leans optimistic by
construction — accepted anyway because the alternative truck-dense grounds
all duplicate Vuosaari harbour character already in train. Proximity caveat:
1,069 m from `r4-tattarisuo` (train). Staged 90 candidates: 19 truck,
14 van, 57 car.

**`r4-toukola` added, then moved to train, 2026-08-14.** Toukola by Hermannin
rantatie, hand-picked as a candidate replacement so `r4-orakas` could move to
train. Review settled it against that: 13 trucks against orakas's 21, at a
6.6 m length median against 10.0 m, and its *darkest* truck (107.9 mean
luminance) is brighter than orakas's *median* (110.8) — 1 of 13 in shadow
against 10 of 21. It cannot carry orakas's job, so orakas stays held out and
toukola joins train for its 212 cars and 41 vans of dense urban ground, which
also retires its 553 m proximity to `r1-kylasaari` (train). Validation returns
to seven areas. What it cost: 286 candidates reviewed to establish that the
swap does not work.

**Shadow mining, 2026-08-14.** The round-4 miss analysis found orakas's
failures are appearance, not class: 0 misnamed, 7 boxed nowhere at all, and
the missed trucks average 90.9 mean luminance against 140.6 for the found
ones. Ranking mining cells by truck-detection count finds the opposite of
what that needs — the top cell by count held 31 trucks and 1 dark one — so
205 industrial/commercial/construction cells were swept at conf 0.10 and
ranked by *shadowed* truck detections instead. Two accepted into train:
`r4-vuosaari-terminal` and `r4-vuosaari-ratapiha`. The method's known bias:
in harbour ground the model reads detached semi-trailers as trucks, so
trailer yards rank high on a count the human then rejects — one proposal was
59% the already-rejected Paulig cell.

**The bootstrap guard, 2026-08-14.** `bootstrap` now refuses weights whose
class names are not DOTA's `large vehicle`/`small vehicle`, naming what the
weights emit and pointing at `detect --weights`. The silent-zero recorded on
2026-08-13 had bitten a second time that morning, including a sweep over
`r1-vuosaari-rahtarinkatu` — 55 labelled trucks — that reported 0 candidates
and a clean exit. The check is a pure function on class names
(`sweep.emits_dota_vehicles`) so it is tested against the base install with no
torch. The `new-aoi` skill said `bootstrap` in its sequence, which is how an
agent reached for it twice; it now says `detect`, and both this trap and the
`stage --aoi` empty-file trap are written up under its failure modes.

**Round 4 passes all three gates, 2026-08-14.** Truck recall 0.902, precision
0.906, `r1-kaivoksela` count +1% — measured with `r4-orakas` still held out,
which is the claim worth having: the gates were earned on the hardest ground
in validation, not on validation with that ground removed. The lever was
ground, not tuning. Round 4 went from precision 0.845 (failing on two seeds,
0.845/0.849) to 0.906 at held recall after `r4-toukola`,
`r4-vuosaari-terminal` and `r4-vuosaari-ratapiha` added 42 trucks and 45
rejects — the rejects matter as much, since a rejected tractor or forestry
machine exports as background in exactly the window that held it. Caveat on
the pass: the operating confidence is 0.124, near "keep almost everything",
where round 3 cleared at 0.605. Negative-area detections were 21, above the
watch level.

**`r4-orakas` moves to train, 2026-08-14.** Done *after* the gates above were
measured, so a replay of round 4 will not reproduce them — the recorded
numbers belong to the collection as it stood, and this entry is the reason
they will not come back. The experiment behind the move
(`exp-orakas-in-train`): the gate numbers rise, but validation lost its
hardest area so most of that is the yardstick — the same weights gain
0.845 -> 0.895 precision from the removal alone. What survives the change of
yardstick is the case: the truck operating confidence moves 0.124 -> 0.772
(a curve-shape number, and round 3's real-move benchmark was 0.776),
negative-area detections halve 21 -> 10, and on `r1-jatkasaari` — held out in
both runs, same 16 trucks, both read at conf 0.25 — the misses drop from 4 to
1 and the remaining one is *bright* (147.2) where the four were dark (mean
96.7). Shadow detection generalised from ~10 examples, which contradicted the
prediction that it would not. Two trucks moved from found into misnamed, so
some failure shifted from detection to naming. The cost: validation's shadow
signal is now `r1-jatkasaari` alone — 16 trucks, 1 remaining miss — thin
enough to hide a regression, and the second shadow validation area remains
unfound. Automated mining could not locate one: ranking cells by dark truck
detections returns harbour trailer yards (the model reads standing trailers
as trucks), adding a rigid 8-12 m filter returns city-centre street shadow
already covered by `r1-kamppi`, and ranking by raw ground darkness returns
dark roofs and water. The signal wanted — paved yard in shadow — needs
building heights and sun geometry, not the detector's own output.
