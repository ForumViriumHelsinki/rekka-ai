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
