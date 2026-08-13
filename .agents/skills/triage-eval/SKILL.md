---
name: triage-eval
description: Diagnose a failed (or suspiciously passing) eval gate — which diagnostic to run per gate, why relabelling lookalikes as rejects buys nothing, and when seed noise means the result should not be believed at all
type: prompt
whenToUse: When an eval run fails a gate, a gate result looks surprising, or the user asks why precision/recall/counts/negatives failed and what to do about it
---

# Triaging an eval result

`rekka-ai eval` verdicts rest on three gates (since the 2026-08-11 rebuild):
truck recall ≥ 0.90, truck precision ≥ 0.85, and per-area count error ≤ 10%
**only where the area holds ≥ 60 ground-truth trucks** — today exactly one
(`r1-kaivoksela`). The hard-negative check is *reported, not gated*: three
seeds produced 1/7/3 unexplained detections against a threshold of 2, and a
quantity noisier than its own threshold cannot decide a ship question.

## Step 0: is the number even believable?

Single-seed results carry **±2–3 pt noise** (measured 2026-08-11, seeds
0/1/2, identical everything else):

- Trustworthy at single-run resolution: truck recall (spread 0.001), truck
  mAP50 (0.010), truck precision (0.027 — smaller than round-to-round steps).
- **Not** trustworthy: per-area count error (20 pt of spread on jatkasaari),
  the negative-area count, van recall (8 pt), and the operating confidence
  (0.19–0.78 — the PR curve is flat near the recall gate, so the argmax is
  close to arbitrary).

Before diagnosing a failure — and before celebrating a pass — run a second
seed if the verdict hangs on a number in the second list. Also check the
operating confidence the eval picked: gates clearing at keep-almost-nothing
confidence (round 3: 0.776) is a real curve move; gates clearing only near
the 0.05 floor is "keep everything", not an operating point.

## Precision fail → read the most confident errors

Detect over the failing validation area **at the operating confidence the
eval picked, not 0.25**, and read the highest-confidence errors first:

```sh
uv run rekka-ai detect --aoi aois/helsinki.yaml --weights runs/train/roundN/weights/best.pt \
  --confidence <operating-conf> --out data/detections/triage-<area>.geojson
```

Round 1's finding: the *most confident* errors sat on cab-less truck-shaped
objects — containers, stored bodies, tipper skips, trailers.

**The counterintuitive rule an agent will get wrong:** relabelling those
lookalikes as rejects buys nothing. `export` writes labels only for
`confirmed`/`added` (`LABEL_STATUSES` in `export.py`), so a reject and an
omission are the *same background* — what training lacked was their
**windows**, not their labels. Round 2 proved it: three mined areas added
only 52 labelled instances but 27 windows of container/trailer background,
and precision moved 0.773 → 0.822 at the same recall. The improvement tracks
the windows.

Precisely: this is not a reason to stop rejecting during review — rejects
are the project's recorded hard negatives and keep the file a complete
record. It is a reason not to treat *more relabelling of existing ground* as
a precision fix. The fix is new ground containing the failing geography →
see the **mine-annotation-ground** skill.

## Recall fail → suspect the ground truth before the model

A missed truck costs hand-labelling, so recall gates first — but recall
"failures" have twice been label error. The round-3 ground-truth pass: run
the current weights over the *labelled* areas, match detections against
labels, and review what matches nothing. 82 such detections turned out
mostly real (+6 truck, +7 van, +78 car, +34 rejects) — the model was right
and the labels were wrong. A model is only as gated as its ground truth.
Only after the labels survive audit does a recall fail mean "add ground with
more of the missed thing".

## Count fail → check whether the area even gates

`count_gate` returns `None` below 60 truth trucks (`GATE_COUNT_MIN_TRUCKS`)
— reported, excluded from the verdict. Measured count noise is
~`0.77*sqrt(n)` boxes against a `0.1*n` tolerance; the curves cross at
n = 60, so below it the seed decides the verdict, not the model. A count
"failure" on a small area is noise by construction; a failure on
kaivoksela-scale ground is real. If it is real, look for count inflaters:
double boxes on one vehicle (cab + body, nose + trailer — the labelling
rules exist precisely because they inflate every count) and seam duplicates.

## Negative-area detections → use unexplained(), not raw counts

`evaluate.unexplained()` forgives every detection sitting at IoU > 0.3
(`NEGATIVE_MATCH_IOU`) onto *any* confirmed/added vehicle, class ignored on
purpose: a negative area is negative about targets, not empty —
r1-puotinharju holds 93 cars and 5 vans. The question is "did the model
invent a vehicle", not "did it name it correctly". Raw detection counts
marked a model down by an order of magnitude for being right. Above
`NEGATIVE_DETECTIONS_WATCH` (2) the report says so loudly and a human looks;
nothing fails on it. Note the sweep that feeds this check runs the pipeline's
`MIN_LENGTH_M` floor (4 m) on purpose — without it, round 2's negative-area
failures were dominated by sub-vehicle slivers of parked cars that no real
sweep would emit (2026-08-11).
