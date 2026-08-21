# Model card — rekka-ai round 4 (z16)

Vehicle detection from Helsinki aerial orthophotos, as oriented bounding
boxes. This card describes the weights the project currently ships; the
development history is `docs/rounds.md`, the design rationale
`docs/DESIGN.md`, and the class definitions `docs/LABELLING.md`.

---

## 1. Identity

| | |
|---|---|
| Weights | `runs/train/round4/weights/best.pt` (118 MB, gitignored; published as `yolo11x-obb-fvh-z16-v4.0.0.pt` on release `model/v4.0.0`; see docs/releasing.md) |
| MLflow run | `round4` / `eval-round4`, experiment `rekka-ai` |
| Architecture | YOLO11x-OBB, 200 layers, 58,740,223 parameters, 203.1 GFLOPs |
| Fine-tuned from | `yolo11x-obb.pt` (Ultralytics, pretrained on DOTAv1) |
| Classes | `truck`, `bus`, `van`, `car` |
| Output | Oriented bounding boxes, EPSG:3879 |

**Every round starts from `yolo11x-obb.pt`.** Rounds are not chained — round 4
does not resume from round 3 — so the rounds differ only in training ground
and stay comparable.

## 2. Intended use

Counting and locating vehicles in **Helsinki** aerial orthophotos at **zoom
16 (12.5 cm/px)**, for city-scale census and siting analysis. The design
target is `truck`; the other three classes exist because they must be
distinguished from trucks, and their metrics are reported but never gate.

**Not intended for**: vehicle tracking, identification of individual vehicles
or plates, per-vehicle enforcement, or any decision about an identifiable
person or business. The imagery resolution does not support it and the
project has never evaluated for it.

## 3. Training configuration

| | |
|---|---|
| Zoom / GSD | z16, 12.5 cm/px |
| Window | 1024 px = 128 m, 30 m overlap |
| Epochs | 100 (patience 30, not triggered) |
| Batch | 4 |
| Image size | 1024 |
| Optimizer | `auto` → AdamW, lr0 0.00125 |
| Seed | 0, `deterministic: true` |
| Augmentation | mosaic 1.0 (closed last 10 epochs), scale 0.5, fliplr 0.5, degrees 0.0 |
| Hardware / time | RTX 5070 Ti 16 GB, **31 minutes**, ~10 GB peak |

`batch 4` is a deliberate default, not a tuning result: Ultralytics'
autobatch picks 1 on this card, which quadruples optimizer steps and takes
BatchNorm statistics from single images. Changing it changes the experiment.

Same seed + same data + same weights reproduces a run exactly. This was
verified on 2026-08-14: a scratchpad dataset and `data/dataset-r4` produced
identical gate numbers to four decimal places.

## 4. Training data

40 areas of 300 m, hand-reviewed in the project's own web tool.

| | |
|---|---|
| Areas | 34 train / 6 validation |
| Roles | 35 positive, 3 hard-negative, 2 sparse |
| Reviewed labels | 6,509 |
| Classes | 532 truck, 162 bus, 829 van, 4,420 car |
| Rejected candidates | 566 |
| Exported windows | 316 train / 54 val |
| Exported instances | 7,849 train / 1,384 val |

**Rejects are training data.** `export` writes a label only for
`confirmed`/`added`, so a rejected candidate becomes *background* in exactly
the window that held it. The 566 rejects — containers, detached trailers,
tipper skips, boats on cradles, tractors, forestry machines — are how the
model learned that cab-less truck-shaped objects are not trucks. Round 2
moved precision 0.773 → 0.822 on 52 new labelled instances and 27 windows of
container background; the improvement tracked the windows.

**The split is by whole area, never random.** Adjacent windows are
near-duplicates, so a random split would make every metric fiction.

**Source imagery:** Helsinki City Survey Services WMTS,
`Ortoilmakuva_2025_5cm` (2025 flight). © Helsingin kaupunki,
Kaupunkimittauspalvelut.

## 5. Evaluation

Validation is six whole areas held out from training: `r1-veturitie`,
`r1-pohjois-haaga`, `r1-kaivoksela`, `r1-jatkasaari`, `r4-mustavuori`
(hard-negative), `r4-kapyla` (sparse).

### Ship gates

| Gate | Threshold | Result |
|---|---|---|
| Truck recall | ≥ 0.90 | **0.911** PASS |
| Truck precision | ≥ 0.85 | **0.945** PASS |
| Count error, `r1-kaivoksela` | ≤ 10% | **+4%** PASS |

At operating confidence **0.772**, chosen from the PR curve. Negative-area
detections are reported, not gated: 10 unexplained.

### Per class, at the max-F1 point

| class | P | R | mAP50 | mAP50-95 | val instances |
|---|---|---|---|---|---|
| truck | 0.926 | 0.947 | 0.971 | 0.896 | 169 |
| bus | 0.939 | 0.943 | 0.976 | 0.934 | 105 |
| van | 0.928 | **0.783** | 0.888 | 0.817 | 138 |
| car | 0.946 | 0.962 | 0.967 | 0.882 | 972 |

### The same model at conf 0.25 — read this before running a sweep

The table above is at each class's best-F1 confidence. A production sweep at
a fixed low confidence is a different operating point, and much weaker.
From the confusion matrix (conf 0.25, IoU 0.45):

| class | recall @0.25 | **precision @0.25** | FPs on background |
|---|---|---|---|
| truck | 0.953 | **0.533** | 109 of 302 predictions (36%) |
| bus | 0.905 | 0.646 | 45 of 147 (31%) |
| van | 0.652 | **0.411** | 100 of 219 (46%) |
| car | 0.958 | 0.844 | 150 of 1103 (14%) |

**At conf 0.25 only about half of predicted trucks are trucks.** Quote
counts at the operating confidence the gates were measured at, or publish the
confidence distribution with them. In the 2025 city sweep, 3,729 trucks at
conf ≥ 0.25 becomes 2,442 at conf ≥ 0.772.

### Inference speed

28.5 ms per 1024 px window on an RTX 5070 Ti. A 208 km² city sweep is 314
cells / 31,400 windows: **22 minutes** with a warm tile cache, 47 minutes
cold.

## 6. Limitations

**Shadowed trucks are still missed.** The known, measured failure. In
`r4-orakas`, 5 of 21 labelled trucks get no box at all — normal dimensions
(6.8–11.5 m), all on dark ground (mean luminance 87 against 141 for found
trucks). Adding shadowed training ground recovered 2 of 7 and improved a
held-out area, but did not solve it. Expect under-counting in yards under
building or tree shadow.

**Van is the weak class, in both directions.** At conf 0.25, 16% of true vans
are called car, 12% truck, and 46% of van predictions are background false
positives. The 5–6 m van band sits between car (4.6–5.0 m) and truck
(8.0–11.3 m), and `LABELLING.md` states that between 6.5 and 8.0 m length
decides nothing. Van/truck confusion also means some predicted trucks are
vans.

**Helsinki only, 2025 imagery.** Trained on the 2025 5 cm flight, which
covers Helsinki (`ref=091`).


## 7. Reproducing

```sh
uv sync --extra train --extra detect
uv run rekka-ai export --aoi data/ablation/r1r2r3r4.yaml --out data/dataset-r4 --zoom 16
uv run rekka-ai train  --data data/dataset-r4/dataset.yaml --name round4
uv run rekka-ai eval   --weights runs/train/round4/weights/best.pt \
    --data data/dataset-r4/dataset.yaml --aoi data/ablation/r1r2r3r4.yaml --name eval-round4
```

`data/ablation/r1r2r3r4.yaml` is a prefix filter of `aois/helsinki.yaml`
(areas named `r1`..`r4`); regenerate it rather than hand-editing.

## 8. Alternatives considered

**z17 (6.25 cm/px)** was trained and evaluated for every round. At round 4 it
fails the gates — recall 0.900, precision 0.884, count +10% (fail) — for
**7.4× the training time** (3.81 h against 0.52 h; the ratio has been 7–14×
in every round). Its truck advantage decays as labelled data accumulates:
+14 pt precision at round 1, reversed by round 4.

**But "worse" is true of the gated classes, not of every class.** The gates
are truck-only, so the choice is made on trucks; mAP50 by class at round 4
(z16 / z17):

| truck | bus | van | car |
|---|---|---|---|
| **0.971** / 0.937 | 0.976 / **0.983** | **0.888** / 0.883 | 0.967 / **0.973** |

z17 is better at **car** (+0.7%, and it wins car in 3 of 4 rounds) and
**bus** (+0.7%), worse at truck (−3.5%) and van (−0.5%, but z16 wins van in
all four rounds, once by 13 points). The split tracks object size: a car is
38 px at z16 and 75 px at z17, so the smallest class gains most from
resolution, while van — the class decided by a shape judgment rather than a
size, per LABELLING.md — never benefits.

So: **z16 is the production model because this project gates on trucks.** A
car-focused task would read the same table the other way, and should weigh
+0.7% against 7–14× the training cost and 4× the sweep cost. Single seed per
cell, and the measured seed spread covers truck metrics only, so the
sub-percent car and bus margins are weaker evidence than the truck gap, which
is five times larger.

## 9. Licensing and attribution

The backend is **AGPL-3.0**, because it builds on Ultralytics YOLO
(AGPL-3.0); weights derived from `yolo11x-obb.pt` inherit that. The `web/`
labelling tool is MIT and communicates only through data files.

- Imagery: © Helsingin kaupunki, Kaupunkimittauspalvelut
- Administrative boundaries, street areas, parking: Helsinki open WFS
- Landuse used for AOI mining: © OpenStreetMap contributors
