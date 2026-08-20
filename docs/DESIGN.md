# rekka-ai — pipeline design

Status: the full loop is built — `rekka-ai fetch`, `aois`, `bootstrap`,
`stage`, `progress`, `export`, `train`, `eval`, `detect`, `mine`, plus the
`web/` labelling tool. The collection holds 27 areas (23 train /
4 validation). Three rounds have trained on it, and round 3 passes all
three ship gates (§7): truck recall 0.904, truck precision 0.884, the
kaivoksela count at +5%. The dated history — the earlier over-large
collection, the restructure into small plots, the per-round numbers — is
in `docs/rounds.md`.

## 1. What this is

Detect trucks in Helsinki aerial orthophotos and emit their locations as
georeferenced vector data.

**What it is not: traffic monitoring.** Orthophotos are flown roughly once a
year, in daylight, in leaf-on summer conditions. A run produces a *snapshot* of
where trucks were standing at one instant on one morning. Any question phrased
as "how much truck traffic is there" cannot be answered by this pipeline.
Questions it *can* answer:

- Where are trucks parked/staged across the city, and on what kind of land?
- How has the footprint of truck storage at a given site changed year over year?
- Which sites hold more vehicles than their permitted use suggests?

This constraint drives everything downstream, and should be settled with
whoever wants the output before model work starts. See §8.

## 2. Shape of the project

This is a model-training project, not a straight-through image pipeline. The
spine is a loop, run repeatedly, not a sequence run once:

```
                    ┌─────────────────────────────────────┐
                    │                                     │
  fetch  ──>  bootstrap  ──>  label  ──>  train  ──>  evaluate
   (z16)      (zero-shot     (correct    (fine-tune   (held-out
              pre-labels)    in web/)     at z16)      blocks)
                    │                                     │
                    └──── mine hard cases, relabel ───────┘
```

Nobody labels from scratch. A pretrained model pre-annotates, a human corrects,
and each round the improved model proposes the next batch — concentrating human
attention where the model is uncertain or wrong. Correcting pre-annotations is
several times faster than drawing boxes on blank imagery, and it is the whole
reason the first pass uses an off-the-shelf detector.

### Two decisions that are expensive to reverse

**Labels live in geographic coordinates.** Store them as GeoJSON polygons and
convert to YOLO-OBB pixel coordinates at dataset-export time. Pixel labels are
welded to one zoom, window size, and tiling scheme; change any of those — and
this project will — and they are worthless. Geographic labels survive every
re-tiling decision made later.

**Those coordinates are EPSG:3879, not WGS84.** RFC 7946 mandates WGS84, so
this is a deliberate deviation, declared by a `crs` member (`geo.crs_member`).
The reason is that nothing else here works in degrees: the detector emits grid
metres, the tile grid is grid metres, the labelling tool's map is grid metres,
and every measurement is metres. Storing degrees put a reprojection on both
sides of the file, and reprojection is not exactly reversible — so an operator
who opened an area and changed one box's class got a diff touching every
coordinate in the file, because the save re-derived what the load had just
converted. In the grid CRS, reading and writing move nothing: OpenLayers skips
the transform when the projections match, both writers round to
`geo.COORD_DECIMALS`, and an untouched box round-trips as identical bytes.

What it costs: GDAL honours the `crs` member, so QGIS and `ogr2ogr` read these
files correctly (verified — including the axis order, which EPSG:3879 declares
as north-east while the files are written east-north like every other GeoJSON).
Anything that assumes RFC 7946 without looking will place the boxes off the
coast of Africa; GitHub's inline GeoJSON preview is the one that actually does.
Hand geometry to such a tool via `ogr2ogr -t_srs EPSG:4326`.

**Train/val/test split by geographic block, never at random.** Adjacent tiles
are near-duplicates, so a random split puts the same parking lot in both train
and validation, and the reported metrics become fiction. Carve the AOI into
spatial blocks and assign whole blocks to a split.

The corollary for the repo: **labels are the only irreplaceable artifact.**
Tiles refetch, datasets regenerate, models retrain — labels are human hours.
So labels are version-controlled; caches, datasets, weights, and run outputs are
all gitignored and reproducible.

## 3. Imagery source (verified)

Helsinki City Survey Services WMTS. Confirmed against the live service on
2026-08-04 and exercised end to end by `rekka-ai fetch`.

| | |
|---|---|
| Endpoint | `https://kartta.hel.fi/ws/geoserver/avoindata/gwc/service/wmts` |
| Protocol | WMTS 1.0.0, KVP (`REQUEST=GetTile`) |
| TileMatrixSet | `ETRS-GK25` → **EPSG:3879** (ETRS89 / GK25FIN) |
| Tile size | 256 × 256, `image/jpeg` |
| Grid origin (top-left) | N = 8388608.0, E = 24451424.0 |
| Fees / access constraints | `NONE` (per GetCapabilities) |

Zoom is a plain power-of-two ladder; `z17` is the deepest. Tile indices from a
projected coordinate, where `span = resolution × 256`:

```
col = floor((E - 24451424.0) / span)
row = floor((8388608.0 -  N) / span)
```

Two traps, both encoded as tests:

- EPSG:3879 declares axis order **(north, east)**, so `TopLeftCorner` reads
  `northing easting`. Swapping them returns imagery from the wrong place
  without erroring.
- The grid origin is top-left, so a tile's *inclusive* corner is its
  north-west one. AOI upper edges are treated as exclusive, so an AOI aligned
  exactly to tile boundaries does not drag in a spurious extra row and column.

### Three CRSs, kept separate

EPSG:3879 is used internally for tile arithmetic, not by preference but because
it is what the grid uses. Nothing outside `imagery/` should assume it.

| Role | CRS | |
|---|---|---|
| Config / input | **EPSG:3067** | ETRS-TM35FIN, the Finnish national standard |
| Tile arithmetic | EPSG:3879 | ETRS89 / GK25FIN, forced by the tile grid |
| Stored geometry | EPSG:3879 | labels and candidates, via a `crs` member |
| Hand-typed bboxes | WGS84 | what a person copies off a map |

Three consequences, all of which cost real correctness if ignored:

- **EPSG:3067 declares (east, north) but EPSG:3879 declares (north, east).**
  Every transform uses `always_xy=True` so both normalise; without it exactly
  one of the two comes back swapped.
- **A box in one CRS is not axis-aligned in the other.** The central meridians
  differ (27°E vs 25°E), so a 2 km EPSG:3067 square arrives in EPSG:3879
  rotated by ~1.74°, and its envelope grows to ~2060 m. All four corners are
  projected, not just min/max — projecting the diagonal alone under-covers by
  ~120 m, roughly four z16 tiles missing from each edge.
- **pyproj returns infinities rather than raising** for coordinates outside a
  CRS's domain. Reading projected metres as WGS84 degrees therefore yields
  `inf` bounds and a `nan`-wide AOI that surfaces much later, so `Bounds`
  rejects non-finite values with a message naming the likely cause.

### Layers

RGB orthophotos, one per flight year: `Ortoilmakuva_2014` … `Ortoilmakuva_2025_5cm`.
Recent years are 5 cm native (2021, 2023, 2024, 2025); older ones 8–20 cm.
**There is no 2022 flight.** Naming is irregular, so `imagery/layers.py` maps
year → layer explicitly rather than deriving it.

Present but unused: `Vaaravariortoilmakuva_*` (false-colour IR, most years —
could cheaply mask vegetation if canopy false positives appear) and
`TosiOrtoilmakuva_2017_8cm` (the only *true* ortho; standard layers let tall
structures lean and occlude vehicles).

## 4. Resolution strategy

What a detector cares about is object size in **pixels**, not GSD:

| Zoom | GSD | Semi-trailer | Car | 1024 px window covers |
|---|---|---|---|---|
| z15 | 25 cm | 66 px | 18 px | 256 m |
| **z16** | **12.5 cm** | **132 px** | **36 px** | **128 m** |
| z17 | 6.25 cm | 264 px | 72 px | 64 m |

Sampled tiles confirm the imagery easily supports this: at 12.5 cm/px
individual cars, boat hulls and rail wagons are all resolved.

### Which zoom to bootstrap at — measured, not reasoned

The obvious guess was z15: Ultralytics OBB weights are pretrained on DOTAv1 at
`imgsz=1024`, `large-vehicle` instances there are roughly 40–70 px, and z15
puts a semi-trailer at 66 px — apparently right on the pretrain distribution,
for a quarter of z16's tiles.

**That guess is wrong.** Zero-shot `yolo11x-obb` over the `r1-tattariharjuntie` area:

| Zoom | conf 0.25 | conf 0.10 | After the 6 m gate |
|---|---|---|---|
| z15 | 7 | 8 | 7 |
| **z16** | **41** | 47 | **22** |
| z17 | 47 | — | — |

z16 finds three times as many real trucks. Matching the pretrain *GSD* matters
less than giving the detector enough pixels on the object; at 66 px a truck is
simply a small object, and small-object recall is where detectors are weakest.
z17 adds little over z16 for four times the tiles.

**So: bootstrap, train, and run all at z16.** The earlier plan to sweep cheaply
at z15 and fetch z16 only where candidates were found is dropped — it would
have missed two thirds of the trucks.

z17 remains useful for annotation and error analysis, where the extra detail
helps a human adjudicate a marginal case.

### First full bootstrap run

`--role positive`, 14 areas, z16, conf 0.25, 6 m gate: **286 candidates**.

| Area | n | median len | ≥12 m | median conf |
|---|---|---|---|---|
| r1-vuosaari-rahtarinkatu | 64 | 12.4 m | 33 | 0.73 |
| r1-vuosaari-channel-road | 45 | 17.0 m | 41 | 0.84 |
| r1-tattariharju | 22 | 10.9 m | 5 | 0.73 |
| r1-tattariharjuntie | 22 | 7.6 m | 2 | 0.80 |
| r1-malmi-airport | 20 | 8.7 m | 0 | 0.78 |
| r1-kamppi | 18 | 12.3 m | 10 | 0.67 |
| r1-valimo, r1-kylasaari | 17 | ~10 m | 1–4 | 0.59–0.76 |
| r1-kivikko | 16 | 8.2 m | 6 | 0.76 |
| r1-myllypuro | 13 | 9.3 m | 0 | 0.82 |
| r1-hermanni | 10 | 9.6 m | 1 | 0.75 |
| r1-patola | 8 | 7.2 m | 0 | 0.61 |
| r1-pohjois-haaga, r1-siltamaki | 7 | ~7 m | 0–2 | 0.60–0.77 |

The two harbour areas hold 38% of all candidates. That looked like container
false positives — a 40 ft container is 12.2 m, almost exactly `r1-vuosaari-rahtarinkatu`'s
median — but inspecting the imagery shows otherwise: the boxes sit on real
trailer rigs in the RoRo staging lanes and the harbour-gate yard, and the
stacked containers alongside them were **not** detected. Those areas are simply
the truck-densest in the collection.

Precision looks usable; **recall is the weak side**, with visible trailers left
unboxed in every area. So the first correction round is add-heavy, not just
delete-heavy — which is the expected shape of zero-shot transfer, and the
reason for fine-tuning.

### The negative areas produce almost nothing — and that is the finding

Same settings, over the non-positive areas:

| Area | Role | Candidates | With the gate off |
|---|---|---|---|
| `vuosaari` | hard-negative | 1 | 1 |
| `r1-puotinharju` | sparse | 0 | 0 |

Both zeros are real, not a silent failure: `r1-puotinharju` yields 9 windows of
genuine imagery (luminance 20–255) and 0 detections even at confidence 0.05,
while the same model on a `r1-tattariharjuntie` window returns 9.

**The premise behind the hard negative does not hold.** `vuosaari` was chosen
because containers "look identical to trailers from above". The AOI is packed
with hundreds of containers in exactly that shape and alignment, and the
zero-shot model found one 6.6 m vehicle by a crane. It does not make the
mistake the area exists to correct. `r1-puotinharju` likewise contains a car park
full of clearly visible cars and produced nothing — a measurement from before
bootstrap kept DOTA's `small vehicle` class (§5); re-running it now would find
those cars, which is the point of that change.

Consequences:

- **Recall, not precision, is the problem.** Effort belongs in the trucks the
  model misses, not in suppressing false positives it does not generate.
- The §9 worry that validation cannot see a hard negative is much less urgent
  than it looked: there is little to measure.
- Keep both areas anyway, and **re-run this check after fine-tuning**. Training
  on truck-shaped objects is exactly what could start pulling containers in, so
  these areas are cheap insurance against a regression rather than dead weight.

The prediction held — the dated story is in `docs/rounds.md`. The present
state: the `hard-negative` role is held by `r1-marjaniemi` (a marina — boat
hulls on cradles are the most truck-like presentation in the collection) and
`r2-vuosaari-harbour-road` (a container terminal), both in train, so
validation has no `hard-negative` area for the regression check to measure
against (§11).

### The length gate is a noise floor, not a class exclusion

At z16, 19 of 41 raw candidates in `r1-tattariharjuntie` are under 6 m —
**vans and small delivery vehicles**, not noise; `large vehicle` is loose at
the van end of the range, and vans only appear where vans are. Cars used to
be excluded here too, by never being detected at all: bootstrap kept only
DOTA's `large vehicle` class, so `r1-puotinharju`'s car park produced nothing
regardless of length. Now that `car` is an annotated class (§5), bootstrap
also keeps DOTA's `small vehicle` class, and the length gate's only job is
filtering raw noise below real-vehicle size — measured at 4 m
(`--min-length`, default `MIN_LENGTH_M`), so a human is not asked to reject
the same sub-4 m noise in every area.

A bobtail tractor unit can fall under the 6 m truck/van boundary described in
the annotation guide below; that is a known limit of that judgement call, not
of the filter.

## 5. Model

**Oriented bounding boxes (OBB), not axis-aligned.** Vehicles are elongated and
arbitrarily rotated; an axis-aligned box around a diagonal semi-trailer is
mostly asphalt, which hurts both NMS and any downstream length/heading estimate.

**Bootstrap: Ultralytics OBB pretrained on DOTAv1.** DOTA-v1.0's 15 classes
include `large vehicle` and `small vehicle`, so the zero-shot pass maps onto
this problem essentially out of the box. The newer `yolo26*-obb` weights follow
the same DOTAv1/1024 recipe, and were measured as an alternative on 2026-08-04
(`r1-tattariharjuntie`, z16, conf 0.25, 6 m gate): 17 candidates against `yolo11x-obb`'s 22,
and 16 of the 17 are boxes 11x also finds. The differences are one marginal
26x-only box (conf 0.34) against six 11x-only boxes, three of them at conf
0.77–0.81 — a quarter of the recall gone for nothing gained. Recall is the
metric bootstrap lives by, so **`yolo11x-obb` stays the default**; a one-area
sample, but the gap is too lopsided to chase further before fine-tuning.

**Then fine-tune on corrected Helsinki labels at z16.** The domain is narrow —
one city, one sensor, one season — so a few hundred well-chosen annotated
windows plausibly suffice. Do not ship the zero-shot model; it exists to
generate the first round of pre-annotations and to calibrate difficulty.

### Annotation guide

This section is the *why*; `docs/LABELLING.md` is the one-page card a labeller
actually works from, and per-area guidance lives in the AOI `notes`. Change a
rule here and change it there in the same commit.

Consolidated from the field-survey notes in `aois/helsinki.yaml` and the
client's specification. Two gates decide almost every case:

1. **A visible cab.** The unit must be a road vehicle, not a detached load.
2. **A separated cab with a load body behind it.** This is the client's own
   definition and it is what separates `truck` from `van`: a truck — including
   the small ones — has a cab that *stops* and a load body that *starts*,
   boxy at the rear, open or closed. A van is one continuous shell from
   windscreen to rear doors. The break between cab and body is the thing to
   look for from above.

**Length is a sanity check, not the rule.** Recorded 2026-08-11 because the
guide said "length ≥ 6 m" for three rounds and the labels never agreed with
it: 88% of what was labelled between 6.0 and 6.5 m is a `van`. Measured over
2,990 labels, length only decides at the ends — below 6.5 m a separated body
is rare, and at 8 m and over it is 293 trucks against 1 van. **Between 6.5 and
8.0 m length decides nothing** (48 van, 77 truck) and the shape is the whole
answer. That band is 4% of the dataset, it is where `van` recall has been
stuck at 0.73 for two rounds, and it is where the model's most confident
class errors live — a model cannot learn a boundary the labels do not draw.

| Keep (as `truck`) | Exclude |
|---|---|
| Box trucks with a visible cab | Bare platforms / swap bodies (*lavat*) with no cab |
| Straight trucks (*kuorma-autot*) | Full trailers (*perävaunut*) — coupled or parked alone |
| Semi-trailer rigs, incl. timber loads | Shipping containers |
| Concrete mixers | Long vans (their own class — see §5 Classes) |
| Platforms **with** a cab attached | |

**Known-hard cases**, recorded so they are decided once rather than per
annotator:

- *Van-fronted motorhomes and caravans* — **decided 2026-08-11: reject them.**
  Earlier guidance was to "avoid annotating them either way", which is not a
  verdict a label file can hold: every candidate needs one before `export`
  will run. They are the hardest confuser and they are not confined to
  `r1-rastila` — the round-1 reject audit found white 6–8 m RVs parked in
  ordinary industrial car parks in `r1-kivikko`, `r1-tattariharjuntie`,
  `r1-kylasaari` and `r1-malmi-airport`, each one a plausible van at a
  glance.
- *Bobtail tractor units* (no trailer) — may fall under the 6 m gate, so the
  gate alone does not settle them.
- *Truck + full trailer combinations* (the common Finnish rig, and most of
  `r1-kaivoksela`) — **box the truck unit only; the trailer is a negative,
  coupled or not.** The cab gate decides it, and this keeps a trailer's
  verdict from depending on whether something is hitched to it. A
  semi-trailer rig is the opposite case and stays one box: the trailer is
  carried by the tractor and cannot stand alone. Confirmed from the labels
  on 2026-08-11 — all 106 trucks in `r1-kaivoksela` measure 6.1–12.5 m,
  so no combination has ever been boxed whole, while the harbour areas'
  16–18 m boxes are single semi rigs. The cost is real and belongs in the
  error budget: the model must separate a tipper body with a cab from an
  identical body 3 m behind it without one.
- *Containers from above* look identical to trailers. This is why the Vuosaari
  container terminal is a `hard-negative` area rather than simply unlabelled.
- *Pickups* — used to simply fall out below the length gate; now that cars are
  detected too, a pickup shows up as a candidate and needs a call.
  **Decided 2026-08-11, on the real examples the first 4-class round
  surfaced: `car`.** They sit in the car length band and are used like cars;
  calling them trucks would put passenger vehicles inside the one class the
  ship gates measure.

### Classes

Four annotated classes:

| Class | |
|---|---|
| `truck` | Everything in the Keep column above |
| `bus` | Buses and coaches |
| `van` | Vans and small delivery vehicles |
| `car` | Sedans and other small passenger vehicles |

`van` and `car` are **classes, not exclusions**. The detector fires on both,
so labelling them explicitly beats leaving them as unlabelled background the
model has to guess about — and it keeps the truck/large-vehicle boundary a
reporting decision rather than one baked into the data.

This changes the length gate for labelling: the 6 m rule was cutting off a
spike of 94 candidates in the 5–6 m band, which is exactly the van population.
The labelling floor is 4.0 m — the `MIN_LENGTH_M` default — measured at 392
candidates versus 286 at 6 m; the 9 detections below 4 m are noise. `car`
reuses the same 4 m floor — real cars clear it naturally, and it was already
the measured noise line, not a value chosen for this class specifically.

`car` was initially scoped as a separate model, but DOTA's pretrain already
carries a `small vehicle` class alongside `large vehicle` — bootstrap was
filtering it out on purpose (see "length gate" above), not lacking the
capability. One model with a 4th class reuses the existing pipeline end to
end rather than standing up a second one, at the cost of a full retrain from
round 1: a class-count change means a new model head, so weights from the
3-class rounds cannot keep being fine-tuned.

**Buses are annotated separately, not merged into `truck` and not skipped.**
Drawing the box is the expensive part, and the box gets drawn either way — so
labelling its class costs almost nothing now and keeps both readings available
later: report trucks alone, or trucks and buses together as large vehicles.
Merging them now would be irreversible without relabelling, and dropping them
would waste the two bus-depot areas already surveyed.

This also gives the model an explicit label for the single most convincing
truck confuser, rather than leaving buses as unlabelled background in
`positive` areas — which would actively teach it that bus-shaped objects are
negatives in some places and unmarked in others.

The bus areas therefore stay `role: positive`: they hold targets, just of the
`bus` class. `r1-ruskeasuo` (depot) and `r1-kamppi` (station kerb) are in
train; `r1-veturitie` (Pohjolan Liikenne's yard in Pohjois-Pasila) is the
held-out one, having taken that job from `r1-kamppi` on 2026-08-11 (§7).

Its honest caveat: `r1-veturitie` sits 1,196 m from `r1-ruskeasuo` in train
and shares its character — same operator, same fleet, same yard geometry —
so held-out bus recall reads optimistically. Helsinki offers no bus ground
of a *different* character (the alternatives were a kerb, whose imagery
cannot be labelled reliably, and a corridor with three instances), so the
choice was an optimistic measurement or none at all. Taken knowingly, and
survivable only because `bus` never gates: it is reported, and no ship
decision rests on it.

### Licensing

Ultralytics is **AGPL-3.0**, weights included — so the Python backend is
AGPL-3.0 too, network-use clause and all. For a public-sector organisation
publishing open source that is acceptable, and it was accepted: the root
`LICENSE` is AGPL-3.0 and `pyproject.toml` declares it. The labelling web
app never touches Ultralytics code and talks to the backend only through
data files, so it ships as a separate work under MIT (`web/LICENSE`) and can
be reused in other projects. Keeping that boundary — data files and HTTP
between the two, never code imports — is what keeps the split defensible.
MMRotate (Apache-2.0) remains the usual escape hatch for OBB if the AGPL
ever becomes a problem.

## 6. Inference pipeline

```
AOI (GeoJSON / bbox)
  └─> tile enumeration      z16 tile indices covering AOI
      └─> fetch + cache     on-disk, keyed by (layer, z, col, row)
          └─> windowing     ~1024 px windows, overlapping
              └─> inference batched OBB detector
                  └─> merge global NMS across window seams
                      └─> georeference px -> EPSG:3879
                          └─> output GeoJSON / GeoPackage
```

- **Cache.** Tiles for a past flight year never change, so the cache needs no
  invalidation and re-runs are free. That is what makes model iteration
  tolerable, so it is not an optimisation to add later.
- **Windowing.** A 256 px tile is too small for a detector and cuts trucks in
  half. Stitch tiles into ~1024 px windows overlapping by more than the longest
  expected object (≥ 160 px at z16) so every truck appears whole in at least one
  window.
- **Merge.** Overlap guarantees duplicates. Global NMS over the union of
  detections in *projected* coordinates — not per window — removes them.
  Class-agnostic: one vehicle read as two classes (the 7–8 m van/truck
  boundary is the usual case) is still one vehicle; the more confident reading
  wins. Candidate pairs come from an STRtree rather than a full pairwise scan,
  so the cost grows with actual neighbours, and a city-wide sweep stays
  feasible.
- **Georeference.** Window pixel → EPSG:3879 is exact and analytic from §3; no
  warping, and no reprojection at output either — that is what §2 buys.

## 7. CLI surface

Commands map onto the loop in §2. Every command below exists.

```
rekka-ai aois       --aoi <file> [--zoom]         # list a collection, report overlaps
rekka-ai fetch      --aoi <file|bbox> [--name] [--crs] [--year] [--zoom] [--dry-run]
rekka-ai bootstrap  --aoi <file|bbox> --out candidates.geojson
                    [--name] [--role] [--zoom] [--weights] [--confidence] [--min-length]
rekka-ai stage      --candidates <file> [--labels labels/] [--force] [--aoi <file>]
rekka-ai progress   [--labels labels/]
rekka-ai export     --aoi <file> [--out data/dataset] [--labels labels/]
                    [--name] [--role] [--year] [--zoom]
rekka-ai train      [--data data/dataset/dataset.yaml] [--weights] [--epochs]
                    [--batch] [--device] [--name]

rekka-ai eval       --weights <path> [--data] [--aoi <file>] [--min-recall]
                    [--min-confidence]
rekka-ai detect     --aoi <file> --weights <path> --out detections.geojson
                    [--confidence] [--name] [--role] [--crs] [--min-length]
rekka-ai mine       --weights <path> --operating-confidence <float>
                    [--existing aois/helsinki.yaml] [--out data/mining/round2.yaml]
                    [--municipality Helsinki] [--profile industrial]
                    [--count 12] [--pool-size 100] [--dry-run] [--refresh-osm]
```

`stage` splits one candidate file into per-AOI label files. It **refuses to
overwrite an existing file** unless `--force`, and reports how many reviewed
features it would have destroyed: re-running the detector must never quietly
discard human correction. That refusal is the single most important safety
property in the pipeline. `--aoi` also stages an empty label file for every
collection entry that produced no candidates — quiet mined cells still need a
reviewed file before export will accept them as background.

`export` turns the reviewed files into a YOLO-OBB dataset (`images/train`,
`images/val`, `labels/…`, `dataset.yaml`). It is the one place geographic
labels become pixel labels, so its rules are strict:

- **Split by whole AOI** from the collection's `split` field — never random,
  so adjacent near-duplicate windows cannot straddle train and validation.
- **Full containment:** a box is written only into windows that hold all four
  corners. The 30 m overlap exceeds the longest rig, so every vehicle is whole
  in at least one window; boxes in the overlap are written twice, which skews
  instance counts mildly (≈1.6× in the first export) but never produces a
  clamped, distorted label.
- **`rejected` is not a label.** Those windows export anyway, so a rejected
  container or van-fronted lookalike teaches "background" rather than
  vanishing. `hard-negative` and `sparse` areas export as pure background
  windows; their reviewed label files record the non-target vehicles the
  ground really holds — what the negative-area check forgives detections
  against — and a missing file for these roles is tolerated.
- **Refuses to run dirty:** any unreviewed candidate, schema problem
  (`labels.validate`), or box whose centre has drifted outside its area
  (`labels.displaced`) blocks the export — pixel labels are baked to a zoom
  and tiling, so errors baked with them are expensive to find later.

`train` fine-tunes the bootstrap weights on the exported dataset — Ultralytics
owns the loop, MLflow owns the record (see Experiment tracking). Defaults are
the dataset's own 1024 px windows and batch 4; autobatch (`--batch -1`) sizes
to free memory but measured badly here, so it is deliberately not the default.
On a small or display-shared GPU, pass `--batch 2`
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by the command,
which helps but is not magic). Needs the `train` extra.

`progress` reports reviewed/total per area, class tallies, a per-area
rejection rate (`rejected / (confirmed + rejected)`, `added` excluded since a
hand-drawn miss is a recall signal, not the model over-triggering; areas with
nothing judged yet sort last as `n/a`), and any schema problems that would
block training — a confirmed feature with no class, a ring that is not a
rectangle. Sorted worst-rate first, so the areas most worth re-sweeping or
deprioritising surface at a glance.

`bootstrap` needs the optional `detect` extra (`uv sync --extra detect`), which
pulls in torch. It is optional so that `fetch` and `aois` — and CI — stay usable
without a multi-gigabyte install. `bootstrap` fetches any tiles it needs, so
running `fetch` first is an optimisation rather than a requirement.

`--name` runs one area, `--role` runs a whole class of them
(`--role positive` for the labelling batch). Worth running over the
`hard-negative` areas too, and separately: what the model finds in the
container terminal is precisely the false-positive set worth recording, and
keeping it in its own file stops those candidates being corrected as if they
were trucks.

### Experiment tracking

Training runs are tracked with **MLflow**, backed by a SQLite file at
`runs/mlflow.db` with artifacts under `runs/mlartifacts/` — inside `runs/`,
so gitignored like every other run output: experiment history is valuable but
not version-controllable. `src/rekka_ai/track.py` is the single place that
configures this; an explicit `MLFLOW_TRACKING_URI` overrides the file
default, so moving to a shared server later is a shell setting. Ultralytics
logs params, per-epoch metrics, and the final weights into the run
automatically once tracking is set up. Inspect runs with:

```sh
uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
```

### What is version-controlled, and what travels another way

There is no shared runner. Labelling happens on one machine through `web/`,
training on another with whatever GPU is to hand, so anything the two people
both need has to travel through the repo — and the temptation is to commit
whatever is currently inconvenient. Two things were considered and left out:

- **Candidate files** (`data/candidates/roundN.geojson`). Already in the repo,
  under another name: candidates *are* the label files before review, so the
  staging commit pins round one's proposals permanently as the initial state
  of `labels/`. `git show <staging-sha>:labels/r1-kamppi.geojson` is what the
  detector proposed for Kamppi. Committing the candidates file would duplicate
  committed data, and it regenerates byte-identically anyway (§2). The
  labeller never touches it; `stage` has already split it for them.
- **The MLflow store** (`runs/mlflow.db`, `runs/mlartifacts/`). A SQLite file
  in git is a binary blob that conflicts on concurrent writes and never
  shrinks, and the artifacts include weights.

What the second machine actually needs is narrower than either:

- **Trained weights**, so the labeller can run `detect` for the next round.
  ~110 MB, regenerable only with the GPU that made them — a **release asset on
  a tag**, not git and not LFS.
- **The decision**, not the store: which weights, which gates passed, what the
  numbers were. `eval` already computes exactly this and prints it before
  handing it to MLflow, so a few KB of committed record per round would make
  experiment history reviewable in a diff while MLflow stays local and
  disposable. Not built — see the open question below.

The tile cache stays ignored and unshared: the labelling tool streams tiles
from the WMTS directly, so only training and `detect` ever need it.

### Evaluation

`eval --weights <path>` answers one question: is this model good enough to
sweep the city unsupervised? Two measurement layers, because they answer
different questions:

1. **Standard metrics — for comparing runs.** Ultralytics `val` over the
   exported validation split: per-class mAP50, precision, recall. Training
   already logs these per epoch to MLflow; `eval` re-measures a weights file
   standalone, so a model can be judged without knowing its training run.
2. **Operational gates — for the ship decision.** The product is "which sites
   hold how many vehicles", not box-perfect geometry, so the gates are:
   - truck recall ≥ **0.90** and precision ≥ **0.85** at the operating
     confidence — recall first, because a missed truck costs hand-labelling,
     a false box costs a glance;
   - per-area truck **count error ≤ 10%**, for areas holding at least
     `GATE_COUNT_MIN_TRUCKS` = **60** trucks. That floor is measured, not
     derived: three seeds of one dataset moved `r1-jatkasaari`'s count error by
     20 points (15 trucks) and `r1-kaivoksela`'s by 6.5 (107), so noise runs at
     about `0.77*sqrt(n)` boxes against `0.1*n` of tolerance, and those cross at
     n = 60. **One area in the collection clears it today**, which is a fact
     about the validation split rather than about the rule;
   - negative-role areas: **reported, not gated** since 2026-08-11. The check
     counts *unexplained* detections — ones matching no vehicle the area really
     holds, since a negative area is negative about **targets**, not empty
     (`r1-puotinharju` has 107 confirmed cars). Class is ignored in the match:
     the question is whether the model invented a vehicle, not whether it named
     it right. Rejected boxes forgive nothing. It was a gate at ≤ 2 until three
     seeds of one dataset produced **1, 7 and 3** — a quantity noisier than its
     own threshold cannot decide a ship question. It now prints under "reported,
     not gated", loudly above a watch level of 2. Restore it when validation has
     a held-out negative area large enough to resolve it (§11.3),
   - `bus`, `van` and `car` are reported but never gate: they are auxiliary
     classes, and the truck/van boundary is genuinely ambiguous at the short
     end.

   Both operational gates sweep with the pipeline's `MIN_LENGTH_M` floor, so
   they count what `detect` would emit rather than every raw proposal — the
   standard metrics above keep the raw model. Without that, sixteen of round
   2's twenty negative-area failures were 2–4 m slivers of parked cars that no
   sweep would ever report (2026-08-11).

   The operating confidence is chosen from the PR curve, not left at the 0.25
   default. Whatever is chosen is logged to MLflow; `detect`'s default
   confidence follows it by hand (currently 0.77, from round 4's 0.772 —
   re-check against each round's eval), so the number eval reports is the
   number a sweep ships at. It had lagged at 0.15 — a round-1 value — through
   rounds 3 and 4, both of which operate near 0.77: measured over the
   125,882-detection city sweep, the two thresholds disagree by **34% of the
   truck census** (3,729 trucks at conf 0.25 against 2,442 at 0.772), which is
   the size of the error a stale default hides (2026-08-16).

   That default is a **census** threshold, and `detect` has a second job the
   census number does not fit: proposing candidates for review. Staging wants
   the near-misses in front of the reviewer — the trucks the operating point
   discards are exactly the ones the next round has to fix — so the round loop
   passes an explicit low `--confidence` (0.25 through round 4) rather than
   taking the default.
   The search never picks below `--min-confidence` (default 0.05, even as
   the F1 fallback): a curve whose recall only clears the gate at conf≈0 has
   found "keep every raw proposal," not an operating point, so the recall
   gate fails by construction instead of silently passing at a useless
   confidence.

Caveat from the round-1 export: 27 validation windows, ~330 instances — one
box is several points of recall. The gates read trends, not decimals; widen
the validation split before trusting them with a big decision.

**An area too small to gate must not gate.** A 10% count error over an area
holding 8 trucks is 0.8 of a truck: the area passes or fails on a single
box, which measures the imagery rather than the model. `count_gate` therefore
skips any area below `GATE_COUNT_MIN_TRUCKS` — the measured floor of 60
above, where the 10% tolerance first exceeds the seed noise — and reports its
count without a verdict. The no-trucks case goes the same way: a bus yard
holding no trucks says nothing about truck counting either. That leaves no
blind spot, because a false positive
anywhere in the split still lands on the precision gate, which is measured
over every validation window.

This rule is why `r1-kamppi` moved to train on 2026-08-11. It was the bus
validation site, but downtown Helsinki is the collection's worst imagery —
building shadow, building lean, and mosaic seams that smear moving vehicles
— and it carried
only 8 of validation's trucks against 16 of its buses. Two things followed
from that mix: the truck gates barely saw it, and its unreadable-but-real
vehicles, which a labeller can only reject, scored as false positives
against a model that was right. Hard imagery belongs in validation; hard
imagery whose *ground truth* is a guess does not — it belongs in train,
where a wrong class is one noisy example among many rather than a verdict.

`r1-veturitie` took over the held-out bus job the same day, on the opposite
principle: it is the cleanest imagery in the collection, so its ground truth
can be trusted, which is what a held-out area needs most. It holds 56 buses
and one truck, so the floor above is what keeps it from distorting the
count gate — the two changes are one decision.

Validation is now four areas: `r1-kaivoksela`, `r1-jatkasaari`,
`r1-pohjois-haaga`, `r1-veturitie`. It is thinner than that sounds — at the
finished round `r1-kaivoksela` alone holds 107 of validation's 123 trucks
and `r1-pohjois-haaga` holds none, so the truck gates rest almost
entirely on one area. `r1-veturitie` widens the *bus* side, not the truck
side; a second truck-dense, cleanly-imaged validation area is still the
outstanding work here (§11).

### The detect round

`detect --aoi <region> --weights <path> --out <file>` is bootstrap's machinery
with the framing flipped: the same windowing, seam merging, and
georeferencing, but pointed at new ground with trained weights, producing
results rather than guesses for a human. Its output stages into label files
for the next correction round — which is the loop from §2 closing: detect →
stage → correct → export → train → eval.

The region can be **real polygons, not just bounding boxes**: postcode areas,
districts, a hand-drawn study area. GeoJSON is read directly (`--crs`
applies, default WGS84 per RFC 7946); FlatGeobuf, GeoPackage, shapefile and
friends go through geopandas (part of the `detect` extra) and **carry their
own CRS**, which then wins over `--crs`. Note the asymmetry with §2: a region
file is someone else's data and is read as RFC 7946 says, so feeding one of
*this* project's own GeoJSONs back as a region needs `--crs EPSG:3879`.
The polygons are unioned, windows
sweep the union's bounds, and detections are kept only where the box
*centre* falls inside a polygon. A vehicle straddling a boundary belongs to
whichever side its centre sits on — a rule, so it is decided once. Trained
models keep every class they know (`YoloObb(keep=None)`); only the zero-shot
bootstrap filters, to DOTA's `large vehicle` and `small vehicle`.

### Choosing the next AOIs (`mine`)

After a round is trained and evaluated, the next labelling batch should come
from where the model is uncertain or wrong — not from another hand-picked
yard. `mine` does that:

1. Fetch (and cache under `data/osm/`) the OpenStreetMap landuse polygons of
   the selected `--profile` (`industrial` by default; `PROFILES` in `osm.py`)
   inside Helsinki (municipal `ref=091`), via one Overpass query with
   an identifying User-Agent. Attribution: © OpenStreetMap contributors.
2. Grid them into 300 m cells in EPSG:3067, anchored at the national origin so
   indices are stable. Drop cells that overlap the existing collection (plus a
   50 m buffer), and cells with less than 15% industrial coverage.
3. Disperse a pool of ~100 eligible cells across the city, then sweep each with
   the trained weights at a detection floor below the operating confidence.
4. Select ~12 proposals by strata quotas rather than a single opaque score:
   ~40% near the operating threshold, ~25% in the 4–8 m truck/van band,
   ~20% dense novel yards, ~15% quiet industrial ground (misses or useful
   negatives). Minimum centre separation keeps the shortlist geographic.
5. Write a standalone proposal YAML and a sibling GeoJSON report. **Never**
   edit `aois/helsinki.yaml` or `labels/`. Proposals are `role: positive`,
   `split: train` — model-mined areas must not silently become validation.

Helsinki only. The current `Ortoilmakuva_2025_5cm` WMTS covers Helsinki; public
HSY metro imagery for Espoo/Vantaa is a different provider and resolution
(2023, 25 cm open mosaic) and needs its own imagery abstraction before those
municipal codes can be enabled. Beyond the default `industrial`, `PROFILES`
already carries `commercial`, `construction` and `camping` — added for
class-targeted mining (van hunting lives in commercial ground) — and is
extensible for warehouse / depot / marina confusers later.

The recorded rounds are in `docs/rounds.md`; the failure shape that motivates
mining is precision lost to truck-shaped background — containers, stored
bodies, trailers — which more *windows* of that geography fix, not more
labels on the same ground.

### Fetch etiquette

`fetch` is deliberately separate: it is the slow, network-bound,
rate-limit-sensitive part, and running it once lets everything else iterate
offline. `--dry-run` reports the tile count before committing to a sweep.
A tile that fails after its retries is collected and reported at the end
(exit 1), not raised mid-sweep: a long run must finish and say what it
missed. `bootstrap` follows the same rule — it skips a window whose tiles
never landed rather than dying on it, since the neighbouring windows still
cover most of that ground.

Verified end to end on 2026-08-04: a from-scratch rebuild (cold cache, empty
`labels/`) reproduces the same 392 candidates exactly, per area. The chain is
deterministic, so when a later model changes the candidate set, the difference
is the model, not run-to-run noise.

### AOI configuration

AOIs live in a YAML collection under `aois/`, in EPSG:3067, as an ordered list:

```yaml
crs: "EPSG:3067"
aois:
  - name: r1-tattariharjuntie
    bbox: [391857, 6680141, 392157, 6680441]
    role: positive          # positive | hard-negative | sparse
    split: validation       # optional, defaults to train
    notes: Transport yard, rows of parked trucks.
```

`crs` is **required, not defaulted**. The same four numbers are plausible as
EPSG:3067, as EPSG:3879, or as WGS84 degrees, and the readings differ by
hundreds of kilometres. A collection's declared CRS always wins; a stray
`--crs` cannot reinterpret it.

`role` carries the sampling strategy: `positive` holds trucks to label,
`hard-negative` holds convincing lookalikes (containers read as trailers from
above), and `sparse` is ordinary city that keeps the model from firing
everywhere. `role` and `split` are validated against fixed sets — a misspelled
`postive` would silently drop an area out of the positives, and a misspelled
split would silently move ground between train and validation.

`split` is a **field, not a note**. It decides which numbers are honest, so it
must not depend on a phrase in free text surviving an edit.

The file holds **geometry only**: year and zoom stay CLI flags, so there is
never a question of which of two places set them.

`--name` runs one entry; omitting it runs all. `--aoi` also accepts a GeoJSON
path or a bare `min_x,min_y,max_x,max_y` bbox, both of which take `--crs`
(default WGS84, since that is what a person copies off a map).

### The label schema

`labels/<aoi>.geojson`, EPSG:3879, one file per area. Beyond the detector's own
properties, each feature carries two editable fields:

| Field | Values |
|---|---|
| `class` | `""` (unreviewed), `truck`, `bus`, `van`, `car` |
| `status` | `candidate`, `confirmed`, `rejected`, `added` |

The state machine is small on purpose:

```
candidate ──T/B/V/C──> confirmed (class set)
    └──────X───────> rejected  (class cleared)
(drawn by hand) ───> added     (class set at creation)
```

`candidate` is the detector's unreviewed guess. `confirmed` and `rejected` are
human verdicts on one; `added` is a vehicle the detector missed. Given §4 —
recall is the weakness — `added` is expected to be the common case, not the
exception.

**Rejects are kept, not deleted.** They are the only hard negatives the project
actually has: the container terminal produced one candidate across the whole
area, so the false positives worth learning from are the ones that turn up
inside positive areas.

`rekka-ai progress` enforces the invariants: a `confirmed` or `added` feature
must carry a class, and every ring must be a closed 4-corner rectangle —
a freehand polygon cannot become a YOLO-OBB label. Measurements
(`length_m`, `width_m`, `heading_deg`) are **always recomputed from geometry**,
never trusted from the file: an editor moves a vertex and a stored length is
silently wrong from that moment on.

### Overlapping areas

`rekka-ai aois` reports areas that share ground, measured **in the collection's
own CRS** — reprojection rotates a box, and its EPSG:3879 envelope grows ~9 m
per edge for a 300 m area, enough to invent or hide a narrow overlap.

Overlaps matter for three reasons: the same vehicles get annotated twice and
train twice; a role conflict makes the same ground both a positive and a
negative; a split conflict leaks validation ground into training.

The collection currently has two, both between areas that agree on role and
split, so they cost duplicate annotation rather than contradictory labels:

| Areas | Shared | |
|---|---|---|
| `r1-tattariharju` × `r1-malmi-airport` | 20,640 m² | Duplicate annotation (23% of a 300 m area) |
| `r1-tattariharju` × `r1-kivikko` | 6,477 m² | Duplicate annotation |

A third, `vuosaari` × `r1-vuosaari-channel-road` at 2,278 m², was a **role conflict** — the same
ground was both `hard-negative` and `positive`. Fixed by stopping the `r1-vuosaari-channel-road`
north edge exactly at the `vuosaari` south edge. A test asserts no role or
split conflict remains, so this cannot come back unnoticed.

The two remaining duplicates are left as-is: the same vehicles will appear
twice in training, which skews instance counts mildly but does not make any
label wrong. Worth trimming before the annotation count matters.

Layout:

```
src/rekka_ai/
  cli.py              # thin typer layer
  geo.py              # CRS transforms, one place only
  labels.py           # labelling schema, measurements, validation
  export.py           # geographic labels -> YOLO-OBB pixel dataset
  train.py            # Ultralytics fine-tuning wrapper
  evaluate.py         # the ship-decision gates
  track.py            # MLflow configuration, one place only
  osm.py              # Overpass industrial geometry, cached under data/osm/
  mine.py             # grid / exclude / stratum selection for the next AOIs
  imagery/
    tiles.py          # grid arithmetic, pure
    wmts.py           # fetch + on-disk cache
    windows.py        # tiles -> model windows, pixel <-> ground
    aoi.py            # AOI collections, overlap detection
    layers.py         # flight year -> layer name
  detect/
    sweep.py          # detector protocol, YOLO adapter, area sweep
    detections.py     # oriented geometry, seam merging, GeoJSON
web/                  # labelling tool (§10); src/lib/obb.ts mirrors the geometry,
                      # grid.ts the tile grid, projection.ts the proj4 defs
aois/                 # AOI collections, EPSG:3067
labels/               # ← version-controlled. the valuable thing.
data/                 # gitignored: cache/ candidates/ osm/ mining/
models/  runs/        # gitignored
```

## 8. Output

GeoJSON `FeatureCollection`, EPSG:3879 (§2), one oriented footprint per
detection. The collection carries the `crs` member; each feature looks like:

```json
{
  "type": "Feature",
  "geometry": { "type": "Polygon", "coordinates": [[[25496000.123, 6673000.877], "..."]] },
  "properties": {
    "label": "large vehicle",
    "confidence": 0.87,
    "length_m": 16.2,
    "width_m": 3.0,
    "heading_deg": 143.0,
    "aoi": "r1-vuosaari-channel-road",
    "source_layer": "Ortoilmakuva_2025_5cm",
    "zoom": 16
  }
}
```

`length_m`, `width_m` and `heading_deg` fall out of the oriented box for
free. Carrying `source_layer` per feature makes multi-year comparison
possible without separate bookkeeping.

## 9. Risks and open questions

**Risks**

- *Annual cadence.* Restated because it invalidates whole use cases rather than
  merely degrading accuracy.
- *Not true orthos.* Building lean occludes vehicles near tall structures — and
  tall structures are common exactly where logistics happen.
- *Shadow and occlusion.* Summer morning flights cast long shadows; tree canopy
  hides vehicles on yard edges.
- *Annotation cost.* The dominant cost of the project, and the easiest to
  underestimate.
- *Class ambiguity.* See §5.
- *Spatial leakage.* A random split silently inflates metrics. See §2. The
  current split is by area and geographically separated, but three pairs are
  closer than the rest: `r1-pohjois-haaga` (validation) sits **935 m** from
  `r1-valimo` (train), `r1-jatkasaari` (validation) **1,185 m** from
  `r1-kamppi` (train, since 2026-08-11), and `r1-veturitie` (validation)
  **1,196 m** from `r1-ruskeasuo` (train). All three gaps are nearest-edge
  and far wider than a 128 m window, so no window straddles the split; the
  risk is shared character, not shared pixels. The last pair is the one that
  really carries that risk — two bus depots of the same operator (§5).
- *A thin validation split.* Four areas, and one of them (`r1-kaivoksela`)
  holds most of the trucks. See §7 — the gates read trends, not decimals.
- *Validation used to be blind to negatives.* Partly resolved since the
  original assessment: `r1-marjaniemi` (a marina — boat hulls on cradles)
  gives training its first negative area. `r1-rastila` (rows of van-fronted
  motorhomes, the annotation guide's hardest confuser) held that role in
  validation until review turned up real trucks and vans and it moved to
  `positive`/train — so validation is currently blind to negatives again, a
  deliberate tradeoff (see `docs/rounds.md`).
- *Service etiquette.* Access constraints are `NONE`, but a city-wide z16 sweep
  is a large number of requests. The fetcher caps concurrency, retries only
  transient failures, and sends an identifying `User-Agent`. Worth a note to
  Kaupunkimittauspalvelut before any full-city run.

**Open questions**

1. Who consumes the output, and does an annual snapshot answer their question?
2. ~~Do buses count?~~ **Decided:** annotated as a separate `bus` class, so
   trucks can be reported alone or with buses as large vehicles. See §5.
3. AOI: full city, or specific logistics/industrial zones?
   **Partially answered:** round-2+ areas are proposed by `mine` from OSM
   industrial landuse inside Helsinki, reviewed by a human before entering
   the collection. Espoo/Vantaa wait on imagery coverage.
4. Single year, or a back-series to 2014 for change detection?
5. ~~Annotation tool — QGIS (labels are already geospatial, and the team works in
   it) or CVAT (purpose-built for OBB)?~~ **Decided:** neither — the project
   built its own (`web/`, §10), shaped around the centreline drawing model
   and the review hot path that neither general tool optimises for.
6. Confirm the open-data license and required attribution string. The service
   reports no access constraints; Helsinki open data is normally CC BY 4.0,
   attributed to *Helsingin kaupunki, Kaupunkimittauspalvelut*, but this should
   be verified rather than assumed.
7. How does a round's provenance survive without a shared runner? Nothing on
   disk records which weights produced which labels: candidate features carry
   `source_layer` and `zoom` but not the model, and `stage` copies that gap
   into the label files. Two candidate fixes, neither built — have `eval`
   write a small committed record per round (`evals/<run>.json`) beside its
   MLflow log, and/or stamp `weights` onto each feature the way `source_layer`
   already is, which would make it part of the label schema. See §7.

## 10. The labelling tool

`web/` is a local SvelteKit app: OpenLayers over the same orthophoto WMTS,
editing `labels/<aoi>.geojson` in place through two API routes. No database, no
auth, localhost only — git is the version control, which is the whole reason
labels are tracked.

**A centreline per vehicle.** Across 392 candidates, width has a standard
deviation of 0.37 m against length's 4.32 m — trucks are road-legal width by
law. So the operator draws the *centreline*: click the nose, click the tail,
scroll for width from a 3.0 m default, click or Enter to place. Two points
fix the two things that vary (length and heading); four-corner placement
would be four times the work for information the law already supplies.

This is also the right optimisation target: §4 shows recall is the weakness, so
the hot path is *adding* missed vehicles, not triaging existing ones.

**Stack:** SvelteKit + OpenLayers + Tailwind 4, with proj4 for the projection.
Two API routes (`GET`/`PUT /api/labels/[aoi]`, `GET /api/aois`) read and write
the repository directly; writes go to a temporary sibling and rename, so a
crash mid-save cannot truncate hours of work into an unparseable file.
`/api/aois` also returns per-area review counts and the current imagery layer
name, so the page hardcodes neither.

Details that make it usable rather than merely correct:

- The AOI's `notes` are shown beside the map. Those notes *are* the annotation
  guide, so this puts the rules at point of use for free.
- OpenLayers is the right library here, not MapLibre: EPSG:3879 with a custom
  WMTS tile grid is native to it, where a WebMercator-centric library would be
  fought the whole way. `web/src/lib/grid.ts` mirrors the constants from
  `tiles.py` so the two cannot drift.
- The drawing width is shown live in the toolbar, and a scale bar stays on
  the map — the 6 m gate is a judgement made by eye, dozens of times an
  hour. Once a box exists, its selected state shows the measured
  `length × width` on its edges.
- Correction is keyboard-first, not redraw-first: `↑`/`↓` nudge length about
  the midpoint, `←`/`→` rotate (Shift for coarse steps), Shift+scroll sets
  width, and dragging one of the two end handles resizes from that end only
  — for the box whose other end was already right. `Del` deletes, `⌘/⌃Z`
  undoes, and clicking a neighbouring area's boundary jumps straight to it.
  Every shape edit rebuilds the ring from its centreline rather than moving
  vertices, so a box can never stop being a rectangle — `labels.validate`
  rejects anything else at export.
- Almost nothing animates. `T`/`B`/`V`/`C`/`X`/`N`/`P` are the hot path and get
  pressed hundreds of times a session; animating them would make every one feel
  slow. Motion is confined to what you see occasionally — switching area, the
  progress meter, press feedback.
- The sidebar is a file tree, because that is what the areas are: collapsible
  folders per group (train / validation / not staged), one row per
  `<aoi>.geojson`, with an inline progress pill and reviewed count per row and
  per folder.
- The class switcher is a toolbar, not keys alone: a segmented
  `truck / bus / van / car` control over the map, with the drawing width readout
  inside it and the save state as a badge on the right. A failed save stays
  dirty, shows `SAVE FAILED`, and retries — corrections that never reach disk
  are the one failure this tool cannot afford to hide. (The autosave debounce
  is cancelled at the start of every save, so a queued timer cannot write the
  previous area's features under the next area's name.)
- Selection is unmistakable but never a new colour: the selected box's class
  colour is drawn wider over a dark casing, and the two end handles and edge
  measurement labels exist only on the selection. An earlier amber halo was
  retired precisely because amber is the candidate class colour — a marker
  that collides with a classification reads as one. The footer shows `#n/N`
  in the review sequence. `N`/`P` jumps happen hundreds of times a session,
  and "which box am I on" cannot be a guess.
- Typography is Geist, with Geist Mono for every measurement and count. One
  desaturated amber drives the chrome (handles, focus, progress); the five
  class colours are map symbology and stay off the chrome. daisyUI was
  evaluated for this and rejected: its themed components cover only the easy
  third (buttons, badges) while fighting the tuned palette, and the hard parts
  — OpenLayers theming, the map overlay, the drawing interactions — stay
  bespoke regardless.

The geometry is tested on both sides (`bun run test` in `web/`, `tests/test_labels.py`),
and the round trip was checked end to end: a 16.00 × 3.00 m box drawn in the
browser is read back by Python as 16.0 × 3.0 m.

## 11. Next step

The loop is closed and proven on the rebuilt collection. What remains is
doing it well:

1. ~~**Finish the fresh-start labelling round.**~~ **Done** — three rounds
   trained and evaluated on the rebuilt collection (now 27 areas, 23 train /
   4 validation); round 3 passes all three gates (`docs/rounds.md`). Review
   can run on a branch while other work continues on
   `main`: since the label files stopped churning (§2), two people editing
   different areas touch disjoint files, and two people editing the *same*
   area touch a few `status`/`class` lines that merge cleanly. Under the old
   whole-file rewrite every save conflicted.
2. **Stop iterating** when a correction round moves validation recall or
   precision by less than a couple of points — after that, only a *different
   kind* of data buys anything: new area types (marinas, rail yards), or
   another flight year. Use `rekka-ai mine` to propose the next industrial
   cells from the trained model rather than hand-picking yards.
3. **Mind the validation set's honesty.** Unanswered again: `r1-marjaniemi`
   gives training its first `hard-negative` area, but `r1-rastila` — which
   briefly gave validation one — moved to `positive`/train once review
   turned up real trucks and vans, so validation currently has no negative
   area for the regression check to measure against. `r1-kamppi` moved to
   train and `r1-veturitie` replaced it (§7), which fixes the bus side but
   not the truck side: `r1-kaivoksela` still holds nearly all of validation's
   trucks, so the truck gates rest on one area. **Widen validation with a
   second truck-dense, cleanly-imaged area** before the gates decide anything
   big. What `rekka-ai aois`
   still warns about is validation missing the `hard-negative` and `sparse`
   roles. Model-mined proposals
   stay in `split: train` on purpose — do not promote them into validation
   without a separate, untouched hold-out plan.
4. ~~**Hold the toolchain bump for between batches.**~~ **Done** (2026-08-07,
   between batches as prescribed): `web/` now runs Vite 8 with
   `@sveltejs/vite-plugin-svelte` 7, SvelteKit 2.70 and Svelte 5.56; lint,
   check, test, the production build and a dev-server smoke test are all
   green. TypeScript stays on 5.9 until the Svelte tooling declares `^7`.
5. **Metro expansion** (Espoo / Vantaa) needs an HSY imagery source and a
   transfer check at that resolution before `mine --municipality` can leave
   Helsinki.
