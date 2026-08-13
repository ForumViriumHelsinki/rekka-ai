---
name: audit-labels
description: Read-only QA over labels/ — what progress/validate catch and what they don't, class-call audits against the LABELLING.md morphology rule, and the do-no-harm invariants for the one irreplaceable artifact
type: prompt
whenToUse: When the user asks to check/audit/QA label files, suspects label errors, wants class-consistency checks, or is about to touch anything under labels/
---

# Auditing labels/

27 files of human verdicts — the **only irreplaceable artifact** in the
repo. Everything else regenerates; labels are human hours. Audits are
read-only by default: report problems, let the human fix them in the web
tool.

## Do-no-harm invariants (before touching anything)

- **Never delete, bulk-rewrite, or "clean up" label files.** An agent
  tidying `labels/` is the single worst thing that can happen in this repo.
  `rekka-ai stage` refuses to overwrite an existing label file without
  `--force` — that refusal is the most important safety property in the
  pipeline. Do not route around it.
- **Measurements are recomputed, never trusted.** `length_m`, `width_m`,
  `heading_deg` in a file are conveniences; every consumer recomputes them
  from geometry. Any audit must recompute too — never compare the stored
  values.
- **Byte-identical round-trip.** An untouched box must round-trip as
  identical bytes; both writers round to `geo.COORD_DECIMALS` (3). If a
  rewrite is ever justified, preserve this — a full-file reformat destroys
  `git diff` as an audit trail.
- **Rejects are data.** Rejected candidates are the project's hard negatives
  and keep the file a complete record of what was proposed. Reject means
  "not a vehicle"; it is never "I can't tell" and never deleted.

## What the existing checks catch — and what they don't

- `uv run rekka-ai progress` — per-area reviewed counts, reject rates, and
  `labels.validate` problems: status/class outside the vocab, `confirmed`/
  `added` with no class, ring not 5 positions, ring not a rectangle.
- `validate` does **not** catch **displaced boxes**: a box dragged outside
  its area is still a valid rectangle with a valid verdict, but export
  writes a label only into windows that fully contain it — so it silently
  exports as *nothing* and the dataset comes out smaller than the review
  count promised. `labels.displaced(collection, bounds)` checks centres
  against area bounds; `export` runs it and refuses dirty, so an export
  failure on a "clean" file is usually this.
- Nothing checks **class-boundary consistency** — the same object called
  truck in one area and van in another. That is what the recipes below are
  for.
- `export` is the final backstop and refuses to run dirty: unreviewed
  candidates, schema problems, displaced boxes. Do not weaken these guards.

## Recipe: class calls against the morphology rule

The rule (docs/LABELLING.md): a truck has a **cab that stops and a load body
that starts**; a van is one continuous shell. Length only decides at the
ends — under 6.5 m a separated body is rare, at 8 m and over it is a truck
293 times out of 294; **between 6.5 and 8.0 m length says nothing**.

Pull `length_m` distributions (recomputed from geometry) per class per area
and flag outliers against the measured quartiles (2,866 labels):

| class | length p25–p75 | median |
|---|---|---|
| `car` | 4.6–5.0 m | 4.83 m |
| `van` | 5.2–6.1 m | 5.60 m |
| `truck` | 8.0–11.3 m | 9.84 m |
| `bus` | 12.8–14.8 m | 14.52 m |

Expected findings and their legitimate exceptions: trucks under 6.5 m
(bobtail tractor units are real but rare — list them for review), vans ≥ 8 m
(suspect — likely motorhomes/RVs, a recorded confuser in four areas), cars
over 5.5 m (pickups are `car` by convention, so check shape not length). An
area whose distributions sit far off the collection medians is a convention
drift suspect, not a geometry bug.

## Recipe: cross-area convention drift

`progress` already prints per-area class mix and reject rate. Outliers worth
a look: an area with near-zero rejects in dense ground (reviewer approving
everything?), a reject rate far above the ~7% round-1 norm (reviewer
rejecting "can't tell" instead of picking a class?), a van/truck ratio far
off peer areas of the same geography. Convention drift between train and
validation makes the eval measure the labeller, not the model.

## Recipe: model-assisted ground-truth pass

Once weights are good enough, the model audits the labels: detect over
*labelled* areas, match against confirmed/added (IoU > 0.3, the
`unexplained()` rule), and review what matches nothing. The round-3 pass
found 82 such detections, mostly real misses: +6 truck, +7 van, +78 car,
+34 rejects. Two earlier eval "failures" were label error, not model error.
This is the highest-yield audit once a decent model exists — and the
findings go back through the web tool, never through a script editing
`labels/`.
