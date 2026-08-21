---
name: replay-round
description: Replay a training round (or all rounds) end to end — collection filter, export, train, eval — with MLflow hygiene for aborted runs and the timing/seed caveats measured on this project
type: prompt
whenToUse: When the user asks to re-run/replay a training round, reproduce the recorded rounds, train at a different zoom or seed, or clean up an aborted MLflow run
---

# Replaying training rounds

The canonical command list lives in `docs/DESIGN.md` §7 "Replaying the
recorded rounds" (it moved out of the README on 2026-08-21) — read it first;
this skill adds the operational knowledge around it.

## Collections are prefix filters

Per-round training collections are subsets of `aois/helsinki.yaml`: round N
trains on the areas whose names start with `r1`..`rN` (`name.split("-")[0]`
in {r1..rN}). Regenerate them by filtering the collection rather than
hand-editing — verified byte-identical to the recorded ablation YAMLs
(2026-08-12).

## Sequence per round

```
export  (collection + labels -> data/dataset-<tag>[-z17])
train   (--data dataset.yaml --name roundN[-z17])
eval    (--collection aois/helsinki.yaml --weights runs/train/roundN/weights/best.pt --name eval-roundN)
```

- **Every round starts from `yolo11x-obb.pt`** — do NOT chain the previous
  round's `best.pt`. The ablation varies only the training ground, and the
  recorded runs confirm it: MLflow params show `model=yolo11x-obb.pt` for
  round1, round2 and round3 alike (checked 2026-08-13). Chaining would make
  the rounds non-comparable.
- Defaults are deliberate: batch 4, seed 0, patience 30. Changing batch
  changes training dynamics — treat it as an experiment config, not a free
  speed knob, and re-baseline comparisons if you change it.
- z17 variants: `--zoom 17` on export/train/eval, `-z17` suffix on dataset
  and run names so the z16 and z17 histories stay separate in MLflow.

## Timing expectations (RTX 5070 Ti)

- z16: rounds 1-3 in 10/19/14 min — ~43 min for all three.
- z17: rounds 1-3 in ~2/2.4/3.1 h — ~7.4 h total (~10× z16, tracks the 9×
  window count). Fetch is quick when tiles are cached (11.6k tiles in 25 s).
- Eval runs are effectively instant in the MLflow store.

## MLflow hygiene

- Store: `runs/mlflow.db` (sqlite), experiment "rekka-ai" (id 1). Reads: the
  mlflow MCP tools. **The MCP server is read-only** — writes/deletes need
  the CLI:
  `MLFLOW_TRACKING_URI=sqlite:///runs/mlflow.db uv run mlflow runs delete --run-id <id>`
- An aborted training attempt leaves a partial run *and* a
  `runs/train/<name>/` directory. Before re-running under the same name:
  soft-delete the partial run and remove the directory, or the new run's
  artifacts mix with the old.
- Keep run names round-shaped (`roundN`, `eval-roundN`, `-z17` variants) —
  the clean 6/12-run history is a presentation artifact; one-off
  experiments with ad-hoc names are what made the original store messy.

## Interpreting results

- Gates: truck recall ≥ 0.90, truck precision ≥ 0.85, count error ≤ 10% on
  areas with ≥ 60 truth trucks (today only `r1-kaivoksela`), negative-area
  detections reported not gated. Validation is whole AOIs, never a random
  split.
- Single-seed results carry **±2–3 pt noise** (measured). round2-z17 scored
  *worse* than round1-z17 purely from shuffle luck. Before quoting any
  single run — especially a gates-passing one — run a second seed.
- Diagnosing a failed or surprising gate — which diagnostic per gate, and
  why relabelling lookalikes buys nothing — is the **triage-eval** skill.
- Zoom effects decay with rounds: z17's +14 pt precision in round 1 is real
  (beyond seed noise); by round 3 the zooms converge (0.884 vs 0.901, inside
  noise) and z17's remaining edge is cleaner negatives (0 vs 1) and class
  purity (van→truck confusion 32% → 3%).
