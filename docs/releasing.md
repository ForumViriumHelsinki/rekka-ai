# Releasing trained weights

The runbook for publishing a model. The reasoning behind the tag scheme and
the rename is `docs/DESIGN.md` §7; this page is the commands and what to do
when one of them fails.

A release is **weights plus the model card**. It is not a code release: the
tag is a bookmark recording which commit the weights came from, and nobody
installs the CLI from it. Someone consuming a release clones or pulls `main`
as usual and downloads the `.pt` beside it.

## Versioning, in one table

Tags are `model/vMAJOR.MINOR.PATCH`, and **MAJOR is the round**.

| what happened | tag |
|---|---|
| a new round trained | `model/v5.0.0` |
| same round retrained — another seed, a z17 sibling | `model/v4.1.0` |
| same weights, the card was wrong | `model/v4.0.1` |
| a bug in the Python code | **no tag** — push to `main` |

That last row is the one people get wrong. Code and weights move on separate
clocks; a CLI fix reaches everyone who pulls `main` without the model being
touched.

## Before tagging

- `docs/model-card.md` describes *these* weights: gates, per-class table,
  training data counts and the operating confidence all match the MLflow run.
  The whole card becomes the release notes, frozen at tag time.
- The labels and collection behind the weights are committed — the release
  claims the training data regenerates from them.
- The weights exist where you think they do:
  `runs/train/round<N>/weights/best.pt`.

## The three steps

```sh
# 1. Tag the commit the weights came from, and push the tag.
git tag model/v4.0.0
git push origin model/v4.0.0
#    -> the `draft` job opens a draft release and prints step 2 in its
#       Actions job summary.

# 2. Rename and upload, from the machine that has the weights.
cp runs/train/round4/weights/best.pt yolo11x-obb-fvh-z16-v4.0.0.pt
gh release upload model/v4.0.0 yolo11x-obb-fvh-z16-v4.0.0.pt

# 3. Publish the draft in the GitHub UI.
#    -> the `verify` job checks the asset and attaches SHA256SUMS.
```

The rename is not cosmetic: Ultralytics names every model it ever trains
`best.pt`, and the published name has to say which architecture, whose, and
**which zoom** — z16 and z17 weights are not interchangeable and the mismatch
degrades quietly rather than failing.

## When a job fails

**`verify` is an alarm, not a gate.** GitHub fires the `release: published`
event *after* the release is public, so a failed verify leaves a red check on
something people can already download. It cannot block publication — nothing
can. Treat a red verify as "fix this now", not "it was prevented".

| failure | state you are left in | recovery |
|---|---|---|
| tag name rejected | no release, tag exists | delete the tag, re-tag correctly |
| `docs/model-card.md` missing at that commit | no release, tag exists | commit the card, delete the tag, tag the new commit |
| `gh release create` says the release exists | the existing release is untouched | intended — a re-release is a new PATCH tag, not an overwrite |
| published with no weights attached | release is public and empty | upload the asset, then **Re-run jobs** on the failed run |
| published with `best.pt` (unrenamed) | release is public, asset anonymous | delete the asset, upload the renamed copy, re-run the job |
| asset version does not match the tag | release is public, name misleading | same: delete, re-upload correctly named, re-run |

Re-running works because `verify` re-downloads the release's assets from
scratch — it reads the release, not the workflow run's history.

Deleting a tag, when the draft flow has to be redone:

```sh
git tag -d model/v4.0.0
git push origin :refs/tags/model/v4.0.0
# delete the draft release in the UI too, or `gh release delete model/v4.0.0`
```

Deleting a tag that already has a **published** release is not a fix — the
release survives and now points at nothing. Publish a PATCH instead.

## Verifying a download

```sh
gh release download model/v4.0.0
sha256sum -c SHA256SUMS
```

`SHA256SUMS` is attached by CI after publication, so a release whose verify
job never went green will not have one.
