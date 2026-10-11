# Negative results

Real outcomes that did not work. Kept so they are not retried blindly. All
numbers are measured.

## Accuracy flags D, E and F: no change on any scene

Measured on branch exp/accuracy with the semantics worker (cached VLM, so no
3.5 minute model reload per run). Six development scenes each. Baseline and every
flag variant are identical:

| experiment | recall(type) | localized | strict move | name | extras | fragments |
|---|---|---|---|---|---|---|
| baseline | 71% | 86% | 25% (1/4) | 60% | 4 | 3 |
| soft_veto | 71% | 86% | 25% | 60% | 4 | 3 |
| coverage_added | 71% | 86% | 25% | 60% | 4 | 3 |
| both | 71% | 86% | 25% | 60% | 4 | 3 |
| objmatch (re-match) | 71% | 86% | 25% | 60% | 4 | 3 |
| fragment_merge | 71% | 86% | 25% | 60% | 4 | 3 |

- soft_veto changes no confirmed metric. It only moves VLM-denied candidates to
  unverified (scene6 went from 0 to 16 unverified, including the missed apple as
  "blue cup and red apple").
- coverage_added had no effect on any scene.
- objmatch (Hungarian re-match of the existing candidate crops) found no new
  move: the departure and arrival crops do not reach 0.55 similarity.
- fragment_merge (0.35 m global radius) did not join scene5's leg fragments or
  scene3's apple pair.

Under the gates nothing is accepted, so every flag stays OFF. The move misses
(demo remote, scene5 clock, scene6 cup) are not recoverable by re-weighting the
existing candidate crops.

## Track B probe: SAM 2 masking does not rescue the move match

`scripts/probe_sam2.py`, job 11017, one GPU. It refines the existing paired
crops with a SAM 2.1 base-plus mask (point prompt at the crop centre) and
compares the masked crops with DINOv2. The match gate is 0.65.

| scene | object | paired-crop DINOv2 after SAM 2 mask | pass (>= 0.65) |
|---|---|---|---|
| scene6 | cup | 0.190 | no |
| scene5 | clock | 0.219 | no |

Both are far below the gate, so the probe fails. Segmentation itself is not the
problem: SAM 2 needs no training and produces the masks. The problem is that the
candidate departure and arrival crops are not the same object, or the object
looks too different between the two views, so masking does not raise the
similarity. Track B stops here per the written pass gate. objmatch2 was not built.
