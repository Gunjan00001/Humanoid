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

## Track B controls (job 11023): the gate is calibrated, targets still fail

| scene | pair | role | CLS | patch | size_r |
|---|---|---|---|---|---|
| scene4 | apple moved | positive | 0.704 | 0.914 | 0.70 |
| scene4 | scissor vs apple | negative | 0.285 | 0.085 | 0.78 |
| scene6 | cup | target | 0.482 | 0.374 | 0.51 |
| scene5 | clock | target | 0.174 | 0.460 | 0.71 |

The positive control (a correct match) scores high and the negative control
(different objects) scores low, so the 0.65 gate is not miscalibrated. The targets
stay low, and the clock's CLS is below the negative control. Track B is closed as
a real negative result. objmatch2 was not built.
