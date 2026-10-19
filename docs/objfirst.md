# Object-first pipeline (Phase 4, `exp/object-first`)

Wired behind `diff.method: objfirst`, **default OFF**: the default path never
imports these modules, so every committed output stays byte identical.

## What exists

| stage | module | state |
|---|---|---|
| M1 inventory | `src/objfirst/inventory.py` | implemented from the cached crops + VLM labels (no GPU); a true per-frame VLM inventory is the next step |
| M2 segmentation / tracking | — | SAM 3.1 is adopted (autocast route) but not wired into this package yet; SAM 2.1 automatic masks are the fallback |
| M3 lift | `src/objfirst/lift.py` | implemented: joins each instance to the diff stage's 3D box (centroid, extent) so object-first cannot disagree with the detector about location |
| M4 matching | `src/objfirst/match.py` | implemented: DINOv3 ViT-L mean patch descriptor (grid recovered from token count and crop aspect), four-signal cost, Hungarian, written acceptance threshold |
| M5 verify / fuse | — | not implemented; the documented fusion rule is cluster overlap plus a type vote, applied only after the flag passes its gates |

## First end-to-end smoke (scene4, CPU, cached data)

`inventory: a=6 b=2 · instances 6/2 · matched pairs 2 · candidates 6`

* two `moved` candidates, both labelled apple (cost 0.238 and 0.353,
  displacement 0.337 and 0.365 reconstruction units);
* four `removed` candidates, all labelled scissors (the scene's real removal,
  fragmented into four entries);
* no `added` candidates.

## Honest limitations found by the smoke

1. **The inventory inherits the detector's candidate clusters.** It joins the
   cached crops, so it can only see objects the current detector already
   proposed. The spec's M1 is a per-video inventory on sampled frames; that is
   the first thing to replace, and it is why this run has 6 + 2 instances.
2. **Fragments duplicate.** Four scissors fragments and two apple fragments
   survive into matching, so one physical object can produce several candidates.
   Fragment merging must be part of M2/M5, not an afterthought.
3. **Acceptance threshold 0.55 was not tuned on this data**; it is the written
   starting value and needs the Phase 4 gate evaluation.

## Gates (unchanged, from docs/autopilot_batch2.md and the master autopilot)

Accept `objfirst` only if strict move recall improves, type and localized recall
do not drop, extras do not rise, scene4 and demo do not regress, three runs are
stable, outputs are byte identical with the flag OFF, tests pass, no per-scene
values, and the gain does not come from a single scene. Otherwise the flag stays
OFF and the negative result is written.

## Fragment merging (commit after the first smoke)

`src/objfirst/fragments.py` merges same-label instances whose 3D boxes touch or
overlap (written rule: centre distance below half the sum of the extents), with
transitive merging and a union box. Effect on scene4: side A **6 -> 5**
instances (scissors 4 -> 3), side B unchanged at 2, matched pairs still 2.

Honest reading: the rule only merges fragments that touch in 3D, and these
fragments mostly do not touch (the two apple fragments on each side sit apart),
so the effect is partial. Two options are recorded for the next pass, in order:
(1) merge same-label instances whose union box is still "one object" sized
relative to the label's measured extent distribution — a global rule but one
that needs the distribution measured first; (2) merge on matching evidence: two
A fragments that match the same B object are one object, which the matcher
already knows. The second is the cleaner route and belongs with M5.
