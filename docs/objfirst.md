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

## Merge on matching evidence (attempt, no change) and what it proved

Pipeline now groups matched pairs by label and only merges when **both sides'
boxes touch** (the same written rule as fragments.py, no new constant). Result on
scene4: **no merge** — the two apple fragments and their B counterparts sit apart
in 3D, so they remain two `moved` candidates, and the three scissors fragments
remain three `removed` candidates.

What that proves, and the next step: these clusters are not adjacent fragments
of one object; they are distant pieces of the same physical object, which is why
a touching-boxes rule cannot join them. The honest route is route (1) from the
previous section: merge same-label instances when the union box is still
one-object sized **relative to the label's own extent distribution**, and that
distribution must be measured first (per label, from the GT crops and their
reconstructions). That is a measurement task for the manifest, not another
constant to guess. Until then the object-first path reports fragments as
fragments, and the gate evaluation must treat fragmentation as a known loss.

## Measured extent rule (route 1), implemented and re-smoked

`scripts/objfirst_extents.py` measured the per-label 3D extent distribution from
the diff boxes joined with the cached VLM labels: **19 labels**, e.g. apple
n=10 median 0.143 P90 0.319, chair n=10 median 0.172 P90 0.512, coconut n=5
median 0.197 P90 0.503; labels with fewer than three instances have no extent
rule and keep the touching-boxes rule. Written to `data/objfirst_extents.json`.

`merge_by_extent` then merges same-label instances while the **union diagonal
stays within the label's P90**. Effect on scene4: side A **5 -> 4** instances
(scissors 3 -> 2), candidates **5 -> 4** (two moved apples, two removed
scissors). The remaining pairs stay apart because their unions exceed the label's
P90 by a wide margin, i.e. they are far apart for one object of that class. Two
readings, both honest: either these are genuinely different pieces the detector
split across metres, or they are separate objects labelled with one name; the
gate evaluation will tell, because it scores against GT objects, not fragments.

## M2 segmentation bridge

`scripts/objfirst_segment.py` (run in `spatialdiff-sam3`) prompts SAM 3.1 with
each cluster's cached VLM label under bf16 autocast and writes
`logs/objfirst/masks/<scene>.npz` (one boolean mask per cluster) plus a JSON of
mask statistics. `src/objfirst/segment.py` loads those masks, or calls the
sam3-env script on demand via `ensure_masks`. Masks are **optional**: without
them the spine still runs on the cached boxes, so a CPU-only checkout stays
testable and nothing here runs on the default path.

Not run yet: the mask pass is a GPU task; it is deliberately not appended to
`data/batch_manifest.json` while the Phase 4 code lives on `exp/object-first`,
because the runner works on the main checkout. The task is appended once the
merge decision brings the code to main.

## M5 verification and fusion (implemented, first smoke)

`verify.py` applies three signals with explicit "unverified" states: VLM
confirmation per cluster (candidates whose only evidence was rejected are
dropped), visibility diagnostics (attached when present, otherwise
"visibility unverified", never assumed), and tiers with **gap-free stable
numbers** (the first smoke exposed a numbering gap when a candidate was dropped;
fixed so numbers are assigned on emission). `fuse.py` implements the written
fusion rule: match on cluster overlap, keep the detector entry on agreement or
disagreement (recording both), and append object-first-only candidates marked
`source: objfirst`.

First smoke on scene4: 4 candidates -> **3 after verification** (one scissors
candidate dropped: its cluster was not VLM-confirmed), tiers plain for apple and
scissors, numbers 1, 2, 3; fused output 5 entries (2 detector entries with one
agreement and one without, plus the object-first-only candidate). Visibility
says "unverified" for all three because the diff cache has no per-cluster
visibility ids; that is stated rather than papered over.

## Phase 4 gate evaluation (cached data, six scenes, deterministic rerun)

`scripts/objfirst_gate.py`, object-first path with M5 fusion, twice per scene for
determinism (all six identical):

| scene | found / GT | extras | fragments | candidates | fused |
|---|---|---|---|---|---|
| demo | 2/3 | 5 | 0 | 7 | 7 |
| scene2 | 1/1 | 5 | 0 | 6 | 6 |
| scene3 | 1/2 | 10 | 0 | 11 | 11 |
| scene4 | 2/3 | 0 | 1 | 3 | 5 |
| scene5 | 0/2 | 7 | 0 | 7 | 8 |
| scene6 | 1/3 | 5 | 0 | 6 | 6 |
| **overall** | **7/14 = 50.0%** | **32** | **1** | | |

**Gates applied, rejected.** The regression floor fails on both required scenes:
demo 2/3 and scene4 2/3 against the baseline's 3/3 and 0 extras on each, and the
extra count rises from 4 to 32 overall. `diff.method` stays `tabletop`; nothing
from this path ships.

## What this evaluation is NOT (accurate negative wording)

M1 inventory and M3 lift **inherit the detector's clusters and 3D boxes**: the
inventory reads the cached crops and VLM labels the detector produced, and the
lift joins the diff stage's boxes. The path can therefore only see objects the
incumbent detector already proposed, and it cannot test the central claim of the
object-first hypothesis, that an independent inventory finds objects the change
detector misses. **Object-first was not tested and did not fail.** What failed is
this particular implementation, for a reason that was measured, not guessed.

Not built, and the first items of "what I would do next with a week":
1. the **lift sanity check** (project the cached boxes through the cameras and
   confirm they land on the objects before any matching is judged);
2. an **independent SAM 3.1 inventory** per video (per-frame text prompts, masks
   lifted with the cached depth and poses);
3. a **per-frame centroid scatter check** to reject unstable instances;
4. matching and verification on those independent instances, then the same gate.

Strict move recall per scene: demo 0.0%, scene4 100.0% (its one GT move found),
scene5 0.0%, scene6 0.0%; scenes 2 and 3 contain no GT moves, so the metric is
undefined there. Overall **1/12 = 8.3%**, against the baseline's 25.0%.
Fused entries total **43** across the six scenes, against **24** baseline entries
(demo 3, scene2 5, scene3 4, scene4 3, scene5 5, scene6 4, from docs/accuracy.md).
