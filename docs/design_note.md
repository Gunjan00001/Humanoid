# Design note

The short version is in the README. This is the longer note behind the choices,
including what was tried and rejected. Every number here is measured; the
measurements live in the linked docs.

## One reconstruction for both videos

Both recordings are reconstructed together in a single VGGT pass, so every frame
of both videos lands in one coordinate frame. The alternative, reconstruct each
video and register them afterwards, needs a separate alignment step and adds a
failure mode (registration error) that nothing downstream can undo. A joint pass
costs a little more memory but removes that risk. ICP then refines the static
structure only (`src/align.py`), improving static mean nearest-neighbour distance
from 15 mm to 11 mm on the demo scene.

## Detection by geometry and photometry, the model only verifies

Candidates come from two detectors. The geometric detector compares the two point
clouds by nearest-neighbour distance and clusters the differences. The tabletop
detector compares appearance over the fitted dominant plane, because VGGT absorbs
objects thinner than its patch scale (about 1.4 cm here), so the comb and the pen
do not exist in the depth maps at all. Qwen3-VL never proposes a candidate; it
confirms or denies a candidate and names it. This keeps the recall question
(are all changes found) separate from the naming question (are they described
correctly), and it is why a missed change is usually a detection or filtering
failure, not a language failure.

## Telling real changes from viewpoint and occlusion

Three checks do this work, and they are the reason the report is short:

- visibility. A region is called `removed` only if the other video's depth
  observed that space empty. A region merely occluded in the other video is not a
  removal.
- coverage. An addition must sit where the other video saw empty space, so a wall
  feature or a floor reflection present in both videos is not an addition.
- static object suppression. Fragments of objects present in both videos (the
  timer, the highlighter) are lighting or blur differences and are dropped.

A replacement at the same spot (comb to pen) is geometrically invisible; only the
appearance comparison separates it from "nothing changed".

## Summaries are grounded

Sentences are generated only from report fields, and the generator validates each
sentence against the change's label and type action words. A sentence can never
claim an object or a count that is not in the report.

## What was tried and rejected

- soft veto (keep a VLM-denied candidate with strong geometry as "possibly"): no
  confirmed metric changed; it only moved candidates to the unverified list.
- coverage-aware additions as a separate flag: no effect on any scene.
- object-level re-matching of the existing candidate crops (Hungarian, DINOv2): no
  new move; the departure and arrival crops do not reach the similarity gate.
- fragment merging by a global 3D radius: it did not join the scene5 chair
  fragments or the scene3 apple pair.
- SAM 2 masks to isolate the moved object before matching: the controls separate
  (correct match 0.704, different objects 0.285) but the cup and clock candidates
  stay below the gate (0.482 and 0.174 CLS). They did not match under this
  descriptor; that is not proof they cannot be matched.

All of this is in `docs/negative_results.md` with the numbers, and the sprint log
is `docs/accuracy.md`.

## Known limits

Moves are the weakest part: only 1 of 4 development moves is reported as `moved`.
Object names come from a language model and can be wrong (a chair read as a
"black trash can"). Held-out recall is lower than development recall, which is
expected because the development thresholds were tuned on those same recordings.
The recordings were made in matched conditions; different lighting or a mixed
orientation weakens the reconstruction. No absolute distances are reported: the
reconstruction scale is not metric-validated.
