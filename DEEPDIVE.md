# Deep dive

Everything a reviewer, or the author in an interview, needs: what the project
does, how it works, why each decision was made, what worked, what failed, and
what is still unverified. Every claim below traces to a file, a tag or a log in
this repository. Numbers are the measured ones; nothing is rounded up.

Reading time is about forty minutes. If you only have five, read sections 1, 2
and 7.

---

## 1. The problem, in plain words

You film the same small scene twice with a phone, from roughly the same spot,
seconds apart. Someone has moved, added or removed something in between. Looking
at the two videos, say what changed: what the object is, whether it was added,
removed or moved, and where it went.

The challenge this was built for asks for exactly that, from two videos, with no
depth sensor and no markers, and it asks for the answer in plain language. The
hard parts are not "detect change in a video". They are:

1. **You have no 3D.** Two ordinary videos give you pixels, not a scene. You must
   reconstruct one, and you must do it consistently for both videos, otherwise
   everything looks like it moved.
2. **The camera moved between takes.** So did the lighting, slightly.
3. **Everything is small and cluttered.** A table with a cup, a pen, a comb on it.
   The interesting objects are a few centimetres across and partly hidden behind
   each other.
4. **A change in the pixels is not a change in the world.** Moving the camera
   changes every pixel. Shadows move. A reflection appears on a surface. The
   pipeline has to reject all of that.
5. **You must not lie.** A confident wrong answer is worse than a hedge here, so
   the reporting has to be honest about what it verified.

The challenge also explicitly allows AI assistance in building the solution,
which was used heavily. Section 8 covers how.

---

## 2. The big idea

Reconstruct both clips into **one** 3D scene, put a fine 3D grid over the
tabletop, mark the grid cells that look different between the two takes, group
those cells into candidate objects, then ask a vision-language model to confirm
and name each candidate, and print a sentence per surviving change.

One reconstruction, not two, is the core idea: if both clips share a single
coordinate system, "moved" is a geometric fact rather than a guess.

### The example: scene4, followed through every file

`data/scene4/` holds `a.mp4` and `b.mp4` (about 9 MB each). Ground truth, written
before any run, is in `data/scene4/ground_truth.json`: the **apple moved**, the
**scissors were removed**.

1. **`src/extract_frames.py`** scans both clips at 3 frames per second and keeps
   the sharpest 30 per clip, judged by the variance of the Laplacian. Blurry
   frames destroy reconstruction quality, so they are dropped early. Writes
   `cache/scene4/frames/a/*.jpg`, `b/*.jpg` and `index.json`.
2. **`src/reconstruct.py`** runs VGGT over both sets of frames in one pass,
   producing point clouds and camera poses in a shared frame of reference.
   Writes `cache/scene4/reconstruction/points.npz`, `cameras.npz`, `dense.npz`,
   `meta.json`. `cameras.npz` is what later lets a 3D point be projected into a
   specific video frame.
3. **`src/align.py`** optionally refines the alignment between the two halves with
   ICP. On the demo this cut the static mean nearest-neighbour error from 15 mm to
   11 mm, which is why it exists, but it is off by default because it can also
   converge to the wrong thing on cluttered scenes.
4. **`src/tabletop.py`** fits the dominant plane (the table), builds a 5 mm voxel
   grid aligned to it, and decides whether to use the plane at all. This is the
   "tabletop" mode. The alternative, "geometric", skips the plane and uses raw 3D
   distances.
5. **`src/diff.py`** compares the two takes voxel by voxel. For each cell it asks
   whether the points there exist in both takes, whether their colour and
   geometry agree, and whether the cell is actually visible in both (a cell behind
   the cup in the second video is not evidence of anything). Writes
   `cache/scene4/diff/points.npz`, `clusters.json`, `visibility.json`.
6. **`src/describe.py`** takes each candidate cluster and cuts crops of it from
   the frames where it is visible, in both takes. DINOv2 embeddings match the
   "before" crop of one cluster against the "after" crop of another, so that a
   moved object is reported as one `moved` entry rather than a removal plus an
   addition. Then Qwen3-VL looks at the crops and answers two questions: is this a
   real change, and what is the object called. Writes
   `cache/scene4/semantics/crops.json`, `vlm.json`, `matches.json`.
7. **`src/report.py`** writes `outputs/scene4/report.json`, the single source of
   truth: one entry per change with type, label, 3D centroid and bounding box, the
   confidence, and the frames used as evidence.
8. **`src/summary.py`** turns `report.json` into prose. It reads nothing else, by
   design, so the summary cannot invent a fact the report does not contain.
9. **`src/visualize.py`** and **`src/viewer.py`** produce the evidence a human
   checks: per-change composites, `change.gif`, `checking_sheet.png` and
   `viewer.html`.

The result for scene4, from `outputs/scene4/report.json`: three entries, the apple
`moved` and the scissors `removed`, with zero extra entries. That is 3 of 3
ground-truth changes with the correct type, and it is the cleanest scene in the
repository.

---

## 3. The concepts, with analogies

**Feed-forward 3D reconstruction (VGGT).** Normal photogrammetry solves for
cameras and points together, slowly, with many iterations. A feed-forward model
has learned the answer and produces it in one pass. Think of the difference
between solving a maze step by step and having seen a million mazes. Used in
`src/reconstruct.py`; chosen over MapAnything and DA3 after the comparison in
`docs/results.md`.

**Joint reconstruction of both videos.** Reconstructing each clip separately gives
two scenes that are each internally consistent but unrelated to each other, like
two maps drawn by different people. Reconstructing both together forces one
coordinate system, so a static object lands in the same place in both.

**Point maps, cameras and depth.** A point map is the depth image's richer cousin:
instead of one distance per pixel, one 3D point per pixel. Camera poses say where
the camera was for each frame. Together they let any 3D point be projected back
into any frame, which is how the evidence overlays are drawn and how the visibility
check works.

**Voxels and the 5 mm grid.** Comparing point clouds directly means comparing
millions of points with slightly different positions, which is noisy. Instead the
space is diced into 5 mm cubes and each cube is asked one question: does this cube
contain something in video A, in video B, or both? The size is a config value
(`configs/default.yaml`) and it is the resolution limit of the whole method: a
change smaller than a cell can be missed.

**Tabletop plane versus geometric mode.** If the scene has a table, finding its
plane gives two useful things: a natural coordinate system ("12 mm above the
table") and a way to say where things are relative to each other. If the scene has
no table, that assumption is worse than useless. `tabletop` is the default,
`geometric` is the fallback, and the mode is chosen automatically per scene.

**Visibility and coverage.** A cell cannot be judged if it was hidden in one of the
two takes. A cell outside the camera's view, or behind another object, is
"unverified", not "changed". This single check removes most false positives and it
is why a moved object reported by the pipeline can never be an object that simply
went behind something. Implemented in `src/diff.py`.

**Clustering.** Changed cells are connected into groups so that one physical
change becomes one report entry, so that they can be cropped and shown. The
failure mode is fragmentation: a chair becomes three "leg" entries. That happened
on held-out scene5, and it is reported honestly.

**Photometric deviation.** Colour disagreement between the two takes in a cell, in
addition to geometric disagreement. This catches objects that moved slightly but
are still there, and it is also what makes a strong shadow look like a change.

**Vision-language model verification and naming.** Qwen3-VL looks at the before and
after crops and answers "is this a real change" and "what is it". It is a second
opinion that geometry cannot give: geometry knows something changed, only a
semantic model knows it was a remote control. The pipeline deliberately lets it
verify, not discover: it never sees the whole scene, only the candidates geometry
already produced. That keeps a hallucinating model from inventing changes out of
nothing, and it is why a missed change is a geometry failure, not a model failure.

**DINOv2 and SigLIP 2 embeddings.** An embedding turns an image into a vector, so
that "are these the same object" becomes "are these vectors close". DINOv2 is used
inside `src/describe.py` to match a cluster in video A with the same object in
video B, which is how a move is recognised. SigLIP 2 is used in the naming work
because it has a text encoder: the same vector space holds images and words, so a
crop can be compared against a list of object names. That is the rerank.

**SAM 2.** A segmentation model that turns a rough box into a precise mask. It was
probed for mask refinement (`scripts/probe_sam2.py`) and the probe came back
negative; see section 6.

**Leave one scene out.** Tuning a setting on all six scenes and then reporting on
one of them is cheating, because the setting has seen the scene. Leave-one-out
tunes on five and scores the sixth, rotating through all six. Used for the naming
knobs.

**Development versus held-out scenes.** Development scenes (demo, scene2, scene3,
scene4) were looked at, tuned on and debugged against. Held-out scenes (scene5,
scene6) were frozen, pre-registered, run exactly once each, and are the only
honest estimate of how the pipeline behaves on a scene it has never seen.

**Oracle control.** Feed the ground truth through your own code and see what score
you get. If pushing perfect masks through the SceneDiff adapter scores 0.1147, then
no amount of detection tuning reaches a good score, and the ceiling is elsewhere.
This one experiment redirected the whole external benchmark effort.

**Wilson intervals.** With 19 samples, "10 wrong versus 2 wrong" looks decisive and
is not: the intervals overlap. Wilson intervals are the correct small-sample
interval for a proportion, and they are why this project says "the sample is too
small to settle it" instead of claiming a win.

---

## 4. Stage by stage

Each stage writes to `cache/<scene>/<stage>/` and can be re-run alone. The table of
what each reads and writes is in [docs/code_map.md](docs/code_map.md).

**extract_frames** (`src/extract_frames.py`). Samples the video, scores sharpness
by Laplacian variance, keeps the sharpest frames evenly spaced. Why it exists:
reconstruction quality collapses on blurry frames. What we chose: 3 fps sampling
and a cap of 30 frames per clip, both in the config. What we tried instead: keeping
every frame (too slow, no better) and a fixed absolute blur threshold only, which
failed on short clips and led to the adaptive fallback in section 6.

**reconstruct** (`src/reconstruct.py`). Runs VGGT over both clips together.
Inputs: frames. Outputs: `points.npz`, `cameras.npz`, `dense.npz`, `meta.json`.
What we tried instead: MapAnything (slower, no better on these scenes) and DA3
(patchy geometry on the demo). The comparison is in `docs/results.md`.

**align** (`src/align.py`). ICP between the two halves of the reconstruction. Off
by default; the demo benefited (15 mm to 11 mm static error), the risks are
documented in `docs/pipeline.md`.

**tabletop** (`src/tabletop.py`). Fits the dominant plane, builds the voxel grid
aligned to it, decides the mode. Why it exists: "12 mm above the table" is a much
better description than a raw 3D coordinate. What we tried instead: no plane at all
(`geometric`), which is kept as the fallback because on room-scale scenes the plane
is usually a floor or a wall.

**diff** (`src/diff.py`). Voxel comparison, visibility, coverage, clustering.
Inputs: the reconstruction. Outputs: changed clusters with their geometry and the
frames that show them. Why it exists: it is the part that actually finds changes.
Every threshold it uses is in `configs/default.yaml` next to a comment.

**describe** (`src/describe.py`). Crops, DINOv2 move matching, Qwen3-VL
verification and naming, confidence scoring, rejection of weak candidates.
Inputs: clusters and frames. Outputs: `crops.json`, `embeddings.npz`, `vlm.json`,
`matches.json`. Why the model only verifies: see section 3. What we tried instead:
letting the model propose changes (rejected, it invented them), a soft veto instead
of a hard filter (measured no better, kept off), and coverage-aware additions
(measured no better, kept off).

**summary** (`src/summary.py`). Turns `report.json` into sentences, reading nothing
else. Why: grounding. A summary that can read the caches can state a fact the
report does not contain, and then the reading and the checking disagree.

**report** (`src/report.py`). Writes `report.json` and `report.md`. One entry per
change: type, label, centroid, box, confidence, evidence frames, and the plane and
mode the run used.

**visualize** and **viewer** (`src/visualize.py`, `src/viewer.py`, `src/render_gif.py`).
Composites per change, the GIF, the checking sheet and the interactive page. Why:
every claim in the report should be checkable by eye, and the human overlay review
in `docs/heldout.md` is what that evidence was used for.

---

## 5. Decision log

Each entry: the options, the choice, the evidence, the trade-off.

1. **Reconstruction backend: VGGT.** Options were VGGT, MapAnything and DA3.
   Chosen because it produced the most stable geometry on the demo and all
   development scenes. Evidence: `docs/results.md`. Trade-off: 1B parameters, one
   GPU needed, and the model family is frozen, so a better backend later means
   re-validating everything.
2. **One joint reconstruction, not two.** Chosen because two separate scenes cannot
   be compared. Trade-off: clips must overlap in viewpoint, which is a real
   constraint on how the videos can be filmed.
3. **The VLM verifies and names, never discovers.** Chosen after the model invented
   objects when given the whole scene. Trade-off: a change geometry misses is lost,
   and the reported label quality depends on the model.
4. **Tabletop mode by default.** Chosen because the target is a tabletop scene and
   the relative height is far more useful than raw coordinates. Trade-off:
   documented on room-scale scenes in `docs/evaluation.md`, where the plane lands on
   a floor or a wall.
5. **A visibility and coverage check before anything is reported.** Chosen because
   unverified cells were the largest source of false positives. Trade-off: a real
   change in an area that is hidden in one take is silently dropped, which is why
   the reports carry an "unverified" list.
6. **The summary reads only `report.json`.** Chosen for grounding. Trade-off: the
   summary is terse, because it cannot add context the report does not hold.
7. **Moving the held-out runs to their own tag and never re-running them.** Chosen
   so the headline claim cannot be quietly improved after the fact. Trade-off: the
   held-out numbers stay what they were, including the failures.
8. **Humanising the text, and correcting the held-out summaries afterwards.**
   `v1.0.1` regenerated the summaries and accidentally changed the held-out summary
   text; `v1.0.3` restored those files to their `v1.0.0` content and made the tests
   accept both formats. The lesson is in section 6.
9. **Keeping every negative result.** The SAM 2 probe, the four accuracy flags and
   the naming gate are all recorded as failures in `docs/accuracy.md`,
   `docs/negative_results.md` and `docs/naming_v3.md`. Trade-off: the repository
   looks less impressive and is far more trustworthy.

---

## 6. What worked, and what failed, with numbers

**Worked.** The demo (3 of 3 changes, no extras), scene4 (3 of 3, no extras),
scene3 (2 of 2 types correct). Reproducibility: three runs per scene with fresh
caches produced identical entries on every scene tested, with scene5 giving 7
entries in all three runs. Runtime: scene5 7 minutes 7 seconds, scene6 18 minutes
53 seconds, both dominated by the VLM stage (`docs/timing.md`).

**Held out.** scene5: the chair removal was found as `removed` but fragmented into
three entries named "wooden leg", "chair leg" and "table leg", so the labels are
0 of 2 and the clock's move was missed. scene6: the remote removal was found and
named correctly, the cup's move and the apple's addition were missed, and one
extra entry (an electrical outlet) was a false positive. That is 1 of 2 type-only
and 1 of 3. These numbers are frozen at `v1.0.0` and checked by
`scripts/check_heldout_claims.py`, which exists because the numbers were once
restated wrongly, by me, in conversation.

**Accuracy sprint, all negative.** Baseline on six development scenes: type recall
71%, localized recall 86%, strict move recall 25%, name accuracy 60%, 4 extras and
3 fragments. Soft veto, coverage-aware additions, object-level re-matching and
fragment merging each measured identical to baseline, three runs each. All four
flags default OFF. `docs/accuracy.md`.

**SAM 2 probe, negative.** Mask refinement with SAM 2 plus a DINOv2 similarity
gate: the positive control separated (0.914 against a negative control of 0.085),
but the two target objects sat in the middle (cup 0.374, clock 0.460). The honest
statement is "did not match under this descriptor and gate", not "cannot be
matched". `docs/negative_results.md`.

**SceneDiff, closed.** 13 of 20 validation pairs and 10 of 10 test pairs could not
be processed at all at the frozen settings, because they yielded too few sharp
frames, so they are counted as empty predictions. Validation px/im IoU 0.0074
counting them as empty, 0.062 over the 7 processable pairs, obj/sc AP 0.0000. Five
adapter bugs were found and fixed in sequence, each documented in
`docs/evaluation.md`. The decisive experiment was the oracle control: pushing
ground-truth masks through the adapter gives 6 of 7 pairs above 0.5 IoU and an
obj/sc AP of 0.1147. With perfect masks the method scores 0.11 on this benchmark,
so the ceiling is the method and the domain, not the adapter. The plane check added
the mechanism: on these room-scale clips the automatic plane is a floor (4 pairs),
a wall (1) or a tilted surface (8), with only 1 to 48 percent of confident points
near it, and switching to the plane-free geometric mode produced identical entry
counts. There is no plane to build boxes on, and the candidates are not there
either.

**Adaptive sharpness, accepted as a usability fix.** Short clips yielded too few
frames and the pipeline refused to run. The fallback takes the sharpest N by rank
when the absolute threshold fails. It is verifiably safe: demo, scene3, scene4,
scene5 and scene6 are byte identical with it on, because it only runs on the failure
path (`scripts/verify_adaptive_identity.py`). scene2 went from aborting to running
with 35 frames, which is why it was enabled by default in `v1.1.0`. On the
SceneDiff pairs it made 12 of 13 processable and added zero true positives, so it
fixes the upstream failure and not the detection quality, which is stated plainly in
`docs/evaluation.md`.

**Naming, failed gate.** On 19 development entries: the labels as printed name
everything with a 47.4% precision and 52.6% wrong; the ladder with abstention names
a third, with 66.7% precision and 15.8% wrong; leave-one-out gives 21.1% coverage,
10.5% wrong, 78.9% abstained. Wilson intervals overlap (52.6% is [31.7, 72.7] and
10.5% is [2.9, 31.4]), so the sample cannot settle it. The larger measurement
explains why: on 317 labelled crops from the SceneDiff validation pairs, the SigLIP
2 rerank puts the correct name in its top 3 only 23.0% of the time, [18.7, 28.0].
The tier design can therefore fire on about a quarter of entries at best. Flags off.

**Mistakes, and how they were caught.**
* Held-out numbers restated wrongly in conversation: caught by the user, fixed by
  adding `scripts/check_heldout_claims.py` and a test, so the documents now fail the
  build if they disagree with the reports.
* `v1.0.1` regenerated the summaries and changed the held-out summary text, which
  should never change after the freeze: caught by diffing against `v1.0.0`,
  corrected in `v1.0.3`, and the diff is now empty.
* The added-change composite swapped before and after images: caught by the user
  looking at the overlay, fixed by choosing the images by tag rather than by
  assuming which is which.
* Five SceneDiff adapter bugs (wrong report path, wrong coordinate space, transposed
  ground-truth masks, tag-local versus reconstruction frame indices, and only 10 of
  20 pairs being extracted): caught because the sanity check refused to let the
  test split be scored before the validation slice was clean.
* A silent coverage collapse in the naming gate: the tuning objective could reach
  "no wrong names" by abstaining on everything. Caught by reading the per-scene
  numbers rather than the headline, and fixed with a coverage floor, which then
  revealed the real 23% rerank ceiling.
* The cluster pipeline came within one step of scoring a test split with a broken
  adapter, which is why the rule now is: no test score before the validation slice
  passes a written sanity check.

---

## 7. Honest limits, and why

* **It is built for a small tabletop scene.** Room-scale scenes break the plane
  assumption, measured in section 6. Why: the whole geometry is relative to a table.
* **Moves are the weak case.** 25% strict move recall. Why: a move is a removal and
  an addition that have to be recognised as the same object, which needs the crop
  matching to work across a viewpoint change.
* **Large objects fragment.** scene5's chair became three legs. Why: clustering
  splits on geometric gaps, and a chair's legs are separated in 3D.
* **Naming is unreliable.** 60% on development scenes, and the rerank cannot do
  better than a 23% top-3 hit rate on the labelled crops. Why: the vocabulary is
  coarse and SigLIP 2 does not separate a jar from a bottle.
* **The external benchmark is out of reach.** obj/sc AP 0.1147 with perfect masks.
  Why: room-scale scenes, a tabletop method, and a frame budget that the dataset's
  clips do not meet.
* **Not metric.** Distances are in reconstruction units, not centimetres, so the
  reports avoid absolute units. Why: the reconstruction has no scale reference.
* **Verified on six scenes.** Six is not a benchmark. Every claim in the README is
  scoped to those scenes, and the held-out numbers are the honest ones.

**With a week:** the room-scale scene first (the template is in
`docs/room_scale.md`), because the plane assumption is the largest single
limitation; then a naming fix that starts from the 23% rerank ceiling rather than
from the ladder, most likely a better vocabulary and a real segmentation model for
the crops; then move matching, to attack the 25%.

---

## 8. How it was built

**An AI coding agent wrote the implementation, under human direction.** The
challenge explicitly allows this. The human set the goals, the constraints and the
gates, reviewed the outputs by eye, and made every call that changed scope: what to
freeze, what to release, what to abandon.

What the agent did: read the environment and the cluster rules, built and ran the
pipeline, wrote the evaluation and scoring tools, ran the experiments, and wrote the
documentation including this file. What the human did: approved each irreversible
step, caught the swapped overlay images and the wrongly restated held-out numbers,
rejected results that did not hold up, and insisted on the honesty rules that
shaped the repository, including keeping every negative result.

**How the work was verified**, because that is the part that matters:
* **Frozen digests.** The detection files are pinned to `v0.12.0`; their sha256
  digests are recorded in `docs/heldout.md` and re-checked, so a later change cannot
  silently alter the released behaviour.
* **Held-out runs.** scene5 and scene6 were pre-registered, run once each, and
  never re-run, with the outputs committed at `v1.0.0` and never regenerated.
* **Golden tests.** `tests/test_summary.py` checks the summary invariants against
  the committed reports; `tests/test_heldout_claims.py` checks the documents against
  the data.
* **Equivalence evidence for behaviour-preserving changes.**
  `scripts/verify_adaptive_identity.py` proves the one accepted behaviour change
  cannot touch a run that already worked, by comparing every extracted frame by
  sha256.
* **Reproducibility runs.** Three runs per scene with fresh caches.
* **Oracle controls.** Ground truth pushed through the project's own code, to
  separate "our detection is bad" from "our plumbing is bad".

---

## 9. Interview prep

### Ten numbers to remember

1. Demo and scene4: 3 of 3 changes, 0 extras.
2. Held-out scene5: 1 of 2, type only; labels 0 of 2.
3. Held-out scene6: 1 of 3; one extra, a false positive.
4. Development baseline: type recall 71%, localized 86%, move 25%, naming 60%.
5. Reproducibility: identical entries across 3 runs per scene, fresh caches.
6. Runtime: scene5 7:07, scene6 18:53, VLM stage dominant.
7. SceneDiff: 13 of 20 validation and 10 of 10 test pairs unprocessable.
8. SceneDiff oracle ceiling: obj/sc AP 0.1147 with perfect masks.
9. Adaptive fallback: byte identical on 5 of 6 scenes; scene2 went from abort to 35 frames.
10. Naming: 52.6% wrong as printed, 10.5% wrong with abstention at 21% coverage, and a 23.0% rerank top-3 ceiling that caps it.

### Two minute script

"We take two phone videos of the same tabletop scene and say what changed. Both
clips are reconstructed into one 3D scene with a feed-forward model, so both takes
share one coordinate system. That is the core idea: with a shared frame, a moved
object is a geometric fact, not a guess. We put a 5 mm voxel grid over the table,
keep the cells whose contents differ between the takes, drop the ones that were not
visible in both, and cluster the rest into candidates. A vision-language model then
verifies each candidate and names it, and we print one sentence per change. On the
development scenes it finds 71% of changes with the right type. The held-out scenes
are the honest number: one of two on scene5, one of three on scene6. The known
weaknesses are moves at 25% and object naming at 60%, and I have the failure
analysis for both. Everything is reproducible, and every negative result is in the
repository."

### Five minute script

The two minute script, then: the visibility check and why it removes most false
positives; the move matching with DINOv2 and why moves are the weak case; the
external benchmark and the oracle control that showed the ceiling is 0.11 with
perfect masks, which is why I stopped investing there; the naming work, where the
ladder cut wrong names from 52.6% to 10.5% and I nevertheless kept the flag off
because the intervals overlap and the rerank tops out at a 23% top-3 hit rate; and
the process, which is the part I would emphasise: frozen digests, pre-registered
held-out runs, a sanity check that refuses to score a test split first, and the
mistakes I made and how they were caught, including a documentation error that now
fails the test suite.

### Twenty-five questions, with short honest answers

1. **What does it do?** Two videos in, a 3D reconstruction, the changed objects,
   and sentences describing them.
2. **How do you get 3D from two videos?** A feed-forward reconstruction model run
   over both clips at once.
3. **Why one reconstruction and not two?** Separate reconstructions have no common
   frame, so nothing can be compared.
4. **How do you decide something changed?** 5 mm voxels, compared between takes,
   with a visibility check.
5. **What was the biggest source of false positives?** Cells that were not visible
   in both takes. The visibility check fixed most of it; one false positive still
   survived on held-out scene6.
6. **How does the pipeline know an object moved rather than being removed and
   added?** DINOv2 crops are matched before and after; if they match well enough, it
   is one `moved` entry.
7. **Why does the model not find the changes itself?** It invented them when given
   the whole scene. It only sees candidates geometry already produced.
8. **What is the accuracy?** 71% type recall on development scenes; 1 of 2 and 1 of
   3 on the two held-out scenes.
9. **What is the worst part?** Moves, 25% strict recall.
10. **Why?** A move has to survive both change detection and cross-view crop
    matching.
11. **How do you know it is reproducible?** Three runs per scene with fresh caches,
    identical entries.
12. **How long does it take?** 7 to 19 minutes per scene, dominated by the VLM.
13. **Does it need a GPU?** Yes for the reconstruction and the VLM; there is a
    documented no-GPU path for reading and re-rendering.
14. **How did you evaluate on an external benchmark?** A SceneDiff adapter,
    validation slice first, test slice once, and a written rule that unprocessable
    pairs count as empty.
15. **How did that go?** Badly, and I can explain why: 13 of 20 validation and 10 of
    10 test pairs could not be processed at all.
16. **Did you not fix the adapter?** Five bugs were found and fixed, and then an
    oracle control showed the ceiling with perfect masks is 0.1147, so the remaining
    gap is the method and the domain.
17. **What is an oracle control?** Feed ground truth through your own pipeline to
    separate a detection failure from a plumbing failure.
18. **What did you try that did not work?** Soft veto, coverage-aware additions,
    object re-matching, fragment merging, a SAM 2 mask refinement probe. All
    measured, all recorded as negative.
19. **Why keep negative results?** Because otherwise I cannot tell a good decision
    from a lucky one, and a reviewer cannot either.
20. **What is the naming work?** A closed vocabulary, a SigLIP 2 rerank and three
    tiers, aimed at never printing a confident wrong name.
21. **Did it work?** Partly. Wrong names fell from 52.6% to 10.5%, but coverage fell
    to 21% and the intervals overlap at 19 samples, so the flag stays off.
22. **What is the real naming limit?** The rerank puts the right name in the top 3
    only 23% of the time over 317 labelled crops.
23. **What would you do with another week?** Room-scale scenes, a better naming
    vocabulary and segmentation, then move matching.
24. **What are you least sure about?** Every number rests on six scenes, two of them
    held out. Twenty-one entries. It is a demonstration, not a benchmark.
25. **What are you most confident about?** The process: frozen digests, one
    pre-registered held-out run each, and mistakes that are written down.

### Questions I should not overclaim on

* Anything implying generalisation beyond the six scenes.
* Any state-of-the-art comparison. There is none, and the external benchmark result
  is poor.
* Metric distances, in centimetres or millimetres. The scale is not validated.
* Speed claims: the numbers are wall clock on a shared cluster, and the VLM stage
  varies with load.
* The naming result: it is a large relative drop on 19 samples, not a settled
  improvement.
* The adaptive fallback as a capability: it is a usability fix that adds zero true
  positives on the external pairs.

---

## 10. Glossary and reading order

**Glossary.** *point map*: a 3D point per pixel. *camera pose*: where the camera was
and which way it looked, per frame. *voxel*: a 5 mm cube of space. *cluster*: a
group of connected changed voxels, the unit the pipeline reports. *unverified
entry*: a change the model did not confirm; listed separately, never counted as a
finding. *extra*: a reported entry with no ground-truth change behind it. *fragment*:
several entries that are really one object. *oracle control*: ground truth pushed
through the pipeline to find the ceiling. *leave one scene out*: tune on five scenes,
score the sixth. *held out*: never looked at during development, run once. *Wilson
interval*: the correct confidence interval for a proportion with small counts.

**Reading order.** `README.md` first screen, then `docs/pipeline.md`, then
`detect.py` and the stage modules in the order listed in
[docs/code_map.md](docs/code_map.md), then `docs/evaluation.md` for the scoring
rules, then section 5 of this file for the decisions, then section 6 for what
actually happened.

**Where the evidence lives.** `docs/heldout.md` for the pre-registered held-out
runs; `docs/results.md` for per-scene results and timings; `docs/accuracy.md` for
the sprint metrics and the negative flags; `docs/negative_results.md` for the SAM 2
probe and the closed tracks; `docs/evaluation.md` for SceneDiff including the five
adapter bugs and the oracle control; `docs/naming_v3.md` and `docs/naming_v4.md` for
the naming round; `docs/timing.md` for measured runtimes; `docs/devlog.md` for the
chronological log.
