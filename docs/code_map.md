# Code map

Where everything lives, what it does, and the order to read it in. For the
concepts behind the code see [DEEPDIVE.md](../DEEPDIVE.md); for how to run it see
the README.

## The tree

```
detect.py              entry point: runs the stages, writes outputs/<scene>/
configs/default.yaml   every threshold in the pipeline, one global file

src/                   the pipeline, one module per stage
  common.py            config loading, logging, small shared helpers
  extract_frames.py    stage 1: scan both videos, keep the sharp frames
  reconstruct.py       stage 2: VGGT (or another backend) to 3D, both videos together
  align.py             stage 3: ICP refinement between the two reconstructions
  geometry.py          shared geometry: load a reconstruction, tagged points, transforms
  tabletop.py          the tabletop plane, the 5 mm voxel grid, the mode choice
  diff.py              stage 4: which voxels changed, clustered into candidates
  describe.py          stage 5: verify and name each candidate, drop the weak ones
  summary.py           stage 6: the plain-language summary, built only from report fields
  report.py            stage 7: writes report.json and report.md
  visualize.py         stage 8: overlays, composites, the change GIF, the checking sheet
  viewer.py            stage 9: the interactive HTML viewer
  render_gif.py        GIF rendering used by visualize.py
  mesh.py              scene mesh export for the viewer
  extract_frames.py    (see above)
  ma_contract.py       the MapAnything backend contract, used by the comparison
  eval_scenediff.py    the SceneDiff adapter, separate from the pipeline

scripts/               tools that are not part of a normal run
  score.py             scoring against the human-resolved matches
  check_numbers.py     every number in the docs must trace to a log or a report
  check_heldout_claims.py  the held-out sentences in README/docs must match the reports
  make_figures.py      the README figures
  plane_check.py       which plane the auto mode picks, per scene
  verify_adaptive_identity.py  proves the sharpness fallback cannot change good runs
  build_naming_set.py  labelled crops from the SceneDiff validation pairs
  naming_set_eval.py   naming measured on those crops, with intervals
  naming_v3_eval.py    the naming ladder measured on the six scenes
  repro_scenes.sh      the reproducibility runs
  job_*.pbs            the PBS job files used on the cluster

tests/                 golden and invariant tests (unittest, no GPU needed)
  test_summary.py      summary invariants against the committed reports
  test_heldout_claims.py  the held-out numbers in the docs
  test_ma_contract.py  the MapAnything contract
  prepare_gpu_smoke.py, compare_backends.py, make_test_videos.py  cluster helpers

docs/                  one topic per file, see docs/README.md for the index
outputs/<scene>/       one directory per run: report.json, summary.txt, viewer.html,
                       change.gif, composites, and for some scenes the mesh
data/<scene>/          the recordings, plus ground_truth.json and matches.json
cache/<scene>/         per-stage caches, the reason a re-run is cheap
```

## Reading order

1. `README.md` first 60 lines: what it does and what it scores.
2. `docs/pipeline.md`: the stages in order, with the files each one passes on.
3. `detect.py`: the stage runner; every stage name in `STAGES` maps to a module.
4. `src/extract_frames.py`, `src/reconstruct.py`, `src/tabletop.py`, `src/diff.py`,
   `src/describe.py`, `src/summary.py`, `src/report.py`: the pipeline in order.
5. `src/geometry.py` and `src/common.py` on demand: shared helpers.
6. `docs/evaluation.md`: how everything is scored, including the failure rule.
7. `DEEPDIVE.md`: the decisions, the numbers and the mistakes.

## Data flow, stage by stage

Each stage reads the previous stage's cache and writes its own, so any stage can
be re-run alone. Paths are `cache/<scene>/<stage>/`.

| stage | reads | writes | notes |
|---|---|---|---|
| extract_frames | the two videos | `frames/a/*.jpg`, `frames/b/*.jpg`, `frames/index.json` | frame index, timestamp and blur score per kept frame |
| reconstruct | `frames/` | `reconstruction/meta.json`, `points.npz`, `cameras.npz`, `dense.npz` | both videos in one optimisation; `cameras.npz` holds the per-frame poses |
| align | `reconstruction/` | `aligned/` transforms | ICP between the two halves, off by default |
| diff | `reconstruction/` | `diff/points.npz`, `diff/clusters.json`, `diff/visibility.json` | the plane, the 5 mm grid, changed voxels, candidate clusters |
| describe | `diff/`, `frames/` | `semantics/crops.json`, `semantics/embeddings.npz`, `semantics/vlm.json`, `semantics/matches.json` | crops per cluster, DINOv2 embeddings, the VLM verdict, move matching |
| summary | `report.json` | `summary.json`, `summary.txt` | wording only; never reads the caches, so it can never add a fact |
| report | `diff/`, `semantics/` | `report.json`, `report.md` | the single source of truth for everything downstream |
| visualize | `report.json`, `frames/`, `reconstruction/` | `changes/*.png`, `change.gif`, `checking_sheet.png`, `scene_mesh.ply` | the evidence a human checks |
| viewer | `report.json`, the renders | `viewer.html` | generated, git-tracked, never LFS |

The SceneDiff adapter (`src/eval_scenediff.py`, `scripts/job_scenediff.pbs`) is a
separate path: it runs `detect.py` per pair and converts the result into the
benchmark's `object_masks.pkl` format. It is not part of a normal run.

## Experimental tooling (not used by detect.py)

These files are evidence and measurement, kept on main so the claims in the docs
can be re-checked. None of them runs during a normal pipeline run, and each one
carries an EXPERIMENTAL header.

| file | what it does |
|---|---|
| `scripts/score.py` | scoring against the human-resolved matches, the metric definitions |
| `scripts/sprint.py`, `scripts/worker.py` | the accuracy-sprint orchestrator and its long-lived semantics-only worker |
| `scripts/probe_sam2.py`, `scripts/probe_sam2_controls.py` | the SAM 2 mask refinement probe and its controls, a recorded negative |
| `scripts/oracle_projection_control.py` | pushes ground truth through the adapter to find the SceneDiff ceiling |
| `scripts/plane_check.py` | which plane the automatic mode picks, per scene |
| `scripts/build_naming_set.py` | labelled crops from the SceneDiff validation pairs |
| `scripts/naming_set_eval.py` | naming measured on those crops, with Wilson intervals |
| `scripts/naming_v3_eval.py` | the naming ladder and tiers measured on the six scenes |
| `src/eval_scenediff.py` | the SceneDiff adapter, separate from `detect.py` |
| `src/naming/` | the naming package: text, vocabulary, descriptors, ladder, tiers, rerank, cache, evaluation |

`scripts/verify_adaptive_identity.py` is in this group too, but note that the
behaviour it protects (`frames.adaptive_sharpness`) *is* part of the default path
since v1.1.0; the script itself only runs when you ask it to.


## The readability pass (v1.2.3)

The six frozen detection files and the harness now carry stage-position,
input and output statements in their module docstrings, `why` reasoning for
every threshold and non-obvious conversion (crop and pad modes, the
occlusion-aware rule, the plane inliers, the alignment's static-subset rule),
and `why` comments at the two-column trickier loops. The digest ledger in
docs/accuracy.md has one pre-to-post row per file: geometry, diff, tabletop,
describe, reconstruct, align. Golden summaries under tests/golden_summary
regenerate byte identical, so the refactor provably changed no behaviour.

## Unattended and experimental tooling (added after the readability pass)

| path | what it does |
|---|---|
| `scripts/run_batch.py`, `scripts/run_batch.pbs`, `data/batch_manifest.json` | the unattended task runner: skips `.done` tasks, pins the idlest GPU with backoff, per-task logs named with the job id, retries failures once, resubmits itself at the walltime |
| `scripts/geometry_eval.py` | one reconstruction backend at one frame budget: extract, reconstruct, static residual, density |
| `scripts/objfirst_gate.py` | Phase 4 gate evaluation on cached data, deterministic rerun |
| `src/objfirst/` | EXPERIMENTAL, not used by detect.py: M1 inventory, M2 SAM 3.1 bridge, M3 lift, M4 DINOv3 matching, M5 verification/fusion |
| `scripts/objfirst_segment.py` | SAM 3.1 mask pass in the `spatialdiff-sam3` env |
| `scripts/objfirst_extents.py` | per-label extent distribution used by the merge rule |
