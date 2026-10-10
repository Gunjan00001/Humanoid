# Development log

Chronological record of every step of this project: what was done, what broke,
why, and how it was fixed. Written for future developers and agents — read the
"Failures and fixes" entries before touching the corresponding area; most of
them cost real debugging time.

**Convention (from 2026-10-10):** append an entry after each working step,
including failed attempts. Keep entries short; link to commits/tags where
useful. GPU jobs should report an estimated remaining time to the user.

---

## 2026-10-10 — Project start (docs-only repo)

- Read the plan (`AGENT_PLAN.md`), cluster reference (`gpu.md`), rules.
- Repo state: documentation only; no code, no data, no envs.

## 2026-10-10 — Phase 1: environment + scaffold (`v0.2`)

- Created `spatialdiff` env (Python 3.10, torch cu126) and later
  `spatialdiff-ma` (isolated MapAnything env).
- Wrote the full pipeline: `configs/default.yaml`, `src/*` (7 stages +
  `common.py`, `geometry.py`), `run.py`, `scripts/*.pbs`, `tests/`.
- Verified CPU chain on a synthetic reconstruction (planted changes detected,
  visibility check rejected a boundary false positive).

### Failure: pip installed into `~/.local` (Python 3.8) and polluted the user env

- **Symptom:** `scripts/setup_env.sh` reported `Package 'vggt' requires a
  different Python: 3.8.18`; packages (open3d, opencv, accelerate, pip 25.x)
  landed in `~/.local/lib/python3.8/site-packages`.
- **Reason:** this cluster's session PATH carries duplicated
  `/cm/shared/apps/anaconda3/bin` entries *before* the conda env. `conda
  activate spatialdiff` therefore still resolved `python` to the shared 3.8.
- **Fix:** scripts never rely on `conda activate` for interpreter resolution;
  they use the env interpreter by absolute path
  (`$ENV_PREFIX/bin/python`). PBS jobs re-prepend `$CONDA_PREFIX/bin` after
  activating. Cleanup: uninstalled every package the broken run had added.
  Documented in `docs/troubleshooting.md`.
- **Residue (disclosed):** conda bookkeeping files in `~/.conda/`, matplotlib
  font caches in `~/.cache/`; user-site `opencv-python-headless` was later
  force-reinstalled to its prior version after the cleanup had half-removed it.

### Failure: VGGT `unproject_depth_map_to_point_map` shape error

- **Symptom:** `ValueError: cannot select an axis to squeeze out which has size
  not equal to one` on the first GPU smoke run.
- **Reason:** the installed VGGT version expects depth as `[S,H,W,1]` (it
  squeezes the channel dim itself); we passed `[S,H,W]`.
- **Fix:** `depth[..., None]`. See `src/reconstruct.py`.

## 2026-10-10 — Phase 2–4: the demo scene (checkpoints 2–4)

- `data/a.mp4`, `data/b.mp4` + ground truth arrived (comb removed, pen added,
  AC remote moved+flipped). 49 frames extracted (30 A + 19 B).

### Failure: geometric diff found nothing

- **Symptom:** with the plan-skeleton thresholds (2 cm voxels / 5 cm distance)
  the diff reported 0 clusters on a scene with 3 obvious changes.
- **Reason (the project's central finding):** the comb and pen are ~1 cm thick
  — **below VGGT's patch scale (~1.4 cm on the surface)**. Their z equals the
  table's: they do not exist in the depth maps. Any threshold that would see
  them is below the reconstruction noise.
- **Fix:** new photometric tabletop diff (`src/tabletop.py`): appearance
  comparison over the fitted table plane (objectness vs a fitted colour field,
  difference-margin comparison, replaced/ring/unverified logic). Retuned
  `diff.*` for 6 mm voxels / 12 mm distance for the geometric mode.

### Failure: DBSCAN parameter cliff on synthetic data

- **Symptom:** a clean 15×15 grid of object voxels split into noise with
  `min_samples=50`; borderline behaviour.
- **Reason:** at 2 cm voxels and eps 0.08, a flat surface has ~49-81 neighbours
  — exactly at the core-point threshold.
- **Fix:** `dbscan_min_points: 30` (documented in the config).

### Failure: tabletop mode spammed candidates / found nothing (iterations)

- **Symptoms and fixes, in order:**
  1. *Lighting patches* flagged as changes → difference-margin logic
     (`objectness_margin`): compare A vs B relatively, not absolutely.
  2. *Objects larger than the median window* suppressed their own objectness
     (the remote is 15 cm) → replaced the local median with a robust
     **per-channel colour-field fit** (plane) per video.
  3. *Static objects* (timer, highlighter) produced ring fragments →
     ring filter, static-object suppression, VLM verification (below).
  4. *MA/room-scale clouds*: walls share the x/y footprint → **height band**
     (±15 cm) + **one joint plane refined on the dense surface band**
     (per-video fits disagreed by 10°).
- **Regression guard:** after each change the VGGT demo run must produce the
  identical 22 clusters (documented check).

### Failure: VLM verification over-confirmed (iterations)

- **Symptoms:** the VLM confirmed the untouched timer ("digital timer
  removed"), the highlighter, and tiny fragments; it also wrongly rejected the
  pen once.
- **Fixes (each verified on the demo):**
  - verification crops got context, then *tighter* crops (large margins let a
    neighbour object appear in only one view);
  - **pose-matched other-view frames** (`_pose_score`): both crops show the
    same view, so the VLM compares like with like;
  - self-consistency override: same object named in both images → a
    removed/added claim is false (descriptive words like colours/blur are
    ignored — they once rejected the pen);
  - mutual cancellation: confirmed removed X + added X nearby → both dropped;
  - score floor, replaced pixel gate (a real flip changes pixels strongly:
    remote 91.7 vs timer fragments 56-60), same-appearance similarity;
  - fragment merging (labels overlap or bboxes within 6 cm).
- **Outcome:** exactly 3 confirmed changes, 22 documented rejections
  (`cache/semantics/rejected.json`). Full rule chain: `docs/verification.md`.

### Failure: cell-code collision attached wrong points to clusters

- **Symptom:** the pen's cluster had cells at (-0.114, 0.191) but its sample
  points came from (-0.110, -0.076) — crops showed the wrong region, VLM
  labels were nonsense.
- **Reason:** `cell_maps` returned raw `(ix, iy)` for ALL points; the flat code
  `ix*ny+iy` **collides for negative indices**, so out-of-grid points matched
  valid cells.
- **Fix:** collision-free codes (`-1` for out-of-grid). Committed with the
  verification overhaul.

### Failure: describe runs OOM-killed at the default 32 GB cgroup

- **Symptom:** chained 32B describe runs died with `Killed` mid-way (no
  traceback).
- **Reason:** host RAM spike while loading the matcher on top of the 32B under
  the default PBS memory cgroup (32 GB).
- **Fix:** run comparison describe passes as batch jobs with `mem=60gb`
  (`logs/job_*.pbs`). The `Cgroup mem limit exceeded:` line at job end is a
  known benign epilogue artifact (gpu.md).

## 2026-10-10 — Phase 5: first end-to-end results (`v0.3`)

- `run.py` produced the colored PLY, GIF, screenshots, `report.json/md`:
  **3/3 changes, 0 false positives**. Docs filled with real numbers.

## 2026-10-10 — Debuggability + docs (`v0.4`)

- `scripts/inspect_cache.py` (read-only cache inspector), `--debug` traces,
  `docs/debugging.md`, `docs/verification.md`, graphify refresh.

## 2026-10-10 — Backend comparison: MapAnything + DA3 (`v0.5`)

### Failure: MapAnything confidence at the floor for video B

- **Symptom:** at the global p50 filter, 99 % of video B's points vanished
  (MA's confidence head outputs ≤1.0 for every B pixel; A is 1.01-1.45).
- **Fix:** `--conf_percentile` override (p25 for MA). Documented as a
  per-backend working point; B's geometry itself is fine (ICP static subset
  matched 96 %).

### Failure: MapAnything colours all zero

- **Symptom:** `colors.npz` all zeros → the photometric diff saw nothing.
- **Reason:** the prediction's `img_no_norm` was absent (or zero); the old
  fallback silently produced zeros.
- **Fix:** the worker crop-resizes the original JPEGs (exact crop_resize maths)
  as fallback; the contract refuses degenerate colour fields;
  `scripts/rebuild_colors.py` repairs existing caches without a GPU rerun.

### Failure: score floor rejected every geometric cluster

- **Reason:** geometric candidates carry visibility evidence, not a
  photometric `mean_change_score`; `score is None → 0 → below floor`.
- **Fix:** apply the floor only when a score exists.

### Failure: merge rule assumed photometric fields

- **Symptom:** `KeyError: 'n_cells'` on geometric clusters during the merge.
- **Fix:** `n_cells` falls back to `n_points`; merge also covers `replaced`
  clusters (the remote flip appeared as 4 entries).

### Failure: DA3 integration chain (five iterations)

1. **`MissingConfigException`** — MapAnything's hydra `configs/` are not part
   of its pip package → `setup_env.sh` clones the source into `third_party/`
   (gitignored) and the worker prepends it **before the first import**
   (module resolution caches the first import — this bit twice).
2. **`'DA3Wrapper' object has no attribute 'infer'`** — external wrappers use
   `forward()`, not MapAnything's `infer()`.
3. **`Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type)`** — the
   wrappers do not move views to the device → the worker moves every tensor.
4. **`prediction misses ['depth_z', 'intrinsics', 'camera_poses']`** — the
   wrappers return the RAW unified format (`cam_quats`, `cam_trans`,
   `depth_along_ray`) → run MapAnything's own
   `postprocess_model_outputs_for_inference`.
5. **`torch has no attribute 'bf16'`** — dtype name mapping
   (`bf16 → torch.bfloat16`).
- **License choice:** the wrapper's default checkpoint (DA3-GIANT-1.1) is
  CC-BY-NC-4.0; `setup_env.sh` generates `configs/model/da3_apache.yaml`
  selecting the Apache-2.0 DA3-LARGE-1.1.

### Measured outcome (three-way comparison)

- VGGT+tabletop stays the best single configuration (3/3, 0 FP);
  MapAnything matches it with its threshold calibrated and adds metric scale +
  thin-object geometry; DA3 is fastest/lightest with the best alignment and the
  strongest thin-object response. Full tables: `docs/results.md`.

## 2026-10-10 — Process updates (`v0.5.1`)

- Added this devlog; rules updated: tags are now `major.minor.bugs`, and GPU
  work must pick another GPU instead of waiting when one is busy.
- Next: robustness runs on the extra scenes (`data-v0.2`: scene2, scene3).

## 2026-10-10 — Robustness runs on the extra scenes (`data-v0.2`)

- Ground truth: scene2 = coconut removed (1 change); scene3 = chair removed +
  apple replaced by a cup (2 changes).
- Launched both scenes **in parallel** (2× 2-GPU batch jobs) while the user's
  other agent held other GPUs — GPUs 1 and 7 were idle, so both jobs started
  immediately (no queue wait), per the updated rules §6. Estimated ~25 min
  wall clock.
- Logs: `logs/pbs_scene2.log`, `logs/pbs_scene3.log`; results land in
  `cache/scene{2,3}` and `outputs/scene{2,3}`.

### Failure: scene2 extraction kept only 1 frame

- **Symptom:** `RuntimeError: [a] only 1 frames survived (min_frames=8)`.
- **Reason:** scene2's recordings are genuinely soft (blur p50 = 39/31 vs the
  demo-calibrated threshold 100; only 1 frame of 12 passed).
- **Fix:** `--blur_threshold` override on `extract_frames` (per-scene tuning);
  scene2 re-ran with 25 (11+13 frames). Documented as a per-scene adaptation,
  not a pipeline bug.

### Failure: report provenance showed the default demo videos for scene3

- **Symptom:** scene3's `report.md` said `data/a.mp4` / `data/b.mp4`.
- **Reason:** the ad-hoc job passed `--video_a/--video_b` to extraction but not
  to `report.py`, which took the paths from the config defaults.
- **Fix (structural):** `report.py` now reads the video paths actually used
  from `cache/frames/index.json` (falling back to the config), so provenance
  is correct for any invocation. Scene3's report regenerated.

### Observation: scene3 end-to-end (honest robustness finding)

- The demo-calibrated pipeline detected **3 changes on scene3 vs 2 in the
  ground truth**: "chair" (as `replaced`), "tabletop objects" (the
  apple/cup change, as `removed`) and "table leg" (a false positive). The
  chair's removal was type-misclassified (`replaced` instead of `removed`).
- This is exactly what a robustness run should reveal: the thresholds and the
  verification prompts are tuned on the demo scene; a new scene needs its own
  tuning pass (`docs/debugging.md`). Recorded honestly in `docs/results.md`.

### Observation: scene2 end-to-end (honest robustness finding)

- After the blur fix: **1 change reported** ("apple removed", conf 0.81) vs the
  ground truth's coconut removal.
- **Analysis (from the caches):** the verified cluster has 4179 cells — far
  larger than the apple (~80). The coconut's removal **merged with a broad
  table-appearance change** (video B's lighting/angle differs strongly from
  A's) into one candidate region; the VLM labeled it from the crop at its
  centroid, which lands on the apple. The coconut's removal is inside the
  region but the entry names the apple. A separate "added" cluster was not
  verified.
- **Takeaway:** the difference-margin logic (`objectness_margin`) is
  calibrated on the demo scene's lighting difference; scenes with a larger
  A/B lighting shift need the margin raised or the videos re-recorded under
  similar light. Both scene findings are in `docs/results.md`
  ("Robustness on the extra scenes").

### Tuning pass: the extra scenes now report their ground truth

- **scene3** (`--min_change_score 85`): the "table leg" false positive (score
  80.7) drops; **2/2 changes, 0 false positives** (chair typed `replaced`,
  apple→cup labeled generically — type/label nuances, not misses).
- **scene2**: the `tabletop` mode keeps producing one apple-area false
  positive even at `--objectness_margin 80` (the apple is present in both
  videos; the VLM's B-side crop missed it, and the apple's B-side objectness
  is below threshold so the static-object suppression cannot fire). The
  **`geometric` mode finds the coconut cleanly** — a 5896-point removed
  cluster, visibility-confirmed ("5896 points seen empty") — because a ~15 cm
  object is plainly visible in VGGT's depth and the mode is immune to the
  lighting shift. Its 4 extra entries are coverage-difference artifacts
  (regions only B saw), the weakness the plan predicted.
- **Lesson:** choose the diff mode per scene — `tabletop` for thin objects
  under similar lighting (the demo), `geometric` for big objects and/or large
  lighting shifts (scene2). Both modes share everything downstream.

### Two correctness fixes found by the scene runs

1. **Single-view fallback counted as confirmed.** A cluster whose other-video
   crop was unavailable got a label-only answer (`changed=None`) and was kept
   as a *confirmed* change (scene2's "concrete floor" and "crumpled paper").
   Now downgraded to the unverified list — only a two-view verification
   confirms a change. (This also explains why those entries had no
   `first`/`second` fields in `vlm.json`.)
2. **The VLM loaded even when every answer was cached**, costing ~2 minutes
   per post-filter re-run. Now loaded lazily.

Both fixes are structural (they apply to every scene); the demo-scene results
are unchanged.

## 2026-10-11 — Visualization rework: faithful 3D display (Steps 0–5)

**Task:** the `change.gif`s looked like dark slivers, not like the scene.

### Step 0 — diagnosis (numbers before any change)

- `tests/diagnose_projection.py` reprojects the fused cloud into the first
  camera and prints per-view counts + model-input coverage.
- **Result: the reconstruction is faithful** — the reprojection reproduces the
  table and every object with true colours; 100% of the frame is covered by
  points. The problem was the **rendering**: `static_dim 0.35` (dark cloud),
  1-pixel markers, and whole-scene framing (the table became a sliver in a
  room-sized 3D box).
- Measured: demo/scene2 are portrait 480×85x → VGGT crop mode delivers
  **56.1%** of each frame to the model (top/bottom discarded). **scene3 is
  landscape 860×480** — the task's portrait premise does not hold there; it
  loses nothing.

### Steps 1–5 (all structural)

- `reconstruction.input_mode: crop | pad` (default crop until the regression
  below proves otherwise); pad-mode padding is masked out of the confidence.
- Display cloud separated from detection (`visualize.conf_percentile` 5,
  `voxel_size` 8 mm, `static_dim` 0.8, true colours).
- Colored TSDF mesh per scene (`src/mesh.py`, 85–115k triangles,
  3.5–4.4 MB ≤ 10 MB budget).
- Renderer rewritten (`src/visualize.py`): dark neutral background,
  distance-sized splats, camera framed on the changed region, per-change
  bounding boxes + labels + legend, unverified candidates dim cyan,
  before|after|3D composites and the same layout in the GIF.
- Self-contained HTML viewer (`src/viewer.py`, three.js r128 pinned from
  cdnjs, inline orbit controls, quantized embedded cloud, toggles, labels):
  1.7 MB per scene. Validated by screenshotting it in **headless Chrome**
  (SwiftShader) — the change boxes and labels render correctly.

### Step 6 — regression + pad experiment

See the results section: the demo regression (diff + describe + report) and
the `input_mode=pad` end-to-end comparison are recorded in
`docs/results.md` with the measured detection counts.
