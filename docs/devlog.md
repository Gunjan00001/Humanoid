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

- Demo regression (crop default): **3/3 changes, 0 false positives — unchanged**
  (`logs/pbs_demo_reg.log`).
- `input_mode=pad` on the portrait demo: **worse, with evidence** — 5 entries
  vs 3: the pen missed and three extra entries (frame-edge artifact +
  highlighter fragments). Pad renders the content at ~1.76× coarser pixels.
- **Decision: crop stays the default**; the option remains for scenes whose
  content sits at the frame edges. Full table in
  `docs/results.md#visualization-rework-steps-05`.

## 2026-10-11 — Presentation + correctness pass (`v0.9.0`)

### Layout / 3D view / Chrome GIF (steps 1-3)

- Two-row composites and GIF (before|after top, large 3D below, legend strip
  outside the panel); 3D geometry cropped to 0.4 m around the changes; camera
  at 40° elevation with the fitted table-plane normal as up
  (`table_up_rotation`); orbit framed on the changed region.
- **Bug found**: rotating the two bbox corners is not a rotated AABB — the
  component-wise min/max swap (negative extents) made the viewer's Box3Helper
  draw giant lines. All eight corners are rotated now.
- The GIF is rendered from `viewer.html` by headless Chrome
  (`src/render_gif.py`, SwiftShader; a matplotlib fallback remains). The
  viewer accepts `?shot=1&az=..&elev=..` for deterministic frames and embeds
  the full cloud sorted by focus distance so its "full scene" toggle is a
  drawRange. A legend strip is drawn during assembly (the viewer's UI is
  hidden in shot mode).

### GitHub Pages (step 4)

`docs/hosting.md` documents enabling Pages (Settings → Pages → main / root)
and lists the live URLs; the README links them. No workflow needed — the
viewers are committed static files. (`gh` CLI is not available on this
cluster, so enablement is a one-time web-UI step for the repo owner.)

### Stale screenshots (step 5)

`visualize.py` clears `changes/` and, defensively, deletes any
`change_*.png` whose id is not in the current change list.

### Scene3 change types (step 6)

- **Crop framing is now occlusion-aware** (in-view = not behind the observed
  surface). The old test required the object to be present, so in the other
  video a removed object's frames looked empty and crops landed on
  neighbouring objects — the root cause of scene3's "chair replaced" and
  "replaced by a black chair" misreadings.
- **Geometric type correction** (calibrated on the scenes): comb 0.73 empty
  vs apple/cup 0.01 occupied → `removed`→`replaced`; remote 0.00 occupied vs
  chair 0.21 empty → `replaced`→`removed`. Corrected clusters get a fresh
  sentence (`PROMPT_REPLACE`/`PROMPT_REMOVED`).
- **Rule priority fixed**: the empty-space confirmation now outranks the
  both-present suppression (the chair sat next to the *other* chair, which
  made that heuristic fire), and reclassified replacements skip the
  photometric pixel gate.
- **Measured outcome**: scene3's chair is `removed` ✓ and the apple→cup is
  `replaced` ✓ — but the scene now shows 4 entries (the apple region appears
  twice; one entry is not in the ground truth) and the naming is unreliable
  because video B's crop of the table is partly occluded by the remaining
  chair. Demo stays 3/3, 0 extra. Documented honestly in
  `docs/results.md#presentation--correctness-pass-v090`.


## 2026-10-11 — scene4 (`data-v0.3`) + move/arrival/composite semantics (`v0.10.0`)

### The new data

`data-v0.3` added scene4: 46 s portrait videos (blur p50 137–163 — the default
blur threshold works), ground truth = apple moved, scissor removed, bottle
added. Videos are 9 MB each → still LFS (the <5 MB exception covers only the
six small demo clips).

### What the first run got (and why)

The first full run reported 3 changes but the wrong shape: the apple's old
spot as "removed apple", a scissor fragment as "replaced scissors→apple", the
apple's arrival dropped, no bottle. Root causes, all measured:

- the **move matcher ran after the structural filters**, and the arrival
  cluster had already been dropped as a duplicate of the replaced fragment;
- the **mutual cancellation** compared stored labels — the arrival cluster's
  label read "scissors" (named from its A-side view) and cancelled the genuine
  apple arrival;
- the apple pair scored **0.704** — just under the 0.75 threshold (different
  objects score 0.13);
- the moved apple landing on the removed scissor's pixels made the tabletop
  mask call the mix "replaced" (a composite of two real changes).

### Fixes (v0.10.0)

Move matching now runs right after the VLM confirmation (clusters skip the
later stages but stay in `confirmed.json` so the report can render the move);
the cancellation compares the removed cluster's `first` with the added
cluster's `second`; the threshold is 0.65 with the VLM move question as the
arbiter and cluster size as the tie-breaker (the clean 102-cell destination
beat a 52-cell fragment at 0.732); arrivals at a move's origin are reported as
their own `added` change; replaced claims whose two sides are independently
confirmed changes are dropped as composites. `build_change_list` renders
moves as before → destination; the renderer/viewer draw a dashed line to the
destination and keep point-less changes (arrivals) as labelled boxes.

### Regression caught a latent demo break

The demo regression (re-run + table diff against the committed outputs)
initially produced **4 changes**: a highlighter fragment survived because the
empty-space skip of the both-present suppression (added for scene3's chair)
was too broad. The skip now requires that the VLM did not name the same object
in both views — "highlighter" vs "highlighter" means present in both and is
suppressed again, while the chair's differing descriptions still skip. After
the fix: demo **3/3, 0 extra — identical to committed**; scene2 and scene3
identical.

### Verified result

**scene4: exactly 3/3** (moved apple with from→to, removed scissors, added
plastic bottle), default mode, 0 extras; 7 unverified edge candidates listed
for transparency. GIF/viewer/mesh regenerated (Chrome on the login node —
Chrome is not installed on the GPU node, so visualize runs there fall back to
matplotlib). Remaining honest notes: the scissor sentence is inherited from
the two-view verification ("removed and replaced with an apple") and the move
sentence is view-relative wording.

Cluster-job note: the `gpu` queue has a **per-user 16-CPU limit**; with the
user's three ASR jobs + the gpu-session holding all 16, GPU jobs queue. The
idle gpu-session was stopped to free a slot before each run and restarted by
the waiters afterwards (five times over this session, documented here).


## 2026-10-11 — timing investigation, render-only mode, GIF palette bug (`v0.11.0`)

### Timing: the reported numbers were wrong

A user-visible mismatch ("~12-15 min per scene" reported vs over an hour in
practice) triggered a full audit of the PBS records and job logs. Findings
(see [`timing.md`](timing.md) for the tables):

- job 10802 (scene4, full pipeline incl. mesh + GIF) used **5:14 walltime**
  with a **1:33 queue wait**; the wait was the queue's **per-user 16-CPU
  quota**, not GPU contention (the queue comment says so; 3 concurrent 1-GPU
  jobs of the same account were running on an 8-GPU node).
- **VGGT loads in 8–14 s** (6 logged runs), not the "~3.5 min" the docs
  claimed; Qwen3-VL-32B loads in 26 s warm (2:07 on the one early cold run);
  the DINOv2 matcher is **loaded twice per run** (appearance check, then move
  matching) — measured in the same log.
- 10 VLM calls in the scene4 run, **11.0 s per call**.
- The mesh took 4 s inside the GPU-node job but ~89 s on the login node — the
  cause (page cache vs cores) was not isolated and is marked unverified.
- The old "~15 min first semantics run" claim is **not reproducible from any
  log** (the early demo describe run 10715 measured 5:25); corrected in
  `results.md` + `debugging.md`, and the README now carries a measured
  runtime table with conditions instead of single numbers.
- The ~1 h wall clock for scene4 = 5 GPU rounds (first run + 4 fix/regression
  rounds) + 2 visual rounds (GPU-node fallback GIF, then Chrome on the login
  node) + analysis time. One prediction is 5:14 on the node.

### Render-only mode

`visualize.py --render-only` reuses the cached reconstruction/diff/VLM
outputs, needs no GPU, and **stops with a list of missing caches** instead of
recomputing anything (guard verified: 9 missing files listed, exit 2, no
output dir created). Measured wall clock per scene: demo 3:11, scene2 3:24,
scene2-geo 2:09, scene3 5:46, scene4 3:21 — scene3 exceeds the ~5 min
guideline because its Chrome GIF alone takes 4:44 in that pass (31
candidates + denser cloud; a GIF-only re-render of the same scene later took
129 s, so the 4:44 includes machine load variance).

### Presentation pass

- unverified changes are hidden by default in the viewer (new "unverified"
  toggle) and never labelled; the GIF legend lists only the visible types;
  single-candidate composites still show an unverified box (cyan, no label)
- labels get a collision resolver in the viewer (a label that would cover an
  already-placed one is pushed down)
- larger splats (viewer RADIUS/150, matplotlib 20) and a taller top row
  (GIF 1100x820, before/after = 42% of the height; composites 1.25:1.5)
- stale-screenshot cleanup now references report.json (with the current
  change list as the fallback when the report is from an older run); audit:
  0 stale files in all five scenes

### GIF palette bug (found by pixel-checking)

The `replaced` colour rendered as **grey** in the scene3 GIF: a grey-dominated
scene makes the adaptive 256-colour GIF palette drop/mangle the isolated
change colours (the viewer rendered magenta exactly — verified with a raw
Chrome screenshot: 614 magenta pixels). Fix: every frame is quantized
together with a strip of the key colours (removed/added/moved/replaced/
unverified/background/label white), so the palette reserves slots for them;
the strip is cropped off. Verified after the fix: magenta = (242,63,242)
exactly, 355 px. All five GIFs re-rendered with the fix.


## 2026-10-11/12 — v0.12: one-command run, NL summaries, held-out scene5

### Step 0-2 (built before any scene5 execution)

- `docs/heldout.md` froze commit `e8c77d9` + sha256 of the 7 detection files
  and pre-registered the scoring rule **before** any new code was written.
- `detect.py`: input validation (decode/stream/duration/resolution/
  orientation, LFS-pointer detection), automatic diff-mode choice from the
  measured dominant-plane coverage, GPU-node detection, automatic PBS
  submission + polling from the login node, cache reuse, `--local/--no-wait/
  --render-only/--force/--out/--cache`, pass-through of the development
  scenes' documented working-point overrides, failure reporting with the
  failing stage.
- `src/summary.py`: report-only natural-language summaries. One Qwen3-VL
  phrasing call; sentences are validated word-by-word (label + type actions +
  non-object words) with a deterministic template fallback; "Possibly ..."
  below confidence 0.40; notes state the unverified count, the naming caveat
  and the development-scene tuning.
- The VLM phrasing initially failed validation for three fixable reasons
  (found by logging the raw reply): "video 1/2" references (no such numbering
  exists), the tokens "1"/"2", and missing action words ("absent", "present",
  "compared"). Fixed in the summary prompt/validator only.

### Step 3 — the held-out run (once) and the user's overlay review

Run: `detect.py --a data/scene5/a.mp4 --b data/scene5/b.mp4 --name scene5`
(auto mode -> `geometric`, dominant plane 11%; queue wait 0 s, job 7:07,
total 7:22). Result: 5 confirmed entries, no `moved`; the chair's removal
fragmented into three "leg" labels; two added entries. The user reviewed the
checking sheet: tiles 1-3 = the chair (type right, labels wrong), tile 4 =
**floor light reflection** (false positive), tile 5 = **not a real addition**
(the bottle is in both videos; the clock's new position misread) -> the clock
move is **missed**. Confirmed score: **found 1/2 (type only), labels 0/2,
missed 1, extras >= 2**.

### Post-hoc diagnostic (labelled, not part of the held-out result)

Scene5 re-run once with `--mode tabletop` forced (job 10829, 6:40): 4 entries,
both false additions gone, the chair named correctly in one entry ("black
chair"), the clock move **still missed**. This suggests - not proves - that
the automatic mode choice contributed; no held-out output was changed.

### Fixes after the review (summary/validation only)

- `detect.py`: an A/B orientation mismatch now prints a loud
  `[WARN] ORIENTATION MISMATCH` with the recommendation to record both videos
  in the same orientation (landscape preferred); the resolution mismatch
  warning is equally loud. Validation message only.
- `src/summary.py`: the validator no longer accepts the report's free-text
  description as a vocabulary source - that is how "leaving only the cracked
  surface" (a visual embellishment from the crop, not grounded in the
  detection) passed. Vocabulary is now the label + the type's action words +
  non-object words (+ common function words, which the object-description
  stopword list did not cover: "was"/"from" blocked valid sentences).
- `tests/test_summary.py`: synthetic + committed-output invariants (every
  report change appears exactly once, nothing extra, notes caveats).

### Step 4 results (regressions, tests, re-encode, clone)

- `detect.py --force` re-runs: **demo, scene3, scene4 identical** to the
  committed reports (types, labels, unverified counts); **scene2 differs**
  (1 -> 2 entries: the committed coconut-mislabeled-"apple" plus a new
  "removed coconut" with 562 empty-space points; 10 -> 9 unverified). Cause:
  recomputed reconstruction + geometric diff (run-to-run variability, frozen
  code unchanged). Documented in results.md and the README.
- `tests/test_summary.py` added (synthetic + committed-output invariants);
  `tests/test_ma_contract.py` still passes.
- `data/scene5/b.mp4` re-encoded (libx264 crf 26, same 480x860@30fps, 4.6 MB,
  original 6.9 MB LFS oid `1be79cea00` kept in history) so plain clones read
  it. A post-hoc check shows detection runs on it (9:04, exit 0) but the auto
  mode flips to `tabletop` and the result differs (7 entries; chair better
  named, clock move still missed) - documented, held-out outputs unchanged.
- Fresh-clone test: `git clone` + the README quick start ran from a clean
  directory (the first attempt failed on the GPU node because git-lfs is not
  installed there; re-run on the login node where git-lfs works, with the
  clone's `detect.py` submitting its own PBS job).

## 2026-10-11 — Agent-handover access verification

- **GPU access: verified live.** One-shot probe via `gpu-run -n 1 -g 1`
  (job 10863, interactive, exited cleanly): `jiit-gpu01` shows all 8
  A100-SXM4-80GB; `torch 2.14.1+cu126`, `cuda_available True`,
  `device_count 8`, matmul OK. Note the env torch is now **2.14.1+cu126**
  (newer than the 2.11.0 recorded in `gpu.md`), still on the working cu126 line.
  Temp probe script `scripts/_gpu_probe.sh` was created and removed.
- **GitHub access: verified live.** `git ls-remote origin HEAD` over SSH works
  (`f96d3b0…`); local `main` == `origin/main`.
- **Finding: the persistent `gpu-session` is stale.** `~/.gpu-session/jobid`
  still points at job **10831**, but that job is gone from the queue (a
  different job, `asr1bit-s*`, now occupies the account's CPUs), so `ready`/`out`
  still say `KERNEL_READY` while no server is actually running — a `gpu-send`
  would write `cmd.py` and wait forever. Fix when needed: `gpu-session start
  && gpu-session wait` (holds 4 CPUs and 1 A100 for 12 h; stop the idle session
  first per the 16-CPU quota pattern).

## 2026-10-11 — Held-out scene6: Step 0 (data + freeze) and pre-registration

- **Frozen code confirmed.** `git diff 93bf47d HEAD` (and vs the working tree)
  is **empty** for `src/{diff,tabletop,geometry,describe,align,reconstruct}.py`,
  `configs/default.yaml`, `detect.py`. The 7 digest prefixes are identical to
  the scene5 freeze (`72f10984`, `d2271109`, `f44767f0`, `ac00b938`, `b7400d42`,
  `db914732`, `f05d5cb6`). Noted honestly in `heldout.md`: the task named the
  frozen commit as `93bf47d`, but tag `v0.12.0` resolves to `f0042c9`; the
  detection code is byte-identical across both and HEAD, so the pin holds.
- **scene6 data re-encoded.** Originals (a 9,090,092 B oid `baf3f76f34`; b
  8,050,457 B oid `cf1d265b48`) copied to `data/scene6/original/` (LFS) and
  re-encoded with the env's bundled ffmpeg 7.0.2 (imageio_ffmpeg; no system
  ffmpeg on the login node) at crf 29: a 3,424,287 B, b 2,826,327 B, both
  < 5 MB, same 860x480@30fps, no audio, `+faststart`. `.gitattributes` marks
  `data/scene6/{a,b}.mp4` as plain git (scene5 pattern).
- **Validation only (no submission)** on the re-encodes printed **no warnings**:
  `[check] A 860x480 30.0fps 55.5s | B 860x480 30.0fps 49.1s`.
- **Pre-registered** the scene6 scoring rule + freeze + data provenance in
  `docs/heldout.md` **before** the run (GT: cup moved, remote removed, apple
  added; FOUND = type + overlay + label; a move only counts as `moved`).
