# Debugging

The pipeline is designed to be debugged from its **caches**: every stage reads
only cached artifacts and writes only new ones, so you can inspect, re-run and
compare stages independently, on CPU, without GPUs, without the videos.

Two tools make this easy:

```bash
# 1. see everything at once (read-only, always safe)
python scripts/inspect_cache.py

# 2. per-cluster / per-cell decision numbers
python src/diff.py --config configs/default.yaml --debug
python src/describe.py --config configs/default.yaml --debug
```

A third tool repairs the most common broken artifact without a GPU rerun:

```bash
# rebuild cache/reconstruction/colors.npz from the original frames using the
# per-frame transforms in meta.json (e.g. after a model stopped returning
# denormalised images - observed with MapAnything's img_no_norm)
python scripts/rebuild_colors.py --config configs/default.yaml --cache cache/ma_run
```

`scripts/inspect_cache.py` prints, for each stage: what the artifact contains,
per-cluster tables with the change-detector's and the VLM's decisions, the
rejection reasons, consistency checks, and the exact command to produce the
stage if it has not run yet.

## The cache map (what each stage reads and writes)

| Stage | Reads | Writes | Re-run with |
|---|---|---|---|
| extract_frames | `data/a.mp4`, `data/b.mp4` | `cache/frames/` | `--force` (clears stale frames) |
| reconstruct | `cache/frames/index.json` | `cache/reconstruction/{meta.json,cameras.npz,dense.npz,colors.npz}` | `--force` (GPU) |
| align | `cache/reconstruction/` | `cache/reconstruction/aligned/` | `--force` |
| diff | reconstruction (aligned when enabled) | `cache/diff/{clusters,uncertain,visibility}.json`, `points.npz` | `--force` |
| describe | diff + reconstruction + frames | `cache/semantics/{crops/,crops.json,vlm.json,rejected.json,confirmed.json,embeddings.npz,matches.json}` | `--force` (GPU; **VLM answers are reused**, see below) |
| visualize | reconstruction + diff + semantics | `outputs/scene.ply`, `outputs/changes/*.png`, `outputs/change.gif` | re-run (overwrites) |
| report | everything above | `outputs/report.{json,md}` | re-run (overwrites) |

**Cache invalidation rules:**

- `--force` re-runs the stage and overwrites its outputs. For `describe` the
  expensive Qwen3-VL answers in `vlm.json` are keyed by model and **always
  reused**; delete `cache/semantics/` to force fresh VLM calls (do this when
  you change the crops or the prompts).
- Changing `diff.tabletop.*` thresholds only needs `diff --force` (describe
  will then re-verify the new candidate set, cached verdicts for surviving
  clusters are reused).
- Changing `semantics.verify_crop_margin` changes the crops, so delete
  `cache/semantics/` before re-running describe.
- Changing `frames.*` invalidates everything downstream: re-run
  `extract_frames --force` then `reconstruct --force` etc.

## Playbook by symptom

### Frame extraction

| Symptom | Look at | Fix |
|---|---|---|
| too few frames kept | `inspect_cache.py` blur stats | lower `frames.blur_threshold`; more `frames.fps` |
| stale frames on disk (index says N, disk has more) | the on-disk count warning | `extract_frames --force` (it clears the tag dirs) |
| wrong frames (scene cut, fingers) | open the JPEGs in `cache/frames/<a|b>/` | re-record or raise the blur threshold |

### Reconstruction

| Symptom | Look at | Fix |
|---|---|---|
| low depth confidence (median near 1.0) | `inspect_cache.py` conf percentiles | normal for soft/low-texture phone video; the diff works in appearance space, so it tolerates it |
| point cloud garbage / degenerate bbox | `dense.npz` bbox print | more frames (`frames.fps`), less blur, more texture in the scene |
| A and B clouds do not overlap | run `align.py` and read the ICP line | enable `reconstruction.use_icp_refinement`; if ICP cannot improve, the two videos may not overlap enough |
| OOM during VGGT | job log | lower `frames.max_frames_per_video`; use `--limit_per_video N` for a quick pass |

### Change detection: misses or spams candidates

`diff.method` decides which detector runs:

- **`geometric`** (voxel/DBSCAN/visibility): good for objects ≥ 2–3 cm thick.
  Misses → lower `diff.voxel_size` and `diff.change_distance` together (they
  must stay within ~2× of each other). Spam → raise `diff.change_distance`,
  `diff.min_cluster_points`, `dbscan_min_points`.
- **`tabletop`** (photometric, default for the demo): compares appearance over
  the fitted table plane. Tuning lives in `diff.tabletop.*`:

| Symptom | Config key | Effect |
|---|---|---|
| a known change is missing | `objectness_margin` ↓ (40 → 25) | accepts weaker A-vs-B differences |
| | `objectness_threshold` ↓ (70 → 50) | accepts less colourful objects |
| | `min_cells` ↓ (18 → 10) | accepts smaller objects (thin pens) |
| too many candidate fragments | `objectness_margin` ↑, `min_cells` ↑ | fewer, stronger candidates |
| static objects keep appearing | `min_change_score` ↑ (in `semantics.*`) and the static-object suppression in describe | fragments of objects present in both videos |
| everything is a candidate | check the plane fit: `--debug` prints the plane coefficients | the dominant plane may not be the table (e.g. a wall dominates) |

**Visual debugging of the diff** (the fastest way to see what the detector
sees): overlay the candidate masks on the colour maps, as done in
`tests/make_fake_reconstruction.py`-style scripts. The masks themselves are
recomputed from `cache/diff/points.npz` (labels) and the reconstruction.

### Semantics (crops, VLM, verification)

| Symptom | Look at | Fix |
|---|---|---|
| a cluster has a wrong label | open `cache/semantics/crops/cluster_XXX.png` and `_other.png` | if the crop misses the object, raise `semantics.verify_crop_margin`; then delete `cache/semantics/` and re-run |
| the VLM denies a real change | `vlm.json` → `first`/`second` fields (what the model saw in each image) | usually a framing problem: check both crops; the pose-matched other-view selection is in `best_crop_for_cluster` |
| the VLM confirms a static object | `rejected.json` vs `confirmed.json` | raise `semantics.min_change_score`; check the static-object suppression ran (`--debug` prints overlap fractions) |
| a "moved" object is split into removed+added | `matches.json` empty; `embeddings.npz` | lower `semantics.match_similarity_threshold` (0.75 → 0.65) or `match_min_distance` |
| a "replaced" claim is wrong | `--debug` prints the replaced pixel-difference | `semantics.replaced_min_pixel_diff` controls the gate |

Every rejection is recorded with a reason in `cache/semantics/rejected.json` , 
start there when a change is missing; the reason tells you which rule fired.

### Report / visualization

| Symptom | Fix |
|---|---|
| change list wrong | inspect `cache/semantics/confirmed.json`, the report is a projection of it |
| confidence looks odd | the heuristic is documented in `docs/results.md` and `geometry.build_change_list` |
| screenshots too dark/small | `visualize.background_voxel`, `visualize.static_dim`, `visualize.gif_frames` |

## GPU debugging

- Everything GPU-related must run inside a GPU allocation: `gpu-run '<cmd>'`
  (one-shot) or the persistent `gpu-session` (see `gpu.md`).
- `CUDA_VISIBLE_DEVICES` is set to an empty string inside PBS jobs, the
  stages handle it: `common.pin_idle_gpus()` picks the idlest GPUs before CUDA
  initialises and logs what it pinned.
- Model loading: measured 8–14 s for VGGT and 26 s for Qwen3-VL-32B with
  warm local files (job 10802); an early cold/network-FS run measured 2:07
  for the VLM load. See [`timing.md`](timing.md). Debug the *stage logic* on
  CPU (crops, rules)
  whenever possible instead of re-loading models.

## Logs

| Log | Contents |
|---|---|
| `logs/setup_and_download.log` | env creation + weight downloads |
| `logs/reconstruct_real*.log` | reconstruction runs |
| `logs/describe_real*.log` | semantics runs (per-cluster verification lines) |
| `logs/gpu_smoke*.log`, `logs/gpu_describe_smoke*.log` | smoke tests |

Run any stage with `--debug` and redirect to a file when you need the full
decision trace:

```bash
python src/describe.py --config configs/default.yaml --debug 2>&1 | tee logs/debug_describe.log
```
