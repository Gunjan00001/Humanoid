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

## Why the privacy guard missed the batch-1 log file (2026-10-18)

Two leaks reached the remote with every test green. The cause was the shape of
the guard, not its strength:

1. **The tests scanned the tracked tree, and ran before `git add`.** The usual
   loop was `run tests` then `git add -A && git commit`. A file created after
   the test run, or created at all while untracked, was invisible to the scan
   and got committed by `git add -A`. `bakeoff_logs.log` (written at the repo
   root because a PBS `-o` directive used a bare relative name) came in exactly
   this way.
2. **Nothing looked at the identity fields.** Author, committer and tagger
   strings were never scanned, so a cluster login could leak through the
   envelope of a commit rather than its content.

Both are closed by `scripts/prepush_audit.sh` and its Python implementation
`scripts/prepush_audit.py`, which scan the staged change set and every added
line in the range about to be pushed (not just the tracked tree), every file at
HEAD regardless of extension (only genuinely binary files are skipped, sniffed
by a NUL byte), and the author, committer and tagger of every commit and
annotated tag in the range. The patterns are built at runtime from
`getpass.getuser()`, `$HOME`, the hostname, the numeric part of the login and an
optional local `.git/info/other_users.txt`, so the guard never contains the
literals it hunts for.

The script is installed as `.githooks/pre-commit` and `.githooks/pre-push`
(`git config core.hooksPath .githooks`; `tests/test_hooks.py` fails if the
hooks are missing, not executable, or not wired into CI). Paths in job scripts
are now relative to `$PBS_O_WORKDIR`; a stray root `*.log` is ignored.

## The Wilson interval that went out wrong (2026-10-18)

A reporting helper computed the Wilson score interval and then returned
`(centre, centre + half)` instead of `(centre - half, centre + half)`, so every
published interval had its lower bound equal to the centre. Nothing in the
pipeline used it (the scorer and the naming evaluation were correct); the damage
was in the tables I quoted from ad-hoc snippets, including
docs/stack_selection.md.

Why it survived the numbers themselves: for 58/204 the wrong interval
[28.8, 35.0] still looked plausible, and the lower bound sitting just above the
point estimate (28.4%) was the only tell. The earlier helper was right, which is
how the 10/19 = [31.7, 72.7] figure quoted in docs/naming_v3.md stayed correct.

What changed:

1. `src/naming/stats.py` is now the single implementation (`wilson`, plus
   `wilson_pair` for the (point, lower, upper) shape the reports use). Callers
   import it; no module re-derives the formula.
2. `tests/test_wilson.py` pins known values: 10/19 = [31.7, 72.7],
   58/204 = [22.7, 35.0], 90/204 = [37.5, 51.0], 0/20 = [0, 16.1],
   20/20 = [83.9, 100], and the invariant that the interval always contains the
   point estimate and stays inside [0, 100].
3. The clamp added after a stats crash was moved to the OUTPUT (`[0, 100]`,
   float noise only: 0/20 used to report -1.4e-15). It never touches the
   proportion, which is the mistake that would hide a real bug.
4. Every interval published since the clamp was recomputed from the raw per-crop
   rows: docs/stack_selection.md and the two `logs/bakeoff_naming_*.json` files
   now carry both denominators (all crops and named crops) with the corrected
   intervals.

## SAM 3.1 text route: attempts and outcome (time boxed)

1. `AutoProcessor.from_pretrained("~/models/sam3.1")` returns
   **`Sam3VideoProcessor`**, whose call signature is
   `(images, segmentation_maps, original_sizes, return_tensors, **kwargs)`, with
   no `text`. Passing `text="apple"` (and `text=["apple"]`) fails with
   "unexpected keyword argument 'text'". The processor's `model_input_names`
   do include `input_ids`/`attention_mask`, which suggests the video/session
   route rather than the plain call.
2. The text-tuned sibling in transformers 5.19 is **`sam3_lite_text`**
   (`Sam3LiteTextModel`, `Sam3LiteTextTextModel`), a different checkpoint from
   the 3.1 weights, so it cannot stand in for SAM 3.1.
3. Remaining routes, in order, for the next time box: (a) the image-model
   session API in the `sam3` documentation, initialising a video session and
   adding a text prompt; (b) a separate env with the official
   `facebookresearch/sam3` package and its own text-prompt API.
4. Status: **blocked on integration, not a model result.** SAM 2.1 base-plus
   stays the segmenter meanwhile. The comparison harness is written and ready
   (`scripts/bakeoff_segmentation.py`).

## SAM 3.1 image route, final probe of the box

`transformers.models.sam3` does contain an image text-prompted path:
**`Sam3Processor`** and **`Sam3Model`**, whose `forward` takes `input_ids`,
`attention_mask`, `text_embeds` and `input_boxes`. That is exactly the route
the review suggested, and it is different from `Sam3VideoProcessor` (which
`AutoProcessor` picks because the checkpoint config says `sam3_video`).

It cannot load this download: `~/models/sam3.1` contains
**`sam3.1_multiplex.pt`** (3.5 GB, Meta's own state dict), a `LICENSE` and a
config with `architectures = ["Sam3VideoModel"]`. There is no
`model.safetensors` / `pytorch_model.bin`, so `Sam3Model.from_pretrained`
raises "no file named model.safetensors". The remaining routes are therefore
(a) a transformers-format image checkpoint, or (b) the official
`facebookresearch/sam3` package which loads this `.pt` (separate env,
`spatialdiff-sam3`, Python 3.12). The per-frame comparison harness is written
and waiting (`scripts/bakeoff_segmentation.py`).

## SAM 3.1: the separate env works (official package imports)

The route from the review (b) succeeded up to a working import:

* env **`spatialdiff-sam3`** created: **Python 3.12.15**, **torch 2.14.1+cu126**
  (the same CUDA build as the other envs).
* official package installed from the repository:
  **`pip install git+https://github.com/facebookresearch/sam3.git`** →
  `sam3 0.1.0`.
* its imports needed five undeclared packages: `setuptools` (pkg_resources;
  version <81 was required), `einops`, `timm`, `omegaconf`, `hydra-core`,
  `opencv-python-headless`, `pillow`, `numpy<2`, `pycocotools`, `psutil`,
  `scipy`, `tqdm`, `imageio`. All installed; `from sam3 import
  build_sam3_image_model, build_sam3_predictor` now succeeds.
* API read from the installed package:
  `build_sam3_image_model(bpe_path=None, device='cpu', eval_mode=True,
  checkpoint_path=None, load_from_HF=True, enable_segmentation=True, ...)` and
  `build_sam3_predictor(checkpoint_path=None, bpe_path=None, version='sam3.1',
  max_num_objects=16, multiplex_count=16, ...)`.
* remaining detail before the per-frame bake-off: the text prompt needs a
  **BPE vocabulary path** (`bpe_path`); the local snapshot has `merges.txt`
  (524 KB) and the package expects its own BPE file, so the next command is to
  pass the correct bpe_path (or let `load_from_HF` fetch it while online) and
  then run the predictor with `~/models/sam3.1/sam3.1_multiplex.pt` on a pinned
  GPU. Nothing about the model is in doubt any more; only the tokenizer file
  location and the per-frame harness run remain.

## SAM 3.1 bake-off attempt: dtype mismatch in the official image path

Two attempts in the official package, both on a pinned GPU, both with the same
error class:

1. as shipped: `RuntimeError: mat1 and mat2 must have the same dtype, but got
   BFloat16 and Float` — the checkpoint loads in bf16 while the processor feeds
   float32 images.
2. after `model.float()`: the same message with the sides swapped (input bf16,
   weights float), so the input side is bf16 (the processor's backbone output)
   and casting only the model is not enough.

Next step, not guessed: either run the whole processor/model under
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` so both sides are
bf16, or cast the processor's tensors to the model dtype after `set_image`.
Then rerun `scripts/bakeoff_segmentation_sam3.py --mode sam3`; the harness, the
job and the SAM 2.1 comparison all work (`logs/bakeoff_seg_sam2.json`, 44
frame/prompt runs, 8-16 automatic masks per frame).

Adoption rule applied: SAM 3.1 produced no segmentation numbers, so it cannot
have "at least as many GT objects as SAM 2.1". **SAM 2.1 base-plus stays the
primary segmenter; SAM 3.1 is recorded as the alternative**, not as a model
result and not as a rejection of the model.
