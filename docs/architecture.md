# Architecture

## Repository layout

Planned layout from [`../AGENT_PLAN.md`](../AGENT_PLAN.md) §4; the root maps to
`Humanoid/` (this repo). Status column shows what exists **today**.

| Path | Purpose | Status |
|---|---|---|
| `README.md` | Project front page | ✅ (results pending) |
| `AGENT_PLAN.md` | Plan, phases, checkpoints | ✅ |
| `LICENSE` | MIT for this repo (models keep their own licenses) | ✅ |
| `requirements.txt` | pip pins | ✅ |
| `environment.yml` | conda env export | ⬜ generated after the env is verified |
| `run.py` | One-command pipeline | ✅ |
| `configs/default.yaml` | All models, paths, thresholds | ✅ |
| `src/common.py` | config/CLI/JSON helpers shared by all stages | ✅ |
| `src/geometry.py` | reconstruction loading, camera math, change list | ✅ |
| `src/tabletop.py` | photometric tabletop diff (thin-object mode) | ✅ |
| `src/extract_frames.py` | video → sharp tagged frames | ✅ |
| `src/reconstruct.py` | VGGT / MASt3R / COLMAP switch | ✅ VGGT (fallbacks raise) |
| `src/align.py` | optional ICP refinement | ✅ |
| `src/diff.py` | voxel diff, DBSCAN, visibility check | ✅ |
| `src/describe.py` | crops, Qwen3-VL labels, DINOv2 matching | ✅ (GPU validation pending) |
| `src/visualize.py` | colored PLY, arrows, viewer | ✅ |
| `src/report.py` | JSON + Markdown report | ✅ |
| `scripts/setup_env.sh` | conda env + pip (cu126) | ✅ |
| `scripts/download_models.sh` | run on login node | ✅ |
| `scripts/job_*.pbs` | PBS job templates | ✅ |
| `tests/` | synthetic smoke tests (not results) | ✅ |
| `data/` | input videos + ground truth | ⬜ videos not recorded yet |
| `cache/` | expensive intermediate artifacts (gitignored) | ✅ produced by stages |
| `outputs/` | PLY, JSON, MD, images, GIF | ✅ produced by stages |
| `docs/` | this documentation | ✅ |

> `src/common.py` and `src/geometry.py` are shared helpers added during
> implementation: `common` holds config/CLI/JSON plumbing (no heavy imports),
> `geometry` holds the reconstruction formats and camera math used by the CPU
> stages. They are the only modules besides the stages in `src/`.

## Module responsibilities

| Module | Responsibility | Must NOT do |
|---|---|---|
| `run.py` | Parse CLI + config, call stages in order, skip stages whose cache exists | contain model logic |
| `common.py` | config loading, path resolution, model-path resolution, JSON/CLI/stage-output plumbing | import torch or heavy deps |
| `geometry.py` | load reconstruction artifacts, voxel/conf helpers, camera projection, model↔original mapping, derived change list | write to cache or files |
| `tabletop.py` | photometric tabletop diff (plane fit, cell maps, objectness, replaced/unverified logic) | touch the GPU |
| `extract_frames.py` | Decode videos, sample fps, blur-filter, tag frames with source video | touch the GPU |
| `reconstruct.py` | Run the configured backend on **all** frames of A and B together; emit poses, depth, tagged cloud | do the diff |
| `align.py` | Optional ICP over static regions only | run unless enabled in config |
| `diff.py` | Voxel downsampling, bidirectional NN distance, outlier removal, DBSCAN, visibility check, added/removed classification | read videos directly; it uses cached frames/poses |
| `describe.py` | Best-view selection, crops, Qwen3-VL structured output, DINOv2 embeddings, moved-matching, final sentences | re-run reconstruction |
| `visualize.py` | Colored PLY (red/green/orange), arrows, screenshots, GIF | classify changes |
| `report.py` | `report.json` + `report.md` | invent data |

## Config reference

`configs/default.yaml` is the single source of truth for every model name,
path and threshold — scripts never hard-code values. Sections:

| Section | Contains |
|---|---|
| `seed` | global RNG seed |
| `paths` | `video_a`, `video_b`, `cache_dir`, `out_dir`, `model_dir` (~/models) |
| `frames` | sampling fps, `max_frames_per_video`, blur threshold, `min_frames`, jpeg quality |
| `reconstruction` | backend switch, optional ICP, device / `gpu_index`, confidence percentile, ICP params |
| `models` | VGGT / VLM / matcher references (HF ids or local paths, resolved via `common.resolve_model`) |
| `diff` | voxel size, change distance, DBSCAN eps/min points, outlier removal, cluster minimum, visibility check params |
| `semantics` | VLM choice (primary/fallback), matcher, match thresholds, crop params |
| `visualize` | background voxel/dimming, GIF frames |
| `gpu` | VLM GPU count, dtype, torch CUDA tag |

Values are starting points and get tuned against the demo scene — **in the
config, not in code**.

## Caches and outputs

See [`pipeline.md`](pipeline.md#caching-strategy) for the full tree. Rule: any
stage that needs a GPU must write its result to `cache/` so later stages never
require a re-run of the expensive part.

## Conventions

- Stage scripts share one interface: `python src/<stage>.py --config configs/default.yaml`.
- No absolute paths in code; paths come from the config or CLI.
- Deterministic filenames (`frame_00042.jpg`, `cluster_003.json`) for diffability.
- Every artifact carries the provenance needed to re-derive it (which video,
  which frames, which config hash once implemented).
