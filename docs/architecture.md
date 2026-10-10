# Architecture

## Repository layout

Planned layout from [`../AGENT_PLAN.md`](../AGENT_PLAN.md) §4; the root maps to
`Humanoid/` (this repo). Status column shows what exists **today**.

| Path | Purpose | Status |
|---|---|---|
| `README.md` | Project front page | ✅ (results pending) |
| `AGENT_PLAN.md` | Plan, phases, checkpoints | ✅ |
| `LICENSE` | MIT for this repo (models keep their own licenses) | ✅ |
| `requirements.txt` | pip pins | ⬜ planned |
| `environment.yml` | conda env export | ⬜ planned |
| `run.py` | One-command pipeline | ⬜ planned |
| `configs/default.yaml` | All models, paths, thresholds | ⬜ planned |
| `src/extract_frames.py` | video → sharp tagged frames | ⬜ planned |
| `src/reconstruct.py` | VGGT / MASt3R / COLMAP switch | ⬜ planned |
| `src/align.py` | optional ICP refinement | ⬜ planned |
| `src/diff.py` | voxel diff, DBSCAN, visibility check | ⬜ planned |
| `src/describe.py` | crops, Qwen3-VL labels, DINOv2 matching | ⬜ planned |
| `src/visualize.py` | colored PLY, arrows, viewer | ⬜ planned |
| `src/report.py` | JSON + Markdown report | ⬜ planned |
| `scripts/setup_env.sh` | conda env + pip (cu126) | ⬜ planned |
| `scripts/download_models.sh` | run on login node | ⬜ planned |
| `scripts/job_*.pbs` | PBS job templates | ⬜ planned |
| `data/` | input videos + ground truth | ⬜ |
| `cache/` | expensive intermediate artifacts (gitignored) | ⬜ |
| `outputs/` | PLY, JSON, MD, images, GIF | ⬜ |
| `docs/` | this documentation | ✅ |

## Module responsibilities

| Module | Responsibility | Must NOT do |
|---|---|---|
| `run.py` | Parse CLI + config, call stages in order, skip stages whose cache exists | contain model logic |
| `extract_frames.py` | Decode videos, sample fps, blur-filter, tag frames with source video | touch the GPU |
| `reconstruct.py` | Run the configured backend on **all** frames of A and B together; emit poses, depth, tagged cloud | do the diff |
| `align.py` | Optional ICP over static regions only | run unless enabled in config |
| `diff.py` | Voxel downsampling, bidirectional NN distance, outlier removal, DBSCAN, visibility check, added/removed classification | read videos directly; it uses cached frames/poses |
| `describe.py` | Best-view selection, crops, Qwen3-VL structured output, DINOv2 embeddings, moved-matching, final sentences | re-run reconstruction |
| `visualize.py` | Colored PLY (red/green/orange), arrows, screenshots, GIF | classify changes |
| `report.py` | `report.json` + `report.md` | invent data |

## Config reference

Everything tunable lives in `configs/default.yaml` (skeleton from the plan):

```yaml
paths:
  video_a: data/a.mp4
  video_b: data/b.mp4
  cache_dir: cache/
  out_dir: outputs/
  model_dir: ~/models/

frames:
  fps: 2
  max_frames_per_video: 30
  blur_threshold: 100.0

reconstruction:
  backend: vggt          # vggt | mast3r | colmap
  use_icp_refinement: false

diff:
  voxel_size: 0.02
  change_distance: 0.05
  dbscan_eps: 0.08
  dbscan_min_points: 50
  visibility_check: true

semantics:
  vlm: qwen3-vl-32b-instruct   # or qwen3-vl-8b-instruct
  matcher: dinov2
  match_similarity_threshold: 0.75
  use_sam2: false

gpu:
  vlm_gpus: 2
  dtype: bfloat16
  torch_cuda: cu126
```

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
