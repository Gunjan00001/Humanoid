# Pipeline

```
Video A ─┐                                         ┌─> colored 3D PLY
         ├─> frames ─> VGGT (joint) ─> point cloud ┤
Video B ─┘            shared frame       A / B tags └─> diff + visibility check
                                                         │
                                         clusters ─> crops ─> Qwen3-VL ─> labels + sentences
                                                         │
                                         DINOv2 matching ─> moved / added / removed
                                                         │
                                                         └─> JSON + Markdown report
```

## Stages

| # | Stage | Script | Runs on | Reads | Writes (cache/output) |
|---|---|---|---|---|---|
| 0 | Inputs | — | — | `data/a.mp4`, `data/b.mp4`, `data/ground_truth.md` | — |
| 1 | Frame extraction | `src/extract_frames.py` | CPU (any) | videos | `cache/frames/<a\|b>/frame_*.jpg`, `cache/frames/index.json` |
| 2 | Joint reconstruction | `src/reconstruct.py` | **GPU (1×)** | all frames (both videos) | `cache/reconstruction/` (poses, depth, tagged point cloud) |
| 3 | Alignment (optional) | `src/align.py` | CPU | reconstruction | `cache/reconstruction/aligned/…` |
| 4 | Change detection | `src/diff.py` | CPU | tagged cloud, poses, depth | `cache/diff/clusters.json`, `cache/diff/visibility.json` |
| 5 | Semantics | `src/describe.py` | **GPU (1–2×)** | clusters, frames, poses | `cache/semantics/crops/`, `vlm.json`, `embeddings.npz`, `matches.json` |
| 6 | Visualization | `src/visualize.py` | CPU | tagged cloud, clusters, matches | `outputs/scene.ply`, `outputs/changes/*.png`, `outputs/change.gif` |
| 7 | Report | `src/report.py` | CPU | everything above | `outputs/report.json`, `outputs/report.md` |
| — | Orchestrator | `run.py` | driver | config + videos | all of the above |

## Stage detail

### 1. Frame extraction
- Sample ~2–3 fps; drop blurry frames using Laplacian variance.
- Target **20–30 frames per video** to start.
- Every frame is tagged with its source video (`a` / `b`); the tag follows the
  frame through reconstruction so the diff can separate the two captures.

### 2. Joint reconstruction (most important)
- **VGGT runs on all frames of both videos together**, producing one coordinate
  frame: camera poses, depth maps, and a colored point cloud.
- Each point carries a source tag (`A` or `B`).
- Backend is selectable: `vggt` (primary) → `mast3r` → `colmap`.
- If GPU memory is tight: fewer frames or lower resolution first; chunking second.

### 3. Alignment (optional)
- ICP refinement over static regions only, if A and B clouds drift apart.

### 4. Change detection
1. Voxel-downsample both clouds (size from config).
2. Nearest-neighbour distance both directions (A→B, B→A); threshold candidates.
3. Statistical outlier removal, then DBSCAN into object-sized clusters.
4. **Visibility check** — project each cluster into the other video's cameras
   using cached poses+depth. Mark *removed* only if the other video observed the
   space as empty. Discard occluded/unseen clusters.
5. Classify: only in B → **added**; only in A → **removed**.

### 5. Semantics
- Pick the best view per cluster (max visible area, least occlusion), crop it.
- Qwen3-VL returns a short label and description (structured JSON, parsed safely).
- DINOv2 embeds crops; a removal + addition with high similarity at a different
  3D position becomes one **moved** object (old and new position).
- Qwen3-VL writes the final one-sentence description per change.
- All VLM outputs are cached so re-runs don't re-spend GPU time.

### 6. Visualization
- PLY coloring: **red = removed, green = added, orange = moved**, background gray
  or true color; arrows from old to new position for moved objects.
- Interactive viewer (Open3D/Rerun) plus static screenshots/GIF for the README.

### 7. Report
- `outputs/report.json`: one entry per change (type, label, description,
  3D centroid, bounding box, confidence).
- `outputs/report.md`: human-readable summary.

## Caching strategy

Everything expensive is written under `cache/` so later stages can re-run on CPU
or on a new GPU allocation without repeating reconstruction or VLM calls:

```
cache/
├── frames/            # extracted + filtered frames, with tags
├── reconstruction/    # poses, depth, tagged point cloud
├── diff/              # clusters, visibility masks
└── semantics/         # crops, VLM answers, DINOv2 embeddings, matches
```

`cache/` is gitignored. Only the final `outputs/report.*` (and selected visuals)
are meant to be shared.
