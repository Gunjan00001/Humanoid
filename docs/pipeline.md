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
- **VGGT or MapAnything runs on all frames of both videos together** (backend
  selectable via `reconstruction.backend`), producing one coordinate frame:
  camera poses, depth maps, and a colored point cloud.
- Each point carries a source tag (`A` or `B`).
- **Input fidelity** (`reconstruction.input_mode`, VGGT only): `crop` (default)
  discards the top/bottom of portrait frames (56% of a 480×854 frame reaches
  the model); `pad` keeps the whole frame at lower resolution with the white
  border masked out of the confidence. See the Step-0 diagnosis in
  [`results.md`](results.md#visualization-rework-steps-05).
- Both backends write the **same artifact contract**
  (`cache/reconstruction/{cameras,dense,colors}.npz` + `meta.json`), so every
  downstream stage works unchanged. MapAnything runs in an isolated conda env
  (`spatialdiff-ma`) as a subprocess - see `scripts/run_mapanything.py` and
  `src/ma_contract.py`.
- Fallback order: `vggt` / `mapanything` → `mast3r` → `colmap`.
- If GPU memory is tight: fewer frames or lower resolution first; chunking
  second (MapAnything: `reconstruction.mapanything.minibatch_size`).

### 3. Alignment (optional)
- ICP refinement over static regions only, if A and B clouds drift apart.

### 4. Change detection
Two modes, selected by `diff.method`:

**`geometric`** (plan default, good for objects ≥ 2–3 cm thick):
1. Voxel-downsample both clouds (size from config).
2. Nearest-neighbour distance both directions (A→B, B→A); threshold candidates.
3. Statistical outlier removal, then DBSCAN into object-sized clusters.
4. **Visibility check** — project each cluster into the other video's cameras
   using cached poses+depth. Mark *removed* only if the other video observed the
   space as empty. Discard occluded/unseen clusters.
5. Classify: only in B → **added**; only in A → **removed**.

**`tabletop`** (photometric; needed when objects are thinner than VGGT's patch
scale — on the demo scene the 1 cm-thick comb and pen are *absent from the
depth maps*, their z equals the table's):
1. Fit the dominant plane of each cloud; rasterise both into 5 mm cells
   (height, mean colour, point count).
2. Per-cell *objectness* = colour distance from a robustly fitted table-colour
   field (insensitive to object size and illumination gradients).
3. Compare the two videos per cell: object in A only → removed; in B only →
   added; different appearance at the same spot → **replaced** (e.g. a flipped
   remote); lighting/shadow patches are rejected by a difference margin.
4. DBSCAN over changed cells, ring filter (static objects only differ at their
   edges), minimum size; cells seen by only one video become clearly flagged
   **unverified** candidates (`cache/diff/uncertain.json`) instead of being
   silently dropped.
5. Output format is identical to the geometric mode, so later stages are shared.

### 5. Semantics
- Pick the best view per cluster (max visible area, least occlusion), crop it;
  verified candidates get a second crop from the *other* video of the same
  region.
- **Qwen3-VL verification**: candidates are confirmed or denied against the
  before/after crops (`"changed": true/false`); lighting/blur artefacts are
  rejected here (`cache/semantics/rejected.json` records every rejection).
  Unverified candidates (one video never saw the area) are labelled but kept
  separate.
- DINOv2 embeds crops; a confirmed removal + addition with high similarity at a
  different 3D position becomes one **moved** object (old and new position).
- Qwen3-VL writes the final one-sentence description per change.
- All VLM outputs are cached so re-runs don't re-spend GPU time.

### 6. Visualization
- Display cloud with its own settings (`visualize.conf_percentile` lower than
  the detection cutoff, `voxel_size`, `static_dim 0.8` — true colours).
- PLY coloring: **red = removed, green = added, orange = moved, magenta =
  replaced**, background in true (slightly dimmed) colours.
- Per-change PNGs: **before frame | after frame | 3D view** with a bounding
  box and label per confirmed change; unverified candidates render dim cyan.
- `change.gif`: the same three-panel layout with the 3D view orbiting, framed
  on the changed region.
- Colored TSDF mesh (`scene_mesh.ply`, ≤ 10 MB, `src/mesh.py`) and a
  self-contained interactive HTML viewer (`viewer.html`, ≤ 5 MB,
  `src/viewer.py`: orbit controls, A/B/change toggles, labels).

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
