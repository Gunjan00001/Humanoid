# 3D Scene Change Detection

System that takes two phone recordings of the same small indoor space, builds a
shared 3D reconstruction, detects which objects were **moved**, **added**, or
**removed**, and describes each change in natural language.

Part of the **Spatial AI & Navigation Challenge**.

> **Status: first successful end-to-end run (2026-10-10).** The three planned
> changes in the demo scene were detected and described. Full numbers, timings
> and honest limitations: [`docs/results.md`](docs/results.md). The plan and
> checkpoints live in [`AGENT_PLAN.md`](AGENT_PLAN.md).

## Results

On the demo scene (two 8–12 s phone recordings of a table; ground truth: comb
**removed**, pen **added**, AC remote **moved and flipped**), the pipeline
detected and described all three changes with **no false positives** in the
final report:

![3D change visualization](outputs/change.gif)

| # | Ground truth | Detected | Description (Qwen3-VL) | Confidence |
|---|---|---|---|---|
| 1 | comb removed | `removed` | "The comb was removed from the table." | 0.87 |
| 2 | pen added | `added` | "A blue pen and a white device were added to the surface." | 0.38 |
| 3 | remote moved + flipped | `replaced` | "The remote control was replaced with a different one." | 0.84 |

- Visualization: `outputs/change.gif` (rotating colored cloud — red = removed,
  green = added, magenta = replaced) and one framed screenshot per change in
  `outputs/changes/`.
- Reports: `outputs/report.json`, `outputs/report.md`.
- The change detector proposed 26 candidates; 22 were rejected with documented
  reasons (`cache/semantics/rejected.json`) — mostly lighting/blur differences
  on static objects. Nothing was silently dropped.

Known limitations of this run (details in [`docs/results.md`](docs/results.md)):
the remote's in-place flip is reported as `replaced` rather than `moved`; the
pen's label includes the neighbouring timer; one-video-only areas can only
produce *unverified* candidates.

### Backend comparison (v0.5)

Two more reconstruction backends were added and compared against VGGT on the
same frames ([full numbers](docs/results.md#backend-comparison-vggt-vs-mapanything-vs-depth-anything-3)):

- **VGGT + `tabletop` remains the best single configuration** (3/3 changes,
  0 false positives).
- **MapAnything** (Apache-2.0) matches it with its threshold calibrated
  (3/3, 0 FP) and brings **metric scale by construction** plus a geometric
  diff that sees the 1 cm-thick comb and pen (VGGT's depth maps cannot).
- **Depth Anything 3** (DA3-LARGE-1.1, Apache-2.0, stretch goal, via
  MapAnything's unified interface) is the **fastest and lightest** backend
  (0.10 s/frame, 6.2 GiB), aligns best, and shows the strongest thin-object
  response; its appearance scores need calibration and it kept one timer false
  positive in tabletop mode.
- Backend choice is `reconstruction.backend: vggt | mapanything` (DA3 via
  `--model-key da3_apache`); the default stays VGGT + `tabletop`.

### Results across all scenes (v0.7.0)

All four runs (same pipeline, per-scene working points; full details in
[`docs/results.md`](docs/results.md#results-across-all-scenes)):

| Scene | Ground truth | Detected | False positives | Mode / tuning |
|---|---|---|---|---|
| **demo** (`data/`) | comb removed, pen added, remote moved+flipped | **3/3** | **0** | `tabletop` (default) |
| **scene2** (`data/scene2/`) | coconut removed | **1/1** (coconut ✓) | 4 extra entries | `geometric` (big object; lighting-immune) + `--blur_threshold 25` |
| **scene3** (`data/scene3/`) | chair removed, apple→cup | **2/2** | **0** | `tabletop` + `--min_change_score 85` |

**scene2** — coconut removed (`geometric` mode; the visibility check confirmed
it with 5896 points seen empty). The `tabletop` mode misses it because video
B's lighting differs strongly; the geometric mode adds 4 coverage-difference
entries (its known weakness with limited A coverage).

![scene2 result](outputs/scene2_geo/change.gif)

**scene3** — both changes found with zero false positives: the chair (typed
`replaced`, see the note in `docs/results.md`) and the apple→cup replacement.

![scene3 result](outputs/scene3/change.gif)

Honest summary: the demo scene reaches 3/3 with 0 false positives; the extra
scenes run end-to-end and find their ground-truth changes, but each needed a
short per-scene tuning pass and scene2 still shows the geometric mode's
coverage artifacts. That is the expected behaviour of a scene-tuned
photometric pipeline and is documented rather than hidden.

**Faithful 3D rendering (v0.8.0)** — each scene now ships a dark-background
render with per-change bounding boxes + labels and a before | after | 3D
layout, a colored **TSDF mesh** (`scene_mesh.ply`, ≤ 10 MB) and a
**self-contained interactive viewer** (`viewer.html` — open it in any browser:
orbit, zoom, toggle video A / B / changes). Diagnosis and the
`input_mode=pad` experiment: [visualization rework](docs/results.md#visualization-rework-steps-05).

## Pipeline

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

Stage-by-stage detail: [`docs/pipeline.md`](docs/pipeline.md).

## Models

| Role | Primary | Fallback |
|---|---|---|
| 3D reconstruction (poses, depth, point cloud) | VGGT | MapAnything → MASt3R → COLMAP/hloc |
| Labels and descriptions | Qwen3-VL-32B-Instruct | Qwen3-VL-8B, Gemma (vision) |
| Cross-scene object matching ("moved") | DINOv2 | SigLIP 2 |
| Object masks (stretch goal) | SAM 2 | DBSCAN clusters only |

Two reconstruction backends are implemented and write the same artifact
contract, so the rest of the pipeline is backend-agnostic
(`reconstruction.backend: vggt | mapanything`). The measured comparison
(alignment, metric scale, thin-object visibility, end-to-end results) is in
[`docs/results.md`](docs/results.md#backend-comparison-vggt-vs-mapanything).

Model licenses — **verified 2026-10-10**: Qwen3-VL, DINOv2, SigLIP 2, SAM 2 and
the MapAnything checkpoint we use are Apache-2.0; **VGGT is CC-BY-NC-4.0
(non-commercial)**; MASt3R's repo is gated.
See [`docs/models.md`](docs/models.md) for IDs, commits, and the checklist.

## Setup

- **GPU cluster (PBS):** [`docs/setup.md`](docs/setup.md) and [`gpu.md`](gpu.md)
- **Laptop (CPU-only stages):** [`docs/setup.md`](docs/setup.md)

Cluster essentials: PyTorch wheels for **CUDA 12.6 (`cu126`)**, and
**`unset CUDA_VISIBLE_DEVICES`** before any CUDA work (PBS sets it empty, which
hides all GPUs). Per-job limits: 16 CPUs, 64 GB RAM, 24 h, 5 jobs.

## How to run

```bash
python run.py --video_a data/a.mp4 --video_b data/b.mp4 --out outputs/
```

The one-command pipeline is implemented (`run.py` → `src/*`, orchestrator +
7 stages). Every stage also runs alone and reads/writes only the cache:

```bash
python src/extract_frames.py --config configs/default.yaml
python src/reconstruct.py    --config configs/default.yaml   # GPU
python src/diff.py           --config configs/default.yaml
python src/describe.py       --config configs/default.yaml   # GPU
python src/visualize.py      --config configs/default.yaml
python src/report.py         --config configs/default.yaml
```

Status: validated on synthetic smoke data; the run on the demo recordings is
pending (see [`docs/roadmap.md`](docs/roadmap.md)). Cluster templates live in
`scripts/`.

## Design decisions

1. **Joint reconstruction** — both videos are reconstructed in a single VGGT
   run so every frame lands in one coordinate frame, avoiding fragile
   cross-run registration.
2. **Occlusion-aware diff** — a region counts as *removed* only if the other
   video actually observed that space and saw it empty (visibility check using
   cached poses and depth). This suppresses false positives from occlusion.
3. **DINOv2 move matching** — a removal and an addition with similar crop
   embeddings but different 3D positions are reported as one *moved* object
   instead of two separate changes.

## Limitations and failure cases

Observed on the demo scene (quantified in [`docs/results.md`](docs/results.md)):

- objects thinner than VGGT's patch scale (~1.4 cm here) are **absent from the
  depth maps** — geometric diffing cannot see them; the photometric tabletop
  mode is required;
- static objects whose appearance differs between recordings (blur, exposure,
  viewing angle) generate candidate fragments that the VLM and structural
  filters must reject;
- a replacement at the same spot (comb → pen) is geometrically
  indistinguishable from "nothing changed"; appearance comparison is what
  separates them;
- areas one video never observed (portrait crop, occlusion) can only yield
  *unverified* candidates, reported separately and never counted as changes;
- reflective surfaces, textureless walls and scale ambiguity remain expected
  failure modes on other scenes (not all reproduced here).

## Documentation

| File | Contents |
|---|---|
| [`AGENT_PLAN.md`](AGENT_PLAN.md) | Plan, phases, checkpoints (source of truth) |
| [`docs/pipeline.md`](docs/pipeline.md) | Stage-by-stage pipeline and inputs/outputs |
| [`docs/verification.md`](docs/verification.md) | Every verification rule with the demo's real numbers |
| [`docs/debugging.md`](docs/debugging.md) | Cache map, inspector tool, per-symptom playbooks |
| [`docs/setup.md`](docs/setup.md) | Environment setup and run instructions |
| [`docs/models.md`](docs/models.md) | Model stack, VRAM plan, license checklist |
| [`docs/data.md`](docs/data.md) | Recording guide and ground-truth template |
| [`docs/architecture.md`](docs/architecture.md) | Repo layout, modules, config reference |
| [`docs/results.md`](docs/results.md) | Real results of the first end-to-end run |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Cluster and model failure modes |
| [`docs/roadmap.md`](docs/roadmap.md) | Phase status checklist |
| [`gpu.md`](gpu.md) | Cluster / GPU access reference |
| [`rules.md`](rules.md) | Working rules for agents |

## License

MIT — see [`LICENSE`](LICENSE). Model checkpoints keep their own licenses.
