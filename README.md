# 3D Scene Change Detection

System that takes two phone recordings of the same small indoor space, builds a
shared 3D reconstruction, detects which objects were **moved**, **added**, or
**removed**, and describes each change in natural language.

Part of the **Spatial AI & Navigation Challenge**.

> **Status: work in progress — no results yet.** The plan and checkpoints live in
> [`AGENT_PLAN.md`](AGENT_PLAN.md). This README follows its §9 requirements and is
> filled in with **real outputs** as phases complete.

## Results

_No results yet._ A result GIF, screenshots, and the example `report.json` /
`report.md` will be linked here after the first successful end-to-end run on the
demo scene. Nothing in this repository will be fabricated; every number comes
from a real run.

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
| 3D reconstruction (poses, depth, point cloud) | VGGT | MASt3R → COLMAP/hloc |
| Labels and descriptions | Qwen3-VL-32B-Instruct | Qwen3-VL-8B, Gemma (vision) |
| Cross-scene object matching ("moved") | DINOv2 | SigLIP 2 |
| Object masks (stretch goal) | SAM 2 | DBSCAN clusters only |

Model licenses — **verified 2026-10-10**: Qwen3-VL, DINOv2, SigLIP 2, SAM 2 are
Apache-2.0; **VGGT is CC-BY-NC-4.0 (non-commercial)**; MASt3R's repo is gated.
See [`docs/models.md`](docs/models.md) for IDs, commits, and the checklist.

## Setup

- **GPU cluster (PBS):** [`docs/setup.md`](docs/setup.md) and [`gpu.md`](gpu.md)
- **Laptop (CPU-only stages):** [`docs/setup.md`](docs/setup.md)

Cluster essentials: PyTorch wheels for **CUDA 12.6 (`cu126`)**, and
**`unset CUDA_VISIBLE_DEVICES`** before any CUDA work (PBS sets it empty, which
hides all GPUs). Per-job limits: 16 CPUs, 64 GB RAM, 24 h, 5 jobs.

## How to run

_Planned interface — not implemented yet:_

```bash
python run.py --video_a data/a.mp4 --video_b data/b.mp4 --out outputs/
```

Per-stage commands and PBS templates will be documented in
[`docs/setup.md`](docs/setup.md) and `scripts/`.

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

Expected items, to be validated and quantified on real runs (see
[`docs/results.md`](docs/results.md)):

- false positives from reflective surfaces, thin objects, or lighting changes;
- small objects missed or merged into larger clusters;
- weak visual matches can split a moved object into one removal + one addition;
- textureless walls reduce reconstruction quality;
- scale ambiguity between runs if alignment is imperfect.

## Documentation

| File | Contents |
|---|---|
| [`AGENT_PLAN.md`](AGENT_PLAN.md) | Plan, phases, checkpoints (source of truth) |
| [`docs/pipeline.md`](docs/pipeline.md) | Stage-by-stage pipeline and inputs/outputs |
| [`docs/setup.md`](docs/setup.md) | Environment setup and run instructions |
| [`docs/models.md`](docs/models.md) | Model stack, VRAM plan, license checklist |
| [`docs/data.md`](docs/data.md) | Recording guide and ground-truth template |
| [`docs/architecture.md`](docs/architecture.md) | Repo layout, modules, config reference |
| [`docs/results.md`](docs/results.md) | Results (placeholder until real runs) |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Cluster and model failure modes |
| [`docs/roadmap.md`](docs/roadmap.md) | Phase status checklist |
| [`gpu.md`](gpu.md) | Cluster / GPU access reference |
| [`rules.md`](rules.md) | Working rules for agents |

## License

MIT — see [`LICENSE`](LICENSE). Model checkpoints keep their own licenses.
