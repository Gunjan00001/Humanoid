# Models

The stack is fixed by [`../AGENT_PLAN.md`](../AGENT_PLAN.md) §3. This document
records where each model runs, what it costs, and the license status that must be
cleared before results are published.

## Stack

| Role | Primary | Fallback | Runs on | GPU |
|---|---|---|---|---|
| 3D reconstruction — poses, depth, point cloud | **VGGT** | MASt3R → COLMAP/hloc | GPU node | 1 |
| Labels + descriptions | **Qwen3-VL-32B-Instruct** | Qwen3-VL-8B, Gemma (vision) | GPU node | 2 (bf16), or 1 if it fits |
| Cross-scene object matching | **DINOv2** | SigLIP 2 | GPU node | 1 (small) |
| Object masks (stretch) | **SAM 2** | none — DBSCAN clusters | GPU node | 1 |
| Diff / clustering / visualization | OpenCV, Open3D, NumPy, SciPy, scikit-learn | — | CPU | 0 |

## Memory notes

| Model | dtype | Rough GPU memory | Notes |
|---|---|---|---|
| VGGT | bf16 | 1 GPU | Reduce frame count / resolution if OOM |
| Qwen3-VL-32B | bf16 | ~64 GB → **2 GPUs** | `device_map="auto"`, `low_cpu_mem_usage=True` |
| Qwen3-VL-8B | bf16 | 1 GPU | Preferred when 2 GPUs are hard to schedule |
| DINOv2 | fp16/bf16 | < 4 GB | Small ViT variants suffice |
| SAM 2 | bf16 | 1 GPU | Optional polish only |

Host RAM is capped at 64 GB per job, so load weights directly onto the GPU
instead of into RAM first.

## Downloads

Run on the **login node** (internet is available there; compute nodes may not
have any):

```bash
export HF_HOME=~/models
huggingface-cli download <model-id>     # fill in the exact IDs below
```

## Exact identifiers and licenses — CHECKLIST

> **Status: not yet verified.** Fill every cell in during Phase 1 and before any
> public release. Some reconstruction checkpoints are known to be
> **non-commercial**; if the challenge requires commercial-friendly licensing,
> the affected backend must be swapped or the limitation stated prominently.

| Model | Exact ID / repo (fill in) | Version / commit | License (fill in) | Commercial use OK? | Verified by / date |
|---|---|---|---|---|---|
| VGGT | TODO | TODO | TODO — check before use | TODO | TODO |
| MASt3R | TODO | TODO | TODO — commonly non-commercial | TODO | TODO |
| COLMAP / hloc | TODO | TODO | TODO | TODO | TODO |
| Qwen3-VL-32B-Instruct | TODO | TODO | TODO | TODO | TODO |
| Qwen3-VL-8B | TODO | TODO | TODO | TODO | TODO |
| DINOv2 | TODO | TODO | TODO | TODO | TODO |
| SigLIP 2 | TODO | TODO | TODO | TODO | TODO |
| SAM 2 | TODO | TODO | TODO | TODO | TODO |

The verified licenses will be summarised in the main README once filled in —
the plan (§3, §9) requires "models used, why they were chosen, and their
licenses" there.

## Why these models

- **VGGT** — feed-forward reconstruction of multiple frames into one shared
  coordinate frame, which is exactly what "reconstruct both videos together"
  needs; no per-frame optimization loop.
- **MASt3R** — strong two-view matching fallback if VGGT misaligns or OOMs.
- **COLMAP** — classical last resort; slow but predictable.
- **Qwen3-VL-32B** — best open VLM available here for short object labels and
  fluent descriptions; the 8B variant is the fast fallback.
- **DINOv2** — robust self-supervised crop embeddings for deciding that two
  detections are the *same* object moved, not two different changes.
- **SAM 2** — optional: tightens cluster masks so crops are cleaner.
