# Models

The stack is fixed by [`../AGENT_PLAN.md`](../AGENT_PLAN.md) §3. This document
records where each model runs, what it costs, and the license status that must be
cleared before results are published.

## Stack

| Role | Primary | Fallback | Runs on | GPU |
|---|---|---|---|---|
| 3D reconstruction, poses, depth, point cloud | **VGGT** | **MapAnything** → MASt3R → COLMAP/hloc | GPU node | 1 |
| Labels + descriptions | **Qwen3-VL-32B-Instruct** | Qwen3-VL-8B, Gemma (vision) | GPU node | 2 (bf16), or 1 if it fits |
| Cross-scene object matching | **DINOv2** | SigLIP 2 | GPU node | 1 (small) |
| Object masks (stretch) | **SAM 2** | none, DBSCAN clusters | GPU node | 1 |
| Diff / clustering / visualization | OpenCV, Open3D, NumPy, SciPy, scikit-learn |, | CPU | 0 |

The reconstruction backend is selectable via `reconstruction.backend`
(`vggt` | `mapanything`); both write the same artifact contract, so every
downstream stage works unchanged. See the backend comparison in
[`results.md`](results.md#backend-comparison-vggt-vs-mapanything).

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

## Exact identifiers and licenses, CHECKLIST

> **Status: verified 2026-10-10** via the Hugging Face / GitHub APIs. Key
> finding: **VGGT (the primary reconstruction model) is CC-BY-NC-4.0 , 
> non-commercial.** The challenge submission is fine, but any commercial use
> would need a swap to a permissive backend (e.g. COLMAP/hloc) or a license
> from Meta. This is stated prominently in the README.

| Model | Exact ID / repo | Version / commit | License | Commercial use OK? | Verified by / date |
|---|---|---|---|---|---|
| VGGT | `facebook/VGGT-1B`, code `facebookresearch/vggt` | HF sha `860abec7937d`, repo commit `a288dd0f` | **CC-BY-NC-4.0** | ❌ non-commercial | agent, 2026-10-10 (HF API) |
| MapAnything (used) | `facebook/map-anything-apache`, code `facebookresearch/map-anything` | HF sha `00f9c245bbcb`, v1.1.4 | **Apache-2.0** | ✅ | agent, 2026-10-10 (HF API) |
| MapAnything (default) | `facebook/map-anything` |, | **CC-BY-NC-4.0** | ❌ non-commercial | agent, 2026-10-10 (HF API) |
| Depth Anything 3 (used, stretch) | `depth-anything/DA3-LARGE-1.1` via MapAnything's unified interface | HF sha, (fetched at run time) | **Apache-2.0** | ✅ | agent, 2026-10-10 (HF API) |
| Depth Anything 3 (MapAnything default) | `depth-anything/DA3-GIANT-1.1` | HF sha `72ee9f89ce4e` | **CC-BY-NC-4.0** | ❌ non-commercial | agent, 2026-10-10 (HF API) |
| MASt3R | `naver/mast3r` |, | repository **gated**, license not verifiable on this account | unknown | agent, 2026-10-10 (HF API 401) |
| COLMAP | `colmap/colmap` |, | repo LICENSE (BSD-3-Clause style; SPDX not auto-detected) | likely yes, verify before commercial use | agent, 2026-10-10 (GitHub API) |
| hloc | `cvg/Hierarchical-Localization` |, | Apache-2.0 | ✅ | agent, 2026-10-10 (GitHub API) |
| Qwen3-VL-32B-Instruct | `Qwen/Qwen3-VL-32B-Instruct` | sha `0cfaf48183f5` | Apache-2.0 | ✅ | agent, 2026-10-10 (HF API) |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | sha `0c351dd01ed8` | Apache-2.0 | ✅ | agent, 2026-10-10 (HF API) |
| DINOv2 | `facebook/dinov2-large` | sha `47b73eefe95e` | Apache-2.0 | ✅ | agent, 2026-10-10 (HF API) |
| SigLIP 2 | `google/siglip2-base-patch16-224` |, | Apache-2.0 | ✅ | agent, 2026-10-10 (HF API) |
| SAM 2 | `facebook/sam2-hiera-large` |, | Apache-2.0 | ✅ | agent, 2026-10-10 (HF API) |

VGGT is installed in the env via `pip install git+https://github.com/facebookresearch/vggt.git`
(package name `vggt`, commit above). The weights are downloaded separately by
`scripts/download_models.sh` and load offline from `~/models/VGGT-1B/`.

MapAnything is **purpose-built for metric scale** and ships two checkpoints
that differ only in training data and license; this project uses the
Apache-2.0 one (`facebook/map-anything-apache`, loaded from
`~/models/map-anything-apache/`) so the whole stack stays commercially usable.
It runs in an **isolated conda env** (`spatialdiff-ma`, created by
`scripts/setup_env.sh`) because its dependency pins (e.g.
`opencv-python-headless==4.10.0.84`) conflict with the main `spatialdiff` env;
`src/reconstruct.py` launches it as a subprocess and the worker writes the
same artifact contract (see `scripts/run_mapanything.py` and
`src/ma_contract.py`).

## Why these models

- **VGGT**, feed-forward reconstruction of multiple frames into one shared
  coordinate frame, which is exactly what "reconstruct both videos together"
  needs; no per-frame optimization loop.
- **MASt3R**, strong two-view matching fallback if VGGT misaligns or OOMs.
- **COLMAP**, classical last resort; slow but predictable.
- **Qwen3-VL-32B**, best open VLM available here for short object labels and
  fluent descriptions; the 8B variant is the fast fallback.
- **DINOv2**, robust self-supervised crop embeddings for deciding that two
  detections are the *same* object moved, not two different changes.
- **SAM 2**, optional: tightens cluster masks so crops are cleaner.
