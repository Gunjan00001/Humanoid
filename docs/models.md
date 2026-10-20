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

## SAM 3.1 (integrated through the official package; not adopted)

* repo: `facebook/sam3.1`, gated (accepted), HF licence field "other". The local
  `LICENSE` (7.3 KB) is the authority; clauses are quoted below.
* local files: `sam3.1_multiplex.pt` (3.5 GB), `LICENSE`, config with
  `architectures = ["Sam3VideoModel"]`, `model_type = "sam3_video"`, and
  `processor_config.json`.
* integration: works through the official `facebookresearch/sam3` package in the
  `spatialdiff-sam3` env with `torch.autocast("cuda", dtype=torch.bfloat16)`. The
  text tokenizer needs the BPE file shipped in that package
  (`sam3/assets/bpe_simple_vocab_16e6.txt.gz`); `merges.txt` is not a substitute.
  The checkpoint is Meta's own format, not transformers format, so
  `Sam3Model.from_pretrained` cannot load it.
* bake-off result (image text-prompt route, 11 GT objects): **4 of 11** objects
  segmented in both videos with a name prompt, 2 of 11 with the generic prompt
  "object", and **14 spurious masks**. Track IDs were not measured because the
  image route has no tracker. **Not adopted as the change segmenter**; SAM 2.1
  remains the fallback and `semantics.use_sam2` stays false.
* **Not cleared for commercial use.** The licence is a limited research licence;
  clauses quoted verbatim from the local copy:
  * Grant: "You are granted a non-exclusive, worldwide, non-transferable and
    royalty-free limited license under Meta's intellectual property or other
    rights owned by Meta embodied in the SAM Materials to use, reproduce,
    distribute, copy, create derivative works of, and make modifications to the
    SAM Materials."
  * Redistribution: "Distribution of SAM Materials, and any derivative works
    thereof, are subject to the terms of this Agreement. If you distribute or
    make the SAM Materials, or any derivative works thereof, available to a third
    party, you may only do so under the terms of this Agreement and you shall
    provide a copy of this Agreement with any such SAM Materials."
  * Acknowledgement: "If you submit for publication the results of research you
    perform on, using, or otherwise in connection with SAM Materials, you must
    acknowledge the use of SAM Materials in your publication."
  * Restrictions: "Your use of the SAM Materials will not involve or encourage
    others to reverse engineer, decompile or discover the underlying components
    of the SAM Materials." and "You agree not to use, or permit others to use,
    SAM Materials for any activities subject to the International Traffic in Arms
    Regulations (ITAR) or end uses prohibited by Trade Controls, including those
    related to military or warfare purposes, nuclear industries or applications,
    espionage, or the development or use of guns or illegal weapons."
  * Termination: "If you institute litigation or other proceedings against Meta
    or any entity ... alleging that the SAM Materials, outputs or results, or any
    portion of any of the foregoing, constitutes infringement of intellectual
    property or other rights owned or licensable by you, then any licenses
    granted to you under this Agreement shall terminate as of the date such
    litigation or claim is filed or instituted."

Two MapAnything checkpoints exist and both are listed in the table above:
`facebook/map-anything-apache` (**Apache-2.0**, the one used here, 13.4 mm static
residual) and `facebook/map-anything` (**CC-BY-NC-4.0**, upstream default, not
used). This project uses only the Apache-2.0 checkpoint.

### SAM 3.1 environment (added)

`spatialdiff-sam3`: Python 3.12.15, torch 2.14.1+cu126, `sam3 0.1.0` from
`github.com/facebookresearch/sam3`, with the undeclared dependencies the
package needs (einops, timm, omegaconf, hydra-core, pycocotools, psutil,
scipy, imageio, numpy<2, setuptools<81). Import verified; the model build takes
`checkpoint_path=~/models/sam3.1/sam3.1_multiplex.pt` and a `bpe_path` for the
text tokenizer, which is the one remaining detail before the per-frame
segmentation bake-off.

## Acknowledgements, licences and non-commercial scope

* **SAM 3.1** (Meta, custom "SAM License"): use in this project is research only.
  The licence asks that publications state the use of SAM Materials in the
  acknowledgement; the local licence text is the authority (docs/models.md
  above). **Non commercial.**
* **DINOv3 ViT-L** (Meta, DINOv3 License): used for the adopted matching
  descriptor; read the local licence before any redistribution. Check the
  commercial terms before product use.
* **DINOv2-large** (Apache-2.0): fallback matcher.
* **Qwen3-VL-32B** and **Qwen3.5** (Alibaba, Apache-2.0 per the probed cards):
  naming baseline and challengers.
* **VGGT-1B** (Meta, CC BY-NC 4.0): reconstruction backend. **Non commercial.**
* **MapAnything** (Apache-2.0 variant used here): geometry alternative.
* **Depth Anything 3** (GIANT cc-by-nc-4.0 / BASE apache-2.0) and **Pi3X**
  (cc-by-nc-4.0): probed, not integrated. **Non commercial** where noted.
* **SceneDiff** benchmark: the external evaluation slice, validation results in
  docs/evaluation.md; its own licence applies to the data, which is not
  redistributed in this repository.
* Related work and citations used in the write-up: **UNITE** (arXiv 2512.14364),
  **SceneDiff**, **HD-Epic**, **VGGT**, **MapAnything**, **SAM 2**, **SAM 3D**,
  **PanSt3R**; see docs/related_work.md and the README citations block.
