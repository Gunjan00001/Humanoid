# Setup

Two environments are possible: the **GPU cluster** (all heavy stages) and a
**laptop** (CPU-only stages). The laptop is optional — every stage can run on the
cluster.

Read [`../gpu.md`](../gpu.md) for the verified cluster reference (queues, CUDA,
the `CUDA_VISIBLE_DEVICES` gotcha).

## Which stages run where

| Stage | Laptop (CPU) | GPU cluster |
|---|---|---|
| 1. Frame extraction | ✅ | ✅ |
| 2. Reconstruction (VGGT/MASt3R) | ❌ | ✅ 1 GPU |
| 3. Alignment (ICP) | ✅ | ✅ |
| 4. Change detection (voxel + DBSCAN) | ✅ | ✅ |
| 5. Semantics (Qwen3-VL + DINOv2) | ❌ | ✅ 1–2 GPUs |
| 6. Visualization (Open3D) | ✅ (viewer may need display) | ✅ |
| 7. Report | ✅ | ✅ |

The CPU stages only need the cached outputs in `cache/`, which is why caching
matters: you can reconstruct once on the cluster, then iterate on the diff and
report locally.

## Cluster environment

Python 3.10, PyTorch built for **CUDA 12.6 (`cu126`)** — verified working on the
525 driver.

```bash
conda create -n spatialdiff python=3.10 -y
conda activate spatialdiff
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
pip install open3d opencv-python transformers accelerate \
    scikit-learn scipy pyyaml huggingface_hub
```

Then install **VGGT** from its official repository, following that repo's
README (record the exact URL + commit in [`models.md`](models.md)).

Environment file: `environment.yml` and `scripts/setup_env.sh` (planned, see
[`roadmap.md`](roadmap.md)).

### Verify CUDA inside a GPU job

Never verify on the login node — it has no GPUs and PBS hides them in jobs until
`CUDA_VISIBLE_DEVICES` is unset.

```bash
qsub -I -q gpu
unset CUDA_VISIBLE_DEVICES
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatialdiff
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected: `True <n>` (all 8 A100s are visible unless you pin one).

### Batch jobs

Templates: `scripts/job_reconstruct.pbs`, `scripts/job_describe.pbs`,
`scripts/job_full.pbs` (planned). Every job must include:

```bash
unset CUDA_VISIBLE_DEVICES   # otherwise PBS hides all GPUs
```

Requests available to you (per `gpu.md`): up to 16 CPUs, 64 GB RAM, 24 h
walltime, 5 concurrent jobs. Load large models straight to GPU with
`device_map="auto"` and `low_cpu_mem_usage=True` to stay under the RAM cap.

## Model weights

Compute nodes may have **no internet**. Download everything on the login node
into the shared model folder:

```bash
export HF_HOME=~/models
huggingface-cli download <model-id>        # exact IDs in docs/models.md
```

Then point `paths.model_dir` in `configs/default.yaml` at `~/models/`.

## Laptop environment (CPU stages)

```bash
python -m venv .venv
. .venv/bin/activate
pip install numpy scipy scikit-learn opencv-python open3d pyyaml
```

The GPU stages will refuse to run without CUDA; use the cached artifacts from a
cluster run instead.

## Configuration

Every model name, path, and threshold lives in `configs/default.yaml` — never
hard-code values in scripts. See
[`architecture.md`](architecture.md#config-reference).
