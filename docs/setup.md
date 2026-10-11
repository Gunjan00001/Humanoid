# Setup

Two environments are possible: the **GPU cluster** (all heavy stages) and a
**laptop** (CPU-only stages). The laptop is optional, every stage can run on the
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

Two conda envs, both created by `scripts/setup_env.sh` (idempotent):

| Env | Python | Purpose |
|---|---|---|
| `spatialdiff` | 3.10 | the pipeline: torch cu126, VGGT, Open3D, transformers, … |
| `spatialdiff-ma` | 3.10 | **isolated** MapAnything backend (its dependency pins conflict with the main env); `src/reconstruct.py` launches it as a subprocess |

```bash
bash scripts/setup_env.sh        # creates/updates both envs
```

Manual equivalent for the main env:

```bash
conda create -n spatialdiff python=3.10 -y
conda activate spatialdiff
export PATH="$CONDA_PREFIX/bin:$PATH"        # see the PATH gotcha in troubleshooting.md
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install "vggt @ git+https://github.com/facebookresearch/vggt.git"   # VGGT (needs internet on this node)
```

VGGT installs from its GitHub repository (package name `vggt`); it is not on
PyPI. Run that line on the login node, which has internet. After it, a clean env
passes `python -c "import vggt, torch"` and `python detect.py --help`. The model
weights are downloaded separately (see Model weights below) and are what remain
unverified if you do not download them.

### `environment.yml`, status: **unverified** (cannot be built as-is)

`environment.yml` is a *machine export* (`conda env export`), not the canonical
install path (that is `scripts/setup_env.sh` / the manual block above). An
attempt to build it clean on 2026-10-11 with
`conda env create -f environment.yml -p .venv` **failed**:

- `vggt==0.0.1` is **not on PyPI** (it is installed from the git repo
  `facebookresearch/vggt`, per §"Installing VGGT" below) , 
  `ERROR: Could not find a version that satisfies the requirement vggt==0.0.1`;
- `torch==2.14.1+cu126` exists only on the PyTorch index
  (`https://download.pytorch.org/whl/cu126`), not on default PyPI;
- several pinned versions require **Python ≥ 3.11/3.12** while the export pins
  Python `3.10.22`, so pip ignored them (`Ignored the following versions that
  require a different python version: ...`).

**Verify a clean clone with the canonical path above, not with `environment.yml`.**
In the existing `spatialdiff` env the CLI and stack do work: `python detect.py
--help` exits 0 and `numpy, scipy, sklearn, cv2, open3d, torch, torchvision,
transformers, accelerate, yaml, matplotlib, PIL, huggingface_hub, safetensors`
all import. **The model stages were not re-run in a clean env, unverified
(weights not downloaded).**

The scripts never rely on `conda activate` alone: this cluster's sessions can
carry a PATH where the shared anaconda's python 3.8 wins over the env (see
[`troubleshooting.md`](troubleshooting.md)). They call the env interpreter by
absolute path, and PBS jobs re-prepend `$CONDA_PREFIX/bin` after activating.

### Verify CUDA inside a GPU job

Never verify on the login node, it has no GPUs and PBS hides them in jobs until
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
`scripts/job_full.pbs`. Every job includes:

```bash
unset CUDA_VISIBLE_DEVICES   # otherwise PBS hides all GPUs
```

Submit from the repo root: `qsub scripts/job_full.pbs`.

Requests available to you (per `gpu.md`): up to 16 CPUs, 64 GB RAM, 24 h
walltime, 5 concurrent jobs. Large models load straight onto the GPU with
`device_map="auto"` and `low_cpu_mem_usage=True` to stay under the RAM cap.
PBS does not pin GPUs on this cluster, check `nvidia-smi` and optionally pin
an idle GPU in the job (`export CUDA_VISIBLE_DEVICES=3` after the `unset`).

## Where to clone

Clone into your **shared home directory** (or any shared filesystem path). A
clone under `/tmp` is node-local: `detect.py` on the login node would submit a
PBS job that runs on a different machine and cannot see the files (detect.py
warns about this; `--local` on a GPU node is the alternative).

## Data and Git LFS

The six small development clips (< 5 MB each) and `data/scene5/b.mp4` (a
4.6 MB re-encode of the original 6.9 MB recording) are plain git files, so a
plain clone can read every video. The **originals** of the re-encoded files
and the TSDF meshes stay in Git LFS: if a file appears as a small text pointer
(`version https://git-lfs...`), run

```bash
git lfs pull          # or: git lfs install && git lfs pull
```

The scene5 held-out result was produced from the **original** b.mp4 (LFS oid
`1be79cea00`, 6,957,385 bytes); the shipped re-encode keeps the same
resolution (480x860) and fps (30) and is only for clone-ability.

## Model weights

Compute nodes may have **no internet**. Download everything on the login node
into the shared model folder:

```bash
bash scripts/download_models.sh        # ~88 GB: VGGT, DINOv2, Qwen3-VL 8B + 32B
```

It writes `~/models/<Model-Name>/` (resumable; rerun any time). The pipeline
resolves `models.*` config entries against `paths.model_dir` first, so it loads
from disk offline. Exact IDs/licenses: [`models.md`](models.md).

## Laptop environment (CPU stages)

```bash
python -m venv .venv
. .venv/bin/activate
pip install numpy scipy scikit-learn opencv-python open3d pyyaml
```

The GPU stages will refuse to run without CUDA; use the cached artifacts from a
cluster run instead.

## Configuration

Every model name, path, and threshold lives in `configs/default.yaml`, never
hard-code values in scripts. See
[`architecture.md`](architecture.md#config-reference).
