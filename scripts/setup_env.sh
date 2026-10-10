#!/usr/bin/env bash
# Create / refresh the `spatialdiff` conda env used by every GPU stage.
#
# Safe to run on the login node (no GPU needed). Idempotent: re-running only
# upgrades missing packages. Verified constraints (see gpu.md):
#   - driver 525 / max CUDA 12.3  ->  torch must come from the cu126 index
#   - python 3.10 is what the plan pins (docs/setup.md)
#
# IMPORTANT: this script never relies on `conda activate`. Agent sessions and
# some cluster shells carry a PATH where /cm/shared/apps/anaconda3/bin comes
# first, so activation alone can still resolve the shared python 3.8. All
# commands below use the env's interpreter by absolute path.
#
# Usage:  bash scripts/setup_env.sh
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-spatialdiff}"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
ENV_PY="$ENV_PREFIX/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -x "$ENV_PY" ]; then
    echo "==> creating conda env '$ENV_NAME' (python 3.10)"
    "$CONDA_BASE/bin/conda" create -n "$ENV_NAME" python=3.10 -y
else
    echo "==> conda env '$ENV_NAME' already exists at $ENV_PREFIX, reusing"
fi

"$ENV_PY" -m pip install --upgrade pip

# 1) PyTorch for CUDA 12.6 (do NOT use cu130 wheels: driver max is CUDA 12.3)
echo "==> installing torch / torchvision (cu126)"
"$ENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2) Project dependencies (pipeline stages)
echo "==> installing pipeline dependencies"
"$ENV_PY" -m pip install \
    open3d \
    opencv-python \
    transformers \
    accelerate \
    scikit-learn \
    scipy \
    pyyaml \
    pillow \
    tqdm \
    einops \
    safetensors \
    matplotlib \
    "huggingface_hub[cli]" \
    hf_transfer

# 3) VGGT from the official repository (pip-installable package; see docs/models.md)
echo "==> installing VGGT"
"$ENV_PY" -m pip install "git+https://github.com/facebookresearch/vggt.git"

echo "==> versions"
"$ENV_PY" - <<'PY'
import sys
import torch
print("python     ", sys.version.split()[0], "|", sys.executable)
print("torch      ", torch.__version__, "| cuda build", torch.version.cuda)
print("cuda avail  (False on the login node is expected):", torch.cuda.is_available())
PY

# ---------------------------------------------------------------------------
# 4) Isolated MapAnything env (reconstruction.backend: mapanything)
#
# MapAnything pins dependencies that conflict with the main env
# (e.g. opencv-python-headless==4.10.0.84), so it gets its own conda env and
# runs as a subprocess - see scripts/run_mapanything.py and docs/models.md.
# Its own package requires python >= 3.10 (the 3.12 in its quickstart is just
# a recommendation).
MA_ENV_NAME="${MA_ENV_NAME:-spatialdiff-ma}"
MA_ENV_PREFIX="$CONDA_BASE/envs/$MA_ENV_NAME"
MA_ENV_PY="$MA_ENV_PREFIX/bin/python"

if [ ! -x "$MA_ENV_PY" ]; then
    echo "==> creating conda env '$MA_ENV_NAME' (python 3.10, isolated)"
    "$CONDA_BASE/bin/conda" create -n "$MA_ENV_NAME" python=3.10 -y
else
    echo "==> conda env '$MA_ENV_NAME' already exists at $MA_ENV_PREFIX, reusing"
fi

echo "==> installing torch (cu126) + MapAnything in $MA_ENV_NAME"
"$MA_ENV_PY" -m pip install --upgrade pip
"$MA_ENV_PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
"$MA_ENV_PY" -m pip install "git+https://github.com/facebookresearch/map-anything.git"
"$MA_ENV_PY" -m pip install pyyaml        # used by src/common.py in the worker

# MapAnything's hydra configs are not part of the pip package. A shallow source
# checkout under third_party/ (gitignored) provides them for the external-model
# interface (da3, ...) - the worker prepends it to sys.path.
if [ ! -d "$REPO_ROOT/third_party/map-anything/configs" ]; then
    echo "==> cloning MapAnything source (configs for external models) into third_party/"
    git clone --depth 1 https://github.com/facebookresearch/map-anything.git \
        "$REPO_ROOT/third_party/map-anything"
fi

echo "==> env '$ENV_NAME' ready."
echo "    act with:  conda activate $ENV_NAME  (then: export PATH=\"\$CONDA_PREFIX/bin:\$PATH\")"
echo "==> env '$MA_ENV_NAME' ready (MapAnything backend worker)."
