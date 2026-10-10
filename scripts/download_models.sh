#!/usr/bin/env bash
# Download all model weights on the LOGIN node (compute nodes may have no
# internet) into the shared model folder. See docs/models.md.
#
#   ~/models/
#   ├── VGGT-1B/                  (phase 2: reconstruction)
#   ├── dinov2-large/             (phase 4: crop matching)
#   ├── Qwen3-VL-8B-Instruct/     (phase 4: fallback VLM)
#   └── Qwen3-VL-32B-Instruct/    (phase 4: primary VLM, ~65 GB)
#
# Resumable: rerun any time; finished repos are skipped, partial ones continue.
# Uses the env's CLI by absolute path (see setup_env.sh for why).
#
# Usage:  bash scripts/download_models.sh
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-spatialdiff}"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
HF="$ENV_PREFIX/bin/hf"
[ -x "$HF" ] || HF="$ENV_PREFIX/bin/huggingface-cli"

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
WORKERS="${WORKERS:-4}"

export HF_HOME="$MODEL_DIR/.hf"          # metadata cache, keeps repos tidy
export HF_HUB_ENABLE_HF_TRANSFER=1       # fast download path (hf_transfer)
mkdir -p "$MODEL_DIR"

download() {  # <hf-repo-id> <local-dir-name>
    local repo="$1" name="$2"
    echo "==> ${repo}  ->  ${MODEL_DIR}/${name}"
    "$HF" download "$repo" --local-dir "$MODEL_DIR/$name" --max-workers "$WORKERS"
}

# Order: smallest/essential first so an interruption leaves the most useful set.
download facebook/VGGT-1B              VGGT-1B
download facebook/dinov2-large         dinov2-large
download Qwen/Qwen3-VL-8B-Instruct     Qwen3-VL-8B-Instruct
download Qwen/Qwen3-VL-32B-Instruct    Qwen3-VL-32B-Instruct

# Not downloaded by default:
#   - MASt3R  (fallback backend; HF repo is gated — needs manual access)
#   - SAM 2   (stretch goal; only used if semantics.use_sam2 is enabled)
echo "==> all requested weights are in $MODEL_DIR"
du -sh "$MODEL_DIR"/*/ 2>/dev/null || true
