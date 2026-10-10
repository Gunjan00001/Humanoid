#!/usr/bin/env bash
# GPU smoke test: Phase-1 verification + a real VGGT run on the static kitchen
# frames (tests/prepare_gpu_smoke.py), expected to produce *no* changes.
#
# Run via the cluster GPU wrapper from the repo root:
#     gpu-run -t 00:40:00 -g 1 -f tests/gpu_smoke_test.sh
#
# Everything it writes stays inside the repo (cache/gpu_smoke, logs/).
set -uo pipefail

cd "$(dirname "$0")/.."
PY="$HOME/miniconda3/envs/spatialdiff/bin/python"
[ -x "$PY" ] || { echo "env python missing: $PY"; exit 1; }
mkdir -p logs

echo "=============================================================="
echo "[1/4] Phase-1 verification: CUDA visibility in a GPU job"
echo "=============================================================="
"$PY" - <<'PYEOF'
import torch, sys
print("python      :", sys.executable)
print("torch       :", torch.__version__, "| cuda build", torch.version.cuda)
print("available   :", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}  free {free/2**30:5.1f} / {total/2**30:.0f} GiB")
PYEOF
[ $? -eq 0 ] || { echo "torch/CUDA check failed"; exit 1; }

echo
echo "=============================================================="
echo "[2/4] pick the idlest GPU (shared node - see gpu.md)"
echo "=============================================================="
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="$GPU"
echo "pinning CUDA_VISIBLE_DEVICES=$GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -8

echo
echo "=============================================================="
echo "[3/4] VGGT joint reconstruction on the kitchen smoke frames"
echo "=============================================================="
"$PY" src/reconstruct.py --config configs/default.yaml --cache cache/gpu_smoke --force
[ $? -eq 0 ] || { echo "reconstruct failed"; exit 1; }

echo
echo "=============================================================="
echo "[4/4] diff on the static scene (expected: 0 changes)"
echo "=============================================================="
"$PY" src/diff.py --config configs/default.yaml --cache cache/gpu_smoke --force

"$PY" - <<'PYEOF'
import json, numpy as np
cam = np.load("cache/gpu_smoke/reconstruction/cameras.npz")
dense = np.load("cache/gpu_smoke/reconstruction/dense.npz")
meta = json.load(open("cache/gpu_smoke/reconstruction/meta.json"))
print("--- reconstruction summary ---")
print("frames:", meta["num_frames"], "| image_hw:", meta["image_hw"], "| device:", meta["device"])
print("extrinsic:", cam["extrinsic"].shape, "| intrinsic:", cam["intrinsic"].shape)
print("depth:", dense["depth"].shape, "| points:", dense["world_points"].shape)
print("depth conf: median %.2f p10 %.2f p90 %.2f" % tuple(np.percentile(dense["depth_conf"], [50, 10, 90])))
pts = dense["world_points"].reshape(-1, 3)
print("world bbox min", np.round(pts.min(0), 2).tolist(), "max", np.round(pts.max(0), 2).tolist())
cl = json.load(open("cache/gpu_smoke/diff/clusters.json"))["clusters"]
print("--- diff summary ---")
print("clusters:", len(cl))
for c in cl:
    print(" ", c["id"], c["type"], c["n_points"], c["centroid"])
PYEOF
echo "SMOKE TEST COMPLETE"
