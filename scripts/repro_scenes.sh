#!/bin/bash
# Reproducibility harness: run the frozen pipeline 3x per scene into isolated,
# gitignored output dirs (never touching the committed outputs/<scene>).
#   scripts/repro_scenes.sh scene2 scene4
# Each detect.py run submits its own PBS gpu job and polls it.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/user/miniconda3/envs/spatialdiff/bin/python

for scene in "$@"; do
  for run in 1 2 3; do
    name="repro_${scene}_${run}"
    echo "=== $name START $(date +%F_%H:%M:%S) ==="
    "$PY" -u detect.py --a "data/$scene/a.mp4" --b "data/$scene/b.mp4" \
        --name "$name" --force
    echo "=== $name DONE rc=$? $(date +%F_%H:%M:%S) ==="
  done
done
echo "WORKER_DONE $*"
