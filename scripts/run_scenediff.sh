#!/bin/bash
# Run the pipeline on SceneDiff pairs, then adapt to the evaluator format.
#   scripts/run_scenediff.sh PAIRS_FILE PRED_DIR
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/spatialdiff/bin/python
DATA=data/scenediff_benchmark/data
PAIRS="$1"; PRED="$2"
while read -r p; do
  [ -z "$p" ] && continue
  a=$(ls "$DATA/$p"/original_video1.* 2>/dev/null | head -1)
  b=$(ls "$DATA/$p"/original_video2.* 2>/dev/null | head -1)
  echo "=== $p ==="
  "$PY" detect.py --a "$a" --b "$b" --name "sd_$p" \
      --out "outputs/scenediff/$p" --cache "cache/scenediff/$p" --local
  "$PY" src/eval_scenediff.py --pair "$p" --cache "cache/scenediff/$p" \
      --out "$PRED" --data "$DATA" --report "outputs/scenediff/$p/report.json"
done < "$PAIRS"
echo "SCENEDIFF_DRIVER_DONE"
