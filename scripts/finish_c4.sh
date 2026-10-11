#!/bin/bash
# Runs after the SceneDiff job ends: mechanical validation sanity checks + the
# validation evaluator. Does NOT evaluate the test slice.
set -uo pipefail
cd /home/jiit_992401210050/Desktop/Humanoid
PY=/home/jiit_992401210050/miniconda3/envs/spatialdiff/bin/python

while qstat 11033 >/dev/null 2>&1; do sleep 30; done
echo "=== job 11033 ended $(date +%F_%H:%M:%S) ==="

"$PY" - <<'PY' > logs/c4_sanity.txt 2>&1
import pickle, json
from pathlib import Path
val = [l.strip() for l in open("logs/val20.txt") if l.strip()]
data = Path("data/scenediff_benchmark/data")
pred = Path("logs/eval_pred_val")
ok = 0; bad = []
for p in val:
    f = pred / p / "object_masks.pkl"
    if not f.exists():
        bad.append((p, "missing pkl")); continue
    d = pickle.load(open(f, "rb"))
    seg = pickle.load(open(data / p / "segments.pkl", "rb"))
    gs = None
    for m in list(seg.get("video1_objects", {}).values())[:1]:
        for mm in m.values(): gs = mm["size"]
    if gs and [d["H"], d["W"]] != gs:
        bad.append((p, f"H/W {d['H']}x{d['W']} != GT {gs}")); continue
    if not any(k not in ("H", "W") for k in d):
        bad.append((p, "no objects")); continue
    ok += 1
print("valid pkl:", ok, "/", len(val))
for p, why in bad: print("  BAD", p, why)
PY
cat logs/c4_sanity.txt

echo "=== validation evaluator ==="
"$PY" third_party/scene_diff/scripts/evaluate_multiview.py --pred_dir logs/eval_pred_val \
    --splits val --sets varied --output_path logs/eval_val_result.txt --visualize False 2>&1 \
    | grep -E "Metric|True Positive|False Positive|False Negative" | tee logs/c4_val_metrics.txt

echo "C4_FINISHER_DONE"
