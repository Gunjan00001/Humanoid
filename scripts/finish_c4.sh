#!/bin/bash
# Runs after the SceneDiff job ends: failure list, headline numbers (unprocessable
# pairs counted as empty predictions), processable-only numbers. Never evaluates
# the test slice.
set -uo pipefail
cd /home/user/Desktop/Humanoid
PY=/home/user/miniconda3/envs/spatialdiff/bin/python
JID="${1:-11367}"

while qstat "$JID" >/dev/null 2>&1; do sleep 30; done
echo "=== job $JID ended $(date +%F_%H:%M:%S) ==="

"$PY" - <<'PY' | tee logs/c4_failures.txt
import pickle
from pathlib import Path
val = [l.strip() for l in open("logs/val20.txt") if l.strip()]
data = Path("data/scenediff_benchmark/data")
pred = Path("logs/eval_pred_val")
proc = Path("logs/eval_pred_val_processable")
logtext = Path("logs/job_scenediff.log").read_text(errors="replace") if Path("logs/job_scenediff.log").exists() else ""
for p in val:
    src = pred / p / "object_masks.pkl"
    seg = pickle.load(open(data / p / "segments.pkl", "rb"))
    size = None
    for m in list(seg.get("video1_objects", {}).values())[:1]:
        for mm in m.values(): size = mm["size"]
    if size is None:
        for m in list(seg.get("video2_objects", {}).values())[:1]:
            for mm in m.values(): size = mm["size"]
    H, W = (size or [1080, 1920])
    d = pickle.load(open(src, "rb")) if src.exists() else None
    has_obj = bool(d and any(k not in ("H", "W") for k in d))
    if has_obj:
        (proc / p).mkdir(parents=True, exist_ok=True)
        pickle.dump(d, open(proc / p / "object_masks.pkl", "wb"))
        print(f"OK        {p}")
    else:
        # headline: empty prediction
        (pred / p).mkdir(parents=True, exist_ok=True)
        pickle.dump({"H": H, "W": W}, open(pred / p / "object_masks.pkl", "wb"))
        # reason
        reason = "unknown"
        idx = logtext.find(p)
        if idx >= 0:
            window = logtext[idx:idx + 4000]
            if "frames survived" in window: reason = "too few sharp frames"
            elif "OutOfMemory" in window: reason = "node OOM"
        print(f"UNPROC    {p}  reason={reason}")
PY

echo "=== HEADLINE (unprocessable = empty prediction) ==="
"$PY" third_party/scene_diff/scripts/evaluate_multiview.py --pred_dir logs/eval_pred_val \
    --splits val --sets varied --output_path logs/eval_val_headline.txt --visualize False 2>&1 \
    | grep -E "Metric|True Positive|False Positive|False Negative" | tee logs/c4_val_headline.txt
echo "=== PROCESSABLE ONLY ==="
"$PY" third_party/scene_diff/scripts/evaluate_multiview.py --pred_dir logs/eval_pred_val_processable \
    --splits val --sets varied --output_path logs/eval_val_processable.txt --visualize False 2>&1 \
    | grep -E "Metric|True Positive|False Positive|False Negative" | tee logs/c4_val_processable.txt
echo "C4_FINISHER_DONE"
