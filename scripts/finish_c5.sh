#!/bin/bash
# After the test-slice regeneration job ends: failure list, empty predictions for
# unprocessable test pairs, then the TEST evaluator exactly once.
set -uo pipefail
cd /home/user/Desktop/Humanoid
PY=/home/user/miniconda3/envs/spatialdiff/bin/python
JID="${1:-11368}"

while qstat "$JID" >/dev/null 2>&1; do sleep 30; done
echo "=== test job $JID ended $(date +%F_%H:%M:%S) ==="

"$PY" - <<'PY' | tee logs/c5_failures.txt
import pickle
from pathlib import Path
tst=[l.strip() for l in open("logs/test10.txt") if l.strip()]
data=Path("data/scenediff_benchmark/data"); pred=Path("logs/eval_pred_test"); proc=Path("logs/eval_pred_test_processable")
logtext=Path("logs/job_scenediff.log").read_text(errors="replace")
for p in tst:
    seg=pickle.load(open(data/p/"segments.pkl","rb")); size=None
    for key in ("video1_objects","video2_objects"):
        for m in list(seg.get(key,{}).values())[:1]:
            for mm in m.values(): size=mm["size"]
    H,W=(size or [1920,1080])
    src=pred/p/"object_masks.pkl"
    d=pickle.load(open(src,"rb")) if src.exists() else None
    has=d and any(k not in ("H","W") for k in d)
    proc_p=proc/p; proc_p.mkdir(parents=True,exist_ok=True)
    if has:
        pickle.dump(d,open(proc_p/"object_masks.pkl","wb")); print(f"OK       {p} {len([k for k in d if k not in ('H','W')])} objects")
    else:
        pred.mkdir(parents=True,exist_ok=True); (pred/p).mkdir(parents=True,exist_ok=True)
        pickle.dump({"H":H,"W":W},open(src,"wb")); print(f"UNPROC   {p}")
PY

echo "=== TEST HEADLINE (unprocessable = empty) ==="
"$PY" third_party/scene_diff/scripts/evaluate_multiview.py --pred_dir logs/eval_pred_test \
    --splits test --sets varied --output_path logs/eval_test_headline.txt --visualize False 2>&1 \
    | grep -E "Metric|True Positive|False Positive|False Negative" | tee logs/c5_test_headline.txt
echo "=== TEST PROCESSABLE ONLY ==="
"$PY" third_party/scene_diff/scripts/evaluate_multiview.py --pred_dir logs/eval_pred_test_processable \
    --splits test --sets varied --output_path logs/eval_test_processable.txt --visualize False 2>&1 \
    | grep -E "Metric|True Positive|False Positive|False Negative" | tee logs/c5_test_processable.txt
echo "C5_FINISHER_DONE"
