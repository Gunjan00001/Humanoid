#!/usr/bin/env python3
"""Compile the reproducibility campaign: per-run change counts + types from
outputs/repro_*/report.json (reproducibility runs only, never the committed
outputs). Usage: python scripts/repro_summary.py"""
import glob
import json
import os
from collections import defaultdict

rows = defaultdict(list)
for d in sorted(glob.glob("outputs/repro_*")):
    rp = os.path.join(d, "report.json")
    if not os.path.exists(rp):
        rows[os.path.basename(d)].append(("NO_REPORT", []))
        continue
    rep = json.load(open(rp))
    ch = rep.get("changes", [])
    types = [f"{c['type']}:{c.get('label')}" for c in ch]
    rows[os.path.basename(d)].append((len(ch), types))

print(f"{'run':<26} {'#changes':>8}  types")
for name in sorted(rows):
    for n, types in rows[name]:
        print(f"{name:<26} {str(n):>8}  {types}")
