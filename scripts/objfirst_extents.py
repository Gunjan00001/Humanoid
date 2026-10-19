#!/usr/bin/env python3
"""Measure the per-label 3D extent distribution, so fragment merging has a
written threshold instead of a guessed constant.

For every scene it joins the diff clusters (id, 3D box) with the cached VLM
labels, canonicalises the labels through the closed vocabulary, and records the
extent (box diagonal, reconstruction units) per label. The merge rule that uses
it: two same-label instances merge when their union diagonal is no larger than
the label's P90 extent, i.e. their union still looks like one object of that
class. Labels with fewer than three measured instances get no extent rule and
fall back to the touching-boxes rule only.

  python scripts/objfirst_extents.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.naming.vocabulary import load_vocabulary  # noqa: E402

SCENES = ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]


def main() -> int:
    vocab = load_vocabulary()
    per_label: dict[str, list[float]] = defaultdict(list)
    for scene in SCENES:
        base = ROOT / "cache" / scene
        cpath, vpath = base / "diff" / "clusters.json", base / "semantics" / "vlm.json"
        if not cpath.exists() or not vpath.exists():
            continue
        clusters = json.loads(cpath.read_text())
        clusters = clusters.get("clusters", clusters if isinstance(clusters, list) else [])
        vlm = json.loads(vpath.read_text())
        for c in clusters:
            if "bbox_min" not in c or "bbox_max" not in c:
                continue
            label = (vlm.get(str(c["id"]), {}) or {}).get("label") or ""
            canon = vocab.canonical_of(label)
            if not canon:
                continue
            lo, hi = np.asarray(c["bbox_min"], float), np.asarray(c["bbox_max"], float)
            per_label[canon].append(float(np.linalg.norm(hi - lo)))
    out = {}
    for label, vals in sorted(per_label.items()):
        v = np.asarray(vals)
        out[label] = {"n": int(len(v)), "median": round(float(np.median(v)), 4),
                      "p90": round(float(np.percentile(v, 90)), 4),
                      "max": round(float(v.max()), 4),
                      "rule": "extent" if len(v) >= 3 else "touch-only"}
    dest = ROOT / "data" / "objfirst_extents.json"
    dest.write_text(json.dumps({"unit": "reconstruction units (box diagonal)",
                                "rule": "merge same-label instances when the union "
                                        "diagonal <= P90 of the label; labels with n<3 "
                                        "fall back to touching boxes",
                                "labels": out}, indent=2))
    print(f"labels measured: {len(out)}")
    for label, d in list(out.items())[:10]:
        print(f"  {label:16} n={d['n']:<3} median {d['median']:<8} p90 {d['p90']:<8} {d['rule']}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
