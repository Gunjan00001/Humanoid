"""Merge fragments of the same object before matching.

The first smoke showed one physical object arriving as several instances (four
scissors, two apples), because the detector's clusters split on geometric gaps.
Matching fragments separately would turn one removal into several candidates and
distort both recall and the extra count.

Written rule (global, no per-scene values): two instances with the same canonical
label merge when the distance between their box centres is smaller than half the
sum of their extents, i.e. when their 3D boxes touch or overlap. The merged
instance keeps the union box and the largest crop.
"""
from __future__ import annotations

import numpy as np


def merge_fragments(items: list[dict]) -> list[dict]:
    """Merge same-label instances whose 3D boxes touch or overlap."""
    merged: list[dict] = []
    used = [False] * len(items)
    for i, a in enumerate(items):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        changed = True
        while changed:                       # transitive merge
            changed = False
            for j, b in enumerate(items):
                if used[j] or b["label"] != a["label"]:
                    continue
                for g in group:
                    dist = float(np.linalg.norm(g["centroid"] - b["centroid"]))
                    if dist <= 0.5 * (g["extent"] + b["extent"]):
                        group.append(b)
                        used[j] = True
                        changed = True
                        break
        if len(group) == 1:
            merged.append(a)
            continue
        lo = np.min([g["bbox_min"] for g in group], axis=0)
        hi = np.max([g["bbox_max"] for g in group], axis=0)
        best = max(group, key=lambda g: g["extent"])
        merged.append({**best,
                       "centroid": (lo + hi) / 2.0,
                       "extent": float(np.linalg.norm(hi - lo)),
                       "bbox_min": lo, "bbox_max": hi,
                       "fragments": len(group)})
    return merged
