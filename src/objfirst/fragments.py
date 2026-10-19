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


# ---------------------------------------------------------------------------
# extent-relative merge, using the measured per-label distribution


def merge_by_extent(items: list[dict], extents_path) -> list[dict]:
    """Merge same-label instances while the union still looks like one object.

    Rule, written before the run and using measured numbers only: two or more
    instances of the same label merge when the diagonal of their union box is no
    larger than that label's P90 extent in data/objfirst_extents.json. Labels
    with fewer than three measured instances have no extent rule and are left to
    the touching-boxes rule already applied.
    """
    import json
    from pathlib import Path

    data = json.loads(Path(extents_path).read_text())["labels"]
    by_label: dict[str, list[dict]] = {}
    out: list[dict] = []
    for it in items:
        by_label.setdefault(it["label"], []).append(it)
    for label, group in by_label.items():
        spec = data.get(label)
        if not spec or spec["rule"] != "extent":
            out.extend(group)
            continue
        limit = float(spec["p90"])
        pending = list(group)
        while pending:
            base = pending.pop(0)
            lo, hi = np.asarray(base["bbox_min"]), np.asarray(base["bbox_max"])
            members = [base]
            for other in list(pending):
                ulo = np.minimum(lo, np.asarray(other["bbox_min"]))
                uhi = np.maximum(hi, np.asarray(other["bbox_max"]))
                if float(np.linalg.norm(uhi - ulo)) <= limit:
                    members.append(other)
                    pending.remove(other)
                    lo, hi = ulo, uhi
            if len(members) == 1:
                out.append(base)
            else:
                best = max(members, key=lambda g: g["extent"])
                out.append({**best, "centroid": (lo + hi) / 2.0,
                            "extent": float(np.linalg.norm(hi - lo)),
                            "bbox_min": lo, "bbox_max": hi,
                            "fragments": len(members)})
    return out
