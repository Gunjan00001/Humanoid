"""M3: lift each instance into the shared 3D frame.

The diff stage already measured, for every cluster, its 3D bounding box in the
shared reconstruction frame. Lifting an inventory instance therefore means
joining its cluster id to that box: the centroid is the box centre and the
extent its diagonal, both in reconstruction units. No new geometry is
recomputed, so the object-first path cannot disagree with the detector about
where an object is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_cluster_boxes(scene_dir: Path) -> dict[int, dict]:
    """cluster id -> {centroid, extent, bbox_min, bbox_max} from the diff cache."""
    path = scene_dir / "diff" / "clusters.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    clusters = data.get("clusters", data if isinstance(data, list) else [])
    boxes = {}
    for c in clusters:
        if "bbox_min" not in c or "bbox_max" not in c:
            continue
        lo = np.asarray(c["bbox_min"], dtype=float)
        hi = np.asarray(c["bbox_max"], dtype=float)
        boxes[int(c["id"])] = {"centroid": (lo + hi) / 2.0,
                               "extent": float(np.linalg.norm(hi - lo)),
                               "bbox_min": lo, "bbox_max": hi}
    return boxes


def lift(inventory: dict[str, list[dict]], scene_dir: Path) -> dict[str, list[dict]]:
    """Attach centroid/extent to every instance that has a cluster box."""
    boxes = load_cluster_boxes(scene_dir)
    out: dict[str, list[dict]] = {"a": [], "b": []}
    for side, items in inventory.items():
        for it in items:
            box = boxes.get(it["cluster_id"])
            if not box:
                continue
            out[side].append({**it, **box})
    return out
