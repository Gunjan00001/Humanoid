"""Shared array/geometry helpers for the CPU stages.

Keeps the expensive-artifact formats in one place: everything here reads the
cache written by src/reconstruct.py (or src/align.py) and never touches a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.common import cache_dir, read_json, repo_path

MODEL_TARGET = 518  # VGGT native width (vggt.utils.load_fn)


def recon_dir_for(cfg: dict[str, Any], prefer_aligned: bool = True) -> Path:
    """Aligned reconstruction if enabled and present, else the raw one."""
    base = repo_path(cfg["paths"]["cache_dir"]) / "reconstruction"
    if prefer_aligned and cfg["reconstruction"].get("use_icp_refinement"):
        aligned = base / "aligned"
        if (aligned / "meta.json").exists():
            return aligned
    return base


def load_reconstruction(cfg: dict[str, Any], prefer_aligned: bool = True) -> dict[str, Any]:
    """Load a whole reconstruction into memory (`.npy` semantics)."""
    rdir = recon_dir_for(cfg, prefer_aligned)
    meta = read_json(rdir / "meta.json")
    cameras = np.load(rdir / "cameras.npz")
    dense = np.load(rdir / "dense.npz")
    colors = np.load(rdir / "colors.npz")["images"]
    return {
        "dir": rdir,
        "meta": meta,
        "extrinsic": cameras["extrinsic"].astype(np.float32),   # [S,3,4] camera-from-world
        "intrinsic": cameras["intrinsic"].astype(np.float32),   # [S,3,3] model space
        "depth": dense["depth"].astype(np.float32),             # [S,H,W]
        "depth_conf": dense["depth_conf"].astype(np.float32),   # [S,H,W]
        "world_points": dense["world_points"].astype(np.float32),  # [S,H,W,3]
        "colors": colors,                                       # [S,H,W,3] uint8
        "tags": np.array([f["tag"] for f in meta["frames"]]),   # per frame
        "files": [f["file"] for f in meta["frames"]],
        "image_hw": tuple(meta["image_hw"]),
    }


def tagged_points(recon: dict[str, Any], conf_percentile: float):
    """Flatten the cloud with the confidence filter applied.

    Returns (points [N,3], rgb [N,3] uint8, tag [N] ('a'/'b'), conf [N]).
    """
    q = float(np.percentile(recon["depth_conf"], conf_percentile))
    pts, rgb, tags, conf = [], [], [], []
    for i, t in enumerate(recon["tags"]):
        c = recon["depth_conf"][i].reshape(-1)
        m = c >= q
        pts.append(recon["world_points"][i].reshape(-1, 3)[m])
        rgb.append(recon["colors"][i].reshape(-1, 3)[m])
        conf.append(c[m])
        tags.append(np.full(int(m.sum()), t))
    return (np.concatenate(pts), np.concatenate(rgb).astype(np.uint8),
            np.concatenate(tags), np.concatenate(conf))


def voxel_downsample(points: np.ndarray, *arrays: np.ndarray, voxel: float):
    """Keep one point per voxel (the first in lexicographic order)."""
    if len(points) == 0:
        return (points, *arrays)
    key = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return (points[idx], *[a[idx] for a in arrays])


# ---------------------------------------------------------------------------
# camera math (all in model space = the resolution VGGT processed)


def project_points(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    """World -> pixel. Returns uv [N,2] float, z [N] (camera-frame depth)."""
    cam = points @ extrinsic[:, :3].T + extrinsic[:, 3]
    z = cam[:, 2]
    uv = (cam @ intrinsic.T)[:, :2] / np.where(np.abs(z) < 1e-8, 1e-8, z)[:, None]
    return uv, z


def sample_depth(depth_map: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour depth lookup. Returns (depth, in_bounds)."""
    h, w = depth_map.shape
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.full(len(uv), np.nan, dtype=np.float32)
    out[ok] = depth_map[v[ok], u[ok]]
    return out, ok


def model_to_original_transform(orig_w: int, orig_h: int, target: int = MODEL_TARGET) -> dict:
    """Replicates vggt.utils.load_fn.load_and_preprocess_images(mode='crop'):
    resize so width == target, height rounded to a multiple of 14, center-crop
    height to <= target. Used to map model-space crops back to the JPEG."""
    scale_w = target / orig_w
    new_h = round(orig_h * scale_w / 14) * 14
    scale_h = new_h / orig_h
    crop_top = max(0, (new_h - target) // 2)
    return {
        "orig_w": int(orig_w), "orig_h": int(orig_h),
        "scale_w": round(scale_w, 6), "scale_h": round(scale_h, 6),
        "crop_top": int(crop_top),
        "model_w": int(target), "model_h": int(min(new_h, target)),
    }


def model_bbox_to_original(bbox_xyxy: tuple[float, float, float, float],
                           tr: dict[str, Any]) -> tuple[int, int, int, int]:
    """Model-space pixel bbox -> original-frame pixel bbox (clamped)."""
    x1, y1, x2, y2 = bbox_xyxy
    ox1 = x1 / tr["scale_w"]
    ox2 = x2 / tr["scale_w"]
    oy1 = (y1 + tr["crop_top"]) / tr["scale_h"]
    oy2 = (y2 + tr["crop_top"]) / tr["scale_h"]
    ox1 = int(max(0, min(ox1, tr["orig_w"] - 1)))
    ox2 = int(max(0, min(ox2, tr["orig_w"])))
    oy1 = int(max(0, min(oy1, tr["orig_h"] - 1)))
    oy2 = int(max(0, min(oy2, tr["orig_h"])))
    return ox1, oy1, ox2, oy2


# ---------------------------------------------------------------------------
# derived change list (shared by visualize.py / report.py)


def build_change_list(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge clusters + DINOv2 matches + VLM crops into the final change list.

    A removed cluster and an added cluster matched by src/describe.py become a
    single "moved" entry (old -> new position). Confidence is a documented
    heuristic (see outputs/report.md), not a probability:

        cluster_size_conf = min(1, n_points / 300)
        visibility_conf   = observed_empty_fraction (0.5 when unchecked)
        confidence        = (size + visibility) / 2            [added/removed]
                            (size + visibility + similarity) / 3  [moved]
    """
    cache = repo_path(cfg["paths"]["cache_dir"])
    clusters = read_json(cache / "diff" / "clusters.json")["clusters"]
    by_id = {c["id"]: c for c in clusters}

    matches_path = cache / "semantics" / "matches.json"
    matches = read_json(matches_path) if matches_path.exists() else []
    crops_path = cache / "semantics" / "crops.json"
    crops = read_json(crops_path) if crops_path.exists() else {}

    matched_removed = {m["removed_id"]: m for m in matches}
    matched_added = {m["added_id"]: m for m in matches}

    def size_conf(c: dict) -> float:
        return min(1.0, c["n_points"] / 300.0)

    def vis_conf(c: dict) -> float:
        v = c.get("visibility") or {}
        if v.get("checked", False):
            return float(v.get("observed_empty_fraction", 0.0))
        return 0.5

    def crop_info(c: dict) -> dict:
        return crops.get(str(c["id"]), {})

    changes: list[dict[str, Any]] = []
    for c in clusters:
        if c["type"] == "removed":
            m = matched_removed.get(c["id"])
            if m and m["added_id"] in by_id:
                new = by_id[m["added_id"]]
                conf = round((size_conf(c) + vis_conf(c) + float(m.get("similarity", 0.0))) / 3, 2)
                changes.append({
                    "id": len(changes) + 1,
                    "type": "moved",
                    "label": m.get("label") or crop_info(c).get("label"),
                    "description": m.get("sentence"),
                    "centroid": c["centroid"],
                    "bbox_min": c["bbox_min"],
                    "bbox_max": c["bbox_max"],
                    "from": c["centroid"],
                    "to": new["centroid"],
                    "confidence": conf,
                    "cluster_ids": [c["id"], new["id"]],
                    "evidence": {
                        "visible_in": [x for x in (crop_info(c).get("frame"),
                                                   crop_info(new).get("frame")) if x],
                        "confirmed_empty_in": (c.get("visibility") or {}).get("evidence", []),
                    },
                })
            else:
                conf = round((size_conf(c) + vis_conf(c)) / 2, 2)
                changes.append({
                    "id": len(changes) + 1,
                    "type": "removed",
                    "label": crop_info(c).get("label"),
                    "description": crop_info(c).get("description"),
                    "centroid": c["centroid"],
                    "bbox_min": c["bbox_min"],
                    "bbox_max": c["bbox_max"],
                    "from": c["centroid"],
                    "to": None,
                    "confidence": conf,
                    "cluster_ids": [c["id"]],
                    "evidence": {
                        "visible_in": [x for x in [crop_info(c).get("frame")] if x],
                        "confirmed_empty_in": (c.get("visibility") or {}).get("evidence", []),
                    },
                })
        else:  # added, unless already consumed by a moved match
            if c["id"] in matched_added:
                continue
            conf = round((size_conf(c) + vis_conf(c)) / 2, 2)
            changes.append({
                "id": len(changes) + 1,
                "type": "added",
                "label": crop_info(c).get("label"),
                "description": crop_info(c).get("description"),
                "centroid": c["centroid"],
                "bbox_min": c["bbox_min"],
                "bbox_max": c["bbox_max"],
                "from": None,
                "to": c["centroid"],
                "confidence": conf,
                "cluster_ids": [c["id"]],
                "evidence": {
                    "visible_in": [x for x in [crop_info(c).get("frame")] if x],
                    "confirmed_empty_in": (c.get("visibility") or {}).get("evidence", []),
                },
            })
    return changes

