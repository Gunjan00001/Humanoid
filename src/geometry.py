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
    """Merge clusters + VLM verification + DINOv2 matches into final changes.

    Sources (all optional except the diff output):
      cache/diff/clusters.json    verified candidates (photometric or geometric)
      cache/diff/uncertain.json   one-video-only candidates (photometric mode)
      cache/semantics/confirmed.json  VLM-confirmed subset (after describe.py)
      cache/semantics/vlm.json    labels / sentences per cluster
      cache/semantics/matches.json    DINOv2 removed<->added matches ("moved")

    Confidence is a documented heuristic, not a probability:

        size_conf  = min(1, n_cells / 150)   (falls back to n_points / 300)
        score_conf = min(1, mean_change_score / 150)   (photometric candidates)
        verified   = (size_conf + score_conf [+ similarity]) / 2 or / 3
        unverified = 0.3 * size_conf
    """
    cache = repo_path(cfg["paths"]["cache_dir"])
    sem = cache / "semantics"

    confirmed_path = sem / "confirmed.json"
    if confirmed_path.exists():
        confirmed_doc = read_json(confirmed_path)
        clusters = confirmed_doc["clusters"]
        uncertain = confirmed_doc.get("uncertain", [])
    else:
        clusters = read_json(cache / "diff" / "clusters.json")["clusters"]
        uncertain_path = cache / "diff" / "uncertain.json"
        uncertain = read_json(uncertain_path)["clusters"] if uncertain_path.exists() else []

    matches = read_json(sem / "matches.json") if (sem / "matches.json").exists() else []
    vlm = read_json(sem / "vlm.json") if (sem / "vlm.json").exists() else {}
    crops = read_json(sem / "crops.json") if (sem / "crops.json").exists() else {}

    matched_removed = {m["removed_id"]: m for m in matches}
    matched_added = {m["added_id"]: m for m in matches}

    def info(cid: int) -> dict:
        return vlm.get(str(cid), {})

    def crop(cid: int) -> dict:
        return crops.get(str(cid), {})

    def size_conf(c: dict) -> float:
        if c.get("n_cells") is not None:
            return min(1.0, c["n_cells"] / 150.0)
        return min(1.0, c["n_points"] / 300.0)

    def score_conf(c: dict) -> float:
        return min(1.0, float(c.get("mean_change_score") or 0.0) / 150.0)

    def evidence(c: dict, extra: list[dict] | None = None) -> dict:
        ci = crop(c["id"])
        visible = [x for x in (ci.get("frame"), ci.get("frame_other")) if x]
        return {
            "visible_in": visible,
            "confirmed_empty_in": (c.get("visibility") or {}).get("evidence", []) if extra is None else extra,
        }

    changes: list[dict[str, Any]] = []

    def add(c: dict, ctype: str, verified: bool, label, description, conf: float,
            from_, to_, cluster_ids: list[int], ev: dict) -> None:
        changes.append({
            "id": len(changes) + 1,
            "type": ctype,
            "verified": bool(verified),
            "label": label,
            "description": description,
            "centroid": c["centroid"],
            "bbox_min": c["bbox_min"],
            "bbox_max": c["bbox_max"],
            "from": from_,
            "to": to_,
            "confidence": round(float(conf), 2),
            "cluster_ids": cluster_ids,
            "evidence": ev,
        })

    for c in clusters:
        cid = c["id"]
        if c["type"] == "removed":
            m = matched_removed.get(cid)
            if m:
                new = next((x for x in clusters if x["id"] == m["added_id"]), None)
                sim = float(m.get("similarity", 0.0))
                conf = (size_conf(c) + score_conf(c) + sim) / 3 if c.get("n_cells") else \
                    (size_conf(c) + 0.5 + sim) / 3
                add(c, "moved", True,
                    m.get("label") or info(cid).get("label"),
                    m.get("sentence") or info(cid).get("description"),
                    conf, c["centroid"], (new or c)["centroid"],
                    [cid, m["added_id"]], evidence(c))
            else:
                conf = (size_conf(c) + score_conf(c)) / 2 if c.get("n_cells") else \
                    (size_conf(c) + 0.5) / 2
                add(c, "removed", True, info(cid).get("label"),
                    info(cid).get("description"), conf,
                    c["centroid"], None, [cid], evidence(c))
        elif c["type"] == "added":
            if cid in matched_added:
                continue                      # already reported as moved
            conf = (size_conf(c) + score_conf(c)) / 2 if c.get("n_cells") else \
                (size_conf(c) + 0.5) / 2
            add(c, "added", True, info(cid).get("label"),
                info(cid).get("description"), conf,
                None, c["centroid"], [cid], evidence(c))
        elif c["type"] == "replaced":
            conf = (size_conf(c) + score_conf(c)) / 2
            other = c.get("sample_points_other")
            to_ = (np.asarray(other, dtype=float).mean(axis=0).round(4).tolist()
                   if other else c["centroid"])
            add(c, "replaced", True, info(cid).get("label"),
                info(cid).get("description"), conf,
                c["centroid"], to_, [cid], evidence(c))

    for c in uncertain:
        cid = c["id"]
        conf = 0.3 * size_conf(c)
        add(c, c["type"], False, info(cid).get("label"),
            info(cid).get("description"), conf,
            c["centroid"] if c["type"] == "removed" else None,
            c["centroid"] if c["type"] == "added" else None,
            [cid], evidence(c))
    return changes

