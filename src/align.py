"""Stage 3 - optional ICP refinement (CPU).

VGGT already places both videos in one shared frame; ICP is only a refinement
for cases where A and B drift slightly apart. Only *static* structure (points
that already have a close neighbour in the other cloud) guides the transform.

    python src/align.py --config configs/default.yaml

Reads  cache/reconstruction/{cameras,dense,colors,meta}.npz|json
Writes cache/reconstruction/aligned/  (transformed B cloud + poses + transform.json)

Downstream stages automatically prefer the aligned directory when
reconstruction.use_icp_refinement is true (see src/geometry.recon_dir_for).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    add_common_args,
    apply_overrides,
    cache_dir,
    load_config,
    setup_logging,
    stamp,
    write_json,
)
from src.geometry import load_reconstruction, tagged_points, voxel_downsample  # noqa: E402

log = setup_logging("align")


def static_subset(pts: np.ndarray, other: np.ndarray, threshold: float) -> np.ndarray:
    dist, _ = cKDTree(other).query(pts, k=1, workers=-1)
    return dist < threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    base_dir = cache_dir(cfg, "reconstruction")
    recon_dir = base_dir / "aligned"
    transform_path = recon_dir / "transform.json"
    if transform_path.exists() and not args.force:
        log.info("alignment already cached (%s); use --force to redo", transform_path)
        return 0

    rcfg = cfg["reconstruction"]
    voxel = float(cfg["diff"]["voxel_size"])
    max_corr = float(rcfg["icp_max_correspondence"])

    recon = load_reconstruction(cfg, prefer_aligned=False)
    pts, _, tag, _ = tagged_points(recon, cfg["reconstruction"]["conf_percentile"])
    log.info("cloud: %d points (%.1f%% A)", len(pts), 100.0 * (tag == "a").mean())

    a_pts, = voxel_downsample(pts[tag == "a"], voxel=voxel)
    b_pts, = voxel_downsample(pts[tag == "b"], voxel=voxel)
    log.info("downsampled: A=%d B=%d (voxel %.3f)", len(a_pts), len(b_pts), voxel)

    a_static = static_subset(a_pts, b_pts, max_corr)
    b_static = static_subset(b_pts, a_pts, max_corr)
    log.info("static subsets: A=%d (%.0f%%) B=%d (%.0f%%)",
             a_static.sum(), 100 * a_static.mean(), b_static.sum(), 100 * b_static.mean())

    import open3d as o3d

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b_pts[b_static].astype(np.float64)))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a_pts[a_static].astype(np.float64)))
    before = float(np.mean(cKDTree(a_pts).query(b_pts[b_static], workers=-1)[0]))
    result = o3d.pipelines.registration.registration_icp(
        src, dst, max_corr, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(rcfg["icp_iterations"])),
    )
    moved = np.asarray(result.transformation @ np.c_[b_pts[b_static], np.ones(b_static.sum())].T).T[:, :3]
    after = float(np.mean(cKDTree(a_pts).query(moved, workers=-1)[0]))
    log.info("ICP fitness=%.3f rmse=%.4f | static mean NN before=%.4f after=%.4f",
             result.fitness, result.inlier_rmse, before, after)

    transform = np.asarray(result.transformation, dtype=np.float32)
    if after > before or not np.isfinite(after):  # refinement made things worse: keep identity
        log.warning("ICP did not improve alignment; keeping identity transform")
        transform = np.eye(4, dtype=np.float32)

    # apply to B: world points and camera extrinsics (camera-frame geometry is
    # unchanged, so depth stays valid)
    points = recon["world_points"].copy()
    extr = recon["extrinsic"].astype(np.float64)
    t4_inv = np.linalg.inv(transform.astype(np.float64))
    for i, tag_i in enumerate(recon["tags"]):
        if tag_i != "b":
            continue
        p = points[i].reshape(-1, 3).astype(np.float64)
        points[i] = (transform.astype(np.float64) @ np.c_[p, np.ones(len(p))].T).T[:, :3] \
            .astype(np.float16).reshape(points[i].shape)
        e = np.eye(4)
        e[:3, :4] = extr[i]
        extr[i] = (e @ t4_inv)[:3, :4]

    recon_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(recon_dir / "cameras.npz",
                        extrinsic=extr.astype(np.float32),
                        intrinsic=recon["intrinsic"].astype(np.float32))
    np.savez(recon_dir / "dense.npz",
             depth=recon["depth"].astype(np.float16),
             depth_conf=recon["depth_conf"].astype(np.float16),
             world_points=points)
    np.savez(recon_dir / "colors.npz", images=recon["colors"])
    aligned_meta = dict(recon["meta"])
    aligned_meta["alignment"] = {
        "created": stamp(),
        "config_hash": cfg["_config_hash"],
        "type": "icp_point_to_point",
        "static_threshold": max_corr,
        "voxel": voxel,
        "fitness": round(float(result.fitness), 4),
        "static_mean_nn_before": round(before, 5),
        "static_mean_nn_after": round(after, 5),
        "transform": transform.tolist(),
    }
    write_json(recon_dir / "meta.json", aligned_meta)
    write_json(transform_path, aligned_meta["alignment"])
    log.info("wrote aligned reconstruction to %s", recon_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
