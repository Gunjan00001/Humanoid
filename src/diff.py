"""Stage 4 - change detection (CPU).

Voxel-downsamples both clouds, finds bidirectional nearest-neighbour
disagreements, removes outliers, clusters them with DBSCAN and applies the
occlusion-aware visibility check: a region counts as *removed* only if the
other video actually observed that space and saw it empty.

    python src/diff.py --config configs/default.yaml

Reads  cache/reconstruction/...            (poses, depth, tagged cloud)
Writes cache/diff/clusters.json            (kept clusters with id, type, 3D
                                            centroid and box in reconstruction
                                            units, confidence, evidence frames)
       cache/diff/visibility.json          (per-candidate diagnostics)
       cache/diff/points.npz               (candidate points + cluster labels)

Where it sits in the pipeline: stage 4 of nine, after reconstruction and
describe feeds on its output; see docs/code_map.md. The occlusion-aware rule is
the deliberate core of the method: a voxel only counts as changed when BOTH
takes had a camera ray into that space, and one of them saw it empty, so a
shadow, a camera move or a previously hidden object can never be reported.

The thresholds live in configs/default.yaml (change_distance, the DBSCAN eps
and min_samples, the outlier k and std_ratio, the visibility Unlike limits)
with a comment each; nothing here is per scene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

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
from src.geometry import (  # noqa: E402
    load_reconstruction,
    project_points,
    sample_depth,
    tagged_points,
    voxel_downsample,
)

log = setup_logging("diff")


def remove_statistical_outliers(points: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """Boolean mask: keep points whose mean distance to k NN is 'normal'."""
    if len(points) <= k:
        return np.ones(len(points), dtype=bool)
    dist, _ = cKDTree(points).query(points, k=k + 1, workers=-1)
    mean_dist = dist[:, 1:].mean(axis=1)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    return mean_dist <= threshold


def candidate_points(cloud: np.ndarray, other: np.ndarray, change_distance: float) -> np.ndarray:
    """Points of `cloud` farther than `change_distance` from every `other` point."""
    dist, _ = cKDTree(other).query(cloud, k=1, workers=-1)
    return dist > change_distance


def cluster_points(points: np.ndarray, cfg: dict) -> np.ndarray:
    dcfg = cfg["diff"]
    labels = DBSCAN(eps=float(dcfg["dbscan_eps"]),
                    min_samples=int(dcfg["dbscan_min_points"])).fit_predict(points)
    return labels


def visibility_check(cluster_pts: np.ndarray, frame_idx: list[int], recon: dict, cfg: dict):
    """The stage gate: is the region really observable in both takes?

    Roughly half of all candidates die here, and they are the dangerous
    half: a surface that was hidden in video A (behind the moving object)
    would look like an addition, and a ray that only grazes the region
    would make it look removed. The check projects the cluster into both
    cameras, asks whether the depth image actually saw the space (and how
    confidently), and returns the per-frame diagnostics that
    cache/diff/visibility.json stores for every candidate - so a rejected
    candidate can be audited without a re-run.
    """
    """Fraction of cluster points that the other video saw as empty space.

    A point is "seen empty" in a frame when it projects inside the image and it
    is closer to the camera than the observed depth surface (z < depth - tol):
    the camera looked through that space. Occluded points (behind the observed
    surface) give no evidence.
    """
    dcfg = cfg["diff"]
    tol = float(dcfg["visibility_tol"])
    conf_q = float(np.percentile(recon["depth_conf"], cfg["reconstruction"]["conf_percentile"]))
    seen_empty = np.zeros(len(cluster_pts), dtype=bool)
    per_frame = []
    for i in frame_idx:
        uv, z = project_points(cluster_pts, recon["extrinsic"][i], recon["intrinsic"][i])
        observed, in_bounds = sample_depth(recon["depth"][i], uv)
        conf_ok = sample_depth(recon["depth_conf"][i], uv)[0] >= conf_q
        valid = in_bounds & conf_ok & (z > 1e-6)
        empty = valid & (z < observed - tol)
        seen_empty |= empty
        per_frame.append({"frame": int(i), "file": recon["files"][i],
                          "n_valid": int(valid.sum()), "n_empty": int(empty.sum())})
    fraction = float(seen_empty.mean()) if len(cluster_pts) else 0.0
    return fraction, per_frame


def cluster_entry(cid: int, pts: np.ndarray, tag: str, vis: dict | None) -> dict:
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    sample_idx = np.linspace(0, len(pts) - 1, num=min(200, len(pts))).round().astype(int)
    return {
        "id": cid,
        "type": "removed" if tag == "a" else "added",
        "source_tag": tag,
        "n_points": int(len(pts)),
        "centroid": [round(float(v), 4) for v in pts.mean(axis=0)],
        "bbox_min": [round(float(v), 4) for v in bbox_min],
        "bbox_max": [round(float(v), 4) for v in bbox_max],
        "extent": [round(float(v), 4) for v in (bbox_max - bbox_min)],
        "visibility": vis,
        "sample_points": [[round(float(v), 4) for v in p] for p in pts[sample_idx]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--method", default=None,
                        help="override diff.method (geometric | tabletop)")
    parser.add_argument("--objectness_margin", type=float, default=None,
                        help="override diff.tabletop.objectness_margin (per-scene "
                             "tuning: a larger A/B lighting shift needs a larger "
                             "margin; see docs/results.md)")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    if args.method:
        cfg["diff"]["method"] = args.method
    if args.objectness_margin is not None:
        cfg["diff"]["tabletop"]["objectness_margin"] = args.objectness_margin

    diff_dir = cache_dir(cfg, "diff")
    clusters_path = diff_dir / "clusters.json"
    if clusters_path.exists() and not args.force:
        log.info("diff already cached (%s); use --force to redo", clusters_path)
        return 0

    dcfg = cfg["diff"]
    recon = load_reconstruction(cfg)

    method = dcfg.get("method", "geometric")
    if method == "tabletop":
        from src.tabletop import detect
        log.info("diff method: tabletop (appearance-based, see src/tabletop.py)")
        all_clusters = detect(recon, cfg)
        clusters = [c for c in all_clusters if c.get("verified", True)]
        uncertain = [c for c in all_clusters if not c.get("verified", True)]
        write_json(clusters_path, {"created": stamp(), "config_hash": cfg["_config_hash"],
                                   "method": "tabletop", "clusters": clusters})
        write_json(diff_dir / "uncertain.json",
                   {"created": stamp(), "config_hash": cfg["_config_hash"],
                    "method": "tabletop",
                    "note": "object-like regions in one video where the other video "
                            "has no coverage; NOT verified changes",
                    "clusters": uncertain})
        write_json(diff_dir / "visibility.json",
                   {"created": stamp(), "config_hash": cfg["_config_hash"],
                    "method": "tabletop",
                    "note": "photometric mode: coverage of both videos is checked "
                            "per cell instead of the geometric empty-space test"})
        # points.npz: cluster sample points per video (visualize.py format)
        a_pts, a_lab, b_pts, b_lab = [], [], [], []
        for c in all_clusters:
            if c["source_tag"] in ("a", "both"):
                p = np.asarray(c["sample_points"], dtype=np.float32)
                a_pts.append(p)
                a_lab.append(np.full(len(p), c["id"], dtype=np.int32))
            if c["source_tag"] in ("b", "both"):
                p = np.asarray(c.get("sample_points_other", c["sample_points"]),
                               dtype=np.float32)
                b_pts.append(p)

                b_lab.append(np.full(len(p), c["id"], dtype=np.int32))
        np.savez_compressed(
            diff_dir / "points.npz",
            a_points=np.concatenate(a_pts) if a_pts else np.zeros((0, 3), np.float32),
            a_labels=np.concatenate(a_lab) if a_lab else np.zeros(0, np.int32),
            b_points=np.concatenate(b_pts) if b_pts else np.zeros((0, 3), np.float32),
            b_labels=np.concatenate(b_lab) if b_lab else np.zeros(0, np.int32),
        )
        log.info("wrote %s (%d verified), uncertain.json (%d unverified), points.npz",
                 clusters_path, len(clusters), len(uncertain))
        return 0
    if method != "geometric":
        raise ValueError(f"unknown diff.method: {method!r} (geometric | tabletop)")

    pts, rgb, tags, _ = tagged_points(recon, cfg["reconstruction"]["conf_percentile"])
    log.info("reconstruction: %d points (%.1f%% A)", len(pts), 100.0 * (tags == "a").mean())

    voxel = float(dcfg["voxel_size"])
    a_pts, = voxel_downsample(pts[tags == "a"], voxel=voxel)
    b_pts, = voxel_downsample(pts[tags == "b"], voxel=voxel)
    log.info("downsampled: A=%d B=%d (voxel %.3f)", len(a_pts), len(b_pts), voxel)

    cand_a_mask = candidate_points(a_pts, b_pts, float(dcfg["change_distance"]))
    cand_b_mask = candidate_points(b_pts, a_pts, float(dcfg["change_distance"]))
    cand_a, cand_b = a_pts[cand_a_mask], b_pts[cand_b_mask]
    log.info("candidates: A=%d B=%d (change_distance %.3f)",
             len(cand_a), len(cand_b), dcfg["change_distance"])

    keep_a = remove_statistical_outliers(cand_a, int(dcfg["outlier_nb_neighbors"]),
                                         float(dcfg["outlier_std_ratio"]))
    keep_b = remove_statistical_outliers(cand_b, int(dcfg["outlier_nb_neighbors"]),
                                         float(dcfg["outlier_std_ratio"]))
    cand_a, cand_b = cand_a[keep_a], cand_b[keep_b]
    log.info("after outlier removal: A=%d B=%d", len(cand_a), len(cand_b))

    labels_a = cluster_points(cand_a, cfg) if len(cand_a) else np.array([], dtype=int)
    labels_b = cluster_points(cand_b, cfg) if len(cand_b) else np.array([], dtype=int)

    min_pts = int(dcfg["min_cluster_points"])
    a_frames = [i for i, t in enumerate(recon["tags"]) if t == "a"]
    b_frames = [i for i, t in enumerate(recon["tags"]) if t == "b"]

    clusters, visibility = [], {}
    label_arrays = {"a": np.full(len(cand_a), -1, dtype=np.int32),
                    "b": np.full(len(cand_b), -1, dtype=np.int32)}
    next_id = 1
    # A clusters -> "removed" (must be confirmed empty in B); B clusters -> "added"
    for source, cloud, labels, other_frames in (
        ("a", cand_a, labels_a, b_frames),
        ("b", cand_b, labels_b, a_frames),
    ):
        for lab in sorted(set(labels.tolist())):
            if lab == -1:
                continue
            cluster_pts = cloud[labels == lab]
            if len(cluster_pts) < min_pts:
                visibility[f"cand_{source}_{lab}"] = {
                    "source_tag": source, "n_points": int(len(cluster_pts)),
                    "kept": False, "reason": f"smaller than min_cluster_points={min_pts}",
                }
                continue
            if dcfg["visibility_check"]:
                fraction, per_frame = visibility_check(cluster_pts, other_frames, recon, cfg)
                kept = fraction >= float(dcfg["visibility_min_fraction"])
                best = sorted(per_frame, key=lambda f: -f["n_empty"])[:3]
                vis = {
                    "checked": True,
                    "observed_empty_fraction": round(fraction, 4),
                    "min_fraction": float(dcfg["visibility_min_fraction"]),
                    "kept": kept,
                    "reason": ("space observed empty by the other video"
                               if kept else "space was never clearly observed empty (occluded or unseen)"),
                    "evidence": best,
                }
            else:
                kept, vis = True, {"checked": False, "kept": True, "reason": "visibility check disabled"}
            visibility[f"cand_{source}_{lab}"] = {
                "source_tag": source, "n_points": int(len(cluster_pts)),
                "centroid": [round(float(v), 4) for v in cluster_pts.mean(axis=0)],
                **vis,
            }
            if kept:
                clusters.append(cluster_entry(next_id, cluster_pts, source, vis))
                label_arrays[source][labels == lab] = next_id
                next_id += 1

    log.info("clusters kept: %d (%d removed, %d added)", len(clusters),
             sum(c["type"] == "removed" for c in clusters),
             sum(c["type"] == "added" for c in clusters))

    write_json(clusters_path, {"created": stamp(), "config_hash": cfg["_config_hash"],
                               "clusters": clusters})
    write_json(diff_dir / "visibility.json",
               {"created": stamp(), "config_hash": cfg["_config_hash"],
                "candidates": visibility})
    # candidate clouds + labels for visualization (kept clusters keep their id)
    np.savez_compressed(diff_dir / "points.npz",
                        a_points=cand_a.astype(np.float32), a_labels=label_arrays["a"],
                        b_points=cand_b.astype(np.float32), b_labels=label_arrays["b"])
    log.info("wrote %s, visibility.json, points.npz", clusters_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
