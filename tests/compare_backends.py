#!/usr/bin/env python
"""Backend comparison metrics: VGGT vs MapAnything (CPU, read-only).

Computes, for each reconstruction cache:
  - provenance (backend, model, frames, inference time, peak GPU memory),
  - point counts (confidence-filtered raw and 6mm-voxelised),
  - static A<->B alignment before/after ICP (mean nearest-neighbour distance),
  - thin-object visibility: how far the comb / pen regions rise above the
    locally fitted table plane (VGGT absorbs 1cm objects; a backend that
    resolves them shows a real height difference),
  - metric-scale plausibility: the comb's principal-axis length (a typical
    hair comb is ~20cm; this is a plausibility check, not a calibrated
    measurement - labelled as such in docs/results.md).

    python tests/compare_backends.py \
        --cache-vggt cache --cache-ma cache/ma_run \
        [--out logs/backend_comparison.json]

Never invents numbers: everything is measured from the cached artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, read_json, repo_path, write_json  # noqa: E402
from src.geometry import load_reconstruction, tagged_points, voxel_downsample  # noqa: E402
from src.tabletop import fit_dominant_plane, plane_height  # noqa: E402

# regions measured on the demo scene (world coordinates from the VGGT run's
# tabletop clusters - see cache/diff/clusters.json; used for BOTH backends)
COMB_BBOX = (-0.205, -0.025, 0.045, 0.14)     # x0, x1, y0, y1
PEN_BBOX = (-0.13, -0.095, 0.175, 0.205)


def region_height_above_table(pts: np.ndarray, bbox: tuple,
                              cell: float = 0.006, ring_mm: float = 60.0) -> dict:
    """Height of the object region above its LOCAL table surface.

    Fits the plane on a ring around the region (bbox expanded by 20mm..80mm)
    so surface *warp* does not bias the measurement - important because
    MapAnything's table carries a ~+/-2cm warp on this scene, which a global
    plane would misattribute to the object.
    """
    x0, x1, y0, y1 = bbox
    pad_in, pad_out = 0.02, ring_mm / 1000.0
    inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    ring = ((pts[:, 0] >= x0 - pad_out) & (pts[:, 0] <= x1 + pad_out)
            & (pts[:, 1] >= y0 - pad_out) & (pts[:, 1] <= y1 + pad_out)
            & ~((pts[:, 0] >= x0 - pad_in) & (pts[:, 0] <= x1 + pad_in)
                & (pts[:, 1] >= y0 - pad_in) & (pts[:, 1] <= y1 + pad_in)))
    q, r = pts[inside], pts[ring]
    if len(q) < 50 or len(r) < 500:
        return {"n": int(len(q)), "ring_n": int(len(r)), "median_mm": None, "p90_mm": None}
    coef = fit_dominant_plane(r)                    # local table plane
    h = plane_height(q, coef)
    return {"n": int(len(q)), "ring_n": int(len(r)),
            "median_mm": round(float(np.median(h)) * 1000, 1),
            "p90_mm": round(float(np.percentile(h, 90)) * 1000, 1)}


def cluster_height_above_table(pts: np.ndarray, cluster: dict,
                               ring_in: float = 0.04, ring_out: float = 0.10) -> dict:
    """Height of a detected cluster's points above its LOCAL table surface.

    Uses the backend's OWN cluster location (frames differ between backends, so
    fixed world-space boxes are not comparable). The table plane is fitted on a
    ring around the cluster centroid, excluding the cluster itself, so surface
    warp cannot masquerade as object height.

    Interpretation: an object that is *visible in the depth maps* rises its
    own thickness above the ring plane (comb ~10mm, pen ~10mm). A backend that
    absorbs thin objects shows a median near 0 with no positive tail.
    """
    c = np.asarray(cluster["centroid"], dtype=float)
    d = np.linalg.norm(pts[:, :2] - c[:2], axis=1)
    core = d < ring_in
    ring = (d >= ring_in) & (d <= ring_out)
    q, r = pts[core], pts[ring]
    if len(q) < 30 or len(r) < 300:
        return {"n": int(len(q)), "ring_n": int(len(r)),
                "median_mm": None, "p90_mm": None}
    coef = fit_dominant_plane(r)
    h = plane_height(q, coef)
    return {"n": int(len(q)), "ring_n": int(len(r)),
            "median_mm": round(float(np.median(h)) * 1000, 1),
            "p90_mm": round(float(np.percentile(h, 90)) * 1000, 1)}


def comb_length_mm(pts: np.ndarray) -> float | None:
    """Principal-axis extent of the comb region in mm (plausibility check)."""
    x0, x1, y0, y1 = COMB_BBOX
    m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    q = pts[m]
    if len(q) < 200:
        return None
    xy = q[:, :2] - q[:, :2].mean(axis=0)
    # principal axis via SVD; extent = 5th..95th percentile projection
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    proj = xy @ vt[0]
    return round(float(np.percentile(proj, 95) - np.percentile(proj, 5)) * 1000, 1)


def cluster_heights(pts: np.ndarray, cache: Path) -> list[dict]:
    """Height above the local table for each of the backend's own detected
    clusters (answers: does this backend's depth show the objects?)."""
    cpath = cache / "diff" / "clusters.json"
    if not cpath.exists():
        return []
    out = []
    for c in read_json(cpath)["clusters"]:
        h = cluster_height_above_table(pts, c)
        out.append({"id": c["id"], "type": c["type"], "centroid": c["centroid"],
                    **h})
    return out


def analyse(label: str, cache: Path, cfg: dict, conf_percentile: float) -> dict:
    """Measure one reconstruction cache. `cfg` is copied with cache_dir pointed
    at `cache`, so load_reconstruction resolves that cache's artifacts
    (including its aligned/ dir when it exists). `conf_percentile` is the
    per-backend working point (see --conf-ma)."""
    cfg = {**cfg, "paths": {**cfg["paths"], "cache_dir": str(cache)}}
    cfg["reconstruction"] = {**cfg["reconstruction"], "conf_percentile": conf_percentile}
    recon = load_reconstruction(cfg, prefer_aligned=True)
    rdir = recon["dir"]
    meta = recon["meta"]
    depth_conf = recon["depth_conf"]

    conf_q = float(np.percentile(depth_conf, conf_percentile))
    n_raw = int((depth_conf >= conf_q).sum())
    pts, rgb, ptags, _ = tagged_points(recon, conf_percentile)
    av, = voxel_downsample(pts[ptags == "a"], voxel=0.006)
    bv, = voxel_downsample(pts[ptags == "b"], voxel=0.006)
    da, _ = cKDTree(bv).query(av, k=1, workers=-1)

    out = {
        "label": label,
        "backend": meta.get("backend"),
        "model": meta.get("model"),
        "num_frames": meta.get("num_frames"),
        "image_hw": meta.get("image_hw"),
        "inference_seconds": meta.get("inference_seconds"),
        "peak_gpu_gib": meta.get("peak_gpu_gib")
        if meta.get("peak_gpu_gib") is not None else
        (read_json(cache / "reconstruction" / "meta.json").get("peak_gpu_gib")
         if (cache / "reconstruction" / "meta.json").exists() else None),
        "points_raw": n_raw,
        "voxels_6mm": {"a": int(len(av)), "b": int(len(bv))},
        "a_to_b_nn_mm": {
            "p50": round(float(np.percentile(da, 50)) * 1000, 1),
            "p75": round(float(np.percentile(da, 75)) * 1000, 1),
            "p90": round(float(np.percentile(da, 90)) * 1000, 1),
        },
        "cluster_heights": cluster_heights(pts, cache),
        "comb_length_mm": comb_length_mm(pts),
    }
    icp = cache / "reconstruction" / "aligned" / "transform.json"
    if icp.exists():
        t = read_json(icp)
        out["icp"] = {"static_nn_before_mm": round(float(t["static_mean_nn_before"]) * 1000, 1),
                      "static_nn_after_mm": round(float(t["static_mean_nn_after"]) * 1000, 1),
                      "fitness": t.get("fitness")}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache-vggt", default="cache")
    parser.add_argument("--cache-ma", default="cache/ma_run")
    parser.add_argument("--conf-vggt", type=float, default=50.0,
                        help="confidence percentile for VGGT (its working point)")
    parser.add_argument("--conf-ma", type=float, default=25.0,
                        help="confidence percentile for MapAnything: its head "
                             "sits at the floor for the blurrier video B, so "
                             "p50 discards 99%% of B (see docs/results.md)")
    parser.add_argument("--out", default="logs/backend_comparison.json")
    args = parser.parse_args()
    cfg = load_config(args.config)

    results = [analyse("VGGT", repo_path(args.cache_vggt), cfg, args.conf_vggt),
               analyse("MapAnything", repo_path(args.cache_ma), cfg, args.conf_ma)]

    for r in results:
        print(f"\n=== {r['label']} ({r['backend']}) ===")
        print(f"  frames {r['num_frames']} | model size {r['image_hw']} | "
              f"inference {r['inference_seconds']}s | peak {r.get('peak_gpu_gib')} GiB")
        print(f"  points: raw {r['points_raw']:,} | 6mm voxels A={r['voxels_6mm']['a']:,} "
              f"B={r['voxels_6mm']['b']:,}")
        nn = r["a_to_b_nn_mm"]
        print(f"  A->B NN p50/p75/p90: {nn['p50']}/{nn['p75']}/{nn['p90']} mm")
        if "icp" in r:
            print(f"  ICP static mean NN: {r['icp']['static_nn_before_mm']} -> "
                  f"{r['icp']['static_nn_after_mm']} mm (fitness {r['icp']['fitness']})")
        print("  detected clusters, height above their local table surface:")
        for c in r.get("cluster_heights", []):
            print(f"    #{c['id']} {c['type']:<9} n={c['n']:>6} "
                  f"median {c['median_mm']} mm, p90 {c['p90_mm']} mm "
                  f"at {[round(x, 2) for x in c['centroid']]}")
        print(f"  comb principal-axis length: {r['comb_length_mm']} mm "
              f"(plausibility check, typical comb ~200mm)")
    write_json(repo_path(args.out), {"created": None, "results": results})
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
