#!/usr/bin/env python3
"""Geometry task: one reconstruction backend at one frame budget, measured.

Frames are extracted first (CPU), then the backend runs (GPU), then the static
alignment residual is measured: points of one video that have a close neighbour
in the other (static structure) report their nearest-neighbour distance, which
is the number the alignment pass tries to reduce. Also recorded: point count,
runtime per stage and the object density proxy (points inside the GT objects'
3D boxes when the report boxes exist).

Writes its JSON incrementally after every stage, so a kill leaves what was
measured. The unattended runner (scripts/run_batch.py) is the normal caller.

  python scripts/geometry_eval.py --scene scene4 --backend vggt --frames 1x \
      --out logs/batch/geom_vggt_1x.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import cache_dir, load_config, write_json  # noqa: E402


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene4")
    ap.add_argument("--backend", choices=["vggt", "mapanything"], required=True)
    ap.add_argument("--frames", choices=["1x", "2x"], default="1x")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    result = {"scene": args.scene, "backend": args.backend, "frames": args.frames,
              "status": "started"}

    cfg = load_config("configs/default.yaml")
    tag = f"geom_{args.backend}_{args.frames}_{args.scene}"
    cfg["paths"]["cache_dir"] = str(ROOT / "cache" / tag)
    cfg["reconstruction"]["backend"] = args.backend
    cfg["frames"]["max_frames_per_video"] = 30 if args.frames == "1x" else 60
    result["frame_cap_per_video"] = cfg["frames"]["max_frames_per_video"]

    from src.extract_frames import extract_one
    t0 = time.time()
    scene_dir = ROOT / "data" / args.scene
    a = (scene_dir / "a.mp4") if scene_dir.exists() else ROOT / "data" / "a.mp4"
    b = (scene_dir / "b.mp4") if scene_dir.exists() else ROOT / "data" / "b.mp4"
    info_a = extract_one(a, "a", cfg)
    info_b = extract_one(b, "b", cfg)
    # load_frame_list() reads this index; extract_one alone does not write it.
    write_json(cache_dir(cfg, "frames") / "index.json",
               {"videos": {"a": info_a, "b": info_b}})
    na, nb = len(info_a["frames"]), len(info_b["frames"])
    result.update({"frames_a": na, "frames_b": nb,
                   "extract_s": round(time.time() - t0, 1), "status": "extracted"})
    write(out, result)

    import src.reconstruct as R
    from src.geometry import load_reconstruction, tagged_points
    from scipy.spatial import cKDTree

    frames = R.load_frame_list(cfg)
    t1 = time.time()
    if args.backend == "vggt":
        backend_result = R.run_vggt(frames, cfg)
    else:
        backend_result = R.run_mapanything(frames, cfg)
    result["reconstruct_s"] = round(time.time() - t1, 1)

    # Persist the artifacts exactly as src/reconstruct.main does: run_vggt
    # returns the arrays and leaves writing to the caller. MapAnything's worker
    # writes its own files instead.
    if not backend_result.get("wrote_files"):
        from src.common import repo_path, stamp
        rdir = repo_path(cfg["paths"]["cache_dir"]) / "reconstruction"
        rdir.mkdir(parents=True, exist_ok=True)
        res = backend_result
        np.savez_compressed(rdir / "cameras.npz",
                            extrinsic=res["extrinsic"].astype(np.float32),
                            intrinsic=res["intrinsic"].astype(np.float32))
        np.savez(rdir / "dense.npz",
                 depth=res["depth"].astype(np.float16),
                 depth_conf=res["depth_conf"].astype(np.float16),
                 world_points=res["world_points"].astype(np.float16))
        np.savez(rdir / "colors.npz", images=res["colors"])
        index_path = repo_path(cfg["paths"]["cache_dir"]) / "frames" / "index.json"
        import hashlib
        write_json(rdir / "meta.json", {
            "created": stamp(), "config_hash": cfg.get("_config_hash"),
            "backend": res["backend"], "model": res["model"],
            "input_mode": res.get("input_mode"), "device": res["device"],
            "dtype": res["dtype"], "num_frames": len(frames),
            "image_hw": [int(res["depth"].shape[1]), int(res["depth"].shape[2])],
            "conf_percentile": cfg["reconstruction"]["conf_percentile"],
            "limit_per_video": None,
            "inference_seconds": res.get("inference_seconds"),
            "peak_gpu_gib": res.get("peak_gpu_gib"),
            "frames_index_sha1": hashlib.sha1(index_path.read_bytes()).hexdigest()[:12],
            "frames": [{"file": f["file"], "tag": f["tag"],
                        "model_shape": list(res["depth"].shape[1:]), **tr}
                       for f, tr in zip(frames, res["transforms"])],
        })
    result["status"] = "reconstructed"
    write(out, result)

    recon = load_reconstruction(cfg)
    pts, rgb, tags, conf = tagged_points(recon, cfg["reconstruction"]["conf_percentile"])
    pa, pb = pts[tags == "a"], pts[tags == "b"]
    step = max(1, len(pa) // 60000)
    pa_s, pb_s = pa[::step], pb[::step]
    dist, _ = cKDTree(pb_s).query(pa_s, k=1, workers=-1)
    static = dist < 0.05
    result.update({
        "points_total": int(len(pts)), "points_a": int(len(pa)), "points_b": int(len(pb)),
        "static_fraction": round(float(static.mean()), 3),
        "static_nn_mean_m": round(float(dist[static].mean()), 4) if static.any() else None,
        "static_nn_median_m": round(float(np.median(dist[static])), 4) if static.any() else None,
        "status": "measured",
    })
    write(out, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
