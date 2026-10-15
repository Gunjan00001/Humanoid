#!/usr/bin/env python3
"""Projection control (decisive adapter check, no detection).

For each processable validation pair, lift the GT object masks to 3D with the
cached depth and poses, then send those 3D points through the production adapter
path (mask_rle_for_points: 3D to 2D, frame resampling raw//30, coordinate
conversion, transpose, RLE) and write object_masks.pkl. The official evaluator
then scores them.

  python scripts/oracle_projection_control.py --data DATA --out PRED_DIR
"""
# EXPERIMENTAL, not used by detect.py. Part of the accuracy and naming
# experiments; kept as evidence. See docs/accuracy.md and docs/naming_v4.md.
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config  # noqa: E402
from src.geometry import load_reconstruction  # noqa: E402
from src.eval_scenediff import mask_rle_for_points  # noqa: E402

RESAMPLE = 30
PAIRS = ["airhockeytable_1_airhockeytable_2", "bathroom_11_bathroom_12",
         "bathroom_1_bathroom_2", "bedroom_2_bedroom_3", "bedroom_2_bedroom_4",
         "gym_9_gym_10", "kitchen_8_kitchen_9"]


def rawmap_for(cache: Path):
    import json
    index = json.loads((cache / "frames" / "index.json").read_text())
    out = {}
    for tag in ("a", "b"):
        m = {}
        for i, f in enumerate(index["videos"][tag].get("frames", [])):
            name = f if isinstance(f, str) else f.get("file", "")
            try:
                m[int(Path(name).stem.split("_")[-1])] = i
            except Exception:  # noqa: BLE001
                pass
        out[tag] = m
    return out


def nearest(rdict, raw):
    from src.eval_scenediff import nearest as n
    return n(rdict, raw)


def main() -> int:
    import pycocotools.mask as mu
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    for pair in PAIRS:
        cache = ROOT / "cache" / "scenediff" / pair
        cfg = load_config("configs/default.yaml")
        cfg["paths"]["cache_dir"] = str(cache)
        recon = load_reconstruction(cfg)
        rawmap = rawmap_for(cache)
        own = {t: [i for i, tt in enumerate(recon["tags"]) if tt == t] for t in ("a", "b")}
        seg = pickle.load(open(data / pair / "segments.pkl", "rb"))
        res = {"H": None, "W": None}
        for vid_key, tag in (("video1_objects", "a"), ("video2_objects", "b")):
            for oid, frames in seg.get(vid_key, {}).items():
                for fid, m in frames.items():
                    raw = int(fid)
                    if raw % RESAMPLE != 0:
                        continue
                    gt = mu.decode(m)                 # (orig_w, orig_h) transposed
                    mv = gt.T                         # video coords (orig_h, orig_w)
                    k = nearest(rawmap[tag], raw)
                    if k is None or k >= len(own[tag]):
                        continue
                    ei = own[tag][k]
                    tr = recon["meta"]["frames"][ei]
                    ys, xs = np.nonzero(mv)
                    if len(ys) == 0:
                        continue
                    mx = xs * tr["scale_w"] - tr.get("crop_left", 0)
                    my = ys * tr["scale_h"] - tr.get("crop_top", 0)
                    depth = recon["depth"][ei]
                    ix = np.clip(np.round(mx).astype(int), 0, depth.shape[1] - 1)
                    iy = np.clip(np.round(my).astype(int), 0, depth.shape[0] - 1)
                    z = depth[iy, ix].astype(float)
                    ok = np.isfinite(z) & (z > 0)
                    if not ok.any():
                        continue
                    intr = recon["intrinsic"][ei]
                    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
                    zz = z[ok]
                    cam = np.stack([(mx[ok] - cx) / fx * zz, (my[ok] - cy) / fy * zz, zz], 1)
                    ext = recon["extrinsic"][ei]
                    R, t = ext[:, :3], ext[:, 3]
                    world = (cam - t) @ R            # X_cam = R X_world + t
                    rle = mask_rle_for_points(recon, ei, world)
                    vkey = f"video_{1 if tag == 'a' else 2}"
                    res.setdefault(str(oid), {}).setdefault(vkey, {})[raw // RESAMPLE] = {
                        "mask": rle, "cost": 1.0}
                    res["H"], res["W"] = rle["size"]
        d = out / pair
        d.mkdir(parents=True, exist_ok=True)
        pickle.dump(res, open(d / "object_masks.pkl", "wb"))
        print(f"oracle {pair}: {len([k for k in res if k not in ('H','W')])} objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
