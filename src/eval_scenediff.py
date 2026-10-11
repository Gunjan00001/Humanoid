#!/usr/bin/env python3
"""SceneDiff adapter (Track C, C3).

Turns our report.json for one SceneDiff pair into the evaluator's prediction
format: object first, resampled frame ids, pycocotools RLE masks.

Each detected change is projected into the pair's annotated GT frames for the
video(s) it belongs to (removed -> video 1, added -> video 2, moved/replaced ->
both). The cluster's 3D points are projected with the cached poses and depth
frame and rasterised into a mask.

  python src/eval_scenediff.py --pair NAME --cache CACHE --out PRED_DIR --data DATA_DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config  # noqa: E402
from src.geometry import load_reconstruction, project_points  # noqa: E402

RESAMPLE = 30


def raw_frames(index: dict, tag: str) -> dict:
    """Map raw frame index -> extracted index for one video."""
    out = {}
    for i, f in enumerate(index["videos"][tag].get("frames", [])):
        name = f if isinstance(f, str) else f.get("file", "")
        try:
            out[int(Path(name).stem.split("_")[-1])] = i
        except Exception:  # noqa: BLE001
            pass
    return out


def nearest(rdict: dict, raw: int):
    if raw in rdict:
        return rdict[raw]
    return min(rdict, key=lambda k: abs(k - raw)) if rdict else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    import pycocotools.mask as mask_utils

    cache = Path(args.cache)
    cfg = load_config("configs/default.yaml")
    cfg["paths"]["cache_dir"] = str(cache)
    recon = load_reconstruction(cfg)
    report = json.loads((Path(args.out).parent / args.pair / "report.json").read_text()) \
        if (Path(args.out).parent / args.pair / "report.json").exists() else None
    if report is None:
        report = json.loads((cache / "report.json").read_text()) if (cache / "report.json").exists() else {"changes": []}
    changes = report.get("changes", [])

    index = json.loads((cache / "frames" / "index.json").read_text())
    rawmap = {"a": raw_frames(index, "a"), "b": raw_frames(index, "b")}
    seg = __import__("pickle").load(open(Path(args.data) / args.pair / "segments.pkl", "rb"))
    H = W = None
    for m in list(seg.get("video1_objects", {}).values())[:1]:
        for mm in m.values():
            H, W = mm["size"][0], mm["size"][1]
    if H is None:
        for m in list(seg.get("video2_objects", {}).values())[:1]:
            for mm in m.values():
                H, W = mm["size"][0], mm["size"][1]
    H, W = H or 1080, W or 1920

    diff = np.load(cache / "diff" / "points.npz")
    out = {"H": H, "W": W}
    for ch in changes:
        vid_tags = {"removed": ["a"], "added": ["b"], "moved": ["a", "b"],
                    "replaced": ["a", "b"]}.get(ch.get("type"), ["a", "b"])
        pts = None
        for cid in ch.get("cluster_ids", []):
            for tag in ("a", "b"):
                arr = diff[f"{tag}_points"]
                lab = diff[f"{tag}_labels"]
                sel = arr[lab == cid]
                if len(sel):
                    pts = sel if pts is None else np.vstack([pts, sel])
        if pts is None or not len(pts):
            continue
        obj = out.setdefault(str(ch["id"]), {})
        for tag in vid_tags:
            frames = rawmap[tag]
            gt_frames = set()
            for oid, fr in seg.get(f"video{1 if tag=='a' else 2}_objects", {}).items():
                gt_frames |= {int(k) for k in fr.keys()}
            for raw in sorted(gt_frames):
                if raw % RESAMPLE != 0:
                    continue
                ei = nearest(frames, raw)
                if ei is None or ei >= len(recon["extrinsic"]):
                    continue
                uv, z = project_points(pts, recon["extrinsic"][ei], recon["intrinsic"][ei])
                uv = np.round(uv).astype(int)
                m = np.zeros((H, W), dtype=np.uint8)
                x = np.clip(uv[:, 0], 0, W - 1); y = np.clip(uv[:, 1], 0, H - 1)
                m[y, x] = 1
                import cv2
                m = cv2.dilate(m, np.ones((15, 15), np.uint8))
                rle = mask_utils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                obj.setdefault(f"video_{1 if tag=='a' else 2}", {})[raw // RESAMPLE] = {
                    "mask": {"size": [H, W], "counts": rle["counts"]},
                    "cost": float(ch.get("confidence") or 0.5)}
    d = Path(args.out) / args.pair
    d.mkdir(parents=True, exist_ok=True)
    __import__("pickle").dump(out, open(d / "object_masks.pkl", "wb"))
    print(f"{args.pair}: {len([k for k in out if k not in ('H','W')])} objects -> {d/'object_masks.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
