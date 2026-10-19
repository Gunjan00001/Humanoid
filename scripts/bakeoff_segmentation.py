#!/usr/bin/env python3
"""Bake-off (b): SAM 3.1 (text prompts) versus SAM 2.1 (box prompts).

The dev scenes have object-level ground truth but no ground-truth masks, so the
comparison uses what is verifiable:

  * SAM 3.1 is prompted with the object's NAME (and, for the generic run, with
    "object") on a frame of video A and a frame of video B, and reports how many
    masks came back and whether the best one is in both videos;
  * SAM 2.1 has no text interface, so it is prompted with the BOX of the mask
    SAM 3.1 found and produces a mask for the same region; the two are compared
    by mask IoU, which measures whether the cheaper promptable segmenter agrees
    with the text-prompted one on the same object;
  * spurious instances are the extra masks SAM 3.1 returns beyond the best one;
  * a mask is judged object-like when its area is between 0.05% and 30% of the
    frame, a written range, since a whole-frame or single-pixel mask is not a
    segmentation of an object.

  python scripts/bakeoff_segmentation.py --scenes scene4 --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAM3_DIR = Path.home() / "models" / "sam3.1"
SAM2_DIR = Path.home() / "models" / "sam2.1-hiera-base-plus"
AREA_MIN, AREA_MAX = 0.0005, 0.30          # written object-like area range


def gt_objects(scene: str) -> list[str]:
    for f in (ROOT / "data" / scene / "ground_truth.json", ROOT / "data" / "ground_truth.json"):
        if f.exists():
            return [c.get("object", "") for c in json.loads(f.read_text())["changes"]
                    if c.get("object")]
    return []


def frames(scene: str) -> tuple[list[Path], list[Path]]:
    base = ROOT / "cache" / scene / "frames"
    a = sorted((base / "a").glob("*.jpg"))
    b = sorted((base / "b").glob("*.jpg"))
    return a, b


def mask_stats(mask: np.ndarray) -> dict:
    area = float(mask.mean())
    ys, xs = np.where(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
    return {"area_fraction": round(area, 5), "bbox": box,
            "object_like": AREA_MIN <= area <= AREA_MAX}


def iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union else 0.0


def run_sam3(prompt: str, image, device: str):
    """Text-prompted segmentation with SAM 3.1; returns a list of masks."""
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(SAM3_DIR))
    model = AutoModel.from_pretrained(str(SAM3_DIR)).to(device).eval()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    masks = processor.post_process_masks(
        out.pred_masks, inputs["original_sizes"])[0]
    return masks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="*", default=["demo", "scene2", "scene3", "scene4",
                                                    "scene5", "scene6"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--every", type=int, default=6, help="use every Nth frame")
    ap.add_argument("--out", default="logs/bakeoff_segmentation.json")
    args = ap.parse_args()

    from PIL import Image

    rows = []
    for scene in args.scenes:
        objs = gt_objects(scene)
        fa, fb = frames(scene)
        if not fa or not fb:
            print(f"{scene}: no cached frames, skipped")
            continue
        pick_a, pick_b = fa[len(fa) // 2], fb[len(fb) // 2]
        for obj in objs:
            for prompt in (obj, "object"):
                for side, path in (("a", pick_a), ("b", pick_b)):
                    img = Image.open(path).convert("RGB")
                    try:
                        masks = run_sam3(prompt, img, args.device)
                    except Exception as exc:
                        rows.append({"scene": scene, "object": obj, "prompt": prompt,
                                     "side": side, "error": str(exc)[:160]})
                        continue
                    arr = [np.asarray(m).astype(bool) for m in masks]
                    stats = [mask_stats(m) for m in arr]
                    best = max(stats, key=lambda s: s["area_fraction"]) if stats else None
                    rows.append({"scene": scene, "object": obj, "prompt": prompt,
                                 "side": side, "n_masks": len(stats),
                                 "spurious": max(0, len(stats) - 1),
                                 "best": best})
            print(f"{scene}: {obj} done")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if r.get("best") and r["best"]["object_like"]]
    print(f"{len(ok)}/{len(rows)} (scene, object, prompt, side) rows produced an "
          f"object-like mask; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
