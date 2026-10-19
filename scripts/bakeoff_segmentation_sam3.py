#!/usr/bin/env python3
"""Bake-off (b): SAM 3.1 (official package, text prompts) vs SAM 2.1 automatic masks.

Run inside the `spatialdiff-sam3` env for the SAM 3.1 pass and the
`spatialdiff-sam` env for the SAM 2.1 automatic-mask pass (two subcommands), then
merge the JSONs. Metrics per (scene, GT object, prompt), for frames of video A
and video B:

  * masks returned and masks above the confidence threshold;
  * the best mask's area fraction and whether it is object-like (0.05%-30%);
  * "segmented in both videos": an object-like mask for that prompt on both sides;
  * spurious instances: masks beyond the best one;
  * mask quality for SAM 2.1: best IoU between its automatic masks and the SAM 3.1
    best mask (the boxes come from the same text prompt, so this is an agreement
    measure, not an absolute quality);
  * track-ID stability is reported as not measured: the image route has no
    tracker, and the video predictor needs a session per video, which the time
    box did not reach.

  python scripts/bakeoff_segmentation_sam3.py --mode sam3
  python scripts/bakeoff_segmentation_sam3.py --mode sam2     # in the sam env
  python scripts/bakeoff_segmentation_sam3.py --merge
"""
# EXPERIMENTAL, not used by detect.py. Evidence for the stack selection.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CKPT = str(Path.home() / "models" / "sam3.1" / "sam3.1_multiplex.pt")
SCENES = ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]
AREA_MIN, AREA_MAX = 0.0005, 0.30
OUT3 = ROOT / "logs/bakeoff_seg_sam3.json"
OUT2 = ROOT / "logs/bakeoff_seg_sam2.json"


def gt_objects(scene: str) -> list[str]:
    for f in (ROOT / "data" / scene / "ground_truth.json",
              ROOT / "data" / "ground_truth.json"):
        if f.exists():
            return [c.get("object", "") for c in json.loads(f.read_text())["changes"]
                    if c.get("object")]
    return []


def frames(scene: str):
    base = ROOT / "cache" / scene / "frames"
    return (sorted((base / "a").glob("*.jpg")), sorted((base / "b").glob("*.jpg")))


def run_sam3(device: str) -> list[dict]:
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3 import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=CKPT, device=device,
                                   eval_mode=True, load_from_HF=False)
    proc = Sam3Processor(model, device=device, confidence_threshold=0.3)
    rows = []
    for scene in SCENES:
        fa, fb = frames(scene)
        if not fa or not fb:
            continue
        pick = {"a": fa[len(fa) // 2], "b": fb[len(fb) // 2]}
        for obj in gt_objects(scene):
            for prompt in (obj, "object"):
                for side, path in pick.items():
                    img = Image.open(path).convert("RGB")
                    # Run the official path under bf16 autocast: the checkpoint
                    # is bf16, the processor feeds float32, and autocast is the
                    # documented way to let the two meet.
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        state = proc.set_image(img)
                        state = proc.set_text_prompt(prompt, state)
                    masks = state.get("masks")
                    scores = state.get("scores")
                    n = 0 if masks is None else int(masks.shape[0])
                    best_i, best_area = None, 0.0
                    if n:
                        arr = masks.detach().cpu().numpy().astype(bool)
                        areas = arr.reshape(n, -1).mean(axis=1)
                        best_i = int(np.argmax(areas))
                        best_area = float(areas[best_i])
                        np.save(ROOT / f"logs/_sam3_{scene}_{obj}_{prompt}_{side}.npy".replace(" ", "_"),
                                arr[best_i])
                    rows.append({"scene": scene, "object": obj, "prompt": prompt,
                                 "side": side, "n_masks": n,
                                 "spurious": max(0, n - 1),
                                 "best_area_fraction": round(best_area, 5),
                                 "object_like": AREA_MIN <= best_area <= AREA_MAX,
                                 "best_score": float(scores[best_i]) if n and scores is not None else None})
        print(f"{scene}: done ({len(gt_objects(scene))} objects)")
    return rows


def run_sam2() -> list[dict]:
    """Automatic masks on the same frames; IoU against the saved SAM3 best masks."""
    import cv2
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    ckpt = str(Path.home() / "models" / "sam2.1-hiera-base-plus" / "sam2.1_hiera_base_plus.pt")
    model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", ckpt, device="cuda")
    gen = SAM2AutomaticMaskGenerator(model)
    rows = []
    for scene in SCENES:
        fa, fb = frames(scene)
        if not fa or not fb:
            continue
        pick = {"a": fa[len(fa) // 2], "b": fb[len(fb) // 2]}
        for obj in gt_objects(scene):
            for prompt in (obj, "object"):
                for side, path in pick.items():
                    bgr = cv2.imread(str(path))
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    masks = gen.generate(rgb)
                    n = len(masks)
                    seg = [m["segmentation"] for m in masks]
                    ref_path = ROOT / f"logs/_sam3_{scene}_{obj}_{prompt}_{side}.npy".replace(" ", "_")
                    best_iou = None
                    if seg and ref_path.exists():
                        ref = np.load(ref_path)
                        if ref.shape == seg[0].shape:
                            best_iou = max(
                                float(np.logical_and(ref, m).sum() /
                                      max(1, np.logical_or(ref, m).sum())) for m in seg)
                    rows.append({"scene": scene, "object": obj, "prompt": prompt,
                                 "side": side, "n_masks": n,
                                 "best_iou_vs_sam3": None if best_iou is None else round(best_iou, 3)})
        print(f"{scene}: sam2 done")
    return rows


def summarise(rows: list[dict]) -> dict:
    saw = {}
    for r in rows:
        key = (r["scene"], r["object"], r["prompt"])
        saw.setdefault(key, {"a": None, "b": None})[r["side"]] = r
    groups = []
    for key, sides in saw.items():
        both = sides["a"] and sides["b"] and sides["a"]["object_like"] and sides["b"]["object_like"]
        groups.append({"scene": key[0], "object": key[1], "prompt": key[2],
                       "segmented_both": bool(both),
                       "mean_area": round(float(np.mean([s["best_area_fraction"] for s in sides.values() if s])), 5)
                       if any(sides.values()) else 0.0,
                       "spurious": sum(s["spurious"] for s in sides.values() if s)})
    objs = {(g["scene"], g["object"]) for g in groups if g["prompt"] != "object" and g["segmented_both"]}
    gen = {(g["scene"], g["object"]) for g in groups if g["prompt"] == "object" and g["segmented_both"]}
    return {"named_prompt_objects_segmented_both": len(objs), "generic_prompt_objects_segmented_both": len(gen),
            "total_objects": len({(g["scene"], g["object"]) for g in groups}),
            "spurious_total": sum(g["spurious"] for g in groups), "groups": groups}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sam3", "sam2", "merge"], required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.mode == "sam3":
        rows = run_sam3(args.device)
        OUT3.write_text(json.dumps({"rows": rows, "summary": summarise(rows)}, indent=2))
        s = summarise(rows)
        print(f"SAM3: {s['named_prompt_objects_segmented_both']} named-prompt objects "
              f"segmented in both videos (of {s['total_objects']}), "
              f"{s['generic_prompt_objects_segmented_both']} with the generic prompt, "
              f"{s['spurious_total']} spurious masks")
    elif args.mode == "sam2":
        rows = run_sam2()
        OUT2.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"SAM2: wrote {OUT2}")
    else:
        a = json.loads(OUT3.read_text())
        b = json.loads(OUT2.read_text()) if OUT2.exists() else {"rows": []}
        merged = {"sam3": a["summary"], "sam2_rows": len(b["rows"])}
        (ROOT / "logs/bakeoff_seg_merged.json").write_text(json.dumps(merged, indent=2))
        print(json.dumps(merged["sam3"], indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
