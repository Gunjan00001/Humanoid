#!/usr/bin/env python3
"""Track B probe: can SAM 2 isolate the moved object in both videos and does the
DINOv2 similarity of the paired crops then clear the match gate?

For a target (scene, destination cluster, origin cluster) it takes the existing
pose-matched crops from the semantics cache, refines each with a SAM 2 mask
(point prompt at the crop centre), then compares the masked crops with DINOv2.

  python scripts/probe_sam2.py        # runs the scene6 cup and scene5 clock probes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = str(Path.home() / "models" / "sam2.1-hiera-base-plus" / "sam2.1_hiera_base_plus.pt")
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
GATE = 0.65

# (scene, label, destination cluster in video B, origin cluster in video A)
TARGETS = [
    ("scene6", "cup", 16, 2),
    ("scene5", "clock", 44, 26),
]


def load_crop(path):
    import cv2
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sam_mask(predictor, rgb):
    h, w = rgb.shape[:2]
    predictor.set_image(rgb)
    pts = np.array([[w // 2, h // 2]])
    lbl = np.array([1])
    masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbl,
                                         multimask_output=True)
    i = int(np.argmax(scores))
    return masks[i].astype(np.uint8)


def masked_embed(model, processor, rgb, mask):
    from PIL import Image
    arr = rgb.copy()
    arr[mask == 0] = 0
    img = Image.fromarray(arr)
    import torch
    inputs = processor(images=img, return_tensors="pt").to(model.device, torch.float16)
    with torch.no_grad():
        out = model(**inputs)
    v = torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)
    return v[0].float().cpu().numpy()


def main() -> int:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda"
    sam2 = build_sam2(CFG, CKPT, device=device)
    predictor = SAM2ImagePredictor(sam2)
    mpath = str(Path.home() / "models" / "dinov2-large")
    mproc = AutoImageProcessor.from_pretrained(mpath)
    mmod = AutoModel.from_pretrained(mpath, dtype=torch.float16).to(device).eval()

    results = []
    for scene, label, dest, origin in TARGETS:
        sem = ROOT / "cache" / scene / "semantics"
        crops = json.loads((sem / "crops.json").read_text())
        d = crops.get(str(dest))
        o = crops.get(str(origin))
        if not d or not o or "crop" not in d or "crop" not in o:
            results.append((scene, label, "no crops", None, None))
            continue
        img_d = load_crop(d["crop"])      # destination object (video B)
        img_o = load_crop(o["crop"])      # candidate origin object (video A)
        md = sam_mask(predictor, img_d)
        mo = sam_mask(predictor, img_o)
        ed = masked_embed(mmod, mproc, img_d, md)
        eo = masked_embed(mmod, mproc, img_o, mo)
        sim = float(np.dot(ed, eo))
        results.append((scene, label, "masked", sim, sim >= GATE))

    print(f"{'scene':<8}{'object':<8}{'dest_frac':>10}{'orig_frac':>10}{'dino':>8}{'pass':>6}")
    for scene, label, status, sim, ok in results:
        print(f"{scene:<8}{label:<8}{status:>10}{'':>10}"
              f"{'' if sim is None else f'{sim:.3f}':>8}{'' if ok is None else str(ok):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
