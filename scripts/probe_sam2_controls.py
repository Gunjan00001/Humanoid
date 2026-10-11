#!/usr/bin/env python3
"""Track B controls: is the 0.65 gate calibrated for cross-video matching?

Same SAM 2 masks and cropping code as the probe, but also reports a
viewpoint-robust descriptor: the mean of the DINOv2 patch tokens inside the mask
(with the CLS descriptor for comparison). Adds a 3D size ratio when the cluster
bounds are available.

Pairs: a positive control (the scene4 apple move that scored 0.704 with the old
descriptor), a negative control (two different objects), and the two targets.
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

PAIRS = [
    ("scene4", "apple_moved (positive)", 2, 6, "positive"),
    ("scene4", "scissor vs apple (negative)", 3, 6, "negative"),
    ("scene6", "cup (target)", 16, 2, "target"),
    ("scene5", "clock (target)", 44, 26, "target"),
]


def extent(scene, cid):
    for cand in [ROOT / "cache" / scene / "semantics" / "confirmed.json",
                 ROOT / "cache" / scene / "diff" / "clusters.json"]:
        if not cand.exists():
            continue
        data = json.loads(cand.read_text())
        clusters = data.get("clusters", data if isinstance(data, list) else [])
        for c in clusters:
            if c.get("id") == cid and "bbox_min" in c:
                return float(np.linalg.norm(np.array(c["bbox_max"]) - np.array(c["bbox_min"])))
    return None


def main() -> int:
    import cv2
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda"
    predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT, device=device))
    mpath = str(Path.home() / "models" / "dinov2-large")
    proc = AutoImageProcessor.from_pretrained(mpath)
    mod = AutoModel.from_pretrained(mpath, dtype=torch.float16).to(device).eval()

    def crop_rgb(scene, cid):
        crops = json.loads((ROOT / "cache" / scene / "semantics" / "crops.json").read_text())
        return cv2.cvtColor(cv2.imread(crops[str(cid)]["crop"]), cv2.COLOR_BGR2RGB)

    def mask_of(rgb):
        h, w = rgb.shape[:2]
        predictor.set_image(rgb)
        m, s, _ = predictor.predict(point_coords=np.array([[w // 2, h // 2]]),
                                    point_labels=np.array([1]), multimask_output=True)
        return m[int(np.argmax(s))].astype(np.uint8)

    def feats(rgb, mask):
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(mod.device, torch.float16)
        with torch.no_grad():
            out = mod(**inputs)
        cls = torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)[0].float().cpu().numpy()
        patches = out.last_hidden_state[0, 1:].float().cpu().numpy()          # [256, 1024]
        g = int(round(patches.shape[0] ** 0.5))
        m = cv2.resize(mask, (g, g), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        pm = patches[m].mean(axis=0) if m.any() else patches.mean(axis=0)
        pm = pm / (np.linalg.norm(pm) + 1e-8)
        return cls, pm

    print(f"{'scene':<8}{'label':<28}{'role':<9}{'cls':>7}{'patch':>7}{'size_r':>8}")
    for scene, label, a, b, role in PAIRS:
        ca, cb = crop_rgb(scene, a), crop_rgb(scene, b)
        ma, mb = mask_of(ca), mask_of(cb)
        cla, pma = feats(ca, ma)
        clb, pmb = feats(cb, mb)
        cls = float(np.dot(cla, clb))
        patch = float(np.dot(pma, pmb))
        ea, eb = extent(scene, a), extent(scene, b)
        sr = f"{min(ea, eb) / max(ea, eb):.2f}" if ea and eb else "-"
        print(f"{scene:<8}{label:<28}{role:<9}{cls:7.3f}{patch:7.3f}{sr:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
