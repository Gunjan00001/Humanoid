#!/usr/bin/env python3
"""The earlier SAM 2 mask-tight controls, re-measured with DINOv3 and DINOv2.

Same pairs, same SAM 2 masks and the same crop code as
scripts/probe_sam2_controls.py, so the numbers are comparable with the probe
recorded in logs/probe_controls_run.log (apple 0.704 CLS / 0.914 patch;
scissors-vs-apple 0.285 / 0.085; cup 0.482 / 0.374; clock 0.174 / 0.460).
Both encoders run on exactly the same masked crops in one process.

  python scripts/probe_sam2_controls_dinov3.py        # needs the sam env + a GPU
"""
# EXPERIMENTAL, not used by detect.py. Evidence for the stack selection.
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CKPT = str(Path.home() / "models" / "sam2.1-hiera-base-plus" / "sam2.1_hiera_base_plus.pt")
CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DINOV2 = str(Path.home() / "models" / "dinov2-large")
DINOV3 = str(Path.home() / "models" / "dinov3-vitl16")

PAIRS = [
    ("scene4", "apple_moved (positive)", 2, 6, "positive"),
    ("scene4", "scissor vs apple (negative)", 3, 6, "negative"),
    ("scene6", "cup (target)", 16, 2, "target"),
    ("scene5", "clock (target)", 44, 26, "target"),
]


def main() -> int:
    import cv2
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda"
    predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT, device=device))
    encoders = {}
    for name, path in (("DINOv2", DINOV2), ("DINOv3", DINOV3)):
        encoders[name] = (AutoImageProcessor.from_pretrained(path),
                          AutoModel.from_pretrained(path, dtype=torch.float16).to(device).eval())

    def crop_rgb(scene, cid):
        crops = json.loads((ROOT / "cache" / scene / "semantics" / "crops.json").read_text())
        return cv2.cvtColor(cv2.imread(crops[str(cid)]["crop"]), cv2.COLOR_BGR2RGB)

    def mask_of(rgb):
        h, w = rgb.shape[:2]
        predictor.set_image(rgb)
        m, s, _ = predictor.predict(point_coords=np.array([[w // 2, h // 2]]),
                                    point_labels=np.array([1]), multimask_output=True)
        return m[int(np.argmax(s))].astype(np.uint8)

    def feats(enc, rgb, mask):
        proc, mod = enc
        inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            out = mod(**inputs)
        cls = torch.nn.functional.normalize(out.last_hidden_state[:, 0], dim=-1)[0].float().cpu().numpy()
        patches = out.last_hidden_state[0, 1:].float().cpu().numpy()
        # The patch grid is NOT square for a rectangular crop (DINOv3 gave 200
        # patches on a 20x10 grid). Derive it from the input and patch size.
        pv = inputs["pixel_values"]
        ps = int(getattr(mod.config, "patch_size", 16))
        gh, gw = pv.shape[-2] // ps, pv.shape[-1] // ps
        if gh * gw != patches.shape[0]:
            g = int(round(patches.shape[0] ** 0.5))
            gh = gw = g
        m = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        pm = patches[m].mean(axis=0) if m.any() else patches.mean(axis=0)
        pm = pm / (np.linalg.norm(pm) + 1e-8)
        return cls, pm

    print(f"{'scene':<8}{'label':<28}{'role':<9}"
          f"{'dv2_cls':>8}{'dv2_patch':>10}{'dv3_cls':>8}{'dv3_patch':>10}")
    rows = []
    for scene, label, a, b, role in PAIRS:
        ca, cb = crop_rgb(scene, a), crop_rgb(scene, b)
        ma, mb = mask_of(ca), mask_of(cb)
        row = {"scene": scene, "label": label, "role": role}
        for name, enc in encoders.items():
            cla, pma = feats(enc, ca, ma)
            clb, pmb = feats(enc, cb, mb)
            row[f"{name}_cls"] = round(float(np.dot(cla, clb)), 3)
            row[f"{name}_patch"] = round(float(np.dot(pma, pmb)), 3)
        rows.append(row)
        print(f"{scene:<8}{label:<28}{role:<9}"
              f"{row['DINOv2_cls']:8.3f}{row['DINOv2_patch']:10.3f}"
              f"{row['DINOv3_cls']:8.3f}{row['DINOv3_patch']:10.3f}")
    out = ROOT / "logs/bakeoff_controls_dinov3.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
