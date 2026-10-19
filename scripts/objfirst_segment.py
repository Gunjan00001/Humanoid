#!/usr/bin/env python3
"""M2: SAM 3.1 masks for the pipeline's clusters, in the sam3 env.

For every cluster crop of a scene, prompt SAM 3.1 with the cached VLM label
(name prompt; automatic masks of SAM 2.1 are the documented fallback) under bf16
autocast, then keep the best mask and its statistics. Output per scene:
logs/objfirst/masks/<scene>.npz with one boolean mask per cluster id plus a JSON
summary. Run this in `spatialdiff-sam3`, not in the pipeline env.

  ~/miniconda3/envs/spatialdiff-sam3/bin/python scripts/objfirst_segment.py --scenes scene4
"""
# EXPERIMENTAL, not used by detect.py. Phase 4 (exp/object-first).
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CKPT = str(Path.home() / "models" / "sam3.1" / "sam3.1_multiplex.pt")
OUT = ROOT / "logs" / "objfirst" / "masks"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["demo", "scene2", "scene3",
                                                    "scene4", "scene5", "scene6"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=CKPT, device=args.device,
                                   eval_mode=True, load_from_HF=False)
    proc = Sam3Processor(model, device=args.device, confidence_threshold=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        base = ROOT / "cache" / scene
        crops = base / "semantics" / "crops.json"
        vlm = base / "semantics" / "vlm.json"
        if not crops.exists():
            print(f"{scene}: no crops, skipped")
            continue
        crop_info = json.loads(crops.read_text())
        labels = json.loads(vlm.read_text()) if vlm.exists() else {}
        masks, stats = {}, {}
        for cid, info in crop_info.items():
            path = info.get("crop")
            if not path or not Path(path).exists():
                continue
            label = (labels.get(str(cid)) or {}).get("label") or "object"
            img = Image.open(path).convert("RGB")
            with torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu",
                                dtype=torch.bfloat16):
                state = proc.set_image(img)
                state = proc.set_text_prompt(label, state)
            m = state.get("masks")
            if m is None or not len(m):
                continue
            arr = m.detach().cpu().numpy().astype(bool)
            areas = arr.reshape(len(arr), -1).mean(axis=1)
            best = int(np.argmax(areas))
            masks[str(cid)] = arr[best]
            stats[str(cid)] = {"label": label, "prompt": label,
                               "n_masks": int(len(arr)),
                               "area_fraction": round(float(areas[best]), 5)}
        np.savez_compressed(OUT / f"{scene}.npz", **masks)
        (OUT / f"{scene}.json").write_text(json.dumps(stats, indent=2))
        print(f"{scene}: {len(masks)} masks for {len(crop_info)} clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
