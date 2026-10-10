#!/usr/bin/env python
"""Rebuild cache/reconstruction/colors.npz from the original frames (CPU).

The colours of the point cloud must be the model-resolution images. They are
normally produced during reconstruction, but a model release can stop
returning denormalised images (observed with MapAnything: `img_no_norm` absent
-> zero colours), which silently breaks the photometric diff. This tool
re-derives the colours from the original JPEGs using the per-frame
model<->original transforms stored in meta.json - no GPU, no model needed.

    python scripts/rebuild_colors.py --config configs/default.yaml --cache cache/ma_run

Works for both backends because both store the same transform fields
(scale_w/scale_h + crop_top/crop_left + model_w/model_h).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import add_common_args, apply_overrides, load_config, read_json, repo_path, setup_logging  # noqa: E402

log = setup_logging("rebuild_colors")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    cache = repo_path(cfg["paths"]["cache_dir"])
    recon_dir = cache / "reconstruction"
    # prefer the aligned dir when it exists (its meta carries the same frames)
    if (recon_dir / "aligned" / "meta.json").exists():
        recon_dir = recon_dir / "aligned"
    meta = read_json(recon_dir / "meta.json")
    frames_dir = cache / "frames"

    images = []
    for f in meta["frames"]:
        tr = f
        model_w, model_h = int(tr["model_w"]), int(tr["model_h"])
        scale_w, scale_h = float(tr["scale_w"]), float(tr["scale_h"])
        crop_left, crop_top = int(tr.get("crop_left", 0)), int(tr["crop_top"])
        im = Image.open(frames_dir / f["file"]).convert("RGB")
        resized = im.resize((int(np.floor(im.size[0] * scale_w)),
                             int(np.floor(im.size[1] * scale_h))), Image.LANCZOS)
        resized = resized.crop((crop_left, crop_top,
                                crop_left + model_w, crop_top + model_h))
        arr = np.asarray(resized, dtype=np.uint8)
        if arr.shape[:2] != (model_h, model_w):
            raise RuntimeError(f"{f['file']}: resized to {arr.shape[:2]}, "
                               f"expected {(model_h, model_w)}")
        images.append(arr)
        if len(images) % 10 == 0:
            log.info("  %d/%d frames", len(images), len(meta["frames"]))

    out = recon_dir / "colors.npz"
    np.savez(out, images=np.stack(images))
    log.info("wrote %s (%d frames, %s)", out, len(images), np.stack(images).shape)
    log.info("note: downstream stages read colors.npz from the same directory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
