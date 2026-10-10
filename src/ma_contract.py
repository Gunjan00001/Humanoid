"""MapAnything -> pipeline artifact contract (pure NumPy, no heavy imports).

Why this module exists
----------------------
The MapAnything backend runs in a *separate* conda env (`spatialdiff-ma`,
see docs/models.md) because its dependency pins (e.g. opencv-python-headless
==4.10.0.84) conflict with the main `spatialdiff` env. The GPU worker
(scripts/run_mapanything.py, executed inside that env) therefore converts
MapAnything's outputs into exactly the artifacts `src/reconstruct.py` writes
for VGGT:

    cache/reconstruction/
      cameras.npz   extrinsic [S,3,4] (world->camera, OpenCV), intrinsic [S,3,3]
      dense.npz     depth [S,H,W], depth_conf [S,H,W], world_points [S,H,W,3]
      colors.npz    images [S,H,W,3] uint8 (model-resolution colours)
      meta.json     provenance + per-frame model<->original transform

Keeping the conversion in this *pure NumPy* module means:
  - the worker imports it without any MapAnything dependency,
  - a CPU fixture test (tests/test_ma_contract.py) can validate the maths
    without a GPU or the model,
  - downstream stages (align/diff/describe/visualize/report) work unchanged.

MapAnything output conventions (from its README, verified against the code):
  - `camera_poses` are **cam2world** 4x4 in OpenCV convention (+X right,
    +Y down, +Z forward) -> we invert them to world->camera extrinsics.
  - `pts3d` are world-space points (H,W,3); `depth_z` is camera-frame Z-depth.
  - `conf` is per-pixel confidence; `mask` marks valid pixels.
  - `img_no_norm` are the de-normalised model-resolution images [0,255].
  - Most tensors carry a leading batch dim B=1 (we squeeze it defensively).

MapAnything preprocessing (from mapanything/utils/image.py + utils/cropping.py):
  - a batch-level target resolution is picked from a fixed table by the
    *average* aspect ratio: for our portrait 480x854 frames the closest key is
    0.567 -> target (294, 518)  (W, H);
  - each image is resized with a *uniform* scale
        scale = max(target_w / W, target_h / H) + 1e-8
    (floor-rounded), then centre-cropped to the target.  The scale is the same
    in x and y; the crop offsets are therefore the only anisotropy to record.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# model <-> original-image transform (mirrors mapanything.utils.cropping)


def transform_for(orig_w: int, orig_h: int, model_w: int, model_h: int) -> dict:
    """Per-frame transform between the original JPEG and the model input.

    Reproduces MapAnything's crop_resize_if_necessary maths:

        scale = max(target_w / W, target_h / H) + 1e-8
        resized = floor((W, H) * scale)          # covers the target
        crop    = (resized - target) // 2        # centred

    The ±1 px rounding of `floor`/`//` is irrelevant for downstream crop
    selection (margins there are centimetre-scale), but we keep the exact
    formula so the numbers stay reproducible and comparable with VGGT's.
    """
    scale = max(model_w / orig_w, model_h / orig_h) + 1e-8
    resized_w = int(np.floor(orig_w * scale))
    resized_h = int(np.floor(orig_h * scale))
    crop_left = max(0, (resized_w - model_w) // 2)
    crop_top = max(0, (resized_h - model_h) // 2)
    return {
        "orig_w": int(orig_w), "orig_h": int(orig_h),
        # uniform scale: both fields hold the same value (kept separate for
        # the shared model_bbox_to_original() code path)
        "scale_w": round(scale, 6), "scale_h": round(scale, 6),
        "crop_top": int(crop_top), "crop_left": int(crop_left),
        "model_w": int(model_w), "model_h": int(model_h),
    }


# ---------------------------------------------------------------------------
# prediction -> contract arrays


def _squeeze_batch(x: Any) -> np.ndarray:
    """MapAnything returns tensors with a leading batch dim (B=1).  Accepts
    torch tensors or numpy arrays and returns a numpy array without B."""
    if x is None:
        raise ValueError("expected an array, got None")
    if hasattr(x, "detach"):                      # torch tensor
        x = x.detach().float().cpu().numpy()
    a = np.asarray(x)
    if a.ndim >= 1 and a.shape[0] == 1:
        a = a[0]
    return a


def prediction_to_arrays(pred: dict[str, Any]) -> dict[str, Any]:
    """Convert one MapAnything per-view prediction into contract arrays.

    Raises ValueError with a helpful message when a required field is missing
    or has an unexpected shape - the worker logs the full prediction keys so
    a new MapAnything release can be debugged quickly.
    """
    required = ("pts3d", "depth_z", "conf", "intrinsics", "camera_poses")
    missing = [k for k in required if k not in pred]
    if missing:
        raise ValueError(f"MapAnything prediction misses {missing}; "
                         f"available keys: {sorted(pred.keys())}")

    world_points = _squeeze_batch(pred["pts3d"]).astype(np.float32)        # (H,W,3)
    depth = _squeeze_batch(pred["depth_z"]).astype(np.float32)             # (H,W,1) or (H,W)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    conf = _squeeze_batch(pred["conf"]).astype(np.float32)                 # (H,W)
    intrinsic = _squeeze_batch(pred["intrinsics"]).astype(np.float32)      # (3,3)
    cam2world = _squeeze_batch(pred["camera_poses"]).astype(np.float64)    # (4,4)

    # honour the validity mask: masked pixels must not pass the confidence
    # percentile filter downstream
    if "mask" in pred and pred["mask"] is not None:
        mask = _squeeze_batch(pred["mask"])
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        conf = np.where(mask.astype(bool), conf, 0.0)

    # cam2world (OpenCV) -> world->camera extrinsic [3,4], matching VGGT's
    # convention in cache/reconstruction/cameras.npz
    world2cam = np.linalg.inv(cam2world)
    extrinsic = world2cam[:3, :4].astype(np.float32)

    # colours: de-normalised model-resolution image [0,255] -> uint8
    if "img_no_norm" in pred and pred["img_no_norm"] is not None:
        colors = np.clip(_squeeze_batch(pred["img_no_norm"]), 0, 255).astype(np.uint8)
    else:                                          # fallback: zero colours
        colors = np.zeros((*depth.shape, 3), dtype=np.uint8)

    msf = pred.get("metric_scaling_factor")
    if msf is not None:
        msf = float(_squeeze_batch(msf).reshape(-1)[0])

    # --- shape consistency (fail loudly, with numbers, for easy debugging) --
    h, w = depth.shape
    if world_points.shape != (h, w, 3):
        raise ValueError(f"pts3d {world_points.shape} vs depth {depth.shape}")
    if conf.shape != (h, w):
        raise ValueError(f"conf {conf.shape} vs depth {depth.shape}")
    if colors.shape != (h, w, 3):
        raise ValueError(f"img_no_norm {colors.shape} vs depth {depth.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsics {intrinsic.shape} != (3,3)")

    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth": depth,
        "depth_conf": conf,
        "world_points": world_points,
        "colors": colors,
        "metric_scaling_factor": msf,
        "image_hw": (h, w),
    }
