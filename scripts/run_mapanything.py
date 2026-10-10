#!/usr/bin/env python
"""MapAnything backend worker - runs in the `spatialdiff-ma` env on a GPU node.

Invoked as a subprocess by src/reconstruct.py when
`reconstruction.backend: mapanything`:

    ~/miniconda3/envs/spatialdiff-ma/bin/python scripts/run_mapanything.py \
        --config configs/default.yaml \
        --frames-index cache/frames/index.json \
        --out cache/reconstruction \
        [--limit-per-video N] [--force]

Why a subprocess?
-----------------
MapAnything pins `opencv-python-headless==4.10.0.84` and other versions that
conflict with the main `spatialdiff` env, so it lives in its own conda env
(created by scripts/setup_env.sh). This worker converts its outputs into the
exact artifact contract of the VGGT backend (see src/ma_contract.py), so every
downstream stage works unchanged.

GPU rules (see gpu.md): run only on a GPU node, `unset CUDA_VISIBLE_DEVICES`
(PBS hides the GPUs otherwise), and pin the idlest GPU BEFORE CUDA is
initialised - common.pin_idle_gpus() does that and is called before any torch
import that touches CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import (  # noqa: E402  (path setup must come first)
    load_config,
    pin_idle_gpus,
    read_json,
    repo_path,
    resolve_model,
    setup_logging,
    stamp,
    write_json,
)
from src.ma_contract import prediction_to_arrays, transform_for  # noqa: E402

log = setup_logging("mapanything")


def load_frames(frames_index: Path, limit_per_video: int | None) -> list[dict]:
    """Ordered frames: all of video A, then all of video B - the same order the
    VGGT backend uses, so tags and downstream comparisons stay aligned."""
    index = read_json(frames_index)
    frames_dir = frames_index.parent
    frames: list[dict] = []
    for tag in ("a", "b"):
        entries = index["videos"][tag]["frames"]
        if limit_per_video is not None:
            entries = entries[:limit_per_video]
        for e in entries:
            frames.append({"tag": tag, "path": str(frames_dir / e["file"]),
                           "file": e["file"]})
    if not frames:
        raise RuntimeError(f"no frames found in {frames_index}")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames-index", required=True)
    parser.add_argument("--out", required=True, help="reconstruction output directory")
    parser.add_argument("--limit-per-video", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    macfg = cfg["reconstruction"].get("mapanything", {})
    out_dir = repo_path(args.out)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not args.force:
        log.info("reconstruction already cached (%s); use --force to redo", meta_path)
        return 0

    frames_index = repo_path(args.frames_index)

    # --- pin an idle GPU BEFORE torch initialises CUDA (see module docstring)
    if cfg["reconstruction"].get("gpu_index") is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(cfg["reconstruction"]["gpu_index"]))
    else:
        pin_idle_gpus(1, log)

    import torch
    from PIL import Image
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. In PBS jobs CUDA_VISIBLE_DEVICES is set to an "
            "empty string - `unset CUDA_VISIBLE_DEVICES` before running "
            "(gpu.md, Gotcha 1)."
        )

    frames = load_frames(frames_index, args.limit_per_video)
    tags = [f["tag"] for f in frames]
    log.info("joint reconstruction over %d frames (%d A + %d B)",
             len(frames), tags.count("a"), tags.count("b"))

    # --- load the model (local path preferred: compute nodes may be offline)
    model_ref = resolve_model(cfg["models"]["mapanything"], cfg["paths"]["model_dir"])
    device = torch.device("cuda")
    amp_dtype = str(macfg.get("amp_dtype", "bf16"))
    log.info("loading MapAnything from %s onto %s (%s)", model_ref,
             torch.cuda.get_device_name(device), amp_dtype)
    t0 = time.perf_counter()
    model = MapAnything.from_pretrained(model_ref).to(device).eval()
    load_s = time.perf_counter() - t0
    log.info("model loaded in %.1fs", load_s)

    # --- MapAnything's own preprocessing (518-px fixed mapping + dinov2 norm)
    t0 = time.perf_counter()
    views = load_images([f["path"] for f in frames])
    prep_s = time.perf_counter() - t0
    first_shape = tuple(views[0]["img"].shape)
    log.info("preprocessed %d views in %.1fs (view shape %s)",
             len(views), prep_s, first_shape)

    # --- inference -------------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)
    minibatch = macfg.get("minibatch_size")
    t0 = time.perf_counter()
    try:
        predictions = model.infer(
            views,
            memory_efficient_inference=bool(macfg.get("memory_efficient_inference", True)),
            minibatch_size=int(minibatch) if minibatch else None,
            use_amp=bool(macfg.get("use_amp", True)),
            amp_dtype=amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=bool(macfg.get("apply_confidence_mask", False)),
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "MapAnything ran out of GPU memory. Options: lower "
            "frames.max_frames_per_video, run with --limit_per_video, or set "
            "reconstruction.mapanything.minibatch_size: 1."
        ) from exc
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
    log.info("inference done in %.1fs (%.2fs/frame) | peak GPU memory %.1f GiB",
             infer_s, infer_s / len(frames), peak_gb)

    # --- convert to the shared contract (src/ma_contract.py) --------------
    extrinsics, intrinsics, depths, confs, points, colors, transforms = \
        [], [], [], [], [], [], []
    msf_values = []
    hw = None
    for f, pred in zip(frames, predictions):
        arr = prediction_to_arrays(pred)
        if hw is None:
            hw = arr["image_hw"]
        elif arr["image_hw"] != hw:
            raise RuntimeError(
                f"frame {f['file']} has model size {arr['image_hw']} but earlier "
                f"frames have {hw} - mixed aspect ratios need separate batches"
            )
        with Image.open(f["path"]) as im:
            orig_w, orig_h = im.size
        transforms.append(transform_for(orig_w, orig_h, *arr["image_hw"]))
        extrinsics.append(arr["extrinsic"])
        intrinsics.append(arr["intrinsic"])
        depths.append(arr["depth"])
        confs.append(arr["depth_conf"])
        points.append(arr["world_points"])
        colors.append(arr["colors"])
        msf_values.append(arr["metric_scaling_factor"])

    # --- save: byte-for-byte the same contract as the VGGT backend --------
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "cameras.npz",
                        extrinsic=np.stack(extrinsics).astype(np.float32),
                        intrinsic=np.stack(intrinsics).astype(np.float32))
    np.savez(out_dir / "dense.npz",
             depth=np.stack(depths).astype(np.float16),
             depth_conf=np.stack(confs).astype(np.float16),
             world_points=np.stack(points).astype(np.float16))
    np.savez(out_dir / "colors.npz", images=np.stack(colors))
    meta = {
        "created": stamp(),
        "config_hash": cfg["_config_hash"],
        "backend": "mapanything",
        "model": model_ref,
        "device": str(torch.cuda.get_device_name(device)),
        "dtype": amp_dtype,
        "num_frames": len(frames),
        "image_hw": [int(hw[0]), int(hw[1])],
        "conf_percentile": cfg["reconstruction"]["conf_percentile"],
        "limit_per_video": args.limit_per_video,
        "load_seconds": round(load_s, 1),
        "preprocess_seconds": round(prep_s, 1),
        "inference_seconds": round(infer_s, 1),
        "peak_gpu_gib": round(peak_gb, 1),
        "metric_scaling_factors": msf_values,
        "frames_index_sha1": hashlib.sha1(frames_index.read_bytes()).hexdigest()[:12],
        "frames": [
            {"file": f["file"], "tag": f["tag"],
             "model_shape": [int(hw[0]), int(hw[1])], **tr}
            for f, tr in zip(frames, transforms)
        ],
        "note": "MapAnything metric reconstruction; A and B share this "
                "coordinate frame by construction (single joint run)",
    }
    write_json(meta_path, meta)

    pts = np.stack(points).reshape(-1, 3)
    conf_all = np.stack(confs).reshape(-1)
    log.info("world bbox min %s max %s",
             np.round(pts.min(0), 3).tolist(), np.round(pts.max(0), 3).tolist())
    log.info("conf p10/p50/p90 %s", [round(float(np.percentile(conf_all, q)), 2)
                                     for q in (10, 50, 90)])
    log.info("wrote %s (cameras.npz, dense.npz, colors.npz)", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
