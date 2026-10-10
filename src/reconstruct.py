"""Stage 2 - joint 3D reconstruction (GPU, 1 device).

Runs VGGT on ALL frames of both videos together so everything lands in one
shared coordinate frame: camera poses, depth maps and a colored point cloud
(with a per-point source tag A/B). Backend selectable via config
(`vggt` primary -> `mast3r` -> `colmap` fallbacks, see docs/models.md).

    python src/reconstruct.py --config configs/default.yaml
    python src/reconstruct.py --config configs/default.yaml --limit_per_video 6  # smoke test

Writes cache/reconstruction/:
    meta.json     provenance + per-frame order/transforms
    cameras.npz   extrinsics [S,3,4], intrinsics [S,3,3]   (OpenCV convention)
    dense.npz     depth [S,H,W], depth_conf [S,H,W], world_points [S,H,W,3]
    colors.npz    images [S,H,W,3] uint8 (model-resolution colors)

Memory note: with 60 frames at 518px in bf16 the model fits one A100-80GB.
If it does not (torch.cuda.OutOfMemoryError), lower frames.max_frames_per_video
or use --limit_per_video for a chunked workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    add_common_args,
    apply_overrides,
    cache_dir,
    load_config,
    pin_idle_gpus,
    read_json,
    repo_path,
    resolve_model,
    setup_logging,
    stamp,
    write_json,
)
from src.geometry import model_to_original_transform  # noqa: E402

log = setup_logging("reconstruct")

MODEL_TARGET = 518  # VGGT's native input width (see vggt.utils.load_fn)


# ---------------------------------------------------------------------------
# input bookkeeping


def load_frame_list(cfg: dict, limit_per_video: int | None = None) -> list[dict]:
    """Ordered frames: all of video A, then all of video B (tags preserved)."""
    index = read_json(cache_dir(cfg, "frames") / "index.json")
    frames_dir = repo_path(cfg["paths"]["cache_dir"]) / "frames"
    frames: list[dict] = []
    for tag in ("a", "b"):
        entries = index["videos"][tag]["frames"]
        if limit_per_video is not None:
            entries = entries[:limit_per_video]
        for e in entries:
            frames.append({"tag": tag, "path": str(frames_dir / e["file"]),
                           "file": e["file"], "blur": e.get("blur")})
    if not frames:
        raise RuntimeError("no frames found in cache/frames/index.json")
    return frames


# ---------------------------------------------------------------------------
# backends


def run_vggt(frames: list[dict], cfg: dict) -> dict:
    import os

    import torch
    from PIL import Image
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    rcfg = cfg["reconstruction"]
    # pin a GPU BEFORE any CUDA initialisation (is_available() below would lock
    # the device list); shared node, PBS does not pin - see gpu.md
    if rcfg.get("gpu_index") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(rcfg["gpu_index"]))
    else:
        pin_idle_gpus(1, log)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available in this job. On this cluster PBS sets "
            "CUDA_VISIBLE_DEVICES to an empty string; `unset CUDA_VISIBLE_DEVICES` "
            "before running (see gpu.md, Gotcha 1)."
        )

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    model_ref = resolve_model(cfg["models"]["vggt"], cfg["paths"]["model_dir"])
    log.info("loading VGGT from %s onto %s (%s)", model_ref, torch.cuda.get_device_name(device), dtype)
    t0 = time.perf_counter()
    model = VGGT.from_pretrained(model_ref).to(device).eval()
    log.info("model loaded in %.1fs", time.perf_counter() - t0)

    paths = [f["path"] for f in frames]
    images = load_and_preprocess_images(paths).to(device)  # [S,3,H,W] float32
    s, _, h, w = images.shape
    log.info("%d frames preprocessed to %dx%d", s, h, w)

    # verify our transform model matches the loaded tensor shape
    transforms = []
    for f in frames:
        with Image.open(f["path"]) as im:
            orig_w, orig_h = im.size
        tr = model_to_original_transform(orig_w, orig_h)
        if (tr["model_h"], tr["model_w"]) != (h, w):
            log.warning("preprocessing transform mismatch for %s: %s vs loaded %dx%d",
                        f["file"], (tr["model_h"], tr["model_w"]), h, w)
        transforms.append(tr)

    t0 = time.perf_counter()
    images_batched = images[None]                     # [1,S,3,H,W] (as in demo_colmap)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        aggregated_tokens_list, ps_idx = model.aggregator(images_batched)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batched.shape[-2:])
        depth, depth_conf = model.depth_head(aggregated_tokens_list, images_batched, ps_idx)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t0
    log.info("inference done in %.1fs (%.2fs/frame)", infer_s, infer_s / s)

    extrinsic = extrinsic.squeeze(0).float().cpu().numpy()          # [S,3,4]
    intrinsic = intrinsic.squeeze(0).float().cpu().numpy()          # [S,3,3]
    depth = depth.squeeze(0).squeeze(-1).float().cpu().numpy()      # [S,H,W]
    depth_conf = depth_conf.squeeze(0).float().cpu().numpy()        # [S,H,W]
    # note: this helper squeezes the channel dim itself -> pass [S,H,W,1]
    world_points = unproject_depth_map_to_point_map(depth[..., None], extrinsic, intrinsic)

    colors = (images.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    return {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth": depth,
        "depth_conf": depth_conf,
        "world_points": world_points,
        "colors": colors,
        "transforms": transforms,
        "backend": "vggt",
        "model": model_ref,
        "device": str(torch.cuda.get_device_name(device)),
        "dtype": str(dtype).replace("torch.", ""),
        "inference_seconds": round(infer_s, 1),
    }


def run_mast3r(frames: list[dict], cfg: dict) -> dict:  # pragma: no cover
    raise NotImplementedError(
        "MASt3R fallback backend not implemented yet. The HF repo is gated and "
        "the checkpoint is unavailable to this account; see docs/models.md and "
        "docs/roadmap.md. Use reconstruction.backend: vggt."
    )


def run_colmap(frames: list[dict], cfg: dict) -> dict:  # pragma: no cover
    raise NotImplementedError("COLMAP fallback backend not implemented yet - see docs/roadmap.md.")


BACKENDS = {"vggt": run_vggt, "mast3r": run_mast3r, "colmap": run_colmap}


# ---------------------------------------------------------------------------
# main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--limit_per_video", type=int, default=None,
                        help="debug: use only the first N frames of each video")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    recon_dir = cache_dir(cfg, "reconstruction")
    meta_path = recon_dir / "meta.json"
    if meta_path.exists() and not args.force:
        old = read_json(meta_path)
        if old.get("limit_per_video") is None and args.limit_per_video is None:
            log.info("reconstruction already cached (%s); use --force to redo", meta_path)
            return 0
        log.warning("existing reconstruction is partial (made with --limit_per_video); redoing")

    frames = load_frame_list(cfg, args.limit_per_video)
    tags = [f["tag"] for f in frames]
    log.info("joint reconstruction over %d frames (%d A + %d B)",
             len(frames), tags.count("a"), tags.count("b"))

    backend = cfg["reconstruction"]["backend"]
    if backend not in BACKENDS:
        raise ValueError(f"unknown reconstruction.backend: {backend!r}")
    try:
        result = BACKENDS[backend](frames, cfg)
    except Exception as exc:  # noqa: BLE001 - keep the OOM hint
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "GPU out of memory during reconstruction. Options: lower "
                "frames.max_frames_per_video, run with --limit_per_video, or "
                "reduce frames.fps."
            ) from exc
        raise

    # ---- save -----------------------------------------------------------
    np.savez_compressed(
        recon_dir / "cameras.npz",
        extrinsic=result["extrinsic"].astype(np.float32),
        intrinsic=result["intrinsic"].astype(np.float32),
    )
    np.savez(
        recon_dir / "dense.npz",
        depth=result["depth"].astype(np.float16),
        depth_conf=result["depth_conf"].astype(np.float16),
        world_points=result["world_points"].astype(np.float16),
    )
    np.savez(
        recon_dir / "colors.npz",
        images=result["colors"],
    )

    index_path = repo_path(cfg["paths"]["cache_dir"]) / "frames" / "index.json"
    meta = {
        "created": stamp(),
        "config_hash": cfg["_config_hash"],
        "backend": result["backend"],
        "model": result["model"],
        "device": result["device"],
        "dtype": result["dtype"],
        "num_frames": len(frames),
        "image_hw": [int(result["depth"].shape[1]), int(result["depth"].shape[2])],
        "conf_percentile": cfg["reconstruction"]["conf_percentile"],
        "limit_per_video": args.limit_per_video,
        "inference_seconds": result.get("inference_seconds"),
        "frames_index_sha1": hashlib.sha1(index_path.read_bytes()).hexdigest()[:12],
        "frames": [
            {"file": f["file"], "tag": f["tag"],
             "model_shape": list(result["depth"].shape[1:]),
             **tr}
            for f, tr in zip(frames, result["transforms"])
        ],
        "note": "world_points are metric-ish (VGGT scale); A and B share this "
                "coordinate frame by construction",
    }
    write_json(meta_path, meta)

    pts = result["world_points"]
    log.info("world points bbox min %s max %s",
             np.round(pts.reshape(-1, 3).min(0), 3).tolist(),
             np.round(pts.reshape(-1, 3).max(0), 3).tolist())
    log.info("wrote %s (cameras.npz, dense.npz, colors.npz)", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
