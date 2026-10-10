"""Generate a synthetic reconstruction cache for smoke-testing the CPU stages.

NOT part of the pipeline and NOT a source of results. It fabricates a
`cache/reconstruction` exactly in the format src/reconstruct.py writes, but
from an analytical scene: a wall plane at z=2.0 with two small boxes in front -
one only visible in video A ("removed"), one only in video B ("added").

    python tests/make_fake_reconstruction.py         # -> cache/fake_run/reconstruction

Then:  python src/diff.py       --config configs/default.yaml --cache cache/fake_run
       python src/visualize.py  --config configs/default.yaml --cache cache/fake_run
       python src/report.py     --config configs/default.yaml --cache cache/fake_run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import stamp, write_json  # noqa: E402

W = H = 518
FX = FY = 600.0
CX = CY = (W - 1) / 2.0
OUT = ROOT / "cache" / "fake_run" / "reconstruction"

WALL_Z = 2.0
# boxes: (centre_x, centre_y, front_z, half_size)  - A-only and B-only
BOX_REMOVED = (-0.40, 0.10, 1.50, 0.15)
BOX_ADDED = (0.40, -0.05, 1.60, 0.15)


def make_frame(t: np.ndarray, box: tuple | None) -> tuple[np.ndarray, np.ndarray]:
    """Analytical depth + colour for a camera with identity rotation at t."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dx = (u - CX) / FX
    dy = (v - CY) / FY
    depth = np.full((H, W), WALL_Z - t[2], dtype=np.float32)   # wall plane
    colour = np.stack([(u / W * 180 + 40), (v / H * 180 + 40), np.full_like(u, 90)], axis=-1)

    if box is not None:
        bx, by, bz, hs = box
        d_box = bz - t[2]
        # world x,y of the front face for each pixel
        wx = t[0] + d_box * dx
        wy = t[1] + d_box * dy
        m = (np.abs(wx - bx) <= hs) & (np.abs(wy - by) <= hs)
        depth[m] = d_box
        colour[m] = (np.array([200, 60, 60]) if bx < 0 else np.array([60, 200, 60]))
    return depth, colour.astype(np.uint8)


def unproject(depth: np.ndarray, t: np.ndarray) -> np.ndarray:
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x = (u - CX) / FX * depth + t[0]
    y = (v - CY) / FY * depth + t[1]
    z = depth + t[2]
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    frames, extr, intr, dens, conf, pts, cols = [], [], [], [], [], [], []
    for i in range(4):                                # video A: box at -0.4 present
        t = np.array([-0.30 + 0.20 * i, 0.0, 0.0], dtype=np.float32)
        t += rng.normal(0, 0.005, 3).astype(np.float32)
        d, c = make_frame(t, BOX_REMOVED)
        frames.append({"file": f"a/frame_{i:05d}.jpg", "tag": "a"})
        extr.append(np.hstack([np.eye(3), t[:, None]]))
        intr.append(np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32))
        dens.append(d)
        conf.append(np.ones((H, W), dtype=np.float32))
        pts.append(unproject(d, t))
        cols.append(c)
    for i in range(4):                                # video B: box at +0.4 present
        t = np.array([-0.25 + 0.20 * i, 0.0, 0.0], dtype=np.float32)
        t += rng.normal(0, 0.005, 3).astype(np.float32)
        d, c = make_frame(t, BOX_ADDED)
        frames.append({"file": f"b/frame_{i:05d}.jpg", "tag": "b"})
        extr.append(np.hstack([np.eye(3), t[:, None]]))
        intr.append(np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32))
        dens.append(d)
        conf.append(np.ones((H, W), dtype=np.float32))
        pts.append(unproject(d, t))
        cols.append(c)

    np.savez_compressed(OUT / "cameras.npz",
                        extrinsic=np.stack(extr), intrinsic=np.stack(intr))
    np.savez(OUT / "dense.npz",
             depth=np.stack(dens).astype(np.float16),
             depth_conf=np.stack(conf).astype(np.float16),
             world_points=np.stack(pts).astype(np.float16))
    np.savez(OUT / "colors.npz", images=np.stack(cols))
    write_json(OUT / "meta.json", {
        "created": stamp(), "config_hash": "fake", "backend": "synthetic",
        "model": "analytic", "device": "cpu", "dtype": "float32",
        "num_frames": len(frames), "image_hw": [H, W], "conf_percentile": 50.0,
        "inference_seconds": 0.0, "limit_per_video": None,
        "frames_index_sha1": "fake",
        "frames": [
            {**f, "model_shape": [H, W], "orig_w": W, "orig_h": H,
             "scale_w": 1.0, "scale_h": 1.0, "crop_top": 0,
             "model_w": W, "model_h": H}
            for f in frames
        ],
        "note": "SYNTHETIC test cache - not a real reconstruction",
    })
    print(f"wrote synthetic reconstruction to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
