"""CPU fixture tests for the MapAnything artifact contract.

These tests exercise src/ma_contract.py with synthetic predictions - no GPU,
no MapAnything install, no real model. They validate the maths that the GPU
worker (scripts/run_mapanything.py) relies on:

  1. the model<->original transform reproduces MapAnything's crop_resize
     behaviour for the demo's portrait frames (480x854 -> 294x518),
  2. cam2world poses are inverted into our world->camera extrinsics,
  3. the validity mask zeroes the confidence of masked pixels,
  4. the model-space -> original-image bbox mapping round-trips.

Run:  python tests/test_ma_contract.py     (any env with numpy; fast, no GPU)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.geometry import model_bbox_to_original  # noqa: E402
from src.ma_contract import prediction_to_arrays, transform_for  # noqa: E402

FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAILED
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED += 1


def test_transform() -> None:
    print("transform_for: demo portrait frames 480x854 -> model 294x518")
    tr = transform_for(480, 854, 294, 518)
    # MapAnything: scale = max(294/480, 518/854) + 1e-8; resized = floor(...)
    scale = max(294 / 480, 518 / 854) + 1e-8
    check("uniform scale", abs(tr["scale_w"] - tr["scale_h"]) < 1e-9,
          f"{tr['scale_w']} vs {tr['scale_h']}")
    check("scale value", abs(tr["scale_w"] - scale) < 1e-5, f"{tr['scale_w']:.6f}")
    check("no horizontal crop", tr["crop_left"] == 0, str(tr["crop_left"]))
    # resized height = floor(854 * scale) = 523; centre crop to 518 -> top 2
    resized_h = int(np.floor(854 * scale))
    check("crop_top = (resized-518)//2", tr["crop_top"] == (resized_h - 518) // 2,
          f"crop_top={tr['crop_top']} resized_h={resized_h}")

    print("transform_for: landscape case with a horizontal crop")
    tr2 = transform_for(1920, 1080, 518, 294)     # 16:9 target (518, 294)
    scale2 = max(518 / 1920, 294 / 1080) + 1e-8
    resized_w2 = int(np.floor(1920 * scale2))
    check("crop_left computed", tr2["crop_left"] == (resized_w2 - 518) // 2,
          f"crop_left={tr2['crop_left']} resized_w={resized_w2}")


def test_bbox_mapping() -> None:
    print("model_bbox_to_original with the MapAnything transform")
    tr = transform_for(480, 854, 294, 518)
    # a bbox in the model image maps back into the original frame and stays
    # inside it; the vertical centre of the model maps near the image centre
    ox1, oy1, ox2, oy2 = model_bbox_to_original((100, 200, 200, 300), tr)
    check("bbox inside original", 0 <= ox1 < ox2 <= 480 and 0 <= oy1 < oy2 <= 854,
          f"{(ox1, oy1, ox2, oy2)}")
    # model row 259 (centre of 518) -> original row (259 + crop_top)/scale
    _, oy_c, _, _ = model_bbox_to_original((10, 259, 20, 260), tr)
    expected = (259 + tr["crop_top"]) / tr["scale_h"]
    check("centre row maps correctly", abs(oy_c - expected) < 2,
          f"{oy_c} vs {expected:.1f}")


def _fake_pred(h: int = 6, w: int = 4, mask_value: float = 1.0) -> dict:
    """A synthetic MapAnything per-view prediction (with batch dims, like the
    real one: (1,H,W,...))."""
    rng = np.random.default_rng(0)
    cam2world = np.eye(4, dtype=np.float64)
    cam2world[:3, 3] = [0.1, -0.2, 0.3]           # a translation to invert
    return {
        "pts3d": rng.normal(size=(1, h, w, 3)).astype(np.float32),
        "depth_z": rng.uniform(0.5, 2.0, size=(1, h, w, 1)).astype(np.float32),
        "conf": rng.uniform(0, 10, size=(1, h, w)).astype(np.float32),
        "intrinsics": np.tile(np.eye(3, dtype=np.float32), (1, 1, 1)),
        "camera_poses": cam2world[None],
        "mask": np.full((1, h, w, 1), mask_value, dtype=np.float32),
        "img_no_norm": rng.uniform(0, 255, size=(1, h, w, 3)).astype(np.float32),
        "metric_scaling_factor": np.array([0.9876]),
    }


def test_prediction_conversion() -> None:
    print("prediction_to_arrays: shapes, pose inversion, mask, colours")
    pred = _fake_pred()
    arr = prediction_to_arrays(pred)
    h, w = 6, 4
    check("depth [H,W]", arr["depth"].shape == (h, w), str(arr["depth"].shape))
    check("conf [H,W]", arr["depth_conf"].shape == (h, w))
    check("points [H,W,3]", arr["world_points"].shape == (h, w, 3))
    check("colours uint8 [H,W,3]",
          arr["colors"].shape == (h, w, 3) and arr["colors"].dtype == np.uint8)
    check("intrinsic (3,3)", arr["intrinsic"].shape == (3, 3))
    check("extrinsic (3,4)", arr["extrinsic"].shape == (3, 4))

    # cam2world with translation t -> world2cam must have translation -R^T t
    # (R = I here, so -t)
    check("pose inverted", np.allclose(arr["extrinsic"][:, 3], [-0.1, 0.2, -0.3],
                                       atol=1e-5), str(arr["extrinsic"][:, 3]))
    check("metric scaling carried",
          abs(arr["metric_scaling_factor"] - 0.9876) < 1e-6)
    check("world points match pts3d",
          np.allclose(arr["world_points"], pred["pts3d"][0], atol=1e-6))

    print("masked pixels get confidence 0 (they must not pass the percentile)")
    pred = _fake_pred(mask_value=0.0)
    arr = prediction_to_arrays(pred)
    check("masked conf all zero", float(np.abs(arr["depth_conf"]).max()) == 0.0)

    print("error handling: missing key / shape mismatch")
    bad = _fake_pred()
    del bad["conf"]
    try:
        prediction_to_arrays(bad)
        check("missing key raises", False)
    except ValueError as exc:
        check("missing key raises with keys listed", "conf" in str(exc))
    bad = _fake_pred()
    bad["depth_z"] = bad["depth_z"][:, :, :, :1].repeat(2, axis=2)  # (1,6,8,1)
    try:
        prediction_to_arrays(bad)
        check("shape mismatch raises", False)
    except ValueError:
        check("shape mismatch raises", True)


def main() -> int:
    test_transform()
    test_bbox_mapping()
    test_prediction_conversion()
    print()
    if FAILED:
        print(f"{FAILED} check(s) FAILED")
        return 1
    print("all checks passed (synthetic fixtures; never project results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
