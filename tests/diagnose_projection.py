#!/usr/bin/env python
"""Step-0 diagnosis: reproject the fused cloud into a real camera and compare.

For a given reconstruction cache this script:
  1. loads the (aligned) cloud with a LOW confidence cutoff (5%) so we see what
     the model actually captured, not what detection keeps,
  2. z-buffers those points into the first camera of the scene (nearest point
     wins) with their true colours,
  3. saves a side-by-side: real frame | reprojection | coverage mask,
  4. prints per-view point counts before/after the confidence filters and the
     fraction of the frame the model input actually covered (portrait crop
     check) and the fraction of the reprojected frame with points.

Read-only; writes only the diagnostic image. No GPU.

    python tests/diagnose_projection.py --cache cache            --out cache/diag/demo.png
    python tests/diagnose_projection.py --cache cache/scene2     --out cache/diag/scene2.png
    python tests/diagnose_projection.py --cache cache/scene3     --out cache/diag/scene3.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, read_json, repo_path  # noqa: E402
from src.geometry import load_reconstruction, project_points, tagged_points  # noqa: E402


def coverage_of_model_input(tr: dict) -> float:
    """Fraction of the ORIGINAL frame that reached the model input.

    crop mode (VGGT default): width -> model_w, height center-cropped ->
        model_h / resized_h of the original area.
    pad mode: the whole frame is scaled to fit the target; content occupies
        (content_w x content_h) of the square target.
    """
    orig_w, orig_h = tr["orig_w"], tr["orig_h"]
    scale_w, scale_h = tr["scale_w"], tr["scale_h"]
    resized_w, resized_h = orig_w * scale_w, orig_h * scale_h
    content = min(resized_w, tr["model_w"]) * min(resized_h, tr["model_h"])
    return float(content / (resized_w * resized_h))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["paths"]["cache_dir"] = args.cache

    recon = load_reconstruction(cfg, prefer_aligned=True)
    meta = recon["meta"]
    tags = recon["tags"]
    s, h, w = recon["depth"].shape

    # ---- per-view point counts at three confidence levels ----------------
    conf = recon["depth_conf"]
    print(f"cache: {args.cache} | frames {s} | model size {h}x{w} | backend {meta.get('backend')}")
    print(f"{'frame':>3} {'tag':>3} {'pixels':>9} {'conf>p50':>9} {'conf>p5':>9}")
    for i in range(min(s, 6)):
        c = conf[i].reshape(-1)
        p50 = np.percentile(conf, 50)
        p5 = np.percentile(conf, 5)
        print(f"{i:>3} {tags[i]:>3} {c.size:>9} {(c > p50).sum():>9} {(c > p5).sum():>9}")

    # ---- how much of the original frame reached the model ---------------
    fracs = [coverage_of_model_input(f) for f in meta["frames"]]
    print(f"\nmodel input coverage of the original frame: "
          f"min {min(fracs):.1%} max {max(fracs):.1%} (crop mode discards the "
          f"top/bottom of portrait frames)")

    # ---- reproject into the first camera ---------------------------------
    pts, rgb, ptags, pconf = tagged_points(recon, 5.0)   # LOW cutoff: see everything
    print(f"points with conf > p5: {len(pts):,}")
    extrinsic = recon["extrinsic"][0]
    intrinsic = recon["intrinsic"][0]
    uv, z = project_points(pts, extrinsic, intrinsic)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 1e-6)
    u, v, z, col = u[ok], v[ok], z[ok], rgb[ok].astype(np.float32)

    # z-buffer: sort far-to-near, later writes (nearer) win
    order = np.argsort(-z)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[v[order], u[order]] = col[order]
    coverage = np.zeros((h, w), dtype=bool)
    coverage[v, u] = True
    print(f"reprojected into frame 0: {ok.sum():,} points, "
          f"frame coverage {coverage.mean():.1%}")

    # ---- side-by-side ----------------------------------------------------
    real = np.asarray(Image.open(repo_path(cfg["paths"]["cache_dir"]) / "frames"
                                 / meta["frames"][0]["file"]).convert("RGB")
                      .resize((w, h)))
    panels = [real, img.astype(np.uint8),
              np.repeat((coverage * 255).astype(np.uint8)[..., None], 3, 2)]
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    side = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(side).save(out)
    print(f"wrote {out}  (real frame | reprojection | coverage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
