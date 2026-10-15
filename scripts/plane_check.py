#!/usr/bin/env python3
"""Item 2: what plane did the automatic mode pick on the SceneDiff pairs?

For each pair it loads the cached reconstruction and computes the dominant plane
the same way detect.py's auto mode does (src.tabletop), then reports:
  - the plane normal (unit) and how vertical it is,
  - the fraction of confident points within 2 cm of the plane (the auto-mode test),
  - the plane height above the lowest confident surface,
  - a plain reading: floor-like (normal near vertical), wall-like (normal near
    horizontal), or neither.

CPU only. Reads cache/scenediff*/<pair>/reconstruction.
"""
# EXPERIMENTAL, not used by detect.py. Part of the accuracy and naming
# experiments; kept as evidence. See docs/accuracy.md and docs/naming_v4.md.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config  # noqa: E402
from src.geometry import load_reconstruction, tagged_points  # noqa: E402
from src.tabletop import fit_dominant_plane, plane_height  # noqa: E402

PAIRS = ["airhockeytable_1_airhockeytable_2", "bathroom_11_bathroom_12",
         "bathroom_1_bathroom_2", "bedroom_2_bedroom_3", "bedroom_2_bedroom_4",
         "gym_9_gym_10", "kitchen_8_kitchen_9",
         "bedroom_1_bedroom_2", "gas_station_3_gas_station_4",
         "laundry_room_3_laundry_room_4", "living_room_15_living_room_16",
         "living_room_17_living_room_18", "living_room_1_living_room_2"]


def main() -> int:
    print(f"{'pair':<40}{'normal':>22}{'vert%':>7}{'cov2cm':>8}{'height':>9}  reading")
    for p in PAIRS:
        cache = ROOT / "cache" / "scenediff" / p
        if not (cache / "reconstruction" / "meta.json").exists():
            cache = ROOT / "cache" / "scenediff_adaptive" / p
        if not (cache / "reconstruction" / "meta.json").exists():
            print(f"{p:<40}  (no reconstruction)")
            continue
        cfg = load_config("configs/default.yaml")
        cfg["paths"]["cache_dir"] = str(cache)
        recon = load_reconstruction(cfg)
        pts, _, _, _ = tagged_points(recon, float(cfg["reconstruction"]["conf_percentile"]))
        if not len(pts):
            print(f"{p:<40}  (no confident points)")
            continue
        sub = pts[:: max(1, len(pts) // 100000)]
        coef = fit_dominant_plane(sub)
        n = np.asarray(coef[:3], dtype=float)
        n = n / (np.linalg.norm(n) + 1e-12)
        height = plane_height(sub, coef)
        cov = float((np.abs(height) <= 0.02).mean())
        vertical = abs(float(n[2])) if len(n) >= 3 else 0.0   # |nz|
        low = float(np.percentile(-height, 1))   # plane vs the lowest surface
        reading = ("floor-like (normal vertical)" if vertical > 0.85
                   else "wall-like (normal horizontal)" if vertical < 0.3
                   else "neither (tilted)")
        print(f"{p:<40}{np.array2string(n, precision=2):>22}{100*vertical:>6.0f}%"
              f"{100*cov:>7.0f}%{low:>9.3f}  {reading}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
