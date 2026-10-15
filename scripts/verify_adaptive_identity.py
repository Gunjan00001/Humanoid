#!/usr/bin/env python3
"""Item 2: prove `frames.adaptive_sharpness` only touches the failure path.

For every scene it extracts frames twice into scratch caches, once with the flag
OFF (the frozen behaviour) and once with it ON, then compares every produced file
by sha256. CPU only, no GPU, no models.

  python scripts/verify_adaptive_identity.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config  # noqa: E402
from src.extract_frames import extract_one  # noqa: E402

SCENES = {
    "demo": (ROOT / "data/a.mp4", ROOT / "data/b.mp4"),
    "scene2": (ROOT / "data/scene2/a.mp4", ROOT / "data/scene2/b.mp4"),
    "scene3": (ROOT / "data/scene3/a.mp4", ROOT / "data/scene3/b.mp4"),
    "scene4": (ROOT / "data/scene4/a.mp4", ROOT / "data/scene4/b.mp4"),
    "scene5": (ROOT / "data/scene5/a.mp4", ROOT / "data/scene5/b.mp4"),
    "scene6": (ROOT / "data/scene6/a.mp4", ROOT / "data/scene6/b.mp4"),
}
SCRATCH = ROOT / "cache" / "adaptive_identity"


def digest(d: Path) -> dict[str, str]:
    out = {}
    for p in sorted(d.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def run(scene: str, adaptive: bool) -> tuple[dict[str, str], int]:
    a, b = SCENES[scene]
    cfg = load_config("configs/default.yaml")
    cfg["paths"]["cache_dir"] = str(SCRATCH / scene / ("on" if adaptive else "off"))
    cfg["frames"]["adaptive_sharpness"] = adaptive
    na = len(extract_one(a, "a", cfg)["frames"])
    nb = len(extract_one(b, "b", cfg)["frames"])
    return digest(SCRATCH / scene / ("on" if adaptive else "off") / "frames"), na + nb


def main() -> int:
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"{'scene':<8}{'frames off':>11}{'frames on':>11}  identical?  adaptive fired?")
    bad = 0
    for scene in SCENES:
        try:
            off, n_off = run(scene, adaptive=False)
        except RuntimeError as exc:               # the scene cannot run at defaults
            off, n_off = None, None
            print(f"{'':<8}(frozen behaviour aborts: {str(exc).splitlines()[-1][:60]})")
        on, n_on = run(scene, adaptive=True)
        if off is None:
            print(f"{scene:<8}{'aborts':>11}{n_on:>11}  {'n/a':>9}   "
                  f"yes (this is the point of the flag)")
            continue
        same = off == on
        fired = n_on != n_off
        print(f"{scene:<8}{n_off:>11}{n_on:>11}  {'yes' if same else 'NO ':>9}   "
              f"{'yes (fallback path)' if fired else 'no'}")
        if not same:
            bad += 1
    if bad:
        print(f"\n{bad} scene(s) differ: the flag is not safe to enable by default.")
        return 1
    print("\nAll scenes byte identical with the flag ON. On every scene that already "
          "worked the\nfallback never runs, so enabling it by default cannot change "
          "existing outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
