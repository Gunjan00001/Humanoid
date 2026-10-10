"""Prepare a real-image GPU smoke test for src/reconstruct.py.

Fetches the VGGT repository's example kitchen frames (from the official
facebookresearch/vggt repo - not redistributed here) and lays them out as a
pseudo "video A / video B" frame cache for `src/reconstruct.py`.

The scene is static, so the expected diff result is *no* changes: this is a
null test that exercises the whole GPU path (model load, preprocessing,
inference, artifact saving) with real photographs.

    python tests/prepare_gpu_smoke.py
    python src/reconstruct.py --config configs/default.yaml --cache cache/gpu_smoke
    python src/diff.py        --config configs/default.yaml --cache cache/gpu_smoke

Notes: raw.githubusercontent.com is often blocked/reset on this cluster, so
the repo is fetched as a codeload tarball and only the kitchen images are
extracted, into cache/ (gitignored). The frames are sample data from the VGGT
repo, NOT part of this project's results.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import stamp  # noqa: E402

TARBALL_URL = ("https://codeload.github.com/facebookresearch/vggt/tar.gz/"
               "refs/heads/main")
REPO_DIR = ROOT / "cache" / "vggt_repo"
LOCAL_IMAGES = REPO_DIR / "vggt-main" / "examples" / "kitchen" / "images"
PICK_A = ["00.png", "04.png", "08.png"]   # every 4th frame, split across "videos"
PICK_B = ["02.png", "06.png", "10.png"]
OUT = ROOT / "cache" / "gpu_smoke" / "frames"


def ensure_repo_images() -> None:
    if LOCAL_IMAGES.is_dir() and any(LOCAL_IMAGES.iterdir()):
        return
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    cmd = (f"curl -sL --retry 3 -m 240 '{TARBALL_URL}' | "
           f"tar -xz -C '{REPO_DIR}' --wildcards '*/examples/kitchen/images/*'")
    print(f"fetching VGGT example images via codeload tarball ...")
    if subprocess.run(cmd, shell=True).returncode != 0 or not LOCAL_IMAGES.is_dir():
        raise RuntimeError(
            "could not fetch the VGGT example images. Alternatives: copy the "
            "examples/kitchen/images folder from a machine that has them into "
            f"{LOCAL_IMAGES}."
        )


def main() -> int:
    ensure_repo_images()
    index = {"created": stamp(), "config_hash": "gpu-smoke",
             "note": "VGGT repo kitchen example frames, static scene (null test)",
             "videos": {}}
    for tag, names in (("a", PICK_A), ("b", PICK_B)):
        tag_dir = OUT / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for n, name in enumerate(names):
            local = f"frame_{n:05d}.png"
            dest = tag_dir / local
            if not dest.exists():
                shutil.copy(LOCAL_IMAGES / name, dest)
            frames.append({"file": f"{tag}/{local}", "source_frame": n,
                           "time_s": float(n), "blur": 0.0})
        index["videos"][tag] = {"path": f"vggt-examples/kitchen/{tag}", "src_fps": 30.0,
                                "candidates": len(names), "frames": frames}
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {OUT/'index.json'} (3 + 3 frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
