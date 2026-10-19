"""M2 bridge: use SAM 3.1 masks when the sam3-env script has produced them.

The segmenter lives in its own environment (`spatialdiff-sam3`, Python 3.12,
torch cu126), so the pipeline does not import it. Two ways to get masks:

  * run scripts/objfirst_segment.py with that env's python (documented below);
  * call `ensure_masks(scene_dir, env_python)` from any stage that needs them.

Masks are optional by design: without them the object-first path still runs on
the cached boxes (M3), which keeps the pipeline testable on a CPU-only checkout.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def masks_dir(root: Path) -> Path:
    return root / "logs" / "objfirst" / "masks"


def load_masks(scene_dir: Path, root: Path) -> dict[str, tuple[np.ndarray, dict]]:
    """cluster id -> (boolean mask, stats), empty when the segmenter has not run."""
    d = masks_dir(root)
    npz, meta = d / f"{scene_dir.name}.npz", d / f"{scene_dir.name}.json"
    if not npz.exists():
        return {}
    stats = json.loads(meta.read_text()) if meta.exists() else {}
    out = {}
    with np.load(npz) as z:
        for cid in z.files:
            out[cid] = (z[cid].astype(bool), stats.get(cid, {}))
    return out


def ensure_masks(scene_dir: Path, root: Path, env_python: str,
                 scene: str | None = None) -> dict:
    """Run the sam3-env segmenter for one scene, then load what it wrote."""
    if load_masks(scene_dir, root):
        return load_masks(scene_dir, root)
    cmd = [env_python, str(root / "scripts" / "objfirst_segment.py"),
           "--scenes", scene or scene_dir.name]
    subprocess.call(cmd, cwd=root)
    return load_masks(scene_dir, root)
