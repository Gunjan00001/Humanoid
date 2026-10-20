"""M4: match objects between the two videos.

Chosen components, per the stack-selection decision table:
  * appearance: **DINOv3 ViT-L patch descriptor** (the accepted matcher; AUC
    0.855 patch on GT pairs, paired difference patch +0.115 [+0.100, +0.131]);
  * geometry: 3D size ratio and a location prior;
  * labels: a written compatibility rule (equal canonical name is best, same
    parent category next);
  * assignment: Hungarian on a cost that combines the four, so every object is
    used at most once and no greedy ordering bias is introduced.

The patch descriptor is the mean of the patch tokens inside the mask, with the
grid derived from the token count and the image aspect: a rectangular crop does
NOT produce a square patch grid (that mapping cost us two runs), so the grid is
recovered by searching the factor pairs of the token count for the ratio closest
to the crop's own.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DINOV3 = Path.home() / "models" / "dinov3-vitl16"


def _grid(n_tokens: int, height: int, width: int) -> tuple[int, int]:
    """Factor pair of the token count whose ratio best matches the crop."""
    target = height / max(1, width)
    best = (1e9, 1, n_tokens)
    for gh in range(1, n_tokens + 1):
        if n_tokens % gh:
            continue
        gw = n_tokens // gh
        err = abs((gh / gw) - target)
        if err < best[0]:
            best = (err, gh, gw)
    return best[1], best[2]


def embed_patches(paths: list[str], device: str = "cpu") -> dict[str, np.ndarray]:
    """Mean patch descriptor per crop, L2-normalised."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(str(DINOV3))
    model = AutoModel.from_pretrained(str(DINOV3)).to(device).eval()
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB")
            inputs = proc(images=img, return_tensors="pt").to(device)
            h = model(**inputs).last_hidden_state
            patches = h[0, 1:].float().cpu().numpy()
            gh, gw = _grid(patches.shape[0], img.height, img.width)
            vec = patches.reshape(gh, gw, -1).mean(axis=(0, 1))
            out[p] = vec / (np.linalg.norm(vec) + 1e-8)
    return out


@dataclass
class Instance:
    """One object instance, the unit both videos are inventoried into."""

    key: str                 # stable id inside its video, e.g. "a-3"
    label: str               # canonical name from the inventory
    parent: str | None       # parent category, when the label has one
    crop: str                # mask-tight crop path
    centroid: np.ndarray     # 3D centroid in the shared frame
    extent: float            # 3D bounding-diagonal length
    visible_both: bool = True


def _label_cost(a: Instance, b: Instance) -> float:
    if a.label and b.label and a.label == b.label:
        return 0.0
    if a.parent and b.parent and a.parent == b.parent:
        return 0.35
    return 1.0


def score_matrix(a: list[Instance], b: list[Instance], embs: dict[str, np.ndarray]) -> np.ndarray:
    """Cost matrix over four signals; lower is better."""
    m = np.ones((len(a), len(b)), dtype=np.float32)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            app = 1.0 - float(np.dot(embs[ai.crop], embs[bj.crop]))       # appearance
            size = abs(ai.extent - bj.extent) / max(ai.extent, bj.extent, 1e-6)
            loc = float(np.linalg.norm(ai.centroid - bj.centroid)) / max(ai.extent, 0.01)
            m[i, j] = 0.45 * _label_cost(ai, bj) + 0.30 * app + 0.15 * size + 0.10 * min(loc, 1.0)
    return m


def match_instances(a: list[Instance], b: list[Instance], embs: dict[str, np.ndarray],
                    max_cost: float = 0.55) -> list[tuple[int, int, float]]:
    """Hungarian assignment; pairs above `max_cost` are left unmatched."""
    from scipy.optimize import linear_sum_assignment

    if not a or not b:
        return []
    m = score_matrix(a, b, embs)
    rows, cols = linear_sum_assignment(m)
    return [(int(i), int(j), float(m[i, j])) for i, j in zip(rows, cols) if m[i, j] <= max_cost]
