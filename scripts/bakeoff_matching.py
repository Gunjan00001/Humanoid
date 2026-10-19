#!/usr/bin/env python3
"""Bake-off (c): DINOv3 ViT-L versus DINOv2-large on the project's own pairs.

Control design (no new labels needed): every cluster in a scene's semantics
cache has a crop in one video and a crop of the SAME object in the other video.
Those two crops are a POSITIVE pair (the same object from two viewpoints); two
crops from different clusters are a NEGATIVE pair. A matcher is useful when
positives score clearly above negatives, so the reported number is the
separation (mean positive similarity minus mean negative similarity) plus the
two means, for CLS and for mean patch tokens.

Everything runs on CPU from the local weights, so it costs no GPU time.

  python scripts/bakeoff_matching.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = {
    "DINOv2-large": Path.home() / "models" / "dinov2-large",
    "DINOv3-ViT-L": Path.home() / "models" / "dinov3-vitl16",
}
SCENES = ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]


def clip_roots() -> dict[str, Path]:
    """Where each scene's semantics cache lives (the demo is the odd one out)."""
    roots = {}
    for s in SCENES:
        for candidate in (ROOT / "cache" / s, ROOT / "cache" / "da3_run_tab"):
            if (candidate / "semantics" / "crops.json").exists():
                roots[s] = candidate
                break
    return roots


def build_pairs(limit_pos: int = 30, limit_neg: int = 30, seed: int = 0):
    """(positive, negative) crop-path pairs from the semantics caches."""
    rng = random.Random(seed)
    singles: list[str] = []
    positives: list[tuple[str, str]] = []
    for scene, root in clip_roots().items():
        crops = json.loads((root / "semantics" / "crops.json").read_text())
        for cid, info in crops.items():
            a, b = info.get("crop"), info.get("crop_other")
            if a and b and Path(a).exists() and Path(b).exists():
                positives.append((a, b))
            if a and Path(a).exists():
                singles.append(a)
    positives = positives[:limit_pos]
    negatives = []
    while len(negatives) < limit_neg and len(singles) > 1:
        x, y = rng.sample(singles, 2)
        negatives.append((x, y))
    return positives, negatives


def embed(path: str, model, processor, device: str = "cpu"):
    """(CLS vector, mean patch vector) for one crop, L2-normalised."""
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    h = out.last_hidden_state                      # [1, 1 + patches, dim]
    cls = torch.nn.functional.normalize(h[:, 0], dim=-1)[0]
    patches = torch.nn.functional.normalize(h[:, 1:], dim=-1).mean(dim=1)[0]
    return cls.cpu().numpy(), patches.cpu().numpy()


def run_model(name: str, path: Path, positives, negatives) -> dict:
    from transformers import AutoImageProcessor, AutoModel

    print(f"--- {name} ({'local weights' if path.exists() else 'MISSING'})")
    processor = AutoImageProcessor.from_pretrained(str(path))
    model = AutoModel.from_pretrained(str(path)).eval()
    cache: dict[str, tuple] = {}

    def vec(p):
        if p not in cache:
            cache[p] = embed(p, model, processor)
        return cache[p]

    pos_cls, pos_patch, neg_cls, neg_patch = [], [], [], []
    for a, b in positives:
        ca, pa = vec(a), vec(b)
        pos_cls.append(float(np.dot(ca[0], pa[0])))
        pos_patch.append(float(np.dot(ca[1], pa[1])))
    for a, b in negatives:
        ca, pa = vec(a), vec(b)
        neg_cls.append(float(np.dot(ca[0], pa[0])))
        neg_patch.append(float(np.dot(ca[1], pa[1])))
    out = {
        "model": name, "positive_pairs": len(pos_cls), "negative_pairs": len(neg_cls),
        "cls_positive": round(float(np.mean(pos_cls)), 3),
        "cls_negative": round(float(np.mean(neg_cls)), 3),
        "cls_separation": round(float(np.mean(pos_cls) - np.mean(neg_cls)), 3),
        "patch_positive": round(float(np.mean(pos_patch)), 3),
        "patch_negative": round(float(np.mean(neg_patch)), 3),
        "patch_separation": round(float(np.mean(pos_patch) - np.mean(neg_patch)), 3),
    }
    print(f"    CLS   pos {out['cls_positive']}  neg {out['cls_negative']}  "
          f"separation {out['cls_separation']}")
    print(f"    patch pos {out['patch_positive']}  neg {out['patch_negative']}  "
          f"separation {out['patch_separation']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/bakeoff_matching.json")
    args = ap.parse_args()

    positives, negatives = build_pairs()
    print(f"pairs: {len(positives)} positive (same object, two videos), "
          f"{len(negatives)} negative (different clusters)")
    if not positives:
        print("no positive pairs found: are the semantics caches present?")
        return 1
    results = []
    for name, path in MODELS.items():
        results.append(run_model(name, path, positives, negatives))
    out = ROOT / args.out
    out.write_text(json.dumps({"positives": len(positives), "negatives": len(negatives),
                               "models": results}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
