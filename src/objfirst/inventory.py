"""M1: per-video object inventory, from the pipeline's cached crops and labels.

Why this is legitimate without a GPU: the semantics stage already cut a crop per
cluster and asked the VLM to name it, and both artefacts are cached per scene.
The inventory therefore re-reads them, canonicalises the names through the closed
vocabulary and groups them by video side. A future version can call the VLM
directly on sampled frames; this one deliberately spends no GPU time and is
exactly reproducible from the cache.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.naming.vocabulary import load_vocabulary


def build_inventory(scene_dir: Path) -> dict[str, list[dict]]:
    """{"a": [...], "b": [...]} with one dict per object instance."""
    vocab = load_vocabulary()
    crops = json.loads((scene_dir / "semantics" / "crops.json").read_text())
    vlm = json.loads((scene_dir / "semantics" / "vlm.json").read_text())
    out: dict[str, list[dict]] = {"a": [], "b": []}
    for cid, info in crops.items():
        answer = vlm.get(str(cid), {})
        label = answer.get("label") or ""
        canon = vocab.canonical_of(label) or label.lower()
        side = "b" if info.get("tag") == "b" else "a"
        crop = info.get("crop")
        if not crop or not Path(crop).exists():
            continue
        out[side].append({"cluster_id": int(cid), "label": canon,
                          "raw_label": label, "parent": vocab.parent_of(canon),
                          "crop": crop, "frame": info.get("frame")})
    return out
