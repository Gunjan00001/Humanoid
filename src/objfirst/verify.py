"""M5a: verify object-first candidates and give them tiers and stable numbers.

Three signals, all from caches, each with an explicit "unverified" state so a
missing artefact can never masquerade as a pass:

  * VLM confirmation: the semantics cache stores, per cluster, whether the model
    accepted the change (`changed`). A candidate whose only evidence clusters
    were rejected is dropped; a candidate with no confirmation is marked
    "unverified" and kept, never silently counted as a finding.
  * visibility: the diff cache stores per-candidate diagnostics. When a cluster
    id is present there, its visibility flag is attached; when it is missing the
    candidate says "visibility unverified" instead of assuming anything.
  * tiers and numbering: a specific name when the label is in the closed
    vocabulary, otherwise "possibly <label>", otherwise an honest descriptor with
    its stable index. The number is assigned once, here, and is the number every
    downstream artefact prints.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.naming.descriptor import describe
from src.naming.vocabulary import load_vocabulary


def _cluster_ids(candidate: dict) -> list[int]:
    ids = []
    for key in ("from_cluster", "to_cluster"):
        raw = (candidate.get(key) or "").split("+")
        for part in raw:
            if "-" in part:
                try:
                    ids.append(int(part.split("-", 1)[1]))
                except ValueError:
                    pass
    return ids


def verify(candidates: list[dict], scene_dir: Path) -> list[dict]:
    vocab = load_vocabulary()
    vlm_path = scene_dir / "semantics" / "vlm.json"
    vlm = json.loads(vlm_path.read_text()) if vlm_path.exists() else {}
    vis_path = scene_dir / "diff" / "visibility.json"
    vis = json.loads(vis_path.read_text()) if vis_path.exists() else {}
    vis_ids = {str(k) for k in (vis.get("candidates", vis) if isinstance(vis, dict) else {})}

    ordered = sorted(candidates, key=lambda c: (c.get("type", ""), c.get("label", "")))
    out = []
    for c in ordered:
        ids = _cluster_ids(c)
        confirmations = [(vlm.get(str(i)) or {}).get("changed") for i in ids]
        known = [x for x in confirmations if x is not None]
        if known and not any(known):
            continue                      # every cluster rejected by the VLM
        visibility = "verified" if any(str(i) in vis_ids for i in ids) else "visibility unverified"
        number = len(out) + 1          # numbering must have no gaps
        canonical = vocab.canonical_of(c.get("label"))
        if canonical:
            label, tier = canonical, "plain"
        elif c.get("label"):
            label, tier = f"possibly {c['label']} (#{number})", "possibly"
        else:
            label, tier = describe(number), "unidentified"
        out.append({**c, "number": number, "label_final": label, "tier": tier,
                    "vlm_confirmed": bool(known and any(known)),
                    "visibility": visibility})
    return out
