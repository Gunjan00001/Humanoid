"""Object-first spine: inventory -> lift -> match -> candidates (M1 to M4).

Runs only when the config says `diff.method: objfirst`; the default path never
imports this module's heavy dependencies. Fusion with the existing detector (M5)
is deliberately the last step and not implemented here yet: the candidates this
module writes carry the same fields as the detector's entries, so the documented
fusion rule (cluster overlap + type vote) can be applied without either side
changing the other's data.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.objfirst.inventory import build_inventory
from src.objfirst.fragments import merge_fragments
from src.objfirst.lift import lift
from src.objfirst.match import Instance, embed_patches, match_instances


def run_objfirst(cfg: dict) -> dict:
    """Produce the object-first candidate list for one scene."""
    if cfg["diff"].get("method") != "objfirst":
        raise RuntimeError("run_objfirst called without diff.method=objfirst")
    scene_dir = Path(cfg["paths"]["cache_dir"]).resolve()
    inv = lift(build_inventory(scene_dir), scene_dir)
    inv = {side: merge_fragments(items) for side, items in inv.items()}

    def as_instances(side: str) -> list[Instance]:
        return [Instance(key=f"{side}-{it['cluster_id']}", label=it["label"],
                         parent=it["parent"], crop=it["crop"],
                         centroid=it["centroid"], extent=it["extent"])
                for it in inv[side]]

    a, b = as_instances("a"), as_instances("b")
    crops = [i.crop for i in a + b]
    embs = embed_patches(crops) if crops else {}
    pairs = match_instances(a, b, embs)

    matched_a = {i for i, _, _ in pairs}
    matched_b = {j for _, j, _ in pairs}
    candidates = []
    for i, j, cost in pairs:
        moved = float(((a[i].centroid - b[j].centroid) ** 2).sum() ** 0.5)
        candidates.append({"type": "moved" if moved > 0.02 else "unchanged",
                           "label": a[i].label, "cost": round(cost, 3),
                           "displacement": round(moved, 4),
                           "from_cluster": a[i].key, "to_cluster": b[j].key})
    for i, inst in enumerate(a):
        if i not in matched_a:
            candidates.append({"type": "removed", "label": inst.label,
                               "from_cluster": inst.key})
    for j, inst in enumerate(b):
        if j not in matched_b:
            candidates.append({"type": "added", "label": inst.label,
                               "to_cluster": inst.key})
    out = {"scene": scene_dir.name, "inventory": {k: len(v) for k, v in inv.items()},
           "instances": {"a": len(a), "b": len(b)}, "matched_pairs": len(pairs),
           "candidates": candidates}
    dest = Path(cfg["paths"]["cache_dir"]).parent.parent / "logs" / "objfirst"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{scene_dir.name}.json").write_text(json.dumps(out, indent=2))
    return out
