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

    # Merge on matching evidence: several A fragments matching several B
    # fragments of the same label are one object. The matcher already knows this,
    # so the merge needs no new threshold: group the pairs by label and require
    # the B side boxes to touch (the same written rule as fragments.py).
    def touch(x1, e1, x2, e2) -> bool:
        """Same written rule as fragments.py: boxes touch or overlap."""
        return float(((x1 - x2) ** 2).sum() ** 0.5) <= 0.5 * (e1 + e2)

    merged_pairs: list[list[tuple[int, int, float]]] = []
    for i, j, cost in pairs:
        placed = False
        for group in merged_pairs:
            gi, gj, _ = group[0]
            if (a[i].label == a[gi].label and b[j].label == b[gj].label
                    and touch(b[j].centroid, b[j].extent, b[gj].centroid, b[gj].extent)
                    and touch(a[i].centroid, a[i].extent, a[gi].centroid, a[gi].extent)):
                group.append((i, j, cost))
                placed = True
                break
        if not placed:
            merged_pairs.append([(i, j, cost)])

    candidates = []
    for group in merged_pairs:
        i0, j0, cost = group[0]
        disps = [float(((a[i].centroid - b[j].centroid) ** 2).sum() ** 0.5) for i, j, _ in group]
        moved = max(disps)
        candidates.append({"type": "moved" if moved > 0.02 else "unchanged",
                           "label": a[i0].label, "cost": round(min(c for _, _, c in group), 3),
                           "displacement": round(moved, 4),
                           "from_cluster": "+".join(a[i].key for i, _, _ in group),
                           "to_cluster": "+".join(b[j].key for _, j, _ in group),
                           "matched_fragments": len(group)})
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
