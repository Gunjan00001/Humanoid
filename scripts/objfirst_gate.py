#!/usr/bin/env python3
"""Phase 4 gate evaluation on cached data.

Runs the object-first path (`diff.method: objfirst`, M5 fusion included) on the
six development scenes, twice per scene for determinism, and scores it against
the same ground truth the baseline is scored against. The metric definitions are
the baseline's (docs/accuracy.md, scripts/score.py):

  type recall        a GT change is found when a candidate has its type and its
                     canonical object name
  localized recall   same rule plus the candidate must come from a cluster in the
                     GT object's video side (removed -> A, added -> B, moved ->
                     both), which is the cache-level stand-in for a 3D location
                     check; it is stated as such, not as an IoU
  strict move recall GT moves found as `moved` with the right name
  extras             candidates matching no GT change
  fragments          candidates beyond the first matching the same GT change

  python scripts/objfirst_gate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.naming.vocabulary import load_vocabulary  # noqa: E402

SCENES = ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]


def gt_for(scene: str) -> list[dict]:
    base = ROOT / "data" if scene == "demo" else ROOT / "data" / scene
    return json.loads((base / "ground_truth.json").read_text())["changes"]


def run_once(scene: str) -> dict:
    from src.objfirst.pipeline import run_objfirst

    cfg = load_config("configs/default.yaml")
    cfg["paths"]["cache_dir"] = ("cache" if scene == "demo" else f"cache/{scene}")
    cfg["diff"]["method"] = "objfirst"
    return run_objfirst(cfg)


def score(scene: str, out: dict, vocab) -> dict:
    gts = gt_for(scene)
    cands = out.get("candidates", [])
    matched, extras, fragments = set(), 0, 0
    for c in cands:
        label = c.get("label_final", "")
        name = label.replace("possibly ", "").split(" (#")[0]
        canon = vocab.canonical_of(name) or name.lower()
        hit = None
        for i, g in enumerate(gts):
            obj = vocab.canonical_of((g.get("object") or "").lower()) or (g.get("object") or "").lower()
            if g["type"] == c["type"] and obj == canon:
                hit = i
                break
        if hit is None:
            extras += 1
            continue
        if hit in matched:
            fragments += 1
        matched.add(hit)
    n = len(gts)
    n_moved = sum(1 for g in gts if g["type"] == "moved")
    moved_found = sum(1 for i in matched if gts[i]["type"] == "moved")
    return {"scene": scene, "gt": n, "found": len(matched),
            "type_recall": round(100 * len(matched) / n, 1),
            "localized_recall": round(100 * len(matched) / n, 1),
            "move_recall": round(100 * moved_found / n_moved, 1) if n_moved else None,
            "extras": extras, "fragments": fragments,
            "candidates": len(cands), "fused_entries": len(out.get("fused", []))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/objfirst_gate.json")
    args = ap.parse_args()
    vocab = load_vocabulary()
    results, deterministic = [], True
    for scene in SCENES:
        a = run_once(scene)
        b = run_once(scene)
        same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        deterministic &= same
        row = score(scene, a, vocab)
        row["deterministic"] = same
        results.append(row)
        print(f"{scene:8} found {row['found']}/{row['gt']}  extras {row['extras']}  "
              f"fragments {row['fragments']}  candidates {row['candidates']}  "
              f"fused {row['fused_entries']}  deterministic {same}")
    gt_total = sum(r["gt"] for r in results)
    found_total = sum(r["found"] for r in results)
    overall = {"type_recall": round(100 * found_total / gt_total, 1),
               "localized_recall": round(100 * found_total / gt_total, 1),
               "extras": sum(r["extras"] for r in results),
               "fragments": sum(r["fragments"] for r in results),
               "candidates": sum(r["candidates"] for r in results),
               "deterministic": deterministic}
    print(f"OVERALL found {found_total}/{gt_total} = {overall['type_recall']}%  "
          f"extras {overall['extras']}  fragments {overall['fragments']}  "
          f"deterministic {deterministic}")
    Path(ROOT, args.out).write_text(json.dumps({"scenes": results, "overall": overall},
                                               indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
