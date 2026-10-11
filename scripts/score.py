#!/usr/bin/env python3
"""Score a report.json against a ground_truth.json (Phase A).

Matching (transparent, no guessing):
  1. STRONG match  - report entry type == GT type AND the report label contains
     one of the GT synonyms  -> GT found, named correctly.
  2. WEAK  match   - report entry type == GT type but no synonym; assigned only
     when exactly one unassigned GT of that type and one unassigned report entry
     of that type remain -> GT found, name WRONG.
  3. Anything left -> extras (report entries) / missed (GT changes).

Metrics per scene + overall: change recall (type), named recall, move recall,
extras (precision), fragments, name accuracy. Every ambiguous case is listed.

Usage:
  python scripts/score.py                 # all dev scenes
  python scripts/score.py --scene scene6
  python scripts/score.py --report R --gt G
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENES = {
    "demo":   ("data/ground_truth.json",          "outputs/report.json"),
    "scene2": ("data/scene2/ground_truth.json",   "outputs/scene2/report.json"),
    "scene3": ("data/scene3/ground_truth.json",   "outputs/scene3/report.json"),
    "scene4": ("data/scene4/ground_truth.json",   "outputs/scene4/report.json"),
    "scene5": ("data/scene5/ground_truth.json",   "outputs/scene5/report.json"),
    "scene6": ("data/scene6/ground_truth.json",   "outputs/scene6/report.json"),
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def name_hit(label: str, synonyms) -> bool:
    nl = f" {norm(label)} "
    return any(f" {norm(s)} " in nl for s in synonyms)


def score_scene(scene: str, gt_path: Path, report_path: Path) -> dict:
    gt = json.loads(gt_path.read_text())
    gts = [dict(g, _i=i, _entries=[]) for i, g in enumerate(gt["changes"])]
    if not report_path.exists():
        return {"scene": scene, "error": f"missing report {report_path}"}
    rep = json.loads(report_path.read_text())
    reps = [dict(r, _assigned=None) for r in rep.get("changes", [])]
    ambiguous = []

    # 1. strong matches (greedy, in report order)
    for r in reps:
        hits = [g for g in gts if g["type"] == r["type"]
                and name_hit(r.get("label", ""), g["synonyms"])]
        if len(hits) == 1:
            g = hits[0]
            r["_assigned"] = g["_i"]
            g["_entries"].append((r["id"], r.get("label", ""), "named"))
        elif len(hits) > 1:
            ambiguous.append({"scene": scene, "entry": r["id"], "label": r.get("label"),
                              "issue": "matches multiple GT synonyms",
                              "gt": [h["object"] for h in hits]})

    # name matches a GT object but the TYPE differs -> ambiguous, do not guess
    for r in reps:
        if r["_assigned"] is not None:
            continue
        nm = [g for g in gts if name_hit(r.get("label", ""), g["synonyms"])]
        if nm:
            ambiguous.append({"scene": scene, "entry": r["id"], "label": r.get("label"),
                              "issue": f"label names GT object but type '{r['type']}' "
                                       f"!= GT type(s) {[g['type'] for g in nm]}"})

    # 2. weak matches (only when unique both sides)
    for g in gts:
        if g["_entries"]:
            continue
        cand = [r for r in reps if r["_assigned"] is None and r["type"] == g["type"]]
        if len(cand) == 1:
            r = cand[0]
            r["_assigned"] = g["_i"]
            g["_entries"].append((r["id"], r.get("label", ""), "type-only"))
            ambiguous.append({"scene": scene, "entry": r["id"], "label": r.get("label"),
                              "issue": f"type-only match to GT '{g['object']}' "
                                       f"(name not matched)"})

    extras = [r for r in reps if r["_assigned"] is None]
    n_gt = len(gts)
    found = [g for g in gts if g["_entries"]]
    named = [g for g in gts if any(e[2] == "named" for e in g["_entries"])]
    moves = [g for g in gts if g["type"] == "moved"]
    moves_found = [g for g in moves if g["_entries"]]
    fragments = sum(max(0, len(g["_entries"]) - 1) for g in gts)
    assigned_reps = len(reps) - len(extras)
    return {
        "scene": scene,
        "n_gt": n_gt, "n_report": len(reps),
        "found": len(found), "named_found": len(named),
        "recall_type": len(found) / n_gt if n_gt else None,
        "recall_named": len(named) / n_gt if n_gt else None,
        "move_total": len(moves), "move_found": len(moves_found),
        "move_recall": (len(moves_found) / len(moves)) if moves else None,
        "extras": len(extras),
        "precision": (assigned_reps / len(reps)) if reps else None,
        "fragments": fragments,
        "name_accuracy": (len(named) / len(found)) if found else None,
        "per_gt": [{"id": g["id"], "type": g["type"], "object": g["object"],
                    "found": bool(g["_entries"]),
                    "named": any(e[2] == "named" for e in g["_entries"]),
                    "entries": [{"id": e[0], "label": e[1], "how": e[2]} for e in g["_entries"]]}
                   for g in gts],
        "extras_list": [{"id": r["id"], "type": r["type"], "label": r.get("label")} for r in extras],
        "ambiguous": ambiguous,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES), default=None)
    ap.add_argument("--report")
    ap.add_argument("--gt")
    ap.add_argument("--json", help="write full results JSON here")
    args = ap.parse_args()

    if args.report and args.gt:
        items = [("adhoc", Path(args.gt), Path(args.report))]
    elif args.scene:
        gt, rp = SCENES[args.scene]
        items = [(args.scene, ROOT / gt, ROOT / rp)]
    else:
        items = [(s, ROOT / gt, ROOT / rp) for s, (gt, rp) in SCENES.items()]

    results = [score_scene(s, gt, rp) for s, gt, rp in items]

    hdr = f"{'scene':<8} {'GT':>3} {'rep':>3} {'found':>5} {'named':>5} {'recall':>7} {'named':>7} {'move':>6} {'extra':>5} {'frag':>4} {'name%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['scene']:<8} ERROR {r['error']}")
            continue
        fmt = lambda v: "  n/a" if v is None else f"{100*v:5.0f}%"
        move = f"{r['move_found']}/{r['move_total']}" if r["move_total"] else "  -"
        print(f"{r['scene']:<8} {r['n_gt']:>3} {r['n_report']:>3} {r['found']:>5} "
              f"{r['named_found']:>5} {fmt(r['recall_type']):>7} {fmt(r['recall_named']):>7} "
              f"{move:>6} {r['extras']:>5} {r['fragments']:>4} {fmt(r['name_accuracy']):>6}")

    ok = [r for r in results if "error" not in r]
    if ok:
        tot_gt = sum(r["n_gt"] for r in ok)
        tot_found = sum(r["found"] for r in ok)
        tot_named = sum(r["named_found"] for r in ok)
        tot_rep = sum(r["n_report"] for r in ok)
        tot_extra = sum(r["extras"] for r in ok)
        tot_move = sum(r["move_total"] for r in ok)
        tot_move_found = sum(r["move_found"] for r in ok)
        print("-" * len(hdr))
        print(f"{'OVERALL':<8} {tot_gt:>3} {tot_rep:>3} {tot_found:>5} {tot_named:>5} "
              f"{100*tot_found/tot_gt:5.0f}% {100*tot_named/tot_gt:5.0f}% "
              f"{tot_move_found}/{tot_move} {tot_extra:>4} "
              f"{(tot_named/tot_found if tot_found else 0):5.0%}")

    print("\n=== ambiguous / unmatched (resolve by hand, do not guess) ===")
    any_amb = False
    for r in ok:
        for a in r["ambiguous"]:
            any_amb = True
            print(f"  [{a['scene']}] entry {a['entry']} ('{a['label']}') - {a['issue']}")
    if not any_amb:
        print("  (none)")
    print("\n=== extras (report entries matching no GT change) ===")
    for r in ok:
        if r["extras_list"]:
            print(f"  [{r['scene']}] " + "; ".join(f"{e['id']}:{e['type']}:{e['label']}"
                                                    for e in r["extras_list"]))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
