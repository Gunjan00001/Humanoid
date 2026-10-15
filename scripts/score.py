#!/usr/bin/env python3
"""Score a report.json against ground_truth.json (+ optional human matches.json).

Metrics (per the sprint definitions):
  1. recall(type)        - share of GT changes with >=1 entry of the CORRECT TYPE
  2. recall(localized)   - share of GT changes with >=1 entry on the RIGHT OBJECT
                           (any type/name)
  3. move recall (strict)- share of GT 'moved' changes with an entry typed 'moved'
  4. name accuracy       - correctly-named entries / localized entries
                           (a part name e.g. "chair leg" is NOT correct)
  5. fragments by location - entries resolved to the same GT beyond the first
  6. precision breakdown  - correct / type-wrong / fragment / extra counts over
                           all report entries

If data/<scene>/matches.json exists it is AUTHORITATIVE (human-resolved):
each entry is {entry_id, gt_id|null, type_ok, name_ok, fragment, evidence}.
Otherwise a transparent name-based auto-match is used (for new/unscored scenes).

Usage:
  python scripts/score.py [--scene NAME] [--report R --gt G] [--json OUT]
"""
# EXPERIMENTAL, not used by detect.py. Part of the accuracy and naming
# experiments; kept as evidence. See docs/accuracy.md and docs/naming_v4.md.
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENES = {
    "demo":   ("data/ground_truth.json",        "data/matches.json",        "outputs/report.json"),
    "scene2": ("data/scene2/ground_truth.json", "data/scene2/matches.json", "outputs/scene2/report.json"),
    "scene3": ("data/scene3/ground_truth.json", "data/scene3/matches.json", "outputs/scene3/report.json"),
    "scene4": ("data/scene4/ground_truth.json", "data/scene4/matches.json", "outputs/scene4/report.json"),
    "scene5": ("data/scene5/ground_truth.json", "data/scene5/matches.json", "outputs/scene5/report.json"),
    "scene6": ("data/scene6/ground_truth.json", "data/scene6/matches.json", "outputs/scene6/report.json"),
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def name_hit(label: str, synonyms) -> bool:
    nl = f" {norm(label)} "
    return any(f" {norm(s)} " in nl for s in synonyms)


def auto_resolve(gts: list, reps: list) -> list:
    """Transparent fallback when no human matches.json exists."""
    resolved = []
    assigned = {r["id"]: None for r in reps}
    first = {}
    # strong
    for r in reps:
        hits = [g for g in gts if g["type"] == r["type"]
                and name_hit(r.get("label", ""), g["synonyms"])]
        if len(hits) == 1:
            assigned[r["id"]] = hits[0]["id"]
    # weak (unique both sides)
    for g in gts:
        if any(assigned[r["id"]] == g["id"] for r in reps):
            continue
        cand = [r for r in reps if assigned[r["id"]] is None and r["type"] == g["type"]]
        if len(cand) == 1:
            assigned[cand[0]["id"]] = g["id"]
    gt_by_id = {g["id"]: g for g in gts}
    for r in reps:
        gid = assigned[r["id"]]
        if gid is None:
            resolved.append({"entry_id": r["id"], "gt_id": None, "kind": "extra",
                             "evidence": "auto: no GT match"})
            continue
        g = gt_by_id[gid]
        type_ok = r["type"] == g["type"]
        name_ok = name_hit(r.get("label", ""), g["synonyms"])
        frag = gid in first
        first.setdefault(gid, r["id"])
        resolved.append({"entry_id": r["id"], "gt_id": gid, "type_ok": type_ok,
                         "name_ok": name_ok, "fragment": frag,
                         "evidence": "auto name-based match"})
    return resolved


def score_scene(scene, gt_path, match_path, report_path) -> dict:
    gt = json.loads(gt_path.read_text())
    gts = gt["changes"]
    if not report_path.exists():
        return {"scene": scene, "error": f"missing report {report_path}"}
    reps = json.loads(report_path.read_text()).get("changes", [])
    rep_by_id = {r["id"]: r for r in reps}
    if match_path and match_path.exists():
        resolved = json.loads(match_path.read_text())["entries"]
        source = "human-resolved"
    else:
        resolved = auto_resolve(gts, reps)
        source = "auto"

    localized = [e for e in resolved if e.get("gt_id") is not None]
    correct = [e for e in resolved if e.get("gt_id") is not None and e.get("type_ok") and not e.get("fragment")]
    type_wrong = [e for e in resolved if e.get("gt_id") is not None and not e.get("type_ok") and not e.get("fragment")]
    frag_entries = [e for e in resolved if e.get("fragment")]
    extras = [e for e in resolved if e.get("gt_id") is None]
    named = [e for e in localized if e.get("name_ok")]

    gt_found_type = {e["gt_id"] for e in resolved if e.get("gt_id") is not None and e.get("type_ok")}
    gt_found_loc = {e["gt_id"] for e in localized}
    moved = [g for g in gts if g["type"] == "moved"]
    moved_found = {e["gt_id"] for e in resolved
                   if e.get("gt_id") is not None and e.get("type_ok")
                   and gt_by_id_type(gts, e["gt_id"]) == "moved"}

    n_gt, n_rep = len(gts), len(reps)
    return {
        "scene": scene, "source": source, "n_gt": n_gt, "n_report": n_rep,
        "recall_type": len(gt_found_type) / n_gt if n_gt else None,
        "recall_localized": len(gt_found_loc) / n_gt if n_gt else None,
        "move_total": len(moved), "move_found": len(moved_found),
        "move_recall": (len(moved_found) / len(moved)) if moved else None,
        "name_accuracy": (len(named) / len(localized)) if localized else None,
        "localized_entries": len(localized), "named_entries": len(named),
        "fragments": len(frag_entries),
        "precision": {"correct": len(correct), "type_wrong": len(type_wrong),
                      "fragment": len(frag_entries), "extra": len(extras),
                      "total": n_rep},
        "extras_list": [{"id": e["entry_id"],
                         "type": rep_by_id.get(e["entry_id"], {}).get("type"),
                         "label": rep_by_id.get(e["entry_id"], {}).get("label"),
                         "evidence": e.get("evidence")} for e in extras],
        "per_gt": [{"id": g["id"], "type": g["type"], "object": g["object"],
                    "found_type": g["id"] in gt_found_type,
                    "localized": g["id"] in gt_found_loc,
                    "entries": [e["entry_id"] for e in localized if e["gt_id"] == g["id"]]}
                   for g in gts],
        "resolved": resolved,
    }


def gt_by_id_type(gts, gid):
    return next((g["type"] for g in gts if g["id"] == gid), None)


def fmt(v):
    return "   n/a" if v is None else f"{100*v:5.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES))
    ap.add_argument("--report")
    ap.add_argument("--gt")
    ap.add_argument("--matches")
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.report and args.gt:
        items = [("adhoc", Path(args.gt), Path(args.matches) if args.matches else None,
                  Path(args.report))]
    elif args.scene:
        gt, mm, rp = SCENES[args.scene]
        items = [(args.scene, ROOT / gt, ROOT / mm, ROOT / rp)]
    else:
        items = [(s, ROOT / gt, ROOT / mm, ROOT / rp) for s, (gt, mm, rp) in SCENES.items()]

    results = [score_scene(s, gt, mm, rp) for s, gt, mm, rp in items]

    hdr = (f"{'scene':<8} {'GT':>3} {'rep':>3} {'recall(t)':>9} {'recall(loc)':>11} "
           f"{'move(strict)':>12} {'name%':>6} {'frag':>4} "
           f"{'correct':>7} {'typeW':>5} {'frag':>4} {'extra':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['scene']:<8} ERROR {r['error']}")
            continue
        mv = f"{r['move_found']}/{r['move_total']}" if r["move_total"] else "-"
        p = r["precision"]
        print(f"{r['scene']:<8} {r['n_gt']:>3} {r['n_report']:>3} "
              f"{fmt(r['recall_type']):>9} {fmt(r['recall_localized']):>11} "
              f"{fmt(r['move_recall']):>12} {fmt(r['name_accuracy']):>6} {r['fragments']:>4} "
              f"{p['correct']:>7} {p['type_wrong']:>5} {p['fragment']:>4} {p['extra']:>5}")

    ok = [r for r in results if "error" not in r]
    if ok:
        tg = sum(r["n_gt"] for r in ok)
        def share(key, numkey):
            n = sum(len({e["gt_id"] for e in r["resolved"] if e.get("gt_id") is not None and e.get(key)})
                    for r in ok)
            return n
        rt = sum(len({e["gt_id"] for e in r["resolved"] if e.get("gt_id") is not None and e.get("type_ok")}) for r in ok)
        rl = sum(len({e["gt_id"] for e in r["resolved"] if e.get("gt_id") is not None}) for r in ok)
        loc_e = sum(r["localized_entries"] for r in ok)
        named_e = sum(r["named_entries"] for r in ok)
        mv_t = sum(r["move_total"] for r in ok)
        mv_f = sum(r["move_found"] for r in ok)
        pc = sum(r["precision"]["correct"] for r in ok)
        pw = sum(r["precision"]["type_wrong"] for r in ok)
        pf = sum(r["precision"]["fragment"] for r in ok)
        pe = sum(r["precision"]["extra"] for r in ok)
        print("-" * len(hdr))
        print(f"{'OVERALL':<8} {tg:>3} {pe+pc+pw+pf:>3} {100*rt/tg:8.0f}% {100*rl/tg:10.0f}% "
              f"{100*mv_f/mv_t if mv_t else 0:11.0f}% {100*named_e/loc_e:5.0f}% "
              f"{pf:>4} {pc:>7} {pw:>5} {pf:>4} {pe:>5}")
        print(f"\nlocalized entries {loc_e}, named {named_e}; "
              f"precision correct {pc}/{pc+pw+pf+pe}, type-wrong {pw}, fragment {pf}, extra {pe}")

    print("\n=== missed GT changes (no entry) ===")
    for r in ok:
        miss = [g for g in r["per_gt"] if not g["localized"]]
        if miss:
            print(f"  [{r['scene']}] " + "; ".join(f"#{g['id']} {g['type']} {g['object']}" for g in miss))
    print("=== GT found by type but NOT named ===")
    for r in ok:
        for g in r["per_gt"]:
            if g["found_type"]:
                es = [e for e in r["resolved"] if e.get("gt_id") == g["id"] and e.get("type_ok")]
                if es and not any(e.get("name_ok") for e in es):
                    print(f"  [{r['scene']}] #{g['id']} {g['object']} (entries {g['entries']})")
    print("=== extras ===")
    for r in ok:
        if r["extras_list"]:
            print(f"  [{r['scene']}] " + "; ".join(
                f"{e['id']}:{e['type']}:{e['label']}" for e in r["extras_list"]))
    print("\n=== fragments (entries beyond the first per GT) ===")
    for r in ok:
        for g in r["per_gt"]:
            if len(g["entries"]) > 1:
                print(f"  [{r['scene']}] GT#{g['id']} {g['object']}: entries {g['entries']}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
