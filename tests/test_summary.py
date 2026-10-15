"""Summary invariants: every report.json change appears exactly once in the
natural-language summary, and nothing extra appears.

Two layers:
  1. a synthetic fixture (always runs; no GPU, no caches): build a fake
     report.json, run src.summary.build_summary with the VLM disabled, and
     check the invariants - including the "No changes found" case and the
     template sentence for a removed change that must not mention a
     replacement;
  2. the committed outputs: for every outputs/*/report.json that has a
     summary.json next to it, check the same invariants against the real data.

Run:  python tests/test_summary.py     (any env with numpy; fast, no GPU)
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import load_config, write_json  # noqa: E402
from src.summary import build_summary, validate_sentence  # noqa: E402

FAILED = 0


def check(name: str, ok: bool) -> None:
    global FAILED
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILED += 1


def invariants(report: dict, summary: dict, label: str) -> None:
    rep_ids = [c["id"] for c in report.get("changes", [])]
    sum_ids = [c["id"] for c in summary.get("changes", [])]
    check(f"{label}: every report change appears exactly once",
          sorted(sum_ids) == sorted(rep_ids) and len(sum_ids) == len(set(sum_ids)))
    check(f"{label}: nothing extra in the summary",
          set(sum_ids) <= set(rep_ids))
    check(f"{label}: n_changes matches",
          summary.get("n_changes") == len(rep_ids))
    # one numbered line per change, no more
    numbered = re.findall(r"^\d+\. ", summary["text"], flags=re.M)
    check(f"{label}: text has exactly {len(rep_ids)} numbered change(s)",
          len(numbered) == len(rep_ids))
    if rep_ids:
        check(f"{label}: header says 'I found N change(s)'",
              f"I found {len(rep_ids)} change(s):" in summary["text"])
        for c in report["changes"]:
            entry = next((s for s in summary["changes"] if s["id"] == c["id"]), None)
            line_ok = (f"{str(c['type']).capitalize()}: {c['label']}." in summary["text"]
                       or f"{str(c['type']).capitalize()} - {c['label']}:" in summary["text"])
            check(f"{label}: change {c['id']} line uses its type and label",
                  entry is not None and line_ok)
    else:
        check(f"{label}: empty report -> 'No changes found'",
              "No changes found." in summary["text"])
    check(f"{label}: notes state the naming caveat",
          any("can be wrong" in n for n in summary.get("notes", [])))
    check(f"{label}: notes state the development-scene tuning caveat",
          any("development scenes" in n for n in summary.get("notes", [])))
    if report.get("unverified_changes"):
        check(f"{label}: notes state the unverified count",
              any(str(len(report["unverified_changes"])) in n
                  for n in summary.get("notes", [])))


def test_synthetic() -> None:
    print("synthetic fixture:")
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        cfg["paths"]["out_dir"] = str(out)
        cfg["paths"]["cache_dir"] = str(out)
        report = {
            "meta": {"video_a": "/x/before.mp4", "video_b": "/x/after.mp4"},
            "changes": [
                {"id": 1, "type": "removed", "label": "comb", "confidence": 0.9,
                 "description": "The comb was removed and replaced by a pen."},
                {"id": 2, "type": "added", "label": "pen", "confidence": 0.3,
                 "description": "A pen appeared."},
            ],
            "unverified_changes": [{"id": 9, "type": "removed", "label": "floor"}],
        }
        write_json(out / "report.json", report)
        summary = build_summary(cfg, use_vlm=False)
        invariants(report, summary, "fixture")
        # the removed sentence must not mention the replacement (scene4 case)
        s1 = next(s for s in summary["changes"] if s["id"] == 1)["sentence"]
        check("fixture: removed sentence does not mention 'replaced'",
              "replac" not in s1.lower())
        s2 = next(s for s in summary["changes"] if s["id"] == 2)["sentence"]
        check("fixture: low-confidence change is phrased 'Possibly'",
              s2.startswith("Possibly"))
        # the tightened validator: embellished wording is rejected
        change = report["changes"][0]
        check("validator: 'leaving only the cracked surface' is rejected",
              not validate_sentence(
                  "The comb was removed, leaving only the cracked surface.", change))
        check("validator: a clean sentence passes",
              validate_sentence("The comb was removed in video B.", change))
        # empty report
        write_json(out / "report.json",
                   {"meta": report["meta"], "changes": [], "unverified_changes": []})
        summary = build_summary(cfg, use_vlm=False)
        invariants({"meta": report["meta"], "changes": [],
                    "unverified_changes": []}, summary, "fixture-empty")


def test_committed_outputs() -> None:
    print("committed outputs:")
    found = 0
    for rep_path in sorted(ROOT.glob("outputs/**/report.json")):
        sum_path = rep_path.parent / "summary.json"
        if not sum_path.exists():
            continue
        found += 1
        name = rep_path.parent.name or "demo"
        invariants(json.loads(rep_path.read_text()),
                   json.loads(sum_path.read_text()), name)
    if not found:
        print("  (no committed summary.json found - run detect.py first)")


def main() -> int:
    test_synthetic()
    test_committed_outputs()
    print()
    if FAILED:
        print(f"{FAILED} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
