#!/usr/bin/env python3
"""Check the held-out claims in README.md and docs against the real artifacts.

The held-out scenes were run once and their numbers must never be restated from
memory. This script reads the committed reports, ground truth and human-resolved
matches, and fails if the documents disagree with them.

  python scripts/check_heldout_claims.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Facts taken from outputs/<scene>/report.json and data/<scene>/matches.json,
# written down once here so the prose can be checked against them.
EXPECT = {
    "scene5": {
        "gt_types": ["moved", "removed"],
        "found_gt_ids": {2},        # the chair removal, fragmented into 3 entries
        "extras": 1,                # "small round object", false positive
        "move_found": False,        # the clock move is missed
        "missing": 1,
    },
    "scene6": {
        "gt_types": ["moved", "removed", "added"],
        "found_gt_ids": {2},        # the remote removal
        "extras": 1,                # "electrical outlet", false positive
        "move_found": False,        # the cup move is missed
        "missing": 2,               # cup move, apple add
    },
}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def check_artifacts() -> list[str]:
    bad: list[str] = []
    for scene, exp in EXPECT.items():
        r = load(ROOT / "outputs" / scene / "report.json")
        g = load(ROOT / "data" / scene / "ground_truth.json")
        m = load(ROOT / "data" / scene / "matches.json")

        gt_types = sorted({c["type"] for c in g["changes"]})
        if gt_types != sorted(exp["gt_types"]):
            bad.append(f"{scene}: ground truth types {gt_types}, expected {sorted(exp['gt_types'])}")

        found = {e["gt_id"] for e in m["entries"]
                 if e.get("gt_id") is not None and e.get("type_ok")}
        if found != exp["found_gt_ids"]:
            bad.append(f"{scene}: found GT ids {sorted(found)}, expected {sorted(exp['found_gt_ids'])}")

        n_extra = sum(1 for e in m["entries"] if e.get("gt_id") is None)
        if n_extra != exp["extras"]:
            bad.append(f"{scene}: {n_extra} extras, expected {exp['extras']}")

        move = any(e.get("gt_id") is not None and e.get("type_ok") and
                   g["changes"][e["gt_id"] - 1]["type"] == "moved" for e in m["entries"])
        if move != exp["move_found"]:
            bad.append(f"{scene}: move found is {move}, expected {exp['move_found']}")

        n_missing = len(g["changes"]) - len(found)
        if n_missing != exp["missing"]:
            bad.append(f"{scene}: {n_missing} missed, expected {exp['missing']}")

        if any(c["type"] == "moved" for c in r["changes"]) and not exp["move_found"]:
            bad.append(f"{scene}: report now contains a moved entry, update the docs")
    return bad


def check_prose() -> list[str]:
    bad: list[str] = []
    readme = (ROOT / "README.md").read_text()
    for scene, exp in EXPECT.items():
        rows = [l for l in readme.splitlines()
                if l.startswith("|") and re.search(rf"\b{scene}\b", l)]
        if not rows:
            bad.append(f"README: no table row for {scene}")
            continue
        n_found, n_gt = len(exp["found_gt_ids"]), len(exp["gt_types"])
        if f"{n_found}/{n_gt}" not in rows[0]:
            bad.append(f"README row for {scene} does not state {n_found}/{n_gt}: {rows[0].strip()}")
    heldout = (ROOT / "docs" / "heldout.md").read_text()
    for need in ("clock **moved**", "chair", "cup moved", "remote"):
        if need not in heldout:
            bad.append(f"docs/heldout.md lost the phrase {need!r}")
    results = (ROOT / "docs" / "results.md").read_text()
    for need in ("1/2, type only", "**1/3** (remote `removed`)"):
        if need not in results:
            bad.append(f"docs/results.md lost the phrase {need!r}")
    return bad


def main() -> int:
    problems = check_artifacts() + check_prose()
    for p in problems:
        print(f"FAIL {p}")
    if problems:
        print(f"\n{len(problems)} problem(s): fix the documents, do not change the reports.")
        return 1
    print("OK held-out claims in README and docs match the reports, ground truth and matches")
    print("   scene5: chair removal found (fragmented, labels wrong), clock move missed, 1 extra")
    print("   scene6: remote removal found, cup move and apple add missed, 1 extra")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
