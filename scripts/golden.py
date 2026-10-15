#!/usr/bin/env python3
"""Golden outputs for the summary stage, and how to check them.

Why this exists: every refactor of `src/summary.py` (or anything that feeds it)
must leave the generated sentences byte identical, or the next run would change
the text a reader is shown. The trick is to freeze the current generation once,
commit the result as fixtures, and let the later change prove itself by
regenerating into memory and comparing.

What it records and what it does not:

  * record   rebuilds each scene's summary from its committed report.json on
             CPU (the VLM is never contacted: `use_vlm=False`) and writes the
             result to tests/golden_summary/<scene>/summary.txt
  * check    rebuilds the same way and compares byte for byte, printing a table

The held-out scenes (scene5, scene6) are deliberately out of scope. Their
committed outputs are frozen at v1.0.0 and must never be regenerated, so a
golden check for them would compare the current template against a historical
one and always differ. Their freeze is enforced elsewhere
(`scripts/check_heldout_claims.py`, the digest table in docs/accuracy.md).

  python scripts/golden.py record
  python scripts/golden.py check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import load_config  # noqa: E402
from src.summary import build_summary  # noqa: E402

FIXTURES = ROOT / "tests" / "golden_summary"
# Development scenes only: their committed summaries were produced by the
# current templates, so a regeneration must match byte for byte.
DEFAULT_SCENES = ["demo", "scene2", "scene3", "scene4"]


# The demo is the odd one out: its run used the default output directory, so
# its report lives at outputs/report.json rather than outputs/demo/report.json.
OUTPUT_DIR = {"demo": str(ROOT / "outputs")}


def regenerate(scene: str) -> str:
    """Rebuild one scene's summary text from its committed report.json."""
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["paths"]["out_dir"] = OUTPUT_DIR.get(scene, str(ROOT / "outputs" / scene))
    summary = build_summary(cfg, use_vlm=False)
    return summary["text"] + "\n"


def record(scenes: list[str]) -> int:
    for scene in scenes:
        text = regenerate(scene)
        target = FIXTURES / scene / "summary.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"recorded {scene}: {len(text)} bytes")
    return 0


def check(scenes: list[str]) -> int:
    bad = []
    print(f"{'scene':10}{'bytes':>8}  identical?")
    for scene in scenes:
        fixture = FIXTURES / scene / "summary.txt"
        if not fixture.exists():
            print(f"{scene:10}{'-':>8}  NO FIXTURE (record first)")
            bad.append(scene)
            continue
        current = regenerate(scene)
        same = current == fixture.read_text()
        print(f"{scene:10}{len(current):>8}  {same}")
        if not same:
            bad.append(scene)
    if bad:
        print(f"\n{len(bad)} scene(s) differ: regenerate with --diff, or stop the "
              f"refactor that changed them.")
        return 1
    print("\nall golden summaries match the current generation")
    return 0


def show_diff(scenes: list[str]) -> int:
    """Print the first differing lines, so a change can be understood in place."""
    import difflib
    for scene in scenes:
        fixture = FIXTURES / scene / "summary.txt"
        if not fixture.exists():
            continue
        current = regenerate(scene)
        if current == fixture.read_text():
            continue
        print(f"=== {scene} ===")
        for line in difflib.unified_diff(fixture.read_text().splitlines(),
                                         current.splitlines(),
                                         "golden", "current", lineterm=""):
            print(line)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["record", "check", "diff"])
    p.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    args = p.parse_args()
    return {"record": record, "check": check, "diff": show_diff}[args.command](args.scenes)


if __name__ == "__main__":
    raise SystemExit(main())
