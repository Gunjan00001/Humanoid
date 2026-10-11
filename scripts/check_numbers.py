#!/usr/bin/env python3
"""Assert that a text-only change did not alter any number.

For every file that differs between two git revisions (or two paths), extract
all numbers and compare the multisets; exit non-zero on any mismatch.

  python scripts/check_numbers.py <old_ref> <new_ref> [paths...]
  python scripts/check_numbers.py            # defaults: HEAD HEAD~1 docs+README
"""
from __future__ import annotations

import re
import subprocess
import sys

NUM = re.compile(r"\d+(?:\.\d+)?")


def nums(text: str):
    from collections import Counter
    return Counter(NUM.findall(text))


def show(path: str, ref: str) -> str:
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True,
                          text=True).stdout


def main() -> int:
    a = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"
    b = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    changed = subprocess.run(["git", "diff", "--name-only", a, b],
                             capture_output=True, text=True).stdout.split()
    if len(sys.argv) > 3:
        keep = tuple(sys.argv[3:])
        changed = [c for c in changed if c.endswith(keep)]
    bad = 0
    for f in changed:
        if not (f.endswith(".md") or f.endswith(".txt")):
            continue
        na, nb = nums(show(f, a)), nums(show(f, b))
        if na != nb:
            bad += 1
            only_a = na - nb
            only_b = nb - na
            print(f"[NUMBER-CHANGED] {f}: -{dict(only_a)} +{dict(only_b)}")
    if bad:
        print(f"{bad} file(s) changed numbers")
        return 1
    print(f"OK: no numbers changed across {len(changed)} changed text file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
