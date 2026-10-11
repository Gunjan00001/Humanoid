#!/usr/bin/env python3
"""Style pass: replace em dashes and spaced hyphens used as dashes with plain
punctuation, outside fenced code blocks and inline code. Never changes numbers.

  python scripts/humanize.py FILE [FILE...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def humanize(text: str) -> str:
    out, in_fence = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        # protect inline code spans
        parts = re.split(r"(`[^`]*`)", line)
        for i, p in enumerate(parts):
            if p.startswith("`") and p.endswith("`"):
                continue
            p = p.replace(" — ", ", ").replace("—", ", ")
            # a spaced hyphen surrounded by word chars and spaces reads as a dash
            p = re.sub(r"(?<=\w) - (?=\w)", ", ", p)
            parts[i] = p
        out.append("".join(parts))
    return "".join(out)


def main() -> int:
    for f in sys.argv[1:]:
        p = Path(f)
        old = p.read_text()
        new = humanize(old)
        if new != old:
            p.write_text(new)
            print(f"humanized {f}")
        else:
            print(f"unchanged {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
