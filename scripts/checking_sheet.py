"""Checking sheet for a detection run (v0.12 Step 3).

    python scripts/checking_sheet.py --out outputs/scene5

Builds `checking_sheet.png` (every change's composite stacked, numbered, with
its type/label/confidence header) and `checking_sheet.md` (the same as an
index linking the individual composites), so a human can confirm each
detection against the before/after frames and the 3D box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import repo_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="outputs/<name>")
    args = parser.parse_args()
    out = repo_path(args.out)
    report = json.loads((out / "report.json").read_text())
    changes = report.get("changes", [])

    from PIL import Image, ImageDraw

    tiles = []
    lines = ["# Checking sheet", "",
             "Each tile: before | after crops on top, the 3D view with the "
             "detection box below. Confirm that the box/region shows the right "
             "object.", ""]
    for i, ch in enumerate(changes, 1):
        png = out / "changes" / f"change_{ch['id']:03d}.png"
        if not png.exists():
            continue
        img = Image.open(png).convert("RGB")
        header = (f"{i}. {str(ch['type']).upper()} - {ch.get('label') or '?'} "
                  f"(confidence {ch.get('confidence')})")
        tile = Image.new("RGB", (img.width, img.height + 26), (16, 20, 24))
        tile.paste(img, (0, 26))
        ImageDraw.Draw(tile).text((8, 6), header, fill=(240, 240, 240))
        tiles.append(tile)
        lines.append(f"## {header}")
        lines.append(f"- sentence: {ch.get('description')}")
        lines.append(f"- centroid: {ch.get('centroid')}")
        lines.append(f"- composite: `changes/{png.name}`")
        lines.append("")

    unverified = report.get("unverified_changes", [])
    lines += [f"## Unverified candidates ({len(unverified)}) - not counted as changes", ""]
    for ch in unverified:
        lines.append(f"- {ch['type']}: {ch.get('label')} at {ch.get('centroid')}")
    lines.append("")
    (out / "checking_sheet.md").write_text("\n".join(lines))

    if tiles:
        width = max(t.width for t in tiles)
        sheet = Image.new("RGB", (width, sum(t.height for t in tiles)),
                          (16, 20, 24))
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y))
            y += t.height
        sheet.save(out / "checking_sheet.png")
    print(f"wrote {out}/checking_sheet.png ({len(tiles)} tiles) and "
          f"{out}/checking_sheet.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
