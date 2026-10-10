"""Stage 7 - report (CPU).

    python src/report.py --config configs/default.yaml

Writes outputs/report.json (machine-readable) and outputs/report.md
(human-readable). No numbers are invented: everything comes from the cached
artifacts of the previous stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    add_common_args,
    apply_overrides,
    load_config,
    out_dir,
    repo_path,
    setup_logging,
    stamp,
    write_json,
)
from src.geometry import build_change_list, recon_dir_for  # noqa: E402

log = setup_logging("report")

CONFIDENCE_NOTE = (
    "confidence is a heuristic in [0,1]: cluster size, visibility fraction "
    "(observed-empty ratio from the other video) and - for moved objects - the "
    "DINOv2 crop similarity; it is not a calibrated probability"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    changes = build_change_list(cfg)
    recon_dir = recon_dir_for(cfg)
    recon_meta = {}
    meta_path = recon_dir / "meta.json"
    if meta_path.exists():
        from src.common import read_json
        recon_meta = read_json(meta_path)

    has_semantics = (repo_path(cfg["paths"]["cache_dir"]) / "semantics" / "vlm.json").exists()

    report = {
        "meta": {
            "video_a": str(repo_path(cfg["paths"]["video_a"])),
            "video_b": str(repo_path(cfg["paths"]["video_b"])),
            "backend": recon_meta.get("backend", cfg["reconstruction"]["backend"]),
            "model": recon_meta.get("model"),
            "created": stamp(),
            "config_hash": cfg["_config_hash"],
            "num_frames": recon_meta.get("num_frames"),
            "semantics": "vlm+dinov2" if has_semantics else "not run",
            "confidence_note": CONFIDENCE_NOTE,
        },
        "changes": [
            {
                "id": ch["id"],
                "type": ch["type"],
                "label": ch["label"],
                "description": ch["description"],
                "centroid": ch["centroid"],
                "bbox": ch["bbox_min"] + ch["bbox_max"],
                "from": ch["from"],
                "to": ch["to"],
                "confidence": ch["confidence"],
                "cluster_ids": ch["cluster_ids"],
                "evidence": {
                    "frames_a": [f for f in ch["evidence"]["visible_in"] if f.startswith("a/")],
                    "frames_b": [f for f in ch["evidence"]["visible_in"] if f.startswith("b/")],
                    "confirmed_empty_in": ch["evidence"]["confirmed_empty_in"],
                },
            }
            for ch in changes
        ],
    }

    out = out_dir(cfg)
    write_json(out / "report.json", report)

    counts: dict[str, int] = {}
    for ch in report["changes"]:
        counts[ch["type"]] = counts.get(ch["type"], 0) + 1

    lines = [
        "# Change report",
        "",
        f"- video A: `{report['meta']['video_a']}`",
        f"- video B: `{report['meta']['video_b']}`",
        f"- reconstruction: `{report['meta']['backend']}` ({report['meta']['model']})",
        f"- frames: {report['meta']['num_frames']} | semantics: {report['meta']['semantics']}",
        f"- generated: {report['meta']['created']} | config `{report['meta']['config_hash']}`",
        "",
        f"**{len(report['changes'])} change(s) detected**: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
        "",
        "| # | type | label | centroid (x, y, z) | confidence |",
        "|---|---|---|---|---|",
    ]
    for ch in report["changes"]:
        c = ch["centroid"]
        label = ch["label"] or "_(no label: semantics stage not run)_"
        lines.append(f"| {ch['id']} | {ch['type']} | {label} | "
                     f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) | {ch['confidence']:.2f} |")
    lines += ["", "## Details", ""]
    for ch in report["changes"]:
        lines.append(f"### {ch['id']}. {ch['type'].upper()}"
                     + (f" - {ch['label']}" if ch["label"] else ""))
        if ch["description"]:
            lines.append("")
            lines.append(f"> {ch['description']}")
        if ch["type"] == "moved":
            lines.append("")
            lines.append(f"- from: ({', '.join(f'{v:.2f}' for v in ch['from'])})")
            lines.append(f"- to:   ({', '.join(f'{v:.2f}' for v in ch['to'])})")
        lines.append("")
        lines.append(f"- centroid: ({', '.join(f'{v:.2f}' for v in ch['centroid'])}) | "
                     f"confidence: {ch['confidence']:.2f}")
        lines.append(f"- cluster ids: {ch['cluster_ids']}")
        visible = ch["evidence"]["frames_a"] + ch["evidence"]["frames_b"]
        if visible:
            lines.append(f"- visible in: {', '.join(visible)}")
        if ch["evidence"]["confirmed_empty_in"]:
            top = ch["evidence"]["confirmed_empty_in"][0]
            lines.append(f"- confirmed by: {top.get('file', '?')} "
                         f"({top.get('n_empty', '?')} points seen empty)")
        lines.append("")
    lines += [
        "---",
        "",
        "Notes:",
        f"- {CONFIDENCE_NOTE}.",
        "- change detection is geometric (voxel/DBSCAN) plus an occlusion-aware "
        "visibility check; see docs/pipeline.md for the full description.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")

    log.info("wrote %s and %s (%d change(s))", out / "report.json", out / "report.md",
             len(report["changes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
