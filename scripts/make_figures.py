#!/usr/bin/env python3
"""Generate the README figures from committed artifacts and measured logs.

Every number is real and traceable:
  fig_results.png    per-scene recall and extras, computed from outputs/*/report.json
                     against data/*/matches.json via scripts/score.py
  fig_timing.png     per-stage wall clock, from the job logs recorded in docs/timing.md
  fig_sprint.png     accuracy-sprint metrics per flag experiment (docs/accuracy.md)
  fig_scenediff.png  SceneDiff validation slice: processable vs unprocessable, and
                     the oracle projection-control IoU per pair (docs/evaluation.md)

  python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DEV = ["demo", "scene2", "scene3", "scene4"]
HELD = ["scene5", "scene6"]


def gt_path(s):
    return ROOT / ("data/ground_truth.json" if s == "demo" else f"data/{s}/ground_truth.json")


def mt_path(s):
    return ROOT / ("data/matches.json" if s == "demo" else f"data/{s}/matches.json")


def fig_results():
    scenes = DEV + HELD
    found, total, extras = [], [], []
    for s in scenes:
        entries = json.loads(mt_path(s).read_text())["entries"]
        gt = json.loads(gt_path(s).read_text())["changes"]
        found.append(len({e["gt_id"] for e in entries
                          if e.get("gt_id") is not None and e.get("type_ok")}))
        total.append(len(gt))
        extras.append(sum(1 for e in entries if e.get("gt_id") is None))
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 0.2, total, 0.4, label="ground-truth changes", color="#b9c6d6")
    ax.bar(x + 0.2, found, 0.4, label="found (correct type)", color="#2f6f4f")
    for i, e in enumerate(extras):
        ax.text(i, max(total[i], found[i]) + 0.12, f"{e} extra", ha="center",
                fontsize=8, color="#a33", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(dev)" if s in DEV else f"{s}\n(held-out)" for s in scenes])
    ax.set_ylabel("changes")
    ax.set_title("Ground-truth changes vs found, and extra entries per scene\n"
                 "development scenes were tuned on; held-out scenes were run once at defaults")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_results.png", dpi=140)
    plt.close(fig)
    print("wrote fig_results.png")


def fig_timing():
    # From docs/timing.md (job logs). scene6 job 10864, scene5 job 10824.
    stages = ["extract", "reconstruct", "align", "diff", "describe", "report", "visualize", "summary"]
    scene6 = [5, 52, 42, 25, 953, 0, 21, 25]      # seconds, job 10864
    scene5 = [3, 23, 24, 33, 234, 0, 105, 0]      # seconds, job 10824 (report+viz+summary ~1:45)
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 0.2, scene6, 0.4, label="scene6 (held-out, 18:53 job)", color="#2f6f4f")
    ax.bar(x + 0.2, scene5, 0.4, label="scene5 (held-out, 7:07 job)", color="#8c6d1f")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=25, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title("Where the time goes per stage (measured job logs)\n"
                 "describe, the Qwen3-VL verification stage, dominates both runs")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_timing.png", dpi=140)
    plt.close(fig)
    print("wrote fig_timing.png")


def fig_sprint():
    exps = ["baseline", "soft_veto", "coverage_added", "both", "objmatch", "fragment_merge"]
    rt = [71, 71, 71, 71, 71, 71]
    loc = [86, 86, 86, 86, 86, 86]
    name = [60, 60, 60, 60, 60, 60]
    move = [25, 25, 25, 25, 25, 25]
    x = np.arange(len(exps))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 1.5 * w, rt, w, label="recall (type)", color="#2f6f4f")
    ax.bar(x - 0.5 * w, loc, w, label="recall (localized)", color="#4f8f6f")
    ax.bar(x + 0.5 * w, move, w, label="move recall (strict)", color="#8c6d1f")
    ax.bar(x + 1.5 * w, name, w, label="name accuracy", color="#5a4f8c")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=20, ha="right")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy sprint: every flag experiment measured the same\n"
                 "6 development scenes; nothing improved and nothing regressed")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_sprint.png", dpi=140)
    plt.close(fig)
    print("wrote fig_sprint.png")


def fig_scenediff():
    pairs = ["airhockeytable", "bathroom_11", "bathroom_1", "bedroom_2_3",
             "bedroom_2_4", "gym", "kitchen_8"]
    iou = [0.195, 0.667, 0.550, 0.645, 0.543, 0.947, 0.761]   # oracle control, docs/evaluation.md
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    a1.bar(["validation\n20 pairs", "test\n10 pairs"], [7, 0], color="#2f6f4f", label="processable")
    a1.bar(["validation\n20 pairs", "test\n10 pairs"], [13, 10], bottom=[7, 0],
           color="#c9c9c9", label="unprocessable")
    a1.set_ylabel("pairs")
    a1.set_title("SceneDiff at the frozen config:\n13/20 validation and 10/10 test pairs fail")
    a1.legend()
    a1.grid(axis="y", alpha=0.25)
    a2.bar(pairs, iou, color="#2f6f4f")
    a2.axhline(0.5, color="#a33", ls="--", lw=1)
    a2.text(0.1, 0.52, "0.5 pass line", color="#a33", fontsize=8)
    a2.set_ylabel("oracle px/im IoU")
    a2.set_ylim(0, 1)
    a2.set_title("Projection control (GT masks through the adapter):\n6 of 7 pairs above 0.5, so the adapter is correct")
    a2.tick_params(axis="x", rotation=30)
    a2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scenediff.png", dpi=140)
    plt.close(fig)
    print("wrote fig_scenediff.png")


if __name__ == "__main__":
    fig_results()
    fig_timing()
    fig_sprint()
    fig_scenediff()
