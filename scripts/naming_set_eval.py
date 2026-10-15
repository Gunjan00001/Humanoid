#!/usr/bin/env python3
"""Measure naming on the labelled crop set, independently of detection.

Every crop is a ground-truth object, so the right answer is known by
construction. Two signals are measured:

  1. SigLIP 2 rerank alone, no vision-language model needed, which is what the
     tiers lean on: how often does the correct vocabulary name come out first,
     and how often is it in the top three (the tier gate)?
  2. The tiers, using a supplied VLM answer table. If no table is supplied the
     tiers are skipped and the report says so rather than inventing an answer.

  python scripts/naming_set_eval.py                       # rerank measurement
  python scripts/naming_set_eval.py --vlm-answers a.json  # + tier measurement

Output: coverage and precision per bucket, with Wilson 95% intervals, plus a
per-label breakdown so a failure can be traced to one object type.
"""
# EXPERIMENTAL, not used by detect.py. Part of the accuracy and naming
# experiments; kept as evidence. See docs/accuracy.md and docs/naming_v4.md.
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.naming import RerankModel, load_vocabulary           # noqa: E402
from src.naming.tiers import decide_tier, tier_of             # noqa: E402

log = logging.getLogger("namingset.eval")

MANIFEST = ROOT / "data" / "naming_set" / "manifest.json"
SYNONYMS = ROOT / "data" / "naming_synonyms.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a proportion. With small counts the normal
    approximation is not usable, which is exactly this situation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (centre - half), 100 * (centre + half))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--vlm-answers", default=None,
                   help="JSON mapping crop path -> the VLM's free name for that crop")
    p.add_argument("--out", default="logs/naming_set.json")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")

    vocab = load_vocabulary()
    synonyms = json.loads(SYNONYMS.read_text())
    crops = json.loads(MANIFEST.read_text())["crops"]
    log.info("%d crops, %d mapped, %d unmapped labels",
             len(crops), len(synonyms["mapped"]), len(synonyms["unmapped"]))

    model = RerankModel(vocab, device=args.device)
    if not model.load():
        print("SigLIP 2 unavailable, cannot measure the rerank")
        return 1

    vlm_answers = json.loads(Path(args.vlm_answers).read_text()) if args.vlm_answers else {}

    # Buckets: is the correct name rank 1, in the top 3, or absent? The tier gate
    # is "the VLM answer is inside the top 3", so that bucket is the one that
    # decides how often a plain tier is even possible.
    rank_counts = collections.Counter()
    per_label: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    tier_outcomes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rows: list[dict] = []
    skipped = 0

    for crop in crops:
        truth = synonyms["mapped"].get(crop["label"])
        if truth is None:
            skipped += 1                                  # unmapped, counted not dropped
            continue
        ranked = model.score_image(crop["path"], top_k=args.top_k)
        names = [n for n, _ in ranked]
        rank = names.index(truth) + 1 if truth in names else 0
        bucket = "rank1" if rank == 1 else "top3" if 1 <= rank <= 3 else "missed"
        rank_counts[bucket] += 1
        per_label[crop["label"]]["n"] += 1
        per_label[crop["label"]][bucket] += 1

        row = {"path": crop["path"], "label": crop["label"], "truth": truth,
               "rank": rank, "bucket": bucket, "top3": names[:3]}

        vlm_name = vlm_answers.get(crop["path"])
        if vlm_answers:
            decision = decide_tier(vlm_name or "", [ranked], vocab, index=1)
            tier = decision.tier
            correct = decision.label.split(" (#")[0].replace("possibly ", "") == truth
            tier_outcomes[tier]["n"] += 1
            tier_outcomes[tier]["correct" if correct else "wrong"] += 1
            row.update({"vlm": vlm_name, "tier": tier, "tier_label": decision.label,
                        "tier_correct": correct})
        rows.append(row)

    total = sum(rank_counts.values())
    print(f"\n=== SigLIP 2 rerank on {total} labelled crops "
          f"({skipped} skipped, labels not in the vocabulary) ===")
    print(f"{'bucket':10}{'crops':>7}{'share':>9}   Wilson 95%")
    for bucket in ("rank1", "top3", "missed"):
        k = rank_counts[bucket]
        lo, hi = wilson(k, total)
        print(f"{bucket:10}{k:>7}{100 * k / max(1, total):>8.1f}%   [{lo:.1f}%, {hi:.1f}%]")
    top3_total = rank_counts["rank1"] + rank_counts["top3"]
    lo, hi = wilson(top3_total, total)
    print(f"\ncorrect name in the top 3 (the tier gate): {top3_total}/{total} "
          f"= {100 * top3_total / max(1, total):.1f}%  [{lo:.1f}%, {hi:.1f}%]")

    if vlm_answers:
        print("\n=== tiers, per tier ===")
        print(f"{'tier':14}{'crops':>7}{'share':>9}{'correct':>9}   Wilson 95%")
        for tier in ("plain", "possibly", "unidentified"):
            c = tier_outcomes.get(tier)
            if not c or not c["n"]:
                continue
            lo, hi = wilson(c["correct"], c["n"])
            print(f"{tier:14}{c['n']:>7}{100 * c['n'] / max(1, total):>8.1f}%"
                  f"{c['correct']:>9}   [{lo:.1f}%, {hi:.1f}%]")
    else:
        print("\ntiers not measured: no --vlm-answers table supplied. The rerank "
              "numbers above are the part that does not need a VLM.")

    bad = sorted(((lab, c) for lab, c in per_label.items() if c["rank1"] == 0),
                 key=lambda kv: -kv[1]["n"])
    print(f"\nlabels never ranked first ({len(bad)}):")
    for lab, c in bad[:15]:
        print(f"  {lab:26} n={c['n']:<3} top3={c['top3']:<3} missed={c['missed']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"total": total, "skipped_unmapped": skipped,
                               "buckets": dict(rank_counts),
                               "tiers": {k: dict(v) for k, v in tier_outcomes.items()},
                               "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
