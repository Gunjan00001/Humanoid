#!/usr/bin/env python3
"""Measure the naming tiers on the development scene entries.

The adoption gate in docs/naming_v4.md needs VLM answers. For the 19
development scene entries the answers already exist in
`cache/<scene>/semantics/vlm.json`, the crops are cached, and the ground-truth
objects are in `data/<scene>/ground_truth.json`, all merged by
`src.naming.cache.load_all`. This script runs the tier ladder over those entries
with SigLIP 2 scoring on CPU and reports the per-tier precision against both
truths: the ground-truth objects and the human verdict recorded in
`data/<scene>/matches.json`.

  python scripts/naming_dev_tiers.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.naming import (Outcomes, RerankModel, decide_tier,   # noqa: E402
                        load_all, load_vocabulary)
from src.naming.tiers import tier_of  # noqa: E402

SCORE_CACHE = ROOT / "logs" / "naming_dev_tiers" / "crop_scores.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-votes", type=int, default=2)
    p.add_argument("--out", default="logs/naming_dev_tiers.json")
    return p.parse_args()


def proposal_of(decision, vocab) -> str | None:
    """The object name a tier proposes, or None when it abstains.

    A plain tier proposes the name as printed. A 'possibly X (#N)' tier proposes
    X; the hedge and the stable number are wording, so the object check strips
    them. An unidentified tier proposes nothing: its honesty is that it names
    nothing.
    """
    if decision.tier == "unidentified":
        return None
    name = decision.label.split(" (#", 1)[0]
    return name.replace("possibly ", "", 1) if name.startswith("possibly ") else name


def tier_correct(proposal: str | None, truth: str | None, vocab) -> bool:
    """Is the proposed object the ground-truth object?

    Same canonicalisation as the rest of the naming round, so 'remote' and
    'remote control' count as the same object. An entry with no ground truth (an
    extra) can never be right, which is deliberate: a false positive named
    correctly is still a false positive.
    """
    if proposal is None or truth is None:
        return False
    truth_canon = vocab.canonical_of(truth) or truth.lower()
    return vocab.canonical_of(proposal) == truth_canon


def main() -> int:
    args = parse_args()
    vocab = load_vocabulary()
    entries = load_all()
    if not entries:
        print("no entries found")
        return 1

    model = RerankModel(vocab, device=args.device)
    if not model.load():
        print("SigLIP 2 unavailable")
        return 1

    # Score the crops once and reuse; the reports re-run faster than a full pass.
    if SCORE_CACHE.exists():
        scores = {k: [tuple(x) for x in v] for k, v in json.loads(SCORE_CACHE.read_text()).items()}
        print(f"loaded {len(scores)} cached crop scores from {SCORE_CACHE}")
    else:
        scores = {}
        todo = sorted({p for e in entries for p in e.views})
        for path in todo:
            scores[path] = model.score_image(path, top_k=args.top_k)
        SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SCORE_CACHE.write_text(json.dumps(scores))
        print(f"scored {len(scores)} crops (cached at {SCORE_CACHE})")

    per_tier = collections.defaultdict(lambda: collections.Counter())
    rows = []
    for e in entries:
        views = [scores[p] for p in e.views if p in scores]
        decision = decide_tier(e.vlm_label, views, vocab, index=e.entry_id,
                               min_votes=args.min_votes)
        tier = decision.tier
        proposal = proposal_of(decision, vocab)
        correct = tier_correct(proposal, e.truth, vocab)
        # The human verdict on what the pipeline printed at the time.
        printed_correct = bool(e.name_ok_at_the_time)
        per_tier[tier]["n"] += 1
        per_tier[tier]["correct" if correct else "wrong"] += 1
        rows.append({"scene": e.scene, "entry_id": e.entry_id,
                     "tier": tier, "tier_label": decision.label,
                     "tier_correct": correct,
                     "printed_label": e.printed_label,
                     "printed_correct": printed_correct,
                     "truth": e.truth,
                     "detail": decision.detail})

    total = len(rows)
    printed_correct = sum(1 for r in rows if r["printed_correct"])
    print(f"\n=== tiers on the 19 development scene entries "
          f"(cached VLM labels, SigLIP 2 on {args.device}) ===")
    print(f"{'tier':14}{'entries':>8}{'right':>8}   precision")
    for tier in ("plain", "possibly", "unidentified"):
        c = per_tier.get(tier)
        if not c or not c["n"]:
            print(f"{tier:14}{0:>8}{0:>8}  no entries")
            continue
        precision = 100 * c["correct"] / c["n"]
        print(f"{tier:14}{c['n']:>8}{c['correct']:>8}   {precision:4.1f}%")
    named = per_tier["plain"]["n"] + per_tier["possibly"]["n"]
    named_right = per_tier["plain"]["correct"] + per_tier["possibly"]["correct"]
    print(f"\nas printed by the pipeline: {printed_correct}/{total} labels right "
          f"({100 * printed_correct / total:.1f}%)")
    print(f"tiers named anything: {named}/{total} "
          f"({100 * named / total:.1f}%), right: {named_right}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"entries": total, "rows": rows,
                               "per_tier": {k: dict(v) for k, v in per_tier.items()},
                               "printed_correct": printed_correct}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
