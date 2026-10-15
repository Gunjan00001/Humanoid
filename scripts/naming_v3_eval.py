#!/usr/bin/env python3
"""Evaluate naming with abstention on cached crops and cached VLM answers.

No 32B reruns and no video decoding: it reads what previous runs already wrote,
adds SigLIP 2 scores (CPU by default), and runs the ladder.

  python scripts/naming_v3_eval.py --scenes demo scene5        # one or more scenes
  python scripts/naming_v3_eval.py --loso                      # + leave one scene out
  python scripts/naming_v3_eval.py --debug                     # every rung decision

Outputs a coverage versus precision table and writes the raw rows next to it, so
a disagreement can be traced back to one entry without re-running anything.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.naming import (Outcomes, RerankModel, decide, label_is_correct,  # noqa: E402
                        load_all, load_vocabulary, size_word)
from src.naming.cache import DEFAULT_SCENES  # noqa: E402
from src.naming.evaluation import leave_one_scene_out  # noqa: E402

log = logging.getLogger("naming.eval")

# The knobs the evaluation is allowed to tune, and only globally. Two of them:
# how sure the rerank has to be, and how many views have to agree.
KNOBS = [{"min_sim": s, "min_votes": v}
         for s in (0.05, 0.10, 0.15, 0.20)
         for v in (1, 2)]
SCORE_CACHE = ROOT / "cache" / "naming_v3" / "crop_scores.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    p.add_argument("--device", default="cpu", help="cpu or cuda; CPU is the default")
    p.add_argument("--top-k", type=int, default=5, help="candidates kept per crop")
    p.add_argument("--min-sim", type=float, default=0.10)
    p.add_argument("--min-votes", type=int, default=1)
    p.add_argument("--loso", action="store_true", help="leave one scene out for the knobs")
    p.add_argument("--out", default="logs/naming_v3.json")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def score_crops(entries, model, top_k: int, use_cache: bool = True) -> dict[str, list]:
    """Score every crop once and remember it. Keyed by crop path so a re-run is
    instant, and stored in the repo's cache directory rather than in /tmp."""
    cached: dict[str, list] = {}
    if use_cache and SCORE_CACHE.exists():
        cached = json.loads(SCORE_CACHE.read_text())
        log.info("loaded %d cached crop scores from %s", len(cached), SCORE_CACHE)
    todo = sorted({p for e in entries for p in e.views if p not in cached})
    if todo:
        log.info("scoring %d crops with SigLIP 2 on %s", len(todo), model.device)
    for i, path in enumerate(todo, 1):
        cached[path] = model.score_image(path, top_k=top_k)
        if i % 25 == 0:
            log.info("  scored %d/%d", i, len(todo))
    if todo and use_cache:
        SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SCORE_CACHE.write_text(json.dumps(cached))
        log.info("wrote %s", SCORE_CACHE)
    return cached


def run_ladder(entries, scores: dict[str, list], vocab, knob: dict) -> Outcomes:
    """Apply the ladder to every entry with one global knob setting."""
    out = Outcomes()
    for e in entries:
        views = [scores[p] for p in e.views if scores.get(p)]
        decision = decide(e.vlm_label, views, vocab,
                          index=e.entry_id,
                          size=size_word(None),
                          color=None,
                          anchor=None,
                          min_sim=knob["min_sim"], min_votes=knob["min_votes"])
        correct = label_is_correct(decision.label, decision.kind, e.truth, vocab)
        out.add(kind=decision.kind, correct=correct, label=decision.label,
                truth=e.truth, scene=e.scene, entry_id=e.entry_id,
                detail=decision.detail)
    return out


def baseline(entries, vocab) -> Outcomes:
    """What the pipeline printed at the time, for comparison. Coverage is 100%
    by definition: every entry got a name back then."""
    out = Outcomes()
    for e in entries:
        out.add(kind="specific", correct=bool(e.name_ok_at_the_time),
                label=e.printed_label, truth=e.truth, scene=e.scene,
                entry_id=e.entry_id, detail="label printed by the run under test")
    return out


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")

    vocab = load_vocabulary()
    entries = load_all(args.scenes)
    if not entries:
        print("no entries found for", args.scenes)
        return 1

    # SigLIP 2 is optional: without it the ladder still runs and reports how
    # often it would have had to abstain.
    model = RerankModel(vocab, device=args.device)
    scores = score_crops(entries, model, args.top_k) if model.load() else {}

    base = baseline(entries, vocab)
    knob = {"min_sim": args.min_sim, "min_votes": args.min_votes}
    main_out = run_ladder(entries, scores, vocab, knob)

    print("\n=== naming with abstention, cached crops and cached VLM answers ===")
    print(f"scenes: {', '.join(args.scenes)}   knob: {knob}")
    print(f"baseline (labels as printed)      {base.table()}")
    print(f"naming_v3 (ladder)                {main_out.table()}")
    print("\nkind breakdown of the ladder:")
    print(f"  specific names {main_out.named_specific} (right {main_out.specific_correct})")
    print(f"  categories     {main_out.named_parent} (right {main_out.parent_correct})")
    print(f"  abstentions    {main_out.abstained}")

    result = {"scenes": args.scenes, "knob": knob,
              "baseline": base.summary(), "naming_v3": main_out.summary(),
              "rows": main_out.rows}

    if args.loso:
        loso = leave_one_scene_out(
            [{"scene": e.scene, "entry_id": e.entry_id, "vlm": e.vlm_label,
              "views": e.views, "truth": e.truth} for e in entries],
            KNOBS,
            lambda rows, k: run_ladder(
                [e for e in entries if any(r["entry_id"] == e.entry_id and
                                           r["scene"] == e.scene for r in rows)],
                scores, vocab, k))
        result["loso"] = loso
        print("\nleave one scene out (knobs tuned on the other scenes):")
        for scene, info in loso["per_scene"].items():
            print(f"  held out {scene:8} knob={info['knob']}  named "
                  f"{info['named_specific'] + info['named_parent']:2d}  wrong {info['wrong']}  "
                  f"abstained {info['abstained']}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path} ({len(main_out.rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
