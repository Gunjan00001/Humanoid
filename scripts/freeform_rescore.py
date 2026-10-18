#!/usr/bin/env python3
"""Rescore free-form answers with the normaliser, and report both denominators.

Also produces:
  * a per-class breakdown and the COMMON class group (the classes with at least
    10 crops in the labelled set, so a per-class number is not a coin flip);
  * the borderline list for manual review, marked as agent reviewed pending user
    confirmation.

  python scripts/freeform_rescore.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.naming.freeform import FreeformScorer, normalize, load_synonyms  # noqa: E402
from src.naming.stats import wilson_pair  # noqa: E402
from src.naming.vocabulary import load_vocabulary  # noqa: E402

MODELS = ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B"]
COMMON_MIN_CROPS = 10


def siglip_text_embedder(batch_size: int = 64):
    """Text encoder from the local SigLIP 2 snapshot, on CPU, cached."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = Path.home() / "models" / "siglip2-base"
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModel.from_pretrained(str(path)).eval()
    cache: dict[str, list[float]] = {}

    def embed(text: str):
        if text in cache:
            return cache[text]
        with torch.no_grad():
            inputs = tok(text=[text], padding="max_length",
                         return_tensors="pt")
            feats = model.get_text_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats, dim=-1)
        vec = feats.squeeze(0).tolist()
        cache[text] = vec
        return vec

    return embed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/bakeoff_freeform_normalized.json")
    ap.add_argument("--review", default="docs/needs_review_freeform.md")
    args = ap.parse_args()

    vocab = load_vocabulary()
    scorer = FreeformScorer(vocab, synonyms=load_synonyms(),
                            embedder=siglip_text_embedder())
    per_model, review_rows = {}, []

    for model in MODELS:
        f = ROOT / f"logs/bakeoff_naming_{model}.json"
        if not f.exists():
            print(f"{model}: no result file, skipped")
            continue
        rows = json.loads(f.read_text())["rows"]
        n_all = len(rows)
        counts = collections.Counter()
        per_class: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for r in rows:
            if r.get("error"):
                counts["runtime error"] += 1
                continue
            truth = r.get("truth") or ""
            if not truth:
                counts["no ground truth (extra)"] += 1
                continue
            answer = r.get("name") or ""
            parent = vocab.printed_parent(vocab.canonical_of(truth) or truth)
            verdict = scorer.score(answer, vocab.canonical_of(truth) or truth, parent)
            counts["correct (normalized)" if verdict.correct else "wrong (normalized)"] += 1
            counts[f"via {verdict.step}"] += 1
            bucket = per_class[truth]
            bucket["n"] += 1
            bucket["correct"] += 1 if verdict.correct else 0
            if verdict.step in ("borderline", "no") and verdict.detail.startswith("nearest"):
                review_rows.append({"model": model, "answer": answer, "truth": truth,
                                    "detail": verdict.detail,
                                    "distance": abs(float(verdict.detail.split(" at ")[1].split(",")[0]) - scorer.threshold),
                                    "decision": "counted wrong (below threshold)"})

        correct = counts["correct (normalized)"]
        point, lo, hi = wilson_pair(correct, n_all)
        result = {"model": model, "n_all": n_all, "correct": correct,
                  "accuracy_pct": round(point, 1), "ci": [round(lo, 1), round(hi, 1)],
                  "via": {k[4:]: v for k, v in counts.items() if k.startswith("via ")},
                  "buckets": dict(counts),
                  "per_class_common": {
                      k: {"n": v["n"], "correct": v["correct"],
                          "pct": round(100 * v["correct"] / v["n"], 1)}
                      for k, v in sorted(per_class.items()) if v["n"] >= COMMON_MIN_CROPS},
                  "per_class_all": {k: {"n": v["n"], "correct": v["correct"]}
                                    for k, v in sorted(per_class.items())}}
        common_n = sum(v["n"] for v in result["per_class_common"].values())
        common_c = sum(v["correct"] for v in result["per_class_common"].values())
        cp, clo, chi = wilson_pair(common_c, common_n)
        result["common_group"] = {"n": common_n, "correct": common_c,
                                  "pct": round(cp, 1), "ci": [round(clo, 1), round(chi, 1)],
                                  "definition": f"classes with at least {COMMON_MIN_CROPS} crops"}
        per_model[model] = result
        print(f"{model}: normalized free-form {correct}/{n_all} = {point:.1f}% "
              f"[{lo:.1f}, {hi:.1f}] | via {result['via']} | COMMON group "
              f"{common_c}/{common_n} = {cp:.1f}% [{clo:.1f}, {chi:.1f}]")

    out = ROOT / args.out
    out.write_text(json.dumps({"models": per_model, "borderline": scorer.borderline[:200]},
                              indent=2))
    print(f"wrote {out}")

    review = ROOT / args.review
    lines = ["# Free-form borderline decisions (agent reviewed, pending user confirmation)", "",
             f"Threshold {scorer.threshold}, margin {scorer.margin}; answers whose best match landed",
             "within the margin of the threshold were counted as wrong and listed here for a human",
             "decision. Each row is one crop answer.", "",
             "| # | model | answer | ground truth | similarity detail | agent decision |",
             "|---|---|---|---|---|---|"]
    review_rows.sort(key=lambda r: r.get("distance", 9.9))
    review_rows = review_rows[:50]
    for i, r in enumerate(review_rows, 1):
        lines.append(f"| {i} | {r['model']} | {r['answer']} | {r['truth']} | {r['detail']} | {r['decision']} |")
    lines.append("")
    lines.append(f"Total listed: {len(review_rows)}; full list in logs/bakeoff_freeform_normalized.json.")
    review.write_text("\n".join(lines) + "\n")
    print(f"wrote {review} ({len(review_rows)} rows, agent reviewed pending confirmation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
