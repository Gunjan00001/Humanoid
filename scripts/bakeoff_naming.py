#!/usr/bin/env python3
"""Naming bake-off on the labelled crop set, for one model at a time.

Measures what the adoption gate needs (docs/stack_selection.md rule 1): exact
and parent-category accuracy against the ground-truth object of each crop, with
Wilson intervals, plus latency and peak memory. The crop set is the mask-tight
variant produced by scripts/build_naming_set.py (the background outside the
masks is blanked), so the model sees exactly the object.

One model per invocation; the incumbent baseline (Qwen3-VL-32B) is NOT run
here (its answers are already cached in the pipeline's VLM store), so this
script only handles the challengers.

  python scripts/bakeoff_naming.py --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.naming.vocabulary import Vocabulary, load_vocabulary  # noqa: E402
from src.naming.descriptor import describe  # noqa: E402

MANIFEST = ROOT / "data" / "naming_set" / "manifest.json"
SYNONYMS = ROOT / "data" / "naming_synonyms.json"

# One neutral prompt for every model in the bake off, so the comparison is
# like for like. The answer format is strict JSON because parse failures are
# themselves a result (a model that cannot follow instructions is not a
# naming candidate).
PROMPT = ("Name the single everyday object in this image. "
          'Answer with JSON only: {"name": "<lowercase object name>"}')

LABEL = "// the Axelrod recipe for correctness: exact means the canonical name"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face repo id of the challenger")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None, help="score the first N crops (smoke)")
    p.add_argument("--out", default=None)
    return p.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval; the small-count workhorse of this project."""
    if n == 0:
        return (0.0, 0.0)
    p_ = k / n
    d = 1 + z * z / n
    c = (p_ + z * z / (2 * n)) / d
    h = z * math.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def stats(c: collections_counter, total: int, name: str) -> dict:
    """One correctness row for the results table."""
    exact = c["exact"]
    parent = c["parent"] + exact
    lo_e, hi_e = wilson(exact, total)
    lo_p, hi_p = wilson(parent, total)
    parse_fail = c["unparsable"]
    row = {
        "bucket": name, "n": total,
        "exact": exact, "exact_pct": round(100 * exact / max(1, total), 1),
        "exact_ci": [round(lo_e, 1), round(hi_e, 1)],
        "exact_or_parent": parent, "exact_or_parent_pct": round(100 * parent / max(1, total), 1),
        "exact_or_parent_ci": [round(lo_p, 1), round(hi_p, 1)],
        "unparsable": parse_fail,
    }
    return row


class collections_counter(dict):
    """Tiny counter so the script needs no import gymnastics above."""

    def __missing__(self, key):
        return 0

    def __getitem__(self, key):
        return dict.get(self, key, 0)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)

    def __iadd__(self, other):
        return self


def abs_path(rel: str) -> Path:
    """Manifest paths are stored repo-relative; resolve them from ROOT."""
    p_ = Path(rel)
    return p_ if p_.is_absolute() else ROOT / p_


def load_inputs() -> list[dict]:
    syn = json.loads(SYNONYMS.read_text())
    crops = json.loads(MANIFEST.read_text())["crops"]
    rows = []
    for c in crops:
        truth = syn["mapped"].get(c["label"])
        if truth:
            rows.append({"path": c["path"], "truth": truth, "label": c["label"]})
    return rows


def main() -> int:
    args = parse_args()
    vocab: Vocabulary = load_vocabulary()
    inputs = load_inputs()
    if args.limit:
        inputs = inputs[: args.limit]
    print(f"model {args.model} on {args.device}, "
          f"{len(inputs)} labelled crops")

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=args.device)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.0f}s")

    torch.cuda.reset_peak_memory_stats(args.device)
    latencies, rows = [], []
    counter: collections_counter = collections_counter()
    for i, sample in enumerate(inputs, 1):
        t1 = time.time()
        raw_answer = ""
        name = None
        try:
            image = Image.open(abs_path(sample["path"])).convert("RGB")
            # The processor owns the chat template and the vision pipeline for
            # the Qwen3-VL family (the tokenizer alone cannot place pixels):
            # text template first, then the text+image pass below.
            chat_text = proc.apply_chat_template(
                [{"role": "user",
                  "content": [{"type": "text", "text": PROMPT},
                              {"type": "image"}]}],
                add_generation_prompt=True, tokenize=False)
            inputs = proc(text=[chat_text], images=[image],
                          return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            raw_answer = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        except Exception as exc:                     # a broken crop or generate
            row = {"path": sample["path"], "error": str(exc)[:80],
                   "raw": raw_answer[:80], "name": None}
            rows.append(row)
            counter["unparsable"] += 1
            continue
        latencies.append(time.time() - t1)
        import re
        match = re.search(r'"name"\s*:\s*"([^"]+)"', raw_answer)
        if match:
            name = match.group(1)
        else:
            counter["unparsable"] += 1
        better = (vocab.canonical_of(name) if name else None)
        truth_canon = vocab.canonical_of(sample["truth"]) or sample["truth"].lower()
        if better is None:
            row = {"path": sample["path"], "raw": raw_answer[:60],
                   "name": name, "correct": None}
            rows.append(row)
            continue
        parent_e = vocab.parent_of(better)
        parent_t = vocab.parent_of(truth_canon)
        if better == truth_canon:
            counter["exact"] += 1; correct = "exact"
        elif parent_e is not None and parent_e == parent_t:
            counter["parent"] += 1; correct = "parent"
        else:
            counter["wrong"] += 1; correct = "wrong"
        rows.append({"path": sample["path"], "name": name, "canon": better,
                     "truth": truth_canon, "correct": correct})

    peak = torch.cuda.max_memory_allocated(args.device) / 1e9
    table = [
        stats(counter, len(inputs), "all crops"),
    ]
    lat = sorted(latencies)
    result = {
        "model": args.model, "n": len(inputs),
        "load_s": round(load_s, 1),
        "latency_mean_s": round(sum(lat) / max(1, len(lat)), 2),
        "latency_p50_s": round(lat[len(lat) // 2], 2) if lat else None,
        "latency_max_s": round(max(lat), 2) if lat else None,
        "peak_mem_gb": round(peak or float("nan"), 1),
        "correct": dict(counter),
        "table": table, "rows": rows,
    }
    out = Path(args.out or ROOT / "logs" /
               f"bakeoff_naming_{args.model.replace('/', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2)[:1200])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
