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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.naming.stats import wilson_pair  # noqa: E402
from src.naming.vocabulary import Vocabulary, load_vocabulary  # noqa: E402
from src.naming.descriptor import describe  # noqa: E402

MANIFEST = ROOT / "data" / "naming_set" / "manifest.json"
_SIGLIP = None
SYNONYMS = ROOT / "data" / "naming_synonyms.json"

# One neutral prompt for every model in the bake off, so the comparison is
# like for like. The answer format is strict JSON because parse failures are
# themselves a result (a model that cannot follow instructions is not a
# naming candidate).
PROMPT = ("Identify the single everyday object in this photograph. "
          "No reasoning and no extra words: reply with one JSON object whose "
          'field name is "name" and whose value is the object.')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face repo id of the challenger")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None, help="score the first N crops (smoke)")
    p.add_argument("--out", default=None)
    p.add_argument("--controls", action="store_true",
                   help="run the visual control (12 crops) and stop")
    p.add_argument("--protocol", choices=["free", "loglik"], default="free",
                   help="free generation, or closed-set scoring by log likelihood")
    p.add_argument("--candidates", choices=["all", "top10"], default="all",
                   help="loglik: score the whole vocabulary or the SigLIP top 10")
    p.add_argument("--vocab", choices=["own301", "public"], default="own301",
                   help="closed-set candidate source; public = LVIS+Objects365+COCO names")
    p.add_argument("--manifest", default=str(MANIFEST),
                   help="crop manifest to score (variants share the code)")
    p.add_argument("--blank-calibration", action="store_true",
                   help="loglik: subtract the score of the same label on a blank image")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="loglik: shuffle the candidate order with this seed")
    p.add_argument("--max-memory", default=None,
                   help="e.g. 40GiB per GPU when --device auto (32B with offload)")
    return p.parse_args()




def stats(c: collections_counter, total: int, name: str) -> dict:
    """One correctness row for the results table."""
    exact = c["exact"]
    parent = c["parent"] + exact
    point_e, lo_e, hi_e = wilson_pair(exact, total)
    point_p, lo_p, hi_p = wilson_pair(parent, total)
    parse_fail = c["unparsable"]
    row = {
        "bucket": name, "n": total,
        "exact": exact, "exact_pct": round(point_e, 1),
        "exact_ci": [round(lo_e, 1), round(hi_e, 1)],
        "exact_or_parent": parent, "exact_or_parent_pct": round(point_p, 1),
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


def load_inputs(manifest: str | None = None) -> list[dict]:
    syn = json.loads(SYNONYMS.read_text())
    crops = json.loads(Path(manifest or MANIFEST).read_text())["crops"]
    rows = []
    for c in crops:
        truth = syn["mapped"].get(c["label"])
        if truth:
            rows.append({"path": c["path"], "truth": truth, "label": c["label"]})
    return rows


def run_controls(model, proc, tok, vocab, inputs, device, n=12):
    """Invariants (b) and (d): the model must react to the IMAGE.

    Real, blank and shuffled inputs must produce different answers; a model that
    answers the same thing for all three is text-only and its scores are
    meaningless (this is exactly what the whitelist bug produced). The known
    answer smoke prints the first few answers so a human can see they are
    plausible object names, not a constant.
    """
    import torch
    from PIL import Image
    import numpy as np
    import inspect
    picks = inputs[:n]
    answers = {"real": [], "blank": [], "shuffled": []}
    rng = np.random.default_rng(0)
    fwd = inspect.signature(model.forward).parameters
    for i, sample in enumerate(picks):
        real = Image.open(abs_path(sample["path"])).convert("RGB")
        others = [Image.open(abs_path(inputs[(i + 1 + k) % len(inputs)]["path"])).convert("RGB")
                  for k in range(2)]
        variants = {"real": real,
                    "blank": Image.new("RGB", real.size, (0, 0, 0)),
                    "shuffled": others[rng.integers(0, len(others))]}
        for name, img in variants.items():
            sess = [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                 {"type": "image"}]}]
            try:
                txt = proc.apply_chat_template(sess, add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False)
            except TypeError:
                txt = proc.apply_chat_template(sess, add_generation_prompt=True, tokenize=False)
            ins = proc(text=[txt], images=[img], return_tensors="pt").to(device)
            ins.pop("mm_token_type_ids", None)
            ins = {k: v for k, v in ins.items() if v is not None and k in fwd or k == "pixel_values"}
            with torch.no_grad():
                out = model.generate(**ins, max_new_tokens=32, do_sample=False)
            answers[name].append(tok.decode(out[0][ins["input_ids"].shape[1]:],
                                            skip_special_tokens=True).strip()[:60])
    print("visual control (invariant b):")
    for k, v in answers.items():
        print(f"  {k:8} first three: {v[:3]}")
    distinct = sum(1 for a, b in zip(answers["real"], answers["blank"]) if a != b)
    print(f"  real differs from blank on {distinct}/{len(picks)} crops "
          f"(must be most of them)")
    print("known answers (invariant d):", [a[:30] for a in answers["real"][:5]])
    return distinct >= max(1, len(picks) // 2)


def score_candidates(model, proc, image, candidates, device, topk=3, blank_image=None):
    """Closed-set protocol: score each candidate by mean token log likelihood.

    Removes parsing as a confound entirely: the model never generates free text,
    it only ranks the given labels. Returns (ranked list of (name, score),
    top1, top3) where top1/top3 are the candidate lists.
    """
    import torch
    prompt_texts = []
    for cand in candidates:
        cont = '{"name": "%s"}' % cand
        sess = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": PROMPT}]}]
        try:
            base = proc.apply_chat_template(sess, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
        except TypeError:
            base = proc.apply_chat_template(sess, tokenize=False,
                                            add_generation_prompt=True)
        prompt_texts.append(base + cont)
    scores = []
    with torch.no_grad():
        for i in range(0, len(prompt_texts), 32):
            batch = prompt_texts[i:i + 32]
            ins = proc(text=batch, images=[image] * len(batch),
                       padding=True, return_tensors="pt").to(model.device)
            ins.pop("mm_token_type_ids", None)
            out = model(**ins)
            logits = out.logits[:, :-1]
            ids = ins["input_ids"][:, 1:]
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            picked = logprobs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            # the continuation is the tail of each sequence; score the last 8
            # tokens, which always cover '{"name": "..."}' for short labels
            for j in range(len(batch)):
                tail = picked[j][-8:].mean().item()
                scores.append(tail)
    if blank_image is not None:
        # Calibrated scoring: log p(label | image) minus log p(label | blank
        # image). The blank pass measures how much the model likes a label with
        # no visual evidence at all; subtracting it removes the text-only prior,
        # which is the same idea as the visual control but per candidate.
        blank_scores = [0.0] * len(candidates)
        blank_texts = []
        for cand in candidates:
            sess = [{"role": "user", "content": [{"type": "image", "image": blank_image},
                                                 {"type": "text", "text": PROMPT}]}]
            try:
                base = proc.apply_chat_template(sess, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
            except TypeError:
                base = proc.apply_chat_template(sess, tokenize=False,
                                                add_generation_prompt=True)
            blank_texts.append(base + '{"name": "%s"}' % cand)
        with torch.no_grad():
            for i in range(0, len(blank_texts), 32):
                batch = blank_texts[i:i + 32]
                ins = proc(text=batch, images=[blank_image] * len(batch),
                           padding=True, return_tensors="pt").to(model.device)
                ins.pop("mm_token_type_ids", None)
                out = model(**ins)
                logprobs = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
                picked = logprobs.gather(-1, ins["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
                for j in range(len(batch)):
                    blank_scores[i + j] = picked[j][-8:].mean().item()
        scores = [r - b for r, b in zip(scores, blank_scores)]
    ranked = sorted(zip(candidates, scores), key=lambda t: -t[1])
    return ranked


def main() -> int:
    global _SIGLIP
    args = parse_args()
    vocab: Vocabulary = load_vocabulary()
    inputs = load_inputs(args.manifest)
    if args.limit:
        inputs = inputs[: args.limit]
    print(f"model {args.model} on {args.device}, "
          f"{len(inputs)} labelled crops")

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except ImportError:                       # older transformers
        from transformers import Qwen3VLForConditionalGeneration as ModelCls

    # The proven recipe from src/describe.py's vlm_answer: the
    # ImageTextToText auto class (NOT AutoModelForCausalLM, which loads the
    # text-only path and then rejects pixel_values at generate time - that was
    # the real cause of the failed 4B chain), dtype=bf16, device_map on the
    # pinned slice, processor-owned chat template.
    t0 = time.time()
    if args.device == "auto":
        import torch as _t
        mm = None
        if args.max_memory:
            mm = {i: args.max_memory for i in range(_t.cuda.device_count())}
        model = ModelCls.from_pretrained(args.model, dtype=torch.bfloat16,
                                         device_map="auto", max_memory=mm,
                                         low_cpu_mem_usage=True)
    else:
        model = ModelCls.from_pretrained(args.model, dtype=torch.bfloat16,
                                         device_map=args.device,
                                         low_cpu_mem_usage=True)
    model.eval()
    proc = AutoProcessor.from_pretrained(args.model)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else None
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.0f}s")

    out_partial = Path(args.out or ROOT / "logs" /
                        f"bakeoff_naming_{Path(args.model).name}.json")
    out_partial.parent.mkdir(parents=True, exist_ok=True)
    if args.controls:
        ok = run_controls(model, proc, tok, vocab, inputs, args.device)
        raise SystemExit(0 if ok else 'visual control failed')
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
            # Canonical call, from the class's own docstring in the installed
            # transformers (models/qwen3_5/modeling_qwen3_5.py): the PIL image
            # goes inside the message and apply_chat_template does the whole
            # job with tokenize=True, return_dict=True. The earlier split
            # text-then-image pass produced vision tensors that generate
            # rejected, which is what the "keyboard" run was hiding.
            if args.protocol == "loglik":
                if args.vocab == "public":
                    public = json.loads((ROOT / "data" / "vocabulary_public.json").read_text())["names"]
                    cands = list(public)
                else:
                    cands = list(vocab.canonical)
                if args.shuffle_seed is not None:
                    import random
                    random.Random(args.shuffle_seed).shuffle(cands)
                if args.candidates == "top10":
                    # SigLIP rerank narrows the closed set to ten plausible
                    # labels; run it on CPU when the VLM owns the GPU.
                    global _SIGLIP
                    if _SIGLIP is None:
                        from src.naming.rerank import RerankModel as _RM
                        dev = "cpu" if args.device != "cpu" else "cpu"
                        _SIGLIP = _RM(vocab, device=dev)
                        _SIGLIP.load()
                    views = _SIGLIP.score_image(abs_path(sample["path"]), top_k=10)
                    if views:
                        cands = [n for n, _ in views]
                blank = (Image.new("RGB", image.size, (0, 0, 0))
                         if args.blank_calibration else None)
                ranked = score_candidates(model, proc, image, cands, args.device,
                                          blank_image=blank)
                name = ranked[0][0]
                top3 = [c for c, _ in ranked[:3]]
                raw_answer = json.dumps({"protocol": "loglik", "ranked": ranked[:5]})
                better = vocab.canonical_of(name)
                truth_canon = vocab.canonical_of(sample["truth"]) or sample["truth"].lower()
                correct = ("exact" if better == truth_canon else
                           "parent" if better and vocab.parent_of(better) == vocab.parent_of(truth_canon) else "wrong")
                counter[correct] += 1
                in_top3 = truth_canon in [vocab.canonical_of(c) or c for c in top3]
                counter["top3"] = counter["top3"] + (1 if in_top3 else 0)
                rows.append({"path": sample["path"], "name": name, "canon": better,
                             "truth": truth_canon, "correct": correct, "top3_hit": in_top3,
                             "protocol": "loglik", "ranked": [c for c, _ in ranked[:5]]})
                if (i % 25) == 0:
                    out_partial.write_text(json.dumps({"model": args.model, "rows": rows, "partial": True}, indent=2))
                continue
            chat_sess = [{"role": "user",
                  "content": [{"type": "image", "image": image},
                               {"type": "text", "text": PROMPT}]}]
            try:
                inputs = proc.apply_chat_template(
                    chat_sess, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                    enable_thinking=False)
            except TypeError:                 # template without the flag
                inputs = proc.apply_chat_template(
                    chat_sess, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt")
            inputs = inputs.to(args.device)
            # Checkpoint processors may emit auxiliary keys their model does not
            # accept. Drop ONLY the known-bad key: a whitelist here silently
            # removed pixel_values/image_grid_thw and the model then answered a
            # constant (the "keyboard" run, docs/negative_results.md), which the
            # invariants below now refuse to let happen silently.
            inputs.pop("mm_token_type_ids", None)
            inputs = {k: v for k, v in inputs.items() if v is not None}
            # Invariant (a): the model must actually receive the image. A run
            # without pixel_values (and image_grid_thw where the checkpoint uses
            # it) is a harness bug, not a model result: fail loudly.
            if "pixel_values" not in inputs or inputs["pixel_values"] is None:
                raise RuntimeError("harness invariant: no pixel_values in the "
                                   "processor output; the model would see text only")
            for gridded in ("image_grid_thw", "image_sizes"):
                if gridded in inputs and inputs[gridded] is None:
                    raise RuntimeError(f"harness invariant: {gridded} is None")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            raw_answer = proc.batch_decode(
                [out[0][inputs["input_ids"].shape[1]:]],
                skip_special_tokens=True)[0].strip()
        except Exception as exc:                     # a broken crop or generate
            row = {"path": sample["path"], "error": str(exc)[:300],
                   "raw": raw_answer[:200], "name": None}
            rows.append(row)
            counter["unparsable"] += 1
            continue
        latencies.append(time.time() - t1)
        if (i % 25) == 0:                        # incremental save: a late crash
            out_partial.write_text(json.dumps({  # must not lose the answers
                "model": args.model, "rows": rows, "partial": True}, indent=2))
        import re
        import json as _json
        text = raw_answer.strip()
        fence = re.search(r"```[a-zA-Z]*\s*(.*?)```", text, re.S)
        if fence:
            text = fence.group(1).strip()
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, dict) and parsed.get("name"):
                name = str(parsed["name"])
        except Exception:
            pass
        if name is None:
            match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
            if match:
                name = match.group(1)
        if name is None:
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
                     "truth": truth_canon, "correct": correct,
                     "protocol": "free", "raw": raw_answer[:200]})

    # Invariant (c): a constant answer over many crops means the model is not
    # seeing the images or the prompt is broken. Stop and report instead of
    # writing a result nobody should believe.
    from collections import Counter
    if rows and all(r.get("error") for r in rows):
        print("EVERY crop errored; first three messages:", file=sys.stderr)
        for r in rows[:3]:
            print("  " + str(r.get("error"))[:200], file=sys.stderr)
        raise SystemExit("harness: no crop succeeded")
    named_counts = Counter((r.get("name") or "") for r in rows if not r.get("error"))
    if len(rows) >= 20 and named_counts:
        top, top_n = named_counts.most_common(1)[0]
        if top and top_n / max(1, len(rows)) > 0.5:
            print(f"DEGENERACY ALARM: {top!r} for {top_n}/{len(rows)} answers; "
                  f"refusing to write a result (harness bug, see docs/debugging.md)",
                  file=sys.stderr)
            raise SystemExit("degenerate answer distribution")

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
    out = out_partial
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2)[:1200])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
