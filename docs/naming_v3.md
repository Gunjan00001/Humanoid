# Naming with abstention (`semantics.naming_v3`)

Flag status: **OFF**. It improves the wrong-name rate but the leave-one-scene-out
result collapses coverage, so it does not pass the gate yet.

## What it does

The ladder in `src/naming/ladder.py`, applied to every report entry:

1. print a **specific name** only when the vision-language model name is in the
   closed vocabulary and the SigLIP 2 rerank of the cached crops agrees on
   enough views;
2. otherwise print the **parent category** when the candidates share one;
3. otherwise print an **honest descriptor**, for example
   `unidentified small dark object (#2) near the cup`;
4. **part words are never names.** "leg", "handle", "top" and 43 other part words
   are rejected at every rung, which is what produced "wooden leg" for a whole
   chair before.

The vocabulary is one global file, `data/naming_vocabulary.json`: 301 canonical
objects, 230 accepted aliases, 14 categories, no scene specific entries. The
number in `(#N)` is assigned once and stored in the report, so report.json, the
viewer, the GIF and the summary print the same string.

Module layout, one responsibility each: `text.py`, `vocabulary.py`,
`descriptor.py`, `ladder.py`, `rerank.py`, `cache.py`, `evaluation.py`, with
`scripts/naming_v3_eval.py` driving them.

## Method

Evaluated on cached crops and cached VLM answers only, no 32B reruns and no video
decoding. 19 entries across the six scenes, SigLIP 2 base on CPU, correctness
decided against the ground-truth object of each entry (extras can never be
right). Counts include entries with no cached crops; those abstain.

## Coverage versus precision

| configuration | named | coverage | name precision | wrong names | abstained |
|---|---|---|---|---|---|
| labels as printed by the pipeline | 19/19 | **100%** | 47.4% | **10 (52.6%)** | 0 |
| naming_v3, knob sim 0.10, votes 1 | 6/19 | 31.6% | **66.7%** | **3 (15.8%)** | 13 (68.4%) |
| naming_v3, knob sim 0.15, votes 1 | 4/19 | 21.1% | 66.7% | 1 (5.3%) | 15 (78.9%) |
| naming_v3, knob sim 0.20, votes 1 | 0/19 | 0.0% | not measurable | 0 | 19 (100%) |

Breakdown at the default knob: 3 specific names (2 right), 3 category labels
(1 right), 13 abstentions.

So at a fixed knob the wrong-name rate does fall clearly, 52.6% to 15.8%, and
name precision rises from 47.4% to 66.7%, at the cost of naming only a third of
the entries.

## Why it is not enabled

Leave one scene out, tuning the two global knobs on the other five scenes and
scoring the held-out one, picks `min_sim` 0.15 or 0.20 every time, and at those
values almost everything abstains: 0 named out of 2, 2, 3, 4, 5 and 2 entries on
the six held-out scenes. Zero wrong names, almost zero coverage.

That is a degenerate optimum, not a good result. The tuning objective as written
(fewest wrong names first, then coverage) can always reach "no wrong names" by
refusing to name anything, so the gate as stated is satisfiable for free and the
result says nothing. Before this flag can be judged it needs a coverage floor in
the objective, for example "the wrong-name rate subject to naming at least half
of the entries", and a knob range that reflects the observed similarity
distribution, which sits mostly below 0.2 for these crops.

I am not claiming the gate passed. Measured facts, in one line: at a fixed knob
the ladder cuts wrong names by about two thirds and names a third of the
entries; tuned leave-one-out it abstains almost completely, which is why the flag
stays off.

## Reproducing

```
python scripts/naming_v3_eval.py --loso --out logs/naming_v3.json
python scripts/naming_v3_eval.py --scenes scene5 --debug     # one scene, every rung
python -m src.naming.cache                                    # entry inventory, no models
python -m src.naming.vocabulary                               # vocabulary lookups
```

Per-entry rows land in `logs/naming_v3.json`, so any disagreement can be traced
back to a single entry without re-running anything.
