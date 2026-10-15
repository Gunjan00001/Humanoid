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
| naming_v3, knob sim 0.05, votes 1 (the coverage ceiling) | 7/19 | **36.8%** | 66.7% | 4 (21.1%) | 12 (63.2%) |

Breakdown at the default knob: 3 specific names (2 right), 3 category labels
(1 right), 13 abstentions.

So at a fixed knob the wrong-name rate does fall clearly, 52.6% to 15.8%, and
name precision rises from 47.4% to 66.7%, at the cost of naming only a third of
the entries.

## Why it is not enabled

The tuning objective was degenerate at first: "fewest wrong names, then highest
coverage" can always reach zero wrong names by refusing to name anything, so the
gate was satisfiable for free. `leave_one_scene_out` now takes a coverage floor
(`--min-coverage`), and a knob has to name at least that share of the training
entries before it is considered.

Two facts came out of that, and both matter more than the gate itself:

* **The rerank cannot name more than 36.8% of these entries at any knob.** That is
  the maximum coverage over the whole knob grid, reached at similarity 0.05, and
  it does not improve further. The similarities SigLIP 2 produces for these crops
  sit mostly below 0.2, so there is no setting that names most objects.
* **A 50% coverage floor is unattainable**, so the search falls back to
  "abstain as much as possible" and the result says nothing. With a reachable
  floor of 0.25 the search behaves sensibly.

Leave one scene out with `--min-coverage 0.25`, six folds, held-out results only:

| held-out scene | knob chosen on the other five | named | wrong | abstained |
|---|---|---|---|---|
| demo | sim 0.10 | 2 | 2 | 1 |
| scene2 | sim 0.10 | 1 | 0 | 1 |
| scene3 | sim 0.15 | 0 | 0 | 4 |
| scene4 | sim 0.05, votes 2 | 1 | 0 | 2 |
| scene5 | sim 0.15 | 0 | 0 | 5 |
| scene6 | sim 0.10 | 0 | 0 | 2 |
| **all 19 entries** | | **4 (21.1%)** | **2 (10.5%)** | **15 (78.9%)** |

So the honest verdict, as measured: the wrong-name rate falls from 52.6% to
10.5% under leave-one-out, which is the drop the gate asked for, and it is paid
for in coverage, naming about a fifth of the entries instead of all of them. The
type and localized recall are untouched because this flag only changes the label
text; detection never reads it, and it is not wired into `describe.py` yet.

Whether that trade is worth taking is a product decision, not a measurement. The
flag stays **OFF** until that decision is made; it is not enabled by the gate
alone.

## Reproducing

```
python scripts/naming_v3_eval.py --loso --out logs/naming_v3.json
python scripts/naming_v3_eval.py --scenes scene5 --debug     # one scene, every rung
python -m src.naming.cache                                    # entry inventory, no models
python -m src.naming.vocabulary                               # vocabulary lookups
```

Per-entry rows land in `logs/naming_v3.json`, so any disagreement can be traced
back to a single entry without re-running anything.
