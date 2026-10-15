# Naming on a larger labelled set (item 3)

Flag `semantics.naming_v4`, default **OFF**. This document records what is
measured so far and what is still missing.

## The set

Built by `scripts/build_naming_set.py` from the **SceneDiff validation pairs
only** (the 20 pairs pre-registered in `data/external/scenediff_slice.json`; the
test split is never read). Every crop is a ground-truth object cut from a frame
where its mask says it is visible, with everything outside the mask blanked, so
the label is right by construction and no detector is involved.

* 387 crops, 56 distinct ground-truth labels, from 20 validation pairs.
* Ground-truth mask labels are mapped to the closed vocabulary by a written table
  in `data/naming_synonyms.json`: 42 labels mapped, 14 left unmapped with a
  reason each (a sticker, a magazine, the pyjama family, a mat, a car that might
  be a toy). Unmapped crops are counted and listed, never silently dropped.

## Rerank ceiling, measured on 317 mapped crops

The tier gate in naming_v4 is "the VLM answer is inside the SigLIP 2 top 3". So
the first question is how often the correct name is even in the top 3.

| bucket | crops | share | Wilson 95% |
|---|---|---|---|
| correct name ranked first | 48 | 15.1% | [11.6%, 19.5%] |
| correct name in top 3 | 25 | 7.9% | [5.4%, 11.4%] |
| correct name not in top 5 | 244 | 77.0% | [72.0%, 81.3%] |
| **correct name in the top 3 (the gate)** | **73** | **23.0%** | **[18.7%, 28.0%]** |

70 crops were skipped because their ground-truth label is not in the vocabulary.

This is the important number of this round, and it is consistent with the earlier
19-entry result: **the SigLIP 2 rerank puts the right name in its top 3 about a
quarter of the time on these crops**, so the plain tier of naming_v4 can fire on
at most roughly a quarter of entries, and the rest will be hedged or abstained.
That is a ceiling of the rerank and the vocabulary, not of the tier logic.

Per label, 27 of 42 never rank first, including the ones with the most crops:
`cream_1` (21 crops, never in the top 3), `face_cream_1` (15, never),
`controller_1` (15, top 3 six times), `bag_1` (11, once),
`madicine_jar_1` (10, never). Several are coarse mappings onto `jar` or `bottle`,
which the rerank does not distinguish, so part of this is the mapping's
granularity rather than the model.

## What is not measured yet

* **The VLM comparison and the tier measurement.** The tiers need a VLM answer
  per crop; `scripts/naming_set_eval.py --vlm-answers <file>` takes a table of
  them. Running the frozen 32B over 317 crops needs a GPU slice, which is queued
  behind the other project's jobs, and the 8B plus a third candidate are not
  started either. Until that table exists there are no per-tier precision and
  coverage numbers and the adoption gate cannot be evaluated.
* The dev-scene comparison of the plain tier against the as-printed baseline.

So the honest state of item 3: the set and the synonym table are built and
committed, the rerank ceiling is measured with intervals, and the VLM side is
waiting on the GPU queue.

## Reproducing

```
python scripts/build_naming_set.py --per-object 4      # rebuild the crops (CPU)
python scripts/naming_set_eval.py                      # rerank measurement (CPU)
python scripts/naming_set_eval.py --vlm-answers logs/vlm_answers.json
```
