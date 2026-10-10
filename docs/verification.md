# Verification: from 26 candidates to 3 changes

The change detector (see [`pipeline.md`](pipeline.md#4-change-detection)) is
deliberately *permissive*: on the demo scene it proposes **26 candidate
clusters**, because photometric comparison cannot by itself tell a real change
from a lighting, blur or exposure difference on a static object. This document
explains every rule that `src/describe.py` applies to cut those candidates down
to the final, confirmed change list — with the numbers observed on the demo
scene, so the rules can be re-tuned knowingly.

All rules run in order; every rejection is written to
`cache/semantics/rejected.json` with a human-readable reason, and the survivors
to `cache/semantics/confirmed.json`. **Nothing is ever silently dropped.**

## The rules, in order

### 1. VLM verification (`PROMPT_VERIFY`)

Each candidate gets **two crops**: the best view from its own video and a
pose-matched view from the other video (same region, camera pose closest to
the own view — see `best_crop_for_cluster`). Qwen3-VL is asked to name the
objects in each image and to confirm or deny the change claim.

- Removed/added claim → `changed: false` rejects the candidate.
- The model's answer is cached per cluster **keyed by model** (re-running with
  `--force` reuses it; switching 32B ↔ 8B re-asks).

### 2. Self-consistency override (removed/added)

If the model names the **same object in both images** ("digital timer" vs
"digital timer"), the claim is false *regardless of its own answer* — small
blurry crops make the model over-sensitive. Descriptive words (colours,
shapes, "blurred", "surface"...) are ignored by the comparison so that
"dark surface with a blurred edge" vs "dark surface with a blue pen" does not
count as the same object (this exact pair wrongly rejected the pen once).

### 3. Mutual cancellation

A confirmed `removed X` **and** a confirmed `added X` with the same label
within 12 cm means X is present in both videos — both fragments came from
lighting differences and are dropped. (Demo: cancelled the highlighter pair
#10/#17.)

### 4. Score floor (`semantics.min_change_score`, default 80)

Removed/added candidates need at least this much colour evidence. The real
changes on the demo score 90 (pen) and 110 (comb); weak fragments score 47–72.
Tune down if a real object is faint (e.g. white-on-white).

### 5. Static-object suppression (`src.tabletop.static_object_mask`)

Cells where **both** videos show an object (objectness > threshold in A *and*
B) are the locations of static objects (timer, highlighter). A candidate whose
points lie ≥50 % inside the 2-cell-dilated version of that mask is an
edge/lighting fragment of a static object → rejected. (Demo: killed the
remaining timer fragments #6/#8.)

### 6. Replaced pixel gate (`semantics.replaced_min_pixel_diff`, default 70)

A `replaced` claim must show a strong pixel-level appearance change between the
before/after crops (mean absolute RGB difference, 64×64, 0–255). Demo numbers:

| cluster | what it is | pixel diff | verdict |
|---|---|---|---|
| #18 | remote flip (black back → white face) | **91.7** | kept |
| #19–22 | timer/highlighter blur differences | 56.1–60.5 | rejected |

### 7. Same-appearance rejection (`semantics.same_appearance_threshold`, 0.80)

The before/after crops are embedded with DINOv2 (the same model used for move
matching). Cosine similarity above the threshold means the two views look like
the same appearance → not a change. This is a safety net for cases the VLM
misses; on the demo it removed the last low-evidence fragment.

### 8. Duplicate rules

- A removed/added candidate whose bbox overlaps a confirmed `replaced` change
  is the same physical change seen twice (demo: the remote's old position).
- Unverified clusters (one video never observed the area) overlapping a
  confirmed change are dropped as duplicates; the rest are reported separately
  under "unverified candidates" and never counted as changes.

### 9. Fragment merging

Same-type confirmed fragments whose labels overlap **or** whose bboxes are
within 6 cm (centroids < 25 cm) are merged into one change — the comb's teeth
cluster ("comb") and its handle cluster ("wooden fork") became one entry with
263 cells. The merged entry keeps the larger cluster's id and gains a
`merged_ids` field.

## Final outcome on the demo scene

```
confirmed:  3   (comb removed, pen added, remote replaced)
rejected:  22   (every reason in cache/semantics/rejected.json)
unverified: 0   (all unverified candidates were duplicates of confirmed changes)
moved:      0   (the remote flipped in place → reported as "replaced")
```

## Why this much machinery?

The demo scene is adversarial for a purely geometric pipeline:

- the comb and pen are **1 cm thick** — below VGGT's patch scale, they are
  absent from the depth maps entirely;
- the two phone recordings differ in **blur and exposure**, so the same static
  object (the timer) looks measurably different in A and B;
- the remote **flips in place**, which is neither a removal nor an addition.

Each rule above exists because a specific failure survived the previous ones —
the git history and `logs/describe_real*.log` show the progression. When
adapting the pipeline to a new scene, start from `rejected.json` and the
`--debug` output rather than changing several thresholds at once
(see [`debugging.md`](debugging.md)).
