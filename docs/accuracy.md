# Accuracy sprint — branch `exp/accuracy`

Goal: improve **change recall, move recall, precision (fewer extras) and object
naming**. Rules in force: `rules.md`, `gpu.md`, real numbers only. Work happens
**only on this branch**; `main` and tag `v1.0.0` are untouched, and the committed
held-out outputs of scene5/scene6 stay as the v1.0.0 historical result. On this
branch demo and scene2–6 are **all development scenes**. One global
`configs/default.yaml`; no per-scene overrides, no hard-coded object names, no
scene-specific thresholds. New behaviour ships behind a config flag, default
**OFF**, until the ablation proves it helps. Caches are kept.

---

## Phase A — scorer + baseline (no pipeline changes)

### Ground truth

`data/<scene>/ground_truth.json` written for demo, scene2…scene6 from the
existing `ground_truth.md`, each change listed as `{id, type, object, synonyms}`.
(demo `type`s: moved/removed/added; scene3 has a `replaced`.)

### Scorer — `scripts/score.py`

Matching is transparent; nothing is guessed:

1. **STRONG** — report-entry type == GT type **and** the report label contains a
   GT synonym → GT *found, named correctly*.
2. **WEAK** — report-entry type == GT type but no synonym, assigned **only** when
   exactly one unassigned GT and one unassigned report entry of that type remain
   → GT *found, name wrong*.
3. Left-overs → **extras** (report) / **missed** (GT).

Metrics: change recall (by type), named recall (type+name), move recall, extras
(precision), fragments (entries per GT), name accuracy (named / found). Every
ambiguous case is listed for human resolution.

### Baseline (v1.0.0 reports)

| scene | GT | report | found (type) | named | recall (type) | recall (named) | move | extras | frag | name acc |
|---|---|---|---|---|---|---|---|---|---|---|
| demo | 3 | 3 | 2 | 2 | 67% | 67% | 0/1 | 1 | 0 | 100% |
| scene2 | 1 | 2 | 1 | 1 | 100% | 100% | – | 1 | 0 | 100% |
| scene3 | 2 | 4 | 1 | 0 | 50% | 0% | – | 3 | 0 | 0% |
| scene4 | 3 | 3 | 3 | 3 | 100% | 100% | 1/1 | 0 | 0 | 100% |
| scene5 | 2 | 5 | 1 | 1 | 50% | 50% | 0/1 | 4 | 0 | 100% |
| scene6 | 3 | 2 | 2 | 1 | 67% | 33% | 0/1 | 0 | 0 | 50% |
| **overall** | **14** | **19** | **10** | **8** | **71%** | **57%** | **1/4** | **9** | **0** | **80%** |

Raw reports scored (`outputs/*/report.json`): demo `1:removed:comb`
`2:added:pen and device` `3:replaced:remote control` · scene2 `1:removed:apple`
`2:removed:coconut` · scene3 `1:replaced:black chair` `2:removed:red apple`
`3:removed:green and yellow object` `4:removed:black trash can` · scene4
`1:moved:apple` `2:removed:scissors` `3:added:plastic bottle` · scene5
`1:removed:wooden leg` `2:removed:chair leg` `3:removed:table leg`
`4:added:small round object` `5:added:bottle and clock` · scene6
`1:removed:remote control` `2:added:electrical outlet`.

### Per-scene GT resolution

- **demo** — comb ✓named, pen ✓named; **AC remote missed** (reported as
  `replaced`, not `moved`).
- **scene2** — coconut ✓named; the extra `removed:apple` is the coconut's cluster
  duplicated/mislabelled.
- **scene3** — **chair missed** (`replaced:black chair` / `removed:black trash
  can` name it oddly); apple→cup only weakly matched (name wrong).
- **scene4** — all three ✓named, 0 extras.
- **scene5** — chair ✓named (one entry); **clock missed** (only as `added:bottle
  and clock`); 4 extras.
- **scene6** — remote ✓named; **cup missed**; apple only **weakly** matched by the
  false-positive `electrical outlet`.

### Ambiguous matches — please resolve (not guessed)

| scene | entry | label | issue |
|---|---|---|---|
| demo | 3 | remote control | label names GT "AC remote" but type `replaced` ≠ GT `moved` |
| scene3 | 1 | black chair | names GT "chair" but type `replaced` ≠ GT `removed`; also weak-matched to GT "apple" |
| scene3 | 2 | red apple | names GT "apple" but type `removed` ≠ GT `replaced` |
| scene5 | 5 | bottle and clock | names GT "clock" but type `added` ≠ GT `moved` |
| scene6 | 2 | electrical outlet | weak-matched to GT "apple" (name not matched) — your scene6 overlay says the outlet is a **false positive** and the apple is **missed** |

### Extras (report entries matching no GT change)

- demo: `3 replaced remote control`
- scene2: `1 removed apple`
- scene3: `2 removed red apple`, `3 removed green and yellow object`, `4 removed black trash can`
- scene5: `1 removed wooden leg`, `3 removed table leg`, `4 added small round object`, `5 added bottle and clock`

### Metric limitations (honest)

- Matching is **name-based**, so **fragments are under-counted**: scene5's chair
  removal is 3 `removed` entries (`wooden leg`/`chair leg`/`table leg`) but only
  `chair leg` name-matches, so the other two land in *extras* and `frag` reads 0.
  A location/evidence-based fragment count would catch this (Phase B/F).
- WEAK matches (scene3 apple, scene6 apple) are **provisional** — they count a
  same-type entry as "found, name wrong" and are all listed above for your call.
  If you rule the scene6 outlet a plain extra, scene6 recall(named) is 1/3 and
  extras 1.

---

## CHECKPOINT — Phase A

Baseline table is above. **Please resolve the 5 ambiguous matches** (in
particular scene6 entry 2: extra, or a mis-named apple?). I have not proceeded
to Phase B.
