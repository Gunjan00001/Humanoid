# Accuracy sprint, branch `exp/accuracy`

Goal: improve **change recall, move recall, precision (fewer extras) and object
naming**. Rules in force: `rules.md`, `gpu.md`, real numbers only. Work happens
**only on this branch**; `main` and tag `v1.0.0` are untouched and the committed
held-out outputs of scene5/scene6 stay as the v1.0.0 historical result. On this
branch demo and scene2–6 are **all development scenes**. One global
`configs/default.yaml`; no per-scene overrides, no hard-coded object names, no
scene-specific thresholds. New behaviour ships behind a config flag, default
**OFF**, until the ablation proves it helps. Caches are kept.

---

## Phase A, scorer + baseline (no pipeline changes)

### Ground truth & human resolutions

`data/<scene>/ground_truth.json` (type, object, accepted synonyms) for demo +
scene2…6. Every report entry has been **human-resolved** in
`data/<scene>/matches.json` (`resolution: "human-resolved"`, entry id → GT id,
`type_ok` / `name_ok` / `fragment`, plus evidence). `scripts/score.py` reads a
matches.json as authoritative when present (otherwise a transparent name-based
auto-match is used, for future sealed scenes).

### Human resolutions (authoritative, user 2026-10-11)

| scene | entry | label | resolved to | verdict |
|---|---|---|---|---|
| demo | 3 | remote control (`replaced`) | GT1 (moved AC remote) | right object + name, **wrong type** |
| scene2 | 1 | apple (`removed`) |, | **extra** (the coconut cluster mislabelled) |
| scene3 | 1 | black chair (`replaced`) | GT2 (apple→cup) | **type correct, label wrong** |
| scene3 | 2 | red apple (`removed`) | GT2 | **fragment/duplicate** of GT2 (not a type match, not an extra) |
| scene3 | 3 | green and yellow object (`removed`) |, | **extra** (coconut-like object stays on the table) |
| scene3 | 4 | black trash can (`removed`) | GT1 (chair removed) | **type correct, label wrong** |
| scene5 | 1,2,3 | wooden leg / chair leg / table leg | GT2 (chair removed) | **fragments**, part names → **wrong** |
| scene5 | 4 | small round object (`added`) |, | **extra** (floor reflection in video B) |
| scene5 | 5 | bottle and clock (`added`) | GT1 (moved clock) | **wrong type** (`added` ≠ `moved`); `bottle` is a false object name |
| scene6 | 2 | electrical outlet (`added`) |, | **extra** (wall outlet present in both videos); **apple missed** |

### Metric definitions (authoritative)

1. **recall(type)**, share of GT changes with ≥1 entry of the **correct type**.
2. **recall(localized)**, share of GT changes with ≥1 entry **on the right object**,
   regardless of type/name.
3. **move recall (strict)**, share of GT `moved` changes reported with type `moved`.
4. **name accuracy**, correctly-named entries / localized entries; a **part name**
   ("chair leg") is **not** correct.
5. **fragments (by location)**, entries resolved to the same GT **beyond the first**.
6. **precision breakdown**, every entry classified **correct / type-wrong /
   fragment / extra** (`correct` = resolved with the right type; naming is metric 4).

### Baseline (v1.0.0 reports), human-resolved

| scene | GT | rep | recall(type) | recall(loc) | move(strict) | name% | frag | correct | type-w | frag | extra |
|---|---|---|---|---|---|---|---|---|---|---|---|
| demo | 3 | 3 | 67% | 100% | 0% | 100% | 0 | 2 | 1 | 0 | 0 |
| scene2 | 1 | 2 | 100% | 100% | – | 100% | 0 | 1 | 0 | 0 | 1 |
| scene3 | 2 | 4 | 100% | 100% | – | 33% | 1 | 2 | 0 | 1 | 1 |
| scene4 | 3 | 3 | 100% | 100% | 100% | 100% | 0 | 3 | 0 | 0 | 0 |
| scene5 | 2 | 5 | 50% | 100% | 0% | 0% | 2 | 1 | 1 | 2 | 1 |
| scene6 | 3 | 2 | 33% | 33% | 0% | 100% | 0 | 1 | 0 | 0 | 1 |
| **overall** | **14** | **19** | **71%** | **86%** | **25% (1/4)** | **60%** | **3** | **10** | **2** | **3** | **4** |

- **Missed GT changes:** scene6 `#1 cup`, scene6 `#3 apple` (nothing else).
- **Found by type but not named:** scene3 chair (#1), scene3 apple (#2), scene5 chair (#2).
- **Extras (4):** scene2 `apple`; scene3 `green and yellow object`; scene5 `small round object`; scene6 `electrical outlet`.
- **Fragments:** scene3 GT#2 apple (entries 1,2); scene5 GT#2 chair (entries 1,2,3).

### Reading

The baseline is strong on **localized recall (86%)** and decent on **recall by
type (71%)**, but weak where it matters most: **move recall 25%** (only scene4's
apple), **name accuracy 60%**, **4 extras** and **3 fragments**. The two
completely-missed changes (scene6 cup, apple) and the type/naming failures are
Phase B's targets.

---

## Phase B, failure attribution

Every missed GT change and every extra, attributed to the stage that lost/created
it, from `logs/pbs_*.log`, `cache/<scene>/semantics/{vlm,confirmed,rejected,matches}.json`
(evidence in the table). demo's semantics live at `cache/semantics/`.

### Per-case attribution

| id | scene | case | stage | evidence |
|---|---|---|---|---|
| M1 | scene6 | **cup `moved`, missed** (move) | move-match (arrival lost upstream) | `matches.json` empty (0 pairs); arrivals `#16` "white cup" & `#21` "bottle and cup" were **VLM-confirmed** but rejected by the *structural post-filter*, "inside an object present in both videos (78% / 100% of its points) - lighting/blur artefact" |
| M2 | scene6 | **apple `added`, missed** | VLM denied | candidate `#28` "blue cup and red apple" → VLM `changed=False` (denied); `#19` "plastic bottle" also denied |
| E1 | scene2 | extra `removed apple` | VLM false-positive | cluster `#1` VLM "apple" `changed=True` (apple is unchanged; spurious removal next to the coconut) |
| E2 | scene3 | extra `removed green and yellow object` | VLM false-positive | cluster `#27` VLM `changed=True` (the coconut stays on the table) |
| E3 | scene5 | extra `added small round object` | VLM false-positive / coverage artefact | cluster `#38` VLM `changed=True`, the "floor reflection in video B" you flagged |
| E4 | scene6 | extra `added electrical outlet` | VLM false-positive / coverage artefact | cluster `#20` VLM `changed=True`, wall outlet present in both videos |
| T1 | demo | GT `moved` remote → typed `replaced` | move-match + type logic | `matches.json` empty; cluster `#18` "remote control" `changed=True`; the departure `#2` "remote control" was also rejected by the post-filter ("inside object present in both videos, 100%") |
| T2 | scene5 | GT `moved` clock → typed `added` | move-match | `matches.json` empty; cluster `#44` "bottle and clock" `changed=True` (clock's new position read as an addition) |
| F1 | scene5 | chair removal **fragmented** (3 entries, part names) | clustering (no merge) + VLM naming | clusters `#6` "wooden leg", `#16` "chair leg", `#18` "table leg" all confirmed `removed` |
| F2 | scene3 | apple→cup **duplicated** | clustering (no merge) | cluster `#5` "black chair" (replaced) + cluster `#26` "red apple" (removed) both resolve to GT2 |
| N1 | scene3 | chair named `black trash can` | VLM naming | cluster `#28` VLM "black trash can" |
| N2 | scene3 | apple named `black chair` | VLM naming | cluster `#5` VLM "black chair" |
| N3 | scene5 | clock named `bottle and clock` | VLM naming | cluster `#44` (`bottle` false object) |

### Ranked causes (by count)

1. **VLM false-positive confirmation, 4** (all extras: scene2 apple, scene3
   green/yellow object, scene5 floor reflection, scene6 wall outlet). The VLM
   confirmed `changed=True` on regions that did not change; nothing downstream
   caught them.
2. **VLM naming errors, 6 localized entries** (scene3 `black trash can`/`black
   chair`; scene5 `wooden leg`/`chair leg`/`table leg`/`bottle and clock`).
   Part names and wrong objects.
3. **Move matcher produced 0 pairs, 3 GT moves** (demo remote, scene5 clock,
   scene6 cup). Sub-causes: arrival dropped by the structural post-filter
   (scene6 cup), appearance-change-at-same-spot typed as `replaced` (demo
   remote), no arrival cluster at all (scene5 clock).
4. **Clustering fragmentation, 2 GT, 3 redundant entries** (scene5 chair ×3;
   scene3 apple ×2). No fragment merge by 3D adjacency.
5. **VLM denial of a genuine change, 1** (scene6 apple, candidate `#28`).
6. **Coverage/reflection artefacts, 3 of the extras** (scene5 reflection,
   scene6 wall outlet, scene2 unchanged object): additions/removals accepted
   where the other video's depth did not establish the change.

### Which lever (Phases C–F) targets which cause

| cause | lever |
|---|---|
| 1 VLM false-positive verify (4) | **Phase D** coverage-aware `added`; **Phase C** stronger verify model/prompt |
| 2 VLM naming (6) | **Phase C** VLM bake-off; **Phase F** name the merged object from multi-view |
| 3 move match 0 pairs (3) | **Phase E** object-level matching (`objmatch`) |
| 4 fragmentation (2 GT) | **Phase F** 3D adjacency merge |
| 5 VLM denial of a real change (1) | **Phase D** soft veto ("possibly"/low-confidence) |
| 6 coverage/reflection (3) | **Phase D** coverage-aware `added` |

---

## CHECKPOINT, Phase B

Ranked causes above. By evidence the highest-value levers are **Phase E
(objmatch, 3 GT moves)** and **Phase D (soft veto + coverage-aware added, 5
of the 6 causes touch VLM verify/coverage)**; **Phase F** addresses the 2
fragmented GT and naming; **Phase C** addresses naming + verify quality.
Which should I take first?


## Track B probe result (SAM 2, job 11017)

`scripts/probe_sam2.py` refines the paired crops with a SAM 2.1 base-plus mask and
compares them with DINOv2 against the 0.65 gate.

| scene | object | paired-crop DINOv2 after SAM 2 mask | pass |
|---|---|---|---|
| scene6 | cup | 0.190 | no |
| scene5 | clock | 0.219 | no |

The probe fails, so objmatch2 is not built (the pass gate needs the object
segmented and matched in at least 2 of the 3 targets). Segmented in both videos
was not enough; the crops do not match. See docs/negative_results.md.

## Track B controls (job 11023) and closure

Same SAM 2 masks and cropping code, with controls. CLS is the DINOv2 CLS token,
patch is the mean of the DINOv2 patch tokens inside the mask, size_r is the 3D
extent ratio (min over max).

| scene | pair | role | CLS | patch | size_r |
|---|---|---|---|---|---|
| scene4 | apple moved | positive (matched at 0.704 earlier) | 0.704 | 0.914 | 0.70 |
| scene4 | scissor vs apple | negative (different objects) | 0.285 | 0.085 | 0.78 |
| scene6 | cup | target | 0.482 | 0.374 | 0.51 |
| scene5 | clock | target | 0.174 | 0.460 | 0.71 |

The controls separate: the correct match scores 0.704 CLS (0.914 patch) and two
different objects score 0.285 (0.085 patch). So the 0.65 gate is calibrated for
cross-video matching. The targets stay below it, and the clock's CLS (0.174) is
even below the negative control. The lane between the negative and positive
controls is 0.285 to 0.704 (CLS) and 0.085 to 0.914 (patch). Track B is closed as
a real negative result: the cup and clock candidate crops are not the same
object, so object first matching on these crops cannot recover the moves.
