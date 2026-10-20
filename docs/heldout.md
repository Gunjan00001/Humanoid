# Held-out test, scene5

## FROZEN (Step 0, recorded before any new code was written)

Scene5 results must be produced by exactly this code. The detection-relevant
files are frozen at:

- **frozen commit: `2c9969642775792547150753bfd863cfb72fd650`**
- file digests (sha256, first 16 hex):

| file | sha256 (16) |
|---|---|
| `src/diff.py` | `72f10984090e44d0` |
| `src/tabletop.py` | `d227110953138229` |
| `src/geometry.py` | `f44767f014c55d5b` |
| `src/describe.py` | `ac00b9382d074679` |
| `src/align.py` | `b7400d42938e979e` |
| `src/reconstruct.py` | `db9147329a3c047d` |
| `configs/default.yaml` | `f05d5cb6323bad40` |

Verification at the end of the task: `git diff 2c996964 -- src/diff.py
src/tabletop.py src/geometry.py src/describe.py src/align.py
src/reconstruct.py configs/default.yaml` must be empty.

## Scoring rule (pre-registered, written before the run)

Ground truth (`data/scene5/ground_truth.md`): the clock **moved**, the chair
**was removed**; nothing added or replaced. This scene was never used for
tuning (no thresholds, prompts or defaults were derived from it).

A ground-truth change counts as **FOUND** only if all three hold:

1. **Type matches**, `moved` for the clock, `removed` for the chair.
2. **The evidence overlay shows the right object**, the per-change composite
   (`changes/change_XXX.png`) and the 3D box must sit on the clock / chair,
   respectively.
3. **The label names it correctly**, the report's label for the entry must
   name the clock / chair.

Label correctness is reported **separately** from detection (an entry can
detect the right change but name it wrongly; that counts as found for
detection, not for labelling). Extra entries (confirmed changes not in the
ground truth) are counted separately. Unverified candidates are listed but not
counted as changes.

**Status: CONFIRMED by the user** (review of
`outputs/scene5/checking_sheet.png`). Final scoring with this rule:

| | result |
|---|---|
| ground-truth changes found | **1/2, type only** (the chair's removal is detected as `removed`, fragmented into 3 entries; the clock's move is missed) |
| labels correct | **0/2** (the fragments are named "wooden leg" / "chair leg" / "table leg") |
| missed | **1** (the clock's move) |
| extras | **≥ 2** (entries 4 and 5; entry 3 is a chair fragment that cannot be attributed cleanly) |

The user's overlay review, tile by tile:

- **tiles 1-3** (removed): the chair's area. Type `removed` correct, but
  fragmented into three entries; the labels are wrong.
- **tile 4** (added "small round object"): **false positive, a floor light
  reflection in video B**. Verified against the frames: the B-side crop shows
  the bright reflective floor patch; counted as an extra.
- **tile 5** (added "bottle and clock"): **not a real addition**, the bottle
  is present in both videos; the clock's *new position* was reported as
  "added". The clock move counts as **missed**.

## Result (scene5, run once, the single held-out execution)

Command (exactly once, frozen code, default settings, no overrides):

```
python detect.py --a data/scene5/a.mp4 --b data/scene5/b.mp4 --name scene5
```

The runner validated the inputs and printed:

```
[warn ] A/B resolutions differ (A 860x480, B 480x860) - frames are used at
        their own size; check that the crops look aligned
[warn ] A is landscape, B is portrait - unusual for the same space
[check] A 860x480 30.0fps 26.8s | B 480x860 30.0fps 42.5s
[mode ] geometric (auto: dominant plane covers 11% of confident points
        (threshold 50%))
```

### Per-change table

| GT change | detected type | detected label | type correct? | label correct? | confidence |
|---|---|---|---|---|---|
| clock **moved** | *(no `moved` entry)* |, | **no** |, |, |
| chair **removed** | `removed` (3 fragments, ids 1-3) | "wooden leg", "chair leg", "table leg" | **yes** (fragmented) | **no** (no fragment says "chair") | 0.74 / 0.29 / 0.45 |

Overlay check (pending user confirmation): the three removed fragments sit at
the chair's area next to the table (see `checking_sheet.png` tiles 1-3); the
clock's new position appears only inside the "added bottle and clock" entry,
whose before/after crops both show the clock, see the failure analysis.

### Confirmed entries (5) and extras

| # | type | label | confidence | assessment |
|---|---|---|---|---|
| 1 | removed | wooden leg | 0.74 | chair fragment (type ok, label wrong) |
| 2 | removed | chair leg | 0.29 (possibly) | chair fragment |
| 3 | removed | table leg | 0.45 | chair fragment or extra |
| 4 | added | small round object | 0.34 (possibly) | **extra** (GT: nothing added) |
| 5 | added | bottle and clock | 0.30 (possibly) | **extra**; crops show the clock in both videos |

Extra entries (confirmed, not in the GT): **2** (ids 4, 5); id 3 is likely a
third fragment of the chair but cannot be attributed to a GT change with
confidence.

### Unverified candidates

**0** kept as unverified in the report. Behind that: the VLM denied **48**
candidates ("same object visible in both videos") and the static-object
suppression rejected **3**; `cache/scene5/semantics/rejected.json` lists every
one with its reason. 10 candidates were VLM-confirmed, 5 survived the filters.

### Printed natural-language summary (verbatim)

```
Compared a.mp4 and b.mp4. I found 5 change(s):
1. Removed - wooden leg: The wooden leg was removed from the table surface in video B.
2. Removed - chair leg: Possibly the chair leg and black object were removed from the scene in video B.
3. Removed - table leg: The table leg and seat were removed, leaving only the cracked surface in video B.
4. Added - small round object: Possibly a small round object appeared on the floor in video B.
5. Added - bottle and clock: Possibly a plastic water bottle and a digital clock were added to the table in video B.

Notes: Object names come from a vision-language model and can be wrong. Detection thresholds were tuned on the development scenes (demo, scene2, scene3, scene4). No absolute distances are reported (the reconstruction scale is not metric-validated); directions refer to the before/after videos as stated.
Files: viewer.html, change.gif, report.json
```

(All five sentences were phrased by Qwen3-VL in the one phrasing call and
passed the report-only validation; `summary.json` records the sentence
source per change.)

### Runtime (measured)

| measure | value |
|---|---|
| queue wait (qtime -> stime) | **0 s** (the gpu queue was free) |
| job runtime (`resources_used.walltime`) | **7:07** |
| total wall clock (submit -> summary printed) | **7:22** |
| stages (job log) | extract 0:03 · reconstruct 0:23 · align 0:24 · diff 0:33 · describe 3:54 · report+visualize+summary ~1:45 |

### Failure analysis (written after the run)

1. **The two recordings differ in orientation** (A landscape 860x480, B
   portrait 480x860) and length (26.8 s vs 42.5 s). The runner warned about
   this before submitting. This is **likely** a contributing cause: a joint
   reconstruction across differently oriented recordings sees the clock/chair
   region from two very different viewpoints, which **likely** weakens both
   the cluster geometry and the VLM crops. Not proven.
2. **The auto mode chose `geometric`** (dominant plane 11% of confident
   points). The geometric mode's known weakness, coverage-difference
   artifacts (documented on scene2) - is visible here: 58 candidates, 48
   denied by the VLM, 3 suppressed, 10 confirmed, 5 surviving the filters.
   The **11% plane coverage is likely the second contributing cause**: it both
   selected the geometric mode and indicates that this scene's confident
   points are spread over floor/walls/objects rather than one dominant
   surface, which is exactly the regime where the geometric mode produces
   fragments. Not proven; see the post-hoc diagnostic below.
3. **The chair's removal is fragmented**: three separate clusters (wooden
   leg / chair leg / table leg) instead of one "chair" removal. The VLM
   labelled the fragments by their local appearance (legs), not by the object
   they belong to. Type is right; naming is wrong.
4. **The clock's move was not matched**: the move matcher found 0 pairs. The
   clock's origin cluster did not survive the filters, so there was no
   removed+added pair to match; the arrival shows up only as the dubious
   "added bottle and clock" entry (both its crops show the clock, so the
   claim is a false positive rather than the move).
5. **Two entries are extras** (ids 4, 5); both carry the "possibly" phrasing
   from the low-confidence rule.

**No tuning, no re-run, no prompt or threshold change was made for scene5.**
Any future fix must be labelled "post-hoc, not part of the held-out result"
and reported on scene5 separately.

### POST-HOC DIAGNOSTIC (not part of the held-out result)

Clearly labelled: this run is **not** the held-out result and was executed
after the held-out run, with one variable changed (`--mode tabletop` forced).
Everything else (frozen code, default thresholds, same videos) is identical.

```
python detect.py --a data/scene5/a.mp4 --b data/scene5/b.mp4 \
    --name scene5_tabletop_posthoc --method tabletop
```

Outcome (job 10829, 6:40, queue wait 0 s):

| | held-out (auto -> geometric) | post-hoc (tabletop forced) |
|---|---|---|
| confirmed changes | 5 | 4 |
| removed entries | 3 ("wooden leg", "chair leg", "table leg") | 3 ("wooden chair leg", **"black chair"**, "cracked shelf") |
| replaced entries | 0 | 1 ("wooden table leg") |
| added entries | 2 ("small round object", "bottle and clock") | 0 |
| `moved` entries | 0 | 0 |
| clock move | missed | **still missed** |
| chair named correctly | no | **yes, in one entry** |

**Does it change the outcome? Partially.** The tabletop mode removes both
false additions (the floor reflection and the clock/bottle misread) and names
the chair correctly in one entry, but the clock's move is still missed and the
removal is still fragmented (3-4 entries, including a "cracked shelf" that is
likely an extra). This **suggests** (does not prove) that the automatic mode
choice contributed to the held-out result; it does not fix the missed move.

The held-out outputs remain exactly as produced; this diagnostic is committed
separately in `outputs/scene5_tabletop_posthoc/`.

### POST-HOC: detection on the shipped re-encoded b.mp4 (not the held-out result)

`data/scene5/b.mp4` was later re-encoded (libx264 crf 26, same 480x860@30fps,
4.6 MB) so a plain clone can read it. Detection **runs** on it (exit 0, 9:04)
but the result differs from the held-out one, because the reconstruction from
the re-encoded pixels changes:

| | held-out (original b.mp4) | re-encode check |
|---|---|---|
| auto mode | geometric (plane 11%) | **tabletop** (the coverage crossed 50%) |
| entries | 5 (3 removed, 2 added) | 7 (6 removed, 1 added) |
| chair naming | no entry says "chair" | "chair", "chair backrest", 3x "chair leg" |
| clock move | missed | still missed |
| extras | 2 | >= 1 ("small brown object", "stacked bowls") |

This is recorded so nobody mistakes a clone's scene5 run for the held-out
result: the held-out outputs were produced from the **original** b.mp4 and are
committed unchanged in `outputs/scene5/`; the re-encode check is committed
separately in `outputs/scene5_reencode_check/`. It also shows that the
automatic mode choice is sensitive to the exact input encoding (an honest,
unproven-likely observation: the lossy re-encode shifts the confident-point
distribution across the 50% threshold).

### Checking sheet

`outputs/scene5/checking_sheet.png` (5 numbered tiles: before | after crops +
the 3D box per detection) and `checking_sheet.md` (the same as an index).
Please confirm the overlays against the ground truth.

---

# Held-out test, scene6

## FROZEN (Step 0, recorded before the run)

Scene6 results must be produced by exactly this code. The detection-relevant
files are pinned (identical byte content verified across all three candidates):

- task-stated frozen commit: **`f007d2a3`**
- `v0.12.0` tag actually resolves to: **`f007d2a34c455faad14fcc5fb7fbb59fb31b33a3`**
- `HEAD` at run time: `f96d3b0`, `git diff f007d2a3 HEAD` on the files below is **empty**
- file digests (sha256, first 16 hex), **identical to the scene5 freeze**:

| file | sha256 (16) |
|---|---|
| `src/diff.py` | `72f10984090e44d0` |
| `src/tabletop.py` | `d227110953138229` |
| `src/geometry.py` | `f44767f014c55d5b` |
| `src/describe.py` | `ac00b9382d074679` |
| `src/align.py` | `b7400d42938e979e` |
| `src/reconstruct.py` | `db9147329a3c047d` |
| `configs/default.yaml` | `f05d5cb6323bad40` |

Verification at the end of the task: `git diff f007d2a3 -- src/diff.py
src/tabletop.py src/geometry.py src/describe.py src/align.py
src/reconstruct.py configs/default.yaml detect.py` must be empty.

## Data (Step 0, re-encoded before the run)

Originals are LFS recordings; they were re-encoded so a plain clone can read
them. **Detection on the re-encoded files is what ships and what is scored.**
(Scene5 lesson: re-encoding can move the auto-mode decision; recorded here as an
honest caveat, not as a reason to re-run.)

| file | original (LFS oid) | re-encoded size | sha256 (re-encoded) |
|---|---|---|---|
| `data/scene6/a.mp4` | 9,090,092 B (`baf3f76f3475…`) | 3,424,287 B | `8c22c960edfc5376c43b5e0de346d1e70add6548a13d624f932cfc548c68482b` |
| `data/scene6/b.mp4` | 8,050,457 B (`cf1d265b48a9…`) | 2,826,327 B | `e669a365fe646ab9b72466524da8db0c1844d44eaefa35700fac5b7c18e928fb` |

Re-encode = libx264 (crf 29, medium), same 860x480@30fps (b measured 29.83 fps
originally, 30.0 after), no audio, `+faststart`. Originals preserved in LFS at
`data/scene6/original/` (and in the data-v0.5 history).

Validation (correct `detect.py` validation code, no submission) printed **no
warnings**: `[check] A 860x480 30.0fps 55.5s | B 860x480 30.0fps 49.1s`. No
resolution or orientation mismatch.

## Scoring rule (pre-registered, written before the run)

Ground truth (`data/scene6/ground_truth.md`): the **cup moved**, the **remote
was removed**, the **apple was added**; nothing else changed. This scene was
never used for tuning (no thresholds, prompts or defaults were derived from it).

A ground-truth change counts as **FOUND** only if all three hold:

1. **Type matches**, `moved` for the cup, `removed` for the remote,
   `added` for the apple.
2. **The evidence overlay shows the right object**, the per-change composite
   (`changes/change_XXX.png`) and the 3D box must sit on the cup / remote /
   apple, respectively.
3. **The label names it correctly**, the report's label for the entry must
   name the cup / remote / apple.

A move counts as FOUND only if it is reported as `moved` (an `added` + `removed`
pair at different positions does **not** count as a found move). Label
correctness is reported **separately** from detection (an entry can detect the
right change but name it wrongly; that counts as found for detection, not for
labelling). Extra entries (confirmed changes not in the ground truth) and
fragments are counted separately. Unverified candidates are listed but not
counted as changes.

**Status: CONFIRMED by the user** (overlay review of
`outputs/scene6/checking_sheet.png`; tile 1's 3D box position was **not**
verified by the user).

## Result (scene6, run once, the single held-out execution)

Command (exactly once, frozen code, default settings, no overrides):

```
python detect.py --a data/scene6/a.mp4 --b data/scene6/b.mp4 --name scene6
```

### Per-change table

| GT change | detected type | detected label | type correct? | label correct? | confidence |
|---|---|---|---|---|---|
| cup **moved** | *(no `moved` entry; `move matches: 0`)* |, | **no, missed** |, |, |
| remote **removed** | `removed` | "remote control" | **yes** | **yes** | 0.50 |
| apple **added** | *(no `added` entry for the apple)* |, | **no, missed** |, |, |

Overlay review (user, authoritative):

- **tile 1** (`removed` "remote control"): the before/after frames confirm the
  remote is on the table in A and absent in B, type and label correct.
  **3D box position not verified by the user.**
- **tile 2** (`added` "electrical outlet"): **FALSE POSITIVE**, the outlet is
  on the wall in both videos.

### Scoring (pre-registered rule)

| | result |
|---|---|
| ground-truth changes found | **1/3** (the remote's removal) |
| labels correct | **1/1** (of the found changes) |
| missed | **2** (cup move, apple add) |
| extras | **1** (electrical outlet) |
| fragments | 0 |

### Confirmed entries and extras

| # | type | label | confidence | assessment |
|---|---|---|---|---|
| 1 | removed | remote control | 0.50 | **ground truth found** |
| 2 | added | electrical outlet | 0.33 | **extra / false positive** |

No `moved` and no `replaced` entries. Unverified candidates kept: **0**.

### Printed natural-language summary (verbatim)

```
Compared a.mp4 and b.mp4. I found 2 change(s):
1. Removed - remote control: The remote control was removed from the table.
2. Added - electrical outlet: Possibly an electrical outlet was added to the wall.

Notes: Object names come from a vision-language model and can be wrong. Detection thresholds were tuned on the development scenes (demo, scene2, scene3, scene4). No absolute distances are reported (the reconstruction scale is not metric-validated); directions refer to the before/after videos as stated.
Files: viewer.html, change.gif, report.json
```

(The Qwen3-VL phrasing call failed with CUDA OOM after `describe` had loaded the
32B model, so the sentences are `src/summary.py`'s deterministic templates;
`summary.json` records the sentence source.)

### Runtime (measured)

| measure | value |
|---|---|
| queue wait (qtime -> stime) | **0 s** (15:28:25 -> 15:28:26) |
| job runtime (`resources_used.walltime`) | **18:53** |
| stages (job log) | extract 0:05 · reconstruct 0:52 · align 0:42 · diff 0:25 · **describe 15:53** (35 VLM verifications) · report 0:00 · visualize 0:21 · summary 0:25 |
| auto diff-mode | **geometric** (dominant plane 6% of confident points, < 50%) |

### Failure analysis (written after the run)

1. **Cup move missed, the move matcher found no pair.** `cache/scene6/semantics/matches.json`
   is empty (`move matches: 0 (0 arrivals)`). The cup's *arrival* was seen: two
   `added` candidates were VLM-confirmed (`16` "white cup", `21` "bottle and
   cup"), but the **structural post-filter dropped both** ("inside an object
   present in both videos", 78% / 100% of their points), so no arrival survived
   to pair with a departure. **Likely** contributor: the auto mode chose
   `geometric` (plane coverage 6%), whose coverage-difference artifacts produce
   exactly these fragments, 28 of the 32 rejects are VLM-denied fragments
   ("table leg", "wooden edge", …). Not proven.
2. **Apple add missed, a candidate existed, the VLM denied it.** Candidate `28`
   ("blue cup and red apple") was built; the VLM compared "plastic bottle with
   purple cap" vs "plastic bottle with purple cap, blue cup, red apple" and
   overrode it to `changed=False` (a VLM false negative). Candidate `19`
   ("plastic bottle") was likewise VLM-denied. So the apple was dropped at the
   **VLM confirmation stage**, not at coverage/filters.
3. **The one extra (electrical outlet)** is a wall feature present in both
   videos; the VLM confirmed candidate `20` ("electrical outlet") as an
   addition. It is a false positive under the user's overlay review.
4. **`describe` dominated the runtime** (15:53 of 18:53), 35 Qwen3-VL-32B
   verifications, each a full VLM call.

**No tuning, no re-run and no prompt/threshold change was made for scene6.** The
only post-run change is a **visualization bug fix** (below), which does not
touch detection.

### POST-HOC: visualization bug fix (not part of the held-out detection result)

The user observed that tile 2's "before" crop showed video-B content. Root
cause: `crops.json` stores `frame` for the change's *own* video (its `tag`) and
`frame_other` for the opposite one, but `src/visualize.py` hard-coded
`before=frame, after=frame_other`, for an `added` change (`tag: b`) that swaps
the panels. Fixed in `src/visualize.py` (select before/after by tag) , 
**visualization only; detection untouched**. `outputs/scene6/changes/*.png` and
the checking sheet were re-rendered from the cached results; `report.json` and
every detection number are unchanged.

## Post-hoc redo of all six uploaded scenes (2026-10-20)

A full redo of every uploaded scene was run once on the cluster (job 11880, two
GPUs, current main, config v1.1.0) into fresh folders, `outputs/<scene>_redo` and
`cache/<scene>_redo`. The frozen outputs, including `outputs/scene5` and
`outputs/scene6`, were **not touched** (`git diff` empty). This is a post-hoc
replay, **not part of the held-out result**: scene5 and scene6 were each run once
for the original evaluation and those numbers stand.

| scene | frozen entries | redo entries | quick match, frozen | quick match, redo | redo time |
|---|---|---|---|---|---|
| demo | 3 | 3 | 2/3, 1 extra | 2/3, 1 extra | 15.8 min |
| scene2 | 2 | 5 | 1/1, 1 extra | 1/1, 4 extras | 9.2 min |
| scene3 | 4 | 5 | 0/2, 4 extras | 0/2, 5 extras | 22.6 min |
| scene4 | 3 | 3 | 3/3, 0 extras | 3/3, 0 extras | 10.7 min |
| scene5 | 5 | 7 | 1/2, 4 extras | 1/2, 2 extras | 30.5 min |
| scene6 | 2 | 2 | 1/3, 1 extra | 1/3, 1 extra | 27.7 min |

The "quick match" column is an automatic type plus canonical-name comparison
against ground truth. It is **not** the official scoring, which uses the human
resolutions in `matches.json`; the two disagree where a label is a paraphrase
(for example demo is 3/3 under the official resolution).

scene4 and scene6 reproduced identically. scene2, scene3 and scene5 differ at the
**cluster level**, not only in wording: fresh reconstruction and clustering gave
8 -> 9 clusters on scene2, 29 -> 26 on scene3 and 58 -> 55 on scene5, while scene4
(8) and scene6 (35) matched. The differences are therefore upstream of naming.
Whether the cause is GPU nondeterminism across runs or a different working point
is not established here and would need a third run to separate.
