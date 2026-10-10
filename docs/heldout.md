# Held-out test — scene5

## FROZEN (Step 0, recorded before any new code was written)

Scene5 results must be produced by exactly this code. The detection-relevant
files are frozen at:

- **frozen commit: `e8c77d9292460ff4decb84adbe4e49f142367de3`**
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

Verification at the end of the task: `git diff e8c77d9 -- src/diff.py
src/tabletop.py src/geometry.py src/describe.py src/align.py
src/reconstruct.py configs/default.yaml` must be empty.

## Scoring rule (pre-registered, written before the run)

Ground truth (`data/scene5/ground_truth.md`): the clock **moved**, the chair
**was removed**; nothing added or replaced. This scene was never used for
tuning (no thresholds, prompts or defaults were derived from it).

A ground-truth change counts as **FOUND** only if all three hold:

1. **Type matches** — `moved` for the clock, `removed` for the chair.
2. **The evidence overlay shows the right object** — the per-change composite
   (`changes/change_XXX.png`) and the 3D box must sit on the clock / chair,
   respectively.
3. **The label names it correctly** — the report's label for the entry must
   name the clock / chair.

Label correctness is reported **separately** from detection (an entry can
detect the right change but name it wrongly; that counts as found for
detection, not for labelling). Extra entries (confirmed changes not in the
ground truth) are counted separately. Unverified candidates are listed but not
counted as changes.

**Status: pending user confirmation** — the per-change overlays are collected
in `outputs/scene5/checking_sheet.png` / `checking_sheet.md` for review.

## Result (scene5, run once — the single held-out execution)

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
| clock **moved** | *(no `moved` entry)* | — | **no** | — | — |
| chair **removed** | `removed` (3 fragments, ids 1-3) | "wooden leg", "chair leg", "table leg" | **yes** (fragmented) | **no** (no fragment says "chair") | 0.74 / 0.29 / 0.45 |

Overlay check (pending user confirmation): the three removed fragments sit at
the chair's area next to the table (see `checking_sheet.png` tiles 1-3); the
clock's new position appears only inside the "added bottle and clock" entry,
whose before/after crops both show the clock - see the failure analysis.

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
   this before submitting. A joint reconstruction across differently oriented
   recordings is harder; the clock/chair region is reconstructed from two very
   different viewpoints.
2. **The auto mode chose `geometric`** (dominant plane 11% of confident
   points). The geometric mode's known weakness - coverage-difference
   artifacts (documented on scene2) - is visible here: 58 candidates, 48
   denied by the VLM, 3 suppressed, 10 confirmed, 5 surviving the filters.
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

### Checking sheet

`outputs/scene5/checking_sheet.png` (5 numbered tiles: before | after crops +
the 3D box per detection) and `checking_sheet.md` (the same as an index).
Please confirm the overlays against the ground truth.
