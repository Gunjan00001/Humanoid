# Autopilot final report

One message, saved as this file. Every number below was measured in this
repository; anything not measured is listed under **Unverified**. The close-out
specification is docs/autopilot_closeout.md.

## Step by step

| step | result | evidence |
|---|---|---|
| 1. Phase 4 gate, six dev scenes, cached data, deterministic rerun | **done, rejected** | `scripts/objfirst_gate.py` → `logs/objfirst_gate.json`; all six scenes identical on rerun; table below |
| 2. Negative result written accurately | **done** | docs/negative_results.md and docs/objfirst.md both state that M1/M3 inherit the detector's clusters and boxes, so the hypothesis is not tested; the lift check and independent inventory are listed as the first next steps |
| 3. Close the branch | **done** | annotated tag `archive/object-first` pushed; remote and local `exp/object-first` deleted; new files copied to main; frozen files and config untouched |
| 4. No more experiments | **done** | naming adoption skipped; every pending experiment listed with its evidence in DEEPDIVE.md §12 |
| 5. Docs and checks | **done** | suites pass, ruff clean, audit clean, fresh clone verified, tag pushed; numbers below |
| 6. Final report | **this file** | |

## Phase 4 gate (new this batch)

| scene | found / GT | extras | fragments | candidates | fused | deterministic |
|---|---|---|---|---|---|---|
| demo | 2/3 | 5 | 0 | 7 | 7 | yes |
| scene2 | 1/1 | 5 | 0 | 6 | 6 | yes |
| scene3 | 1/2 | 10 | 0 | 11 | 11 | yes |
| scene4 | 2/3 | 0 | 1 | 3 | 5 | yes |
| scene5 | 0/2 | 7 | 0 | 7 | 8 | yes |
| scene6 | 1/3 | 5 | 0 | 6 | 6 | yes |
| **overall** | **7/14 = 50.0%** | **32** | **1** | | | **yes** |

Written gates: the regression floor requires demo and scene4 to keep 3/3 with
0 extras. Both fall to **2/3**, and extras rise from 4 to 32. **Not accepted**;
`diff.method` stays `tabletop`.

## Before and after, separated by benchmark

**Development set (six scenes, 14 GT changes), v1.0.0 baseline, unchanged:**

| metric | value |
|---|---|
| type recall | 10/14 = 71.4% |
| localized recall | 12/14 = 85.7% |
| strict move recall | 1 of 4 per run (3 repeat runs) |
| naming | 6/10 = 60.0% |
| extras / fragments | 4 / 3 |

**Held out (never regenerated in any rewrite):** scene5 1/2, scene6 1/3, i.e.
**2/5 = 40%**, extras 3. Identical bytes to v1.0.0.

**SceneDiff:** obj/sc AP **0.0000** on both slices; 7 of 20 validation pairs were
processable with no true positives, 13 of 20 and all 10 test pairs were
unprocessable, and even perfect ground truth masks through the same adapter score
0.1147, so the adapter is not the cause (docs/evaluation.md).

**Phase 4:** 7/14 = 50.0%, 32 extras, 1 fragment, rejected (table above).

## Accepted and rejected changes

Accepted: adaptive sharpness as the default fallback (byte-identical on
demo/scene3-scene6); the readability refactor with golden proof; prepush guard and
hooks; the unattended batch runner; the SAM 3.1 integration (official package,
bf16 autocast, packaged BPE); DINOv3 ViT-L as the matching descriptor (paired
intervals exclude a loss); VGGT-1B as the reconstruction backend (4.1 mm vs
MapAnything's 13.4 mm at 1x); Wilson intervals with unit tests.

Rejected or not adopted: the object-first path (gate table); `semantics.naming_v3`
and `naming_v4` (no significant gain, tiers measured but not adopted); SAM 3.1 as
the change segmenter (4/11 objects, 14 spurious, no tracker on the image route;
kept only as a mask source behind a flag); doubling frames (density not
accuracy); frame-major reconstruction (no gain); scene-graph diff (quality loss);
CLIP as a matcher; the multimodal VSATSIM (beaten by plain box masking); Qwen3.5
sizes as naming replacements (28.4/31.5/29.7% free-form over 317, none better
than the 32B baseline); DINOv2-large as primary matcher.

## Models and licences

| model | role | licence | commercial |
|---|---|---|---|
| SAM 3.1 | target segmenter for the flag-gated route (flag OFF; not used by the released path) | Meta SAM License | **non commercial**; acknowledgement requested for publications |
| SAM 2.1 base-plus | fallback segmenter | Apache-2.0 | yes |
| DINOv3 ViT-L | target matcher for the flag-gated route (not used by the released path) | DINOv3 License | check terms |
| DINOv2-large | fallback matcher | Apache-2.0 | yes |
| VGGT-1B | reconstruction backend | CC BY-NC 4.0 | **non commercial** |
| MapAnything | geometry alternative | Apache-2.0 variant | yes |
| Depth Anything 3 / Pi3X | probed, not integrated | cc-by-nc-4.0 | **non commercial** |
| Qwen3-VL-32B | naming baseline | Apache-2.0 per the card | yes |
| Qwen3.5 (4B/9B/27B) | naming challengers | Apache-2.0 per the card | yes |

Related work cited in the README and docs/models.md: SceneDiff, HD-Epic, UNITE,
VGGT, MapAnything, SAM 2, SAM 3D, PanSt3R. The SceneDiff data is not
redistributed here.

## Git state, with proof

- one branch on the remote, `main`; `exp/object-first` deleted after archiving;
- annotated tags present, including `v1.0.0` and `archive/object-first`;
- `git diff v1.0.0 main -- src/diff.py src/tabletop.py src/geometry.py
  src/describe.py src/align.py src/reconstruct.py` shows only the readability
  docstrings recorded in docs/accuracy.md, with the digest table proving the
  outputs unchanged;
- `git diff v1.0.0 main -- outputs/scene5 outputs/scene6` empty;
- `configs/default.yaml` keeps `diff.method: tabletop`;
- `scripts/prepush_audit.sh` clean at HEAD; identities clean; no absolute home
  paths, account names or tokens in tracked content;
- fresh clone from GitHub: tests pass, `detect.py --help` works, hardlinks/LFS
  intact.

## Unverified, stated plainly

1. The reconstruction residuals are reconstruction units treated as millimetres
   of a millimetre-scale scene; they are not metric-validated (no tape measure,
   no calibration target).
2. SceneDiff numbers are one validation slice with the published protocol; no
   test split was used.
3. Naming numbers come from cached crops of the development scenes, not from new
   recordings.
4. SAM 3.1 tracking was not measured (the image route has no tracker).
5. The Phase 4 gate uses cache-level stand-ins for localization (cluster side and
   label), not 3D IoU against GT boxes.
6. The batch runner has completed geometry tasks and wrote `.done` markers; the
   Phase 4 mask pass and the public-vocabulary run were never submitted.
7. The "agent reviewed, pending user confirmation" naming rows are exactly that.

## HUMAN ONLY (never waited on)

1. Send the GitHub Support request (text at `~/purge/support_request.txt`).
2. Enable GitHub Pages for the viewer if wanted.
3. Record or supply the demo video (`DEMO_VIDEO_URL` is intentionally empty).
4. Review frames for faces/documents before publishing any new footage.
5. Optional new sealed scenes for room-scale evaluation (not evaluated).
6. Delete the local backups (`~/backups`, `~/backups2`, `~/backups3`,
   `~/purge/`) once satisfied.
7. Final submission steps: repository description/topics, release notes for the
   tag, and the report link in the README if desired.

### Addendum: strict move recall and fused entries

Strict move recall per scene in the gate: demo 0.0%, scene2 undefined (no GT
moves), scene3 undefined, scene4 100.0% (one GT move, found), scene5 0.0%,
scene6 0.0%; **1 of 4 per run (3 repeat runs)**, below the baseline note. Fused entries
total **43** across the six scenes, against **24** baseline entries (demo 3,
scene2 5, scene3 4, scene4 3, scene5 5, scene6 4; docs/accuracy.md).
