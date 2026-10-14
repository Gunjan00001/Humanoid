# Autopilot status

Progress log for the automated run described in AUTOPILOT.md. Updated after every
step: timestamp, track, step, result, next step.

## State at start (2026-10-11 21:41 IST)

- main is at commit 485d33f (tag v1.0.0 = 5e5185d, plus three docs and CI commits:
  3fbc9dc, a6fc8c0, 485d33f). The held-out scene5 and scene6 outputs are on main
  and are left untouched.
- Branch exp/accuracy holds the accuracy sprint: Phase A and B (scorer, ground
  truth, failure attribution), the semantics worker, and the D, E and F flags.
  soft_veto, coverage_added, objmatch (re-match only) and fragment merge produced
  no change on any scene, so nothing is accepted and every flag stays OFF.
  Baseline: recall(type) 71 percent, localized 86 percent, strict move recall
  25 percent (1 of 4), name accuracy 60 percent, 4 extras, 3 fragments.
- objmatch2 (object first matching) is blocked: SAM 2 is not installed.
- Generated summaries are humanized on exp/accuracy. README, docs and devlog are
  not.
- No data/sealed directory exists, so the optional sealed step is pending.

## Log

### 2026-10-11 21:41 IST, Track A, branch created
- Created branch docs/humanize from main.
- Result: on docs/humanize, ready for the v1.0.1 doc work.
- Next: humanize the docs, cherry pick the summary wording change, apply the fixes,
  validate with scripts/check_numbers.py, run tests, tag v1.0.1.

### 2026-10-11 21:50 IST, Track A, fixes and cherry pick
- Cherry picked the humanized summary format and its tests plus
  scripts/check_numbers.py (no detection files).
- README quick start now uses the HTTPS clone URL and adds the VGGT install step;
  docs/setup.md documents it and what stays unverified.
- Redacted the cluster username from docs/devlog.md and scripts/repro_scenes.sh.
  No tracked file contains it now.
- detect.py prints the working flags when frame extraction finds too few sharp
  frames.
- Regenerated every committed summary from the existing report.json (text only).
- tests/test_summary.py passes. scripts/check_numbers.py reports only expected
  items: the new status file, the username redaction, and a stale scene2_geo
  summary that lacked its list numbering. No result number changed.
- Next: humanize README, docs and devlog prose (large), then fresh clone, tag
  v1.0.1, push, merge main into exp/accuracy.

### Notes on the other tracks (not started)
- Track B (objmatch2): blocked, SAM 2 is not installed.
- Track C (SceneDiff): not started.
- Track D (freeze and sealed): no change accepted, no data/sealed directory.

### 2026-10-11 22:05 IST, Track A, humanization and release
- Humanized README.md and every docs/*.md (em dashes and spaced hyphens to plain
  punctuation) with scripts/humanize.py. No em dashes remain. Numbers identical
  before and after in every file (checked file by file).
- Frozen detection digests on this branch match docs/heldout.md exactly (all 7).
- tests/test_summary.py and tests/test_ma_contract.py pass.
- Fresh clone from the shared filesystem: LFS videos present, detect.py --help OK,
  vggt imports.
- Next: merge docs/humanize into main, tag v1.0.1, push, verify on the remote,
  then merge main into exp/accuracy.



### 2026-10-12 04:00 IST, projection control passes; freeze recorded; test slice running
- Control (scripts/oracle_projection_control.py): GT masks lifted to 3D and pushed
  through the production adapter path. Per-pair oracle IoU above 0.5 on 6 of 7
  processable val pairs (gym 0.947, bathroom_11 0.667, kitchen 0.761, bedroom_2_3
  0.645, bathroom_1 0.550, bedroom_2_4 0.543; airhockeytable 0.195 on a 60x75 px
  object). obj/sc AP 0.1147, 7 TPs. The adapter is verified end to end, so the
  near-zero validation score is a detection and domain result.
- Freeze recorded in docs/accuracy.md: v1.0.0 configs/default.yaml, detection file
  digests (identical to the v1.0.0 held-out freeze).
- Test slice regenerated at the frozen config (job 11368; earlier test predictions
  came from the buggy adapter and no test score was seen). finish_c5.sh counts
  unprocessable test pairs as empty and evaluates the test slice exactly once.
- README Known limitations and docs/evaluation.md gain the honest small-tabletop
  limitation paragraph. main = 139f507, tag v1.0.6.

### 2026-10-12 04:30 IST, control passes, test slice evaluated once, Part C next
- Projection control passes: per-pair oracle IoU above 0.5 on 6 of 7 processable
  val pairs, obj/sc AP 0.1147 (7 TPs). Adapter verified end to end; the near-zero
  validation score is a detection and domain result.
- Freeze recorded (v1.0.0 configs/default.yaml + digests, identical to the held-out
  freeze).
- Test slice (10 pairs) regenerated at the frozen config and evaluated exactly
  once: all 10 pairs unprocessable at the frozen blur setting, headline all zero
  (0 TP, 29 FN). No test score had been seen before. Committed as produced.
- Failure analysis on the 7 processable val pairs recorded (scale mismatch, change
  semantics, frame budget; plumbing ruled out).
- README + evaluation.md carry the honest small-tabletop limitation. main =
  51ee950 (tag v1.0.6).
- Next: Part C on exp/semantics-v2, first frames.adaptive_sharpness (default OFF),
  post-hoc diagnostic on the 13 unprocessable val pairs and scene2, plus a
  byte-identical check of the development outputs with the flag ON.

### 2026-10-12 05:00 IST, Part C item 1: adaptive sharpness fallback
- frames.adaptive_sharpness (default OFF) implemented: when the absolute blur
  threshold leaves fewer than min_frames, the sharpest N frames by rank are used
  (a global rule, bounded two-pass scan).
- Verified: demo frames are byte-identical with the flag ON (both videos); scene2
  now yields 12+23 frames (it fails at defaults with 1); the unprocessable val
  pairs produce frames too (bed_3_bed_4 13+11, bedroom_1_bedroom_2 15+30,
  hallway_1_hallway_2 14+9).
- Job 11370 runs the post-hoc diagnostic: the 13 unprocessable val pairs and
  scene2 through the full pipeline with the flag ON (val only, never the test
  pairs), then scores them.
- Next: items 2 to 6 (mode/plane check, scale, frame budget, freeze v1.1 + a
  second pre-registered test slice if something passes, docs).
