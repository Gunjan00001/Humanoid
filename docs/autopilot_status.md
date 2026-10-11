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


