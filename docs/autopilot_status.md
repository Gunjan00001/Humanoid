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
