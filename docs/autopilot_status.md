# Autopilot status

## Phase 1 complete (privacy purge push)

The authorized history rewrite was executed end to end:

1. source: the rewritten repo `~/purge/work` (checked out from
   `~/purge/staging.git`), HEAD at the purge tip, 33 tags, remote pre-purse
   state verified, `git push --force-with-lease` applied
2. LFS store copied from the working clone, `git lfs fsck` OK without the
   `allowincompletepush` escape hatch, `git lfs push --all origin` clean
3. force-with-lease on main then force tags
4. fresh clone from GitHub verified: HEAD at the purge tip, 33 tags with the
   same names, zero hits of the account name, the bare number, the home path,
   the two hostnames, `otheruser-jobs` and the personal gmail anywhere in any ref,
   commit message or tag message; `git lfs pull` and `git lfs fsck` OK; all 7
   test suites pass; the 7 frozen digests unchanged; held-out outputs byte
   identical to v1.0.0
5. the cited hashes refreshed from the commit map in one normal commit; the
   working repo reset to the rewritten history after verifying nothing was
   unpushed; reflog expired and pruned; the v1.2.2 release tag at this commit
6. rollback materials held at `~/backups/` and not deleted:
   `humanoid_all.bundle` (complete history), `mirror.git` and
   `~/purge/commit-map_real.txt`

NEXT (Phase 2, exact commands):

```
# golden outputs recorded for demo, scene2..scene6 already exist in
# tests/golden_summary (regenerate on CPU when scenes change)
git checkout -b refactor/readability          # frozen files only
# per module (geometry, diff, tabletop, describe, reconstruct, align):
#   edit (headers/docstrings only), then
#   python scripts/golden.py check
#   python -m unittest tests.test_summary -q
#   add a row to docs/accuracy.md per file (sha256 pre -> post)
# merge by fast forward: git checkout main && git merge --ff-only refactor/readability
# then delete the branch and tag a patch version
```

# Autopilot status

Progress log for the automated run described in AUTOPILOT.md. Updated after every
step: timestamp, track, step, result, next step.

## State at start (2026-10-11 21:41 IST)

- main is at commit 28e67580 (tag v1.0.0 = 77908f1, plus three docs and CI commits:
  e25defae, 7d48da28, 28e67580). The held-out scene5 and scene6 outputs are on main
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
  limitation paragraph. main = c30337ec, tag v1.0.6.

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
  798b4603 (tag v1.0.6).
- Next: Part C on exp/semantics-v2, first frames.adaptive_sharpness (default OFF),
  post-hoc diagnostic on the 13 unprocessable val pairs and scene2, plus a
  byte-identical check of the development outputs with the flag ON.
