# Autopilot status

RESUME HERE (batch 1 phase 3; the 4B chain is a recorded negative with the
cause identified; the fix is one line):

```
# 1. in scripts/bakeoff_naming.py, replace the whitelist filter with a blacklist:
#      inputs.pop("mm_token_type_ids", None)      # the only rejected key
#    (the whitelist dropped pixel_values/image_grid_thw, so the model saw text only
#     and answered "keyboard" for all 317 crops - cause proven, see
#     docs/negative_results.md attempt 4/5)
# 2. resubmit: qsub scripts/job_bakeoff_naming.pbs ; read logs/bakeoff_naming_Qwen3.5-4B.json
#    expected: non-constant answers; then exact/parent with Wilson intervals
# 3. same for 9B and 27B (download on the login node, smallest first, 19.3 GB / 55.6 GB)
# 4. bake offs b (SAM 3.1 vs SAM 2.1), c (DINOv3 vs DINOv2), d (geometry) per
#    docs/stack_selection.md rules written before the runs
# 5. decision table, then docs/autopilot_report_batch1.md and the final message
```

# Autopilot status

RESUME HERE (batch 1 phase 3, bake-off attempt 4 plan; the 4B attempt is a
recorded negative: placeholder echo, 301 of 317 replies `{"name": "text"}`,
16 wrong names; no gate decision possible):

```
# attempt 4 (two prompts, in order, stop at the first that works):
#   1. no placeholder anywhere: PROMPT = "Identify the object in this image,

#      reply with JSON {name: object}", and strip ``` fences if present in
#      parse (r"```[a-z]*(.*?)```" then json)
#   2. if parse fails again, run the SAME model on the plain crop variant:
#      scripts/build_naming_set.py --no-blank (append the flag: skip patch[~sub]=0)
#      so the crop keeps its photographic background at 4B scale
# then: bake off (a) for 9B and 27B (download on the login node, smallest first),
#       then bake offs b (SAM 3.1 vs SAM 2.1), c (DINOv3 vs DINOv2 controls),
#       d (geometry: VGGT vs DA3 vs Pi3X vs MapAnything), each per
#       docs/stack_selection.md rules 1-6 written before the runs, then the
#       batch 1 report (docs/autopilot_report_batch1.md) and the final message
```

# Autopilot status

RESUME HERE (batch 1 phase 3, naming bake-off attempt 2 for Qwen3.5-4B:
the run fails at generate with unused model_kwargs; exact resume, in order):

```
# 1. reproduce the full error (the 80-char cap hid it):
cd /home/user/Desktop/Humanoid
GIT_LFS off; qsub scripts/job_bakeoff_naming.pbs; tail logs/bakeoff_naming_job.err
# 2. in scripts/bakeoff_naming.py, strip the INPUT KEYS the model rejects:
#    build local_inputs = {k: v for k, v in inputs.items() if k not in ("mm_token_type_ids",
#    "pixel_values"?? ...)} per the full error's list, and raise the error cap from 80 to 200
# 3. resubmit (qsub scripts/job_bakeoff_naming.pbs), read logs/bakeoff_naming_Qwen3.5-4B.json
# 4. then the same for 9B and 27B (weights: download on the login node, smallest first),
#    then bake offs b/c/d per docs/stack_selection.md, then the decision table there,
#    then docs/autopilot_report_batch1.md and the final message per batch 1
```

# Autopilot status

RESUME HERE (batch 1 phase 3, bake offs RUNNING):

```
qstat | grep bakeoff        # job 11670, 4 GB challenger (Qwen3.5-4B)
tail -f logs/bakeoff_naming_job.log      # done = BAKEOFF_DONE
ls logs/bakeoff_naming_Qwen3.5-4B.json  # results when done
# then: python scripts/make_bakeoff_table.py (TODO or print the json)
# next models: 9B (19.3 GB, download started on the login node), 27B (55.6 GB)
# then phases b/c/d of docs/stack_selection.md and the batch 1 report
```

# Autopilot status

RESUME HERE (batch 1 phase 3: probes and rules done in docs/stack_selection.md;
next the offline bake offs, then Phase 7's batch 1 report):

```
# bake offs per the plan in docs/stack_selection.md:
# a) naming: Qwen3.5 family vs Qwen3-VL-32B cached baseline on the 387 crops
#    + dev crops (mask-tight crop variant included), Wilson intervals
# b) segmentation: SAM 3.1 vs SAM 2.1 on the dev scenes (stable track IDs)
# c) matching: DINOv3 ViT-L vs DINOv2 on the recorded control pairs
# d) geometry: VGGT vs DA3 / Pi3X / MapAnything (2x, 4x), residual, density,
#    metric scale, runtime, memory
# write results into docs/stack_selection.md per decision rule 1-6, then the
# batch 1 report file docs/autopilot_report_batch1.md and the final message
```

# Autopilot status

RESUME HERE (batch 1 phase 2 code COMPLETE: geometry b73eb78, diff 8ed08b9,
tabletop 66a8271, describe d44de88, reconstruct 5e449fc, align 7f23f4f, all
merged into main by fast forward, branch deleted; next: patch tag):

```
git tag -a v1.2.3 -m "readability pass (headers and docstrings only)" && git push origin main
git push origin v1.2.3
# then Phase 3 (stack selection) per docs/autopilot_batch1.md, then the Phase 7 report
```

```
git checkout refactor/readability   # unless already there
# per module (tabletop, describe, reconstruct, align): headers and why-docstrings only
python scripts/golden.py check
python -m unittest tests.test_summary tests.test_golden_summary tests.test_heldout_claims -q
ruff check tests scripts src/naming
git add -A && git commit -m "readability: <module> documented (frozen file)"
python - <<'PY'   # append one digest row: sha256 pre -> post to docs/accuracy.md
PY
git checkout main && git merge --ff-only refactor/readability
git branch -d refactor/readability && git push origin main
git tag -a v1.2.3 -m "readability pass (headers and docstrings only)" && git push origin v1.2.3
# then Phase 3 (stack selection) per docs/autopilot_batch1.md, then the Phase 7 report

# Phase 2 next step: the golden outputs already live in tests/golden_summary.
git checkout -b refactor/readability
# iterate per frozen module (geometry, diff, tabletop, describe, reconstruct, align):
#   headers + why-comments + docstrings only, no behavior change, then in this order:
#   python scripts/golden.py check
#   python -m unittest tests.test_summary tests.test_golden_summary -q
#   python -m unittest tests.test_heldout_claims          # frozen digest rows
#   ruff check tests scripts src/naming                    # E4,E7,E9,F
#   append one digest row (sha256 pre -> post) to docs/accuracy.md
#   git commit -m "readability: <module> documented"
# after the last module:
git checkout main && git merge --ff-only refactor/readability
git branch -d refactor/readability && git push origin main
git tag -a v1.2.3 -m "readability pass (headers and docstrings only)"
git push origin v1.2.3
# then Phase 3 stack selection (docs/stack_selection.md), Phase 4 object-first on
# exp/object-first, Phase 5 naming adoption behind flags, Phase 6 merge decision,
# Phase 7 final report per the master autopilot.
```

utopilot status

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
   the two hostnames, another user's jobs and the personal gmail anywhere in any ref,
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
