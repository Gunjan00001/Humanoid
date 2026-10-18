# AUTOPILOT_BATCH2.md: fix the leak regression, finish Phase 3, run Phases 4 to 7

Saved verbatim (minus the owner's opening sentence) after the Phase 1C cleanup, so
any later session can resume from it.

## 0. What I found when I checked the pushed repo (verify each point yourself)

1. Good: main = 07876e5 is the only branch, 226 commits and 35 annotated tags all carry the
   noreply identity, v1.2.3 exists, held out outputs and the frozen semantics are intact.
2. REGRESSION: the cluster account name and its number are back in tracked files and in
   history. A fresh clone shows tracked-tree hits in bakeoff_logs.log (a log file at the
   repo root), docs/autopilot_status.md (line 49) and scripts/job_bakeoff_naming.pbs
   (absolute home paths in the PBS directives and the python path). They entered in commits
   ad72c8c and cbd737e (2026-10-17). The literal from the job-name list also entered history
   in two commits (b73eb78, 0daea1d). All suites were reported green, so the privacy guard
   has a coverage gap: find it and close it (Phase 1C).
3. The Qwen3.5-4B bake off answered "keyboard" for all 317 crops. Your diagnosis (the kwargs
   whitelist dropped pixel_values and image_grid_thw, so the model never saw the images) is
   right. This is a harness bug, not a model result. It must not be recorded or counted as
   a negative result about Qwen3.5.

## 1. Rails

1. No force push and no history rewrite except the ONE rewrite authorized in Phase 1C
   (same safeguards as before: backups, staging, automatic evidence checks, lease, automatic
   rollback). v1.0.0, the archive tags and the held out outputs of scene5 and scene6 are
   never touched. No backup is deleted. Frozen detection files change only with byte
   identical proof.
2. From now on, before EVERY push run scripts/prepush_audit.sh (built in Phase 1C). If it
   reports a hit, fix it before pushing. No exceptions.
3. Claims are verified against the remote or a fresh clone before you state them. Real
   numbers only. Mark everything unverified.
4. No secrets, usernames, hostnames, enrollment numbers or absolute home paths in any
   tracked file, commit, tag or message. Patterns are built at runtime, never stored as
   literals. Agent logs and run outputs go to ignored folders (logs/, outputs_local/,
   cache/), never to the repo root, and PBS scripts use paths relative to the work
   directory (PBS does not expand variables in directives).
5. Branches only when named here (exp/object-first in Phase 4), at most two open, deleted
   after merge. Everything else goes on main with all tests green at every commit.
6. GPU etiquette: stages one at a time, slices chosen by free memory, one large model per
   GPU, never touch other users' jobs, delete your own jobs when finished, models loaded
   from local paths with HF_HUB_OFFLINE=1 on compute nodes, the token never printed or
   written anywhere.
7. A step that fails twice for the same reason: record it and continue with the next
   independent step.
8. TRUE BLOCKERS (stop and report only for these): a Phase 1C check that fails after the
   automatic rollback, a license that forbids our use, disk quota, the queue unusable for 6
   hours with all jobs waiting.
9. HUMAN ONLY (never wait; list them in the final report): the GitHub Support request,
   enabling GitHub Pages, the demo video, new scene recordings (data/sealed/), frame review
   for faces or documents, deleting backups.

## 2. Phase 1C: leak regression (do this first; authorized second rewrite, no GO gate)

1. Make no other commits before the cleanup commit.
2. Normal cleanup commit on main: remove every hit from the tracked tree, in files of ANY
   extension. Untrack bakeoff_logs.log (git rm --cached) and add the pattern for stray root
   logs to .gitignore. Rewrite scripts/job_bakeoff_naming.pbs to use relative output paths
   (cd $PBS_O_WORKDIR inside the script, python from $HOME or a documented environment
   variable). Fix docs/autopilot_status.md (cd "$(git rev-parse --show-toplevel)"). Remove
   the literal if present.
3. Close the guard gap. Find out why the privacy test and the hooks did not catch these
   files (file extension filters, excluded paths, only HEAD scanned, hook not enabled in the
   agent shell, and so on) and write the answer in docs/debugging.md. Fix it: scan ALL
   tracked files regardless of extension (binary files excluded), scan every ADDED line in
   the commit range about to be pushed (git log origin/main..HEAD -p), scan author,
   committer and tagger fields, and build all patterns at runtime (getpass.getuser(),
   $HOME, hostname, numeric part, other users' names read from a local ignored file).
   Provide scripts/prepush_audit.sh and install it as .githooks/pre-push and
   .githooks/pre-commit (git config core.hooksPath .githooks, documented, and verified
   active with git config --get core.hooksPath). Add the same check to the CI workflow.
   Add a test that fails if the hook directory is not executable.
4. Backups: new bundle and mirror of the CURRENT GitHub state under ~/backups3/ (outside the
   repo, never pushed). Keep the older backups.
5. Rewrite in a FRESH clone with git filter-repo: (a) remove bakeoff_logs.log from all
   history, (b) a replace-text file kept outside the repo that maps the home path prefix,
   the account name, the bare number and the job-name literal to neutral words, for contents
   and for messages, (c) a mailmap check (all identities already noreply). Commits before
   ad72c8c must keep their hashes.
6. Staging and automatic evidence, no push yet: push to a local bare repo, clone it fresh and
   verify: (a) zero hits of every audit item in all commits, messages, tags, tree and
   author, committer, tagger fields; (b) identical hashes for all commits before ad72c8c,
   and only ad72c8c and its descendants (and the tags that point to them) differ;
   (c) 35 tags with the same names; (d) the 7 frozen digests, the golden summaries, the held
   out outputs, all tests, the link and budget tests and git lfs fsck pass. Any failure: do
   not push, report.
7. Push only if everything passed: confirm remote main still equals 07876e5 (otherwise
   stop), push with --force-with-lease=refs/heads/main:<remote sha> from the rewritten
   repo (never from ~/Desktop/Humanoid), force push only the tags that changed. Verify from
   a fresh clone from GitHub with the same checks. On any failure run the rollback from
   the ~/backups3 mirror at once, verify it, stop and report.
8. After success: refresh cited hashes of changed commits in one normal commit, reset the
   working repo to the rewritten history after confirming nothing is unpushed, expire the
   reflog and prune, append the old hashes and the third force push time to
   ~/purge/support_request.txt (outside git). Save this file as docs/autopilot_batch2.md.

## 3. Phase 3 (continued): fix the bake off harness, then finish the bake offs

1. Apply your one line fix (drop only the known bad key, as in the working 32B path).
2. Add harness invariants so this class of bug cannot hide again:
   a. Assert that the processor output contains pixel_values (and image_grid_thw where the
      model uses it) with the expected shapes before every generate call.
   b. Visual control on 12 crops: answers with the real image, a blank image and a shuffled
      image must differ; the blank image answer is the text only prior and is reported as a
      baseline.
   c. Degeneracy alarm: if more than 50 percent of answers over at least 20 crops are the
      same string, stop the run and report.
   d. Known answer smoke test on a handful of unambiguous crops before the full run.
3. Correct the documentation: docs/negative_results.md and docs/stack_selection.md must label
   the Qwen3.5-4B attempts as "harness bug, not a model result" and keep them out of the
   decision table. Do not describe Qwen3.5 as having failed.
4. Rerun the naming bake off: Qwen3.5-4B, 9B and 27B against the Qwen3-VL-32B baseline on the
   317 crops (plain crop vs mask tight crop with numbered marks), exact and parent category
   accuracy with Wilson intervals, per tier precision, latency, memory. Download weights on
   the login node, smallest first, check disk first.
5. Finish the other bake offs per docs/stack_selection.md with the decision rules written
   BEFORE the runs: segmentation (SAM 3.1 vs SAM 2.1: are the GT objects segmented in both
   videos with stable track IDs), matching (DINOv3 vs DINOv2 on the earlier controls and the
   cup and clock targets), geometry (VGGT vs Depth Anything 3 vs Pi3X vs MapAnything, then 2x
   and 4x frames: alignment residual, density on objects, metric scale, runtime, memory).
   Use any gated model that the probe reported OK; use fallbacks for the rest and record
   "blocked on access".
6. Fill the decision table (primary and fallback per role, memory plan for one 80 GB GPU
   and a light profile for a 24 GB GPU) and write docs/autopilot_report_batch1.md.
   If no candidate beats the current baseline by the written rules, keep the baseline,
   record it, and skip Phases 4 and 5 (go to Phase 6 and 7).

## 4. Phase 4: object first pipeline (branch exp/object-first)

All new code in new modules (src/objfirst/), wired behind diff.method: objfirst, default OFF.
1. Per video inventory with the selected VLM; concept segmentation and tracking with the
   selected segmenter; static structure (wall, floor, ceiling, window, outlet, shadow,
   reflection) excluded by a written list; masks cached to disk (subprocess if the env
   differs).
2. Lift each instance into the shared 3D frame with the cached depth and poses: centroid,
   extent, multi view appearance feature.
3. Match A to B by label compatibility, 3D size ratio, appearance (selected matcher) and a
   location prior, Hungarian assignment. moved = matched and displaced beyond a threshold
   derived from the measured alignment residual on static structure and the object size
   (global rule, per scene calibration from objects present in both videos). Unmatched in A
   is a removed candidate, unmatched in B an added candidate.
4. Verify with the existing visibility and coverage checks and with VLM verification on mask
   tight crops from both videos. Keep the confidence tiers and the stable (#n) numbers for
   uncertain names. Fuse with the current detector by a documented rule and keep the old path
   intact.
5. Evaluate with scripts/score.py: dev scenes (leave one scene out), SceneDiff validation
   pairs that process with adaptive sharpness ON (post hoc diagnostic only; never the test
   pairs), and the scenes in data/sealed/ if that folder exists (each exactly once, config
   frozen BEFORE looking at them). If it does not exist, list it as pending.
6. Accept the flag only if ALL hold: strict move recall improves, type and localized recall
   do not drop, extras do not rise, scene4 and demo do not regress, three runs are stable,
   outputs are byte identical with the flag OFF, tests pass, no per scene values, and the
   gain does not come from a single scene. Otherwise keep it OFF and write the negative
   result.

## 5. Phase 5: naming adoption

Wire only the parts that passed: part word ban, stable (#n) numbering, hedged tiers, and the
adopted VLM or segmenter naming. Everything behind flags with the invariant test (every
entry has a label, a number and a tier; summaries only from report fields; byte identical
with the flags OFF).

## 6. Phase 6: automatic merge decision

1. Nothing accepted: main stays as is. Archive exp/object-first as archive/object-first and
   delete the branch. Put the stack selection and the negative results on main as docs only.
   Patch tag.
2. Something accepted: merge to main with the accepted flags ON, re run all dev scenes with
   both configs for a like for like table (held out outputs untouched), refresh outputs,
   composites, GIFs, viewers, README and DEEPDIVE, freeze the new config with digests, minor
   tag. v1.0.0 and the v1.2.x tags stay reachable.
3. Sealed scene results, if any, are reported separately from the dev scenes.

## 7. Phase 7: finish and final report

README first screen, DEEPDIVE.md (decision log, what worked and failed, interview prep, kept
in sync with what is on main), docs/code_map.md, docs/branches.md, models and licenses
table, limitations with the numbers, privacy, link and budget tests green, CI workflow green
(say if you cannot see the run), fresh clone smoke test from GitHub, clone size with and
without --depth 1. Write docs/autopilot_report_batch2.md and send ONE final message with:
per phase done, skipped or failed with evidence; the Phase 1C verification output and the
cause of the guard gap; the harness invariants; before and after tables (dev, SceneDiff
validation diagnostic, sealed) clearly separated; accepted and rejected changes with
numbers; models and licenses added; git state with proof that the rails held; runtimes;
everything unverified; and the HUMAN ONLY list.
