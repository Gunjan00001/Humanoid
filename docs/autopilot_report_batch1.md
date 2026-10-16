# Autopilot batch 1 report

## Phase 1B (author and tagger identity) — DONE, verified on GitHub

Found and fixed, exactly as the finding said: one commit carrying the cluster
login as author and committer, five annotated tags carrying the personal gmail
as tagger, and one literal in the status doc.

1. identity pinned (global + repo, useConfigOnly), verified with git var.
2. tests/test_no_personal_paths.py gained a history check for every commit
   (author, committer) and every annotated tag (tagger name and email); they
   found both leaks, ran as expected failures until the rewrite landed, and now
   pass flat (12 checks on the file).
3. the literal is out of docs/autopilot_status.md, worded as
   "another user's jobs".
4. new backups under ~/backups2/ (bundle + mirror of the pre-rewrite GitHub
   state, 36 refs), older backups kept.
5. fresh clone rewrite with two-entry mailmap (cluster identity -> noreply for
   author and committer; personal gmail -> noreply for the taggers) and the
   another user's jobs replacement rule. The full LFS store answered a first failed push
   (missing objects in the disposable clone), fixed by copying the store.
6. staging at ~/purge/staging2.git, fresh clone verify_b: all audit items at
   zero across pickaxe, messages, tag messages, tree AND identity fields (the
   only identities in the whole history are the public ones, 207 commits and
   34 taggers); the seven frozen digests unchanged; held-out outputs identical
   to v1.0.0; the tests, the link and budget tests all pass.
7. push: force-with-lease against the exact remote sha, then the v1.2.2 tag
   force-updated. Fresh clone from GitHub: main at the rewritten tip, all three
   change commits carry the public identity, 34 tags with 34 noreply taggers,
   git lfs pull and fsck OK, all seven suites pass.
8. follow-up normal commit added the now-passing identity tests and the shared
   wording; the working repo reset to the rewritten history, stale local
   branches (pre-purge copies, content already in the rewritten archive tags)
   deleted, tags force-updated locally, reflog expired, pruned; the pre-purge
   and pre-rewrite objects are gone from the local store
   (0880477... and 9951932... no longer resolve). The support request gained
   the second force push, the changed triple and the tag object.

Final state: remote main at the batch-1 tip, v1.2.2 on the rewritten history,
Desktop equal to it, tests 7/7.

## Phase 2 (readability refactor) — staged, not started

Golden summaries already recorded for demo, scene2, scene3 and scene4
(tests/golden_summary, byte identical, CPU only). The held-out scenes are out
of scope by design. The branch refactor/readability does NOT exist yet: the
batch ran out of execution budget right after the Phase 1B verification, and
no refactor commit was started, so main is consistent and nothing was left
half done. Resume commands are at the top of docs/autopilot_status.md.

## Phase 3 (stack selection) — not started

Not reached. The HF probe plan lives in docs/autopilot_batch1.md for the next
session (probe tiny files first, then download on the login node, then the
bake offs with decision rules written first).

## Could not verify

The GitHub Actions run of the lint and test workflow (no gh CLI here; the
workflow file exists and the same tests pass locally).

## HUMAN ONLY list

GitHub Support request (text ready at ~/purge/support_request.txt), enabling
GitHub Pages, the demo video, new scene recordings, frame review, deleting
backups when you are satisfied.
