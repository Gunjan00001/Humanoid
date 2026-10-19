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


## Phase 3 (stack selection) — probes and rules done; bake offs running

Access probes: all the candidate repos resolve (sam3.1 and sam3 gated-accepted,
Qwen family not gated, DA3-GIANT cc-by-nc-4.0, Pi3X cc-by-nc-4.0, two repo
names corrected in docs/stack_selection.md). Decision rules were written before
any run. Bake off (a) attempted for the smallest challenger
(Qwen/Qwen3.5-4B, apache 2.0, 8.8 GB local): three failed attempts recorded as
a negative in docs/negative_results.md (offline cache miss fixed with the local
weights directory; unused model_kwargs on generate fixed by keeping only the
forward keys; placeholder echo 301 of 317 replies at attempt 3, latency
0.44 s per crop, peak 8.9 GB). Qwen3-VL-32B remains the naming baseline;
attempt-4 plan (no placeholder, and a non-blank crop variant for the small
model) is the resume command at the top of docs/autopilot_status.md.


---

# Batch 1 update (current evidence, supersedes the numbers above where they differ)

## Decision table as it stands (written rules applied)

| role | primary | fallback | evidence state |
|---|---|---|---|
| matching | **DINOv3 ViT-L** | DINOv2-large | accepted: GT-based pairs, AUC 0.805 CLS / 0.855 patch, paired difference excludes a loss on both descriptors (CLS +0.050 [+0.004, +0.100]; patch +0.115 [+0.100, +0.131]) |
| segmentation | **SAM 2.1 base-plus** | — | SAM 3.1 text route blocked on integration; the download is sam3.1_multiplex.pt (Meta format, Sam3VideoModel), not a transformers image checkpoint |
| naming | **Qwen3-VL-32B (incumbent)** | Qwen3.5 family | 4B/9B/27B: all-crops 28.4% / 31.5% / 29.7%; in-vocabulary 44.1% / 46.1% / 41.0%; strict group (identical 63 crops) 25.4% / 25.4% / 23.8%; the 32B over this crop set is not completed |
| public-vocabulary closed set | implemented, not run | 301-word result kept | data/vocabulary_public.json: 1,253 names (LVIS 1,203 + Objects365 v1/v2 + COCO 80), sources and licences recorded; harness flag --vocab public; job budget closed |
| geometry | VGGT-1B (incumbent) | MapAnything (local) | not started |

## Corrections made in this batch
- Wilson intervals: the reporting helper returned the centre as the lower bound; fixed in
  src/naming/stats.py with known-value tests, every published interval recomputed, both
  denominators stated (all crops and named crops).
- The "unparsable" bucket was out-of-vocabulary naming: all three models answered 317/317,
  zero format violations.
- Strict-group denominators differed per model; fixed with one truth table, identical 63
  crops for every model.
- Matching pairs were circular (the pipeline's own matches); rebuilt from ground truth.
- Context ablation (9B): blanked 38.5% [33.1, 44.1], 20% margin 33.1% [28.0, 38.6], frame
  39.5% [34.1, 45.1], n = 299 (18 variant crops failed to load and count as errors).
- SAM 2 mask-tight controls rerun blocked on the non-square patch grid (200 tokens vs 196
  cells); two fixes tried, exact next step recorded.

## Not completed, never inferred
32B free baseline (job 11795 exit 1; 11790 failed on a literal $HOME in qsub -v, later
fixed); full-vocabulary loglik runs (11791/11792/11794 exit 1); SAM 3.1 per-frame text
segmentation (checkpoint format, see above); controls rerun (grid mapping); geometry
(not started).

## Git state and rails
Only the two authorized rewrites (Phase 1B identity, Phase 1C leak regression) used force
pushes, with the tag updates they required; no other force push or history rewrite; no
backup deleted; held-out outputs and v1.0.0/archive tags untouched; the prepush audit ran
clean before every push; suites green at every commit. This batch also removed 774 variant
crop files that `git add -A` had swept in (repo budget back to 412 tracked files, test
green), keeping the three manifests.

## HUMAN ONLY
GitHub Support request (~/purge/support_request.txt), enabling GitHub Pages, the demo video,
new scene recordings under data/sealed/, frame review for faces or documents, deleting the
backups.
