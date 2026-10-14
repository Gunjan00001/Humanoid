# Autopilot status

Progress log for the automated run described in AUTOPILOT.md. Updated after every
step: timestamp, track, step, result, next step.

## Resume here (2026-10-11 22:20 IST)

Track A is done. Track C preflight is partial. Track B has not started.

Next steps and the exact commands:

1. Track B, SAM 2 install (2 hour time box), branch exp/accuracy:
   - `conda create -n spatialdiff-sam python=3.10 -y`
   - `~/miniconda3/envs/spatialdiff-sam/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126`
   - `git clone https://github.com/facebookresearch/sam2` then `pip install -e .`
   - download the SAM 2.1 base-plus checkpoint to ~/models/ on the login node,
     verify license and size, record in docs/models.md
   - run the probe on scene6 (cup) and scene5 (clock): automatic masks in both
     videos, and does the DINOv2 similarity of the paired crops clear the match
     gate. Write numbers to docs/accuracy.md. Pass gate: segmented in both videos
     and matched in at least 2 of the 3 targets.
2. Track C C1 remaining gates: evaluator smoke test on 3 val pairs, the frame
   convention, ten per-pair facts, compute estimate. Needs the 11.9 GB zip under
   data/external/scenediff/ (gitignored).
3. Then C2 (pre-register the slice), C3 to C5, and Track D.

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



### 2026-10-11 22:20 IST, Track A done, Track C preflight partial
- Track A: merged docs/humanize into main, tagged v1.0.1 (98904ed), pushed, tag
  verified on the remote. Merged main into exp/accuracy (summary JSON conflicts
  resolved by taking main's version).
- Track C C1: all sources reachable; real zip is scenediff_benchmark.zip
  (11,919,571,401 bytes); prediction format, metrics and evaluator usage recorded
  in docs/evaluation.md; citations added to README and docs/data.md. Remaining C1
  gates (evaluator smoke test, frame convention, 10-pair facts, compute estimate)
  pending.
- Track B: not started.

### 2026-10-11 22:10 IST, Step 1 done, Step 2 and 3 started
- Step 1 (v1.0.2): README quick start links the weights (docs/setup.md); setup.md
  lists VGGT (CC BY-NC 4.0), Qwen3-VL-32B/8B and DINOv2 with sizes, licenses and
  folders; README top keeps the dev/held-out split table, limitations, heldout link
  and the DEMO_VIDEO_URL placeholder; all seven README viewer URLs are plain git
  (no LFS). Tests pass, fresh clone OK. main = a854181, tag v1.0.2 = b869f28.
- Step 2 (Track B): SAM 2 install running in logs/install_sam2.sh (env
  spatialdiff-sam, torch cu126, facebookresearch/sam2, checkpoint
  facebook/sam2.1-hiera-base-plus into ~/models/).
- Step 3 (Track C): SceneDiff zip downloading to data/external/scenediff/
  (gitignored).
- Next: when the install finishes, run the SAM 2 probe on scene6 (cup) and scene5
  (clock); when the zip finishes, extract splits and the selected pairs and run the
  evaluator smoke test.

### 2026-10-11 22:55 IST, Track B done (negative), Track C download done
- SAM 2 installed (env spatialdiff-sam, facebookresearch/sam2, Apache-2.0, SAM 2.1
  base-plus, 323,606,802 bytes) and recorded in docs/models.md.
- Probe (job 11017) on scene6 cup and scene5 clock: SAM 2 masks the crops, but
  DINOv2 similarity is 0.190 and 0.219, below the 0.65 gate. Probe FAILS, so
  objmatch2 is not built. Written to docs/negative_results.md and docs/accuracy.md.
- SceneDiff zip downloaded (12 GB) to data/external/scenediff/ (gitignored).
- Next: extract splits and the selected pairs, run the evaluator smoke test, then
  complete the C1 gates.

### 2026-10-11 23:10 IST, Track C C1 progress
- Zip downloaded (12 GB). Extracted splits and 10 val varied pairs. Verified the
  real structure and frame convention; recorded in docs/evaluation.md. Key
  discrepancies: raw videos are .MOV, no metadata.json, pair names are real, zip
  needs UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE.
- Evaluator cloned to third_party/scene_diff.
- Next: evaluator smoke test (perfect/empty prediction on 3 val pairs), 10 per-pair
  facts, compute estimate, then C2.

### 2026-10-11 23:25 IST, held-out restore fix (v1.0.3)
- The v1.0.1 summary regeneration had changed the held-out scene5/scene6 summary
  text. Restored those six files to v1.0.0 and made tests/test_summary.py accept
  both summary line formats. main = e4aed51, tag v1.0.3. git diff v1.0.0 main for
  outputs/scene5 and outputs/scene6 is now empty. Merged main into exp/accuracy.
- Status of tracks: A done (v1.0.1/2/3), B done (negative), C C1 partial.

### 2026-10-11 23:45 IST, Track B closed, C1 nearly done
- Track B closed as a real negative result (controls positive 0.704/0.914,
  negative 0.285/0.085, targets below the gate). objmatch2 not built.
- C1: per-pair facts, adapter notes and compute estimate (3 to 4 GPU hours,
  under the cap) recorded in docs/evaluation.md. Evaluator internals documented.
- pycocotools installed.
- Next: build the perfect-prediction smoke test, then C2 (pre-register the slice).

### 2026-10-11 24:10 IST, Track C C1+C2 done
- C1 complete: evaluator smoke test passes (perfect 1.0000 on all four metrics,
  empty 0.0000). The real prediction format is object first with resampled frame
  ids (raw frame // 30), not the README's video-first shape. pycocotools and
  faiss-cpu installed. Recorded in docs/evaluation.md.
- C2 complete: data/external/scenediff_slice.json (20 val + 10 test, varied,
  sorted ids) committed.
- Next (C3): write src/eval_scenediff.py plus scripts/job_scenediff.pbs. For each
  pair, run the pipeline on the two original videos, project each change into the
  annotated GT frames, and write object_masks.pkl in the object-first format with
  resampled frame ids. Dry run on 3 val pairs (logs/eval_pred_* has the format
  example). Then C4 (20 val pairs) and C5 (10 test pairs, once).

### 2026-10-12 00:25 IST, Track C C3 in progress
- Wrote src/eval_scenediff.py (projects each report change into the GT annotated
  frames, writes object_masks.pkl object-first with resampled ids) and
  scripts/run_scenediff.sh (runs detect.py per pair, then the adapter).
- Dry run on 3 val pairs launched (gpu-run, 2 GPUs). Slow: the node was busy so
  the 32B sharded with CPU offload, which makes the describe stage very slow.
- Next: finish the dry run, then C4 (20 val pairs) and C5 (10 test pairs once).

### 2026-10-12 00:45 IST, Track C C3 done, C4/C5 running
- C3 dry run passed: src/eval_scenediff.py produced object_masks.pkl that the
  evaluator accepted and scored (low scores, which is expected and not a gate).
- C4/C5 launched as PBS job 11027 (12 h): runs the 20 val pairs (logs/val20.txt)
  then the 10 test pairs (logs/test10.txt), predicting to logs/eval_pred_val and
  logs/eval_pred_test.
- Next: when 11027 finishes, run the evaluator on the val slice, freeze the
  config (v1.0.0 default, nothing accepted), then run the test evaluator once and
  write the failure analysis.

### 2026-10-12 01:00 IST, test-eval rule recorded; 11027 running
- Rule recorded in docs/evaluation.md: no evaluator run on the test predictions
  until the validation sanity check passes (valid pkl per pair, H/W match,
  scores not degenerate, 3 overlay spot checks). If an adapter bug is found, fix
  as a pure bug fix, document it, re-run both slices, and report the test slice
  as run twice with no test score seen before the fix. If it passes, freeze,
  record digests, then run the test slice once.
- Job 11027 running (val pairs producing predictions; test predictions are built
  but the evaluator will not touch them until the sanity check passes).
- Next: wait for 11027, run the validation sanity check, freeze, evaluate val,
  then evaluate test once, then Track D and the final report.

### 2026-10-12 00:00 IST, C4/C5 job 11033 running (slow)
- Bug fixed: only 10 val pairs were extracted; the driver ran 20. Extracted all 30
  pairs, documented in docs/evaluation.md, re-submitted as job 11033. No test score
  was produced before the fix.
- 11033: 8/20 val predictions done, running slowly (the 32B CPU-offloaded on the
  busy node; the gym pair already has 68 VLM verifications and is still going).
- Next: when 11033 finishes, sanity-check the val predictions (valid pkl, H/W
  match, non-degenerate scores, 3 overlay looks), freeze (v1.0.0 config + digests),
  evaluate the val slice, then evaluate the test slice once, then Track D.

### 2026-10-12 02:55 IST, Part A done (v1.0.4); Part B re-running after adapter bug 2
- Part A: docs/challenge-fit merged to main, tag v1.0.4 pushed and verified.
  README first screen: 3 sentences, DEMO_VIDEO_URL placeholder, headline scene4
  GIF, minimal quick start, "Challenge requirements and where to find them" table.
  Added Examples (scene4 verbatim, scene6 held-out verbatim), Design choices and
  tradeoffs, docs/design_note.md, the occlusion/visibility table with a real
  rejected candidate, Hardware and a no-GPU path, More evidence, Licenses and
  citations; handoff carries the repo description and topics. Tests pass, fresh
  clone OK.
- Part B: the validation sanity check found adapter bug 2 (predictions were empty
  because the adapter looked for report.json in the wrong place). Fixed as a pure
  bug fix, documented (bugs 1 and 2), both slices re-run as job 11367. Six
  validation pairs fail upstream at the frozen blur threshold (fewer than 8 sharp
  frames), reported as unprocessable, not changed.
- Next: finisher runs the sanity check + validation evaluator on 11367's output,
  then freeze and the single test evaluation.
