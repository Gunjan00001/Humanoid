# Handoff, brief for the next agent

**State: `v0.12.0` (2026-10-12), working tree clean, everything pushed.**
Read this first, then [`rules.md`](../rules.md), [`gpu.md`](../gpu.md),
[`docs/heldout.md`](heldout.md) and the tail of [`docs/devlog.md`](devlog.md).

---

## 1. TL;DR

The system takes two phone videos of a small indoor space, builds a joint 3D
reconstruction, detects **moved / added / removed / replaced** objects and
writes a natural-language summary. It runs with **one command**:

```bash
python detect.py --a before.mp4 --b after.mp4 --name NAME
```

Five recordings have been run end-to-end. The demo, scene3 and scene4 reports
reproduce identically under a full re-run; scene2 does not (documented); scene5
is the **held-out** scene and its honest result (1/2 type-only) is in
[`docs/heldout.md`](heldout.md).

**The three things to do first (in order):**

1. **Enable GitHub Pages** (repo owner action, one click, see §5.1) so the
   interactive viewers become public.
2. Read the **held-out rules** (§6) before touching anything detection-related.
3. Pick up the **next steps** in §4 (they are ordered by value).

---

## 2. Where everything is

| What | Where |
|---|---|
| One-command runner | `detect.py` (validation, auto mode, PBS submit+poll, `--render-only`) |
| Pipeline stages | `src/{extract_frames,reconstruct,align,diff,describe,report,visualize,summary}.py` |
| Natural-language summary | `src/summary.py` (report-only, one VLM call + template fallback) |
| Config (all thresholds) | `configs/default.yaml`, **never hard-code** |
| Tests | `tests/test_summary.py`, `tests/test_ma_contract.py` (fast, no GPU) |
| Held-out protocol + result | `docs/heldout.md` |
| Results (all scenes, real numbers) | `docs/results.md` |
| Timing measurements | `docs/timing.md` |
| Every step incl. failures | `docs/devlog.md` |
| Cluster access | `gpu.md`, `docs/setup.md` |
| Outputs (committed) | `outputs/` = demo, `outputs/{scene2,scene2_geo,scene3,scene4,scene5}/`, `outputs/scene5_tabletop_posthoc/`, `outputs/scene5_reencode_check/` |
| Caches (git-ignored, needed for re-runs) | `cache/`, `cache/{scene2,scene3,scene4,scene5,…}` |
| Knowledge graph (optional) | `graphify-out/`, `graphify query "…"`, `graphify update .` after code changes |

---

## 3. Environment cheat-sheet (learned the hard way)

- **Python**: always the absolute interpreter
  `~/miniconda3/envs/spatialdiff/bin/python`, the session PATH puts a shared
  anaconda 3.8 first, so bare `python`/`conda run` may pick the wrong env.
- **GPU access**: `gpu-run -t 00:30:00 -g 2 '<cmd>'` (interactive) or `qsub`
  (`logs/job_*.pbs` templates). `gpu.md` has the full reference.
- **Always `unset CUDA_VISIBLE_DEVICES`** in GPU jobs (PBS sets it empty and
  hides all GPUs). `detect.py` handles this for its own jobs.
- **Queue limits** (`gpu` queue): **16 CPUs per user total**, 64 GB per job,
  5 jobs, max 16 CPUs/24 h per job. `describe` needs `mem=60gb` (the 32B VLM
  otherwise gets OOM-killed by the cgroup).
- **The per-user CPU quota is the real bottleneck**, not GPUs. The persistent
  `gpu-session` holds 4 CPUs; when the user's other jobs fill the 16, new jobs
  queue. Pattern used throughout: **stop the idle session → run → restart it**
  (`~/bin/gpu-session stop|start|status`). Check `qstat -f <id> | grep comment`
  for the exact reason a job is queued.
- **PBS buffers job output**: `-o` log files appear only at job end (or in
  chunks). Progress lines during a run can only show state + elapsed.
- **Clock offset**: the compute node's log timestamps are exactly **3:30:00
  behind** the login node's clock (verified; needed for timing analysis).
- **git-lfs is installed on the login node only**, a clone must live on the
  **shared home filesystem**, never `/tmp` (node-local; the PBS job runs on
  another machine and cannot see it). `detect.py` warns about `/tmp`.
- **Models** (in `~/models`): `VGGT-1B`, `Qwen3-VL-32B-Instruct`,
  `dinov2-large`, DA3, MapAnything (isolated `spatialdiff-ma` env). Load times:
  VGGT 8–14 s, Qwen3-VL 26 s warm (2:07 cold/network FS).
- **Runtime expectations** (measured, `docs/timing.md`): cold scene 5–7 min on
  the node; warm `--force` re-run 2–3 min; `--render-only` 2:09–5:46 per scene
  (CPU, needs Chrome for the GIF).

---

## 4. What to do next (prioritized)

### 4.1 GitHub Pages (5 minutes, owner action)
The five `viewer.html` files are committed and the links are in
[`docs/hosting.md`](hosting.md) and the README. Enable **Settings → Pages →
Deploy from a branch → `main` / (root)**. No `gh` CLI exists on the cluster.

### 4.2 scene5: the missed clock move (post-hoc only, read §6 first)
The held-out result stands as committed. Any improvement must be labelled
*"post-hoc, not part of the held-out result"* and reported on scene5
separately. The failure analysis (`docs/heldout.md`) says: the clock's origin
cluster did not survive the filters, so there was no removed+added pair for the
move matcher. Directions worth exploring:
- a move path that pairs an arrival with a **vanished origin that was filtered
  out** (e.g. re-check filtered clusters for a matching crop/DINOv2 signal);
- the auto-mode **sensitivity to encoding** (the re-encoded b.mp4 flips the
  plane coverage from 11% to >50% → `tabletop`): consider making the mode
  decision robust (e.g. coverage over multiple confidence percentiles) , 
  this is a detection-adjacent change, so it needs the full regression suite
  and a fresh held-out scene to prove it;
- an **orientation-normalization preprocessing step** (scene5's A is landscape,
  B portrait): rotating frames before extraction would change detection, so it
  must be validated on a new held-out scene, not tuned on scene5.

### 4.3 scene2: run-to-run variability (documented, unresolved)
A full `--force` re-run gives 2 entries instead of the committed 1 (the
coconut correctly named + the apple false positive). The cause is the
recomputed reconstruction + threshold-sensitive geometric diff. Useful work:
measure the spread over 3–5 re-runs, check whether torch deterministic flags
change it, and document a stability expectation per mode.

### 4.4 scene3: extras and naming
2 extra entries + unreliable names remain (documented). The 27 unverified
composites in `outputs/scene3/changes/` are heavy, consider rendering
unverified composites only on request.

### 4.5 scene6 is already here, the next held-out run (`data-v0.5`)

`data/scene6/` was added while this handoff was being written and **has never
been run**. Ground truth (`data/scene6/ground_truth.md`): **cup moved, remote
removed, apple added** (all three change types in one scene). Properties
(measured): both videos **landscape 860x480 @ ~30 fps**, 55.5 s / 49.1 s , 
the orientation lesson from scene5 was applied. Both are **> 5 MB and stored
in Git LFS** (9.1 / 8.1 MB), so a plain clone cannot read them.

To run it as a held-out scene, follow the protocol (§6) exactly:

1. **Record a new freeze** at the commit you start from: append to
   `docs/heldout.md` the commit hash + sha256 of the 7 detection files
   (`src/diff.py`, `tabletop.py`, `geometry.py`, `describe.py`, `align.py`,
   `reconstruct.py`, `configs/default.yaml`), the v0.12 files are unchanged
   since the scene5 freeze, but freeze *your* starting point.
2. Run **exactly once**, cold, defaults, no overrides:
   `python detect.py --a data/scene6/a.mp4 --b data/scene6/b.mp4 --name scene6`
3. Commit the outputs as produced, build the checking sheet, write the
   per-change table + extras + unverified + the verbatim summary + a failure
   analysis, and have the user confirm the overlays.
4. Only **after** the held-out run: consider re-encoding both videos to
   < 5 MB (libx264 crf 26 keeps 860x480@30) so plain clones can read them , 
   the scene5 measurement showed a re-encode **changes** the reconstruction
   and the auto-mode choice, so it must never replace the held-out inputs.
   (Alternative: document `git lfs pull` for scene6.)

### 4.6 Optional / stretch
- Qwen3-VL-8B vs 32B ablation (~15–20 min GPU), does the smaller model keep
  the demo's 3/3?
- MASt3R backend (HF repo gated for this account) or COLMAP (unimplemented).

### 4.7 Housekeeping
- `cache/` holds every reconstruction (several GB). Re-runs depend on it , 
  keep it unless you are sure.
- After code changes: `graphify update .` and append to `docs/devlog.md`.
- Tag format `v<major>.<minor>.<bugs>` from `main` after verification; never
  move published tags.

---

## 5. Non-negotiables (from `rules.md` + this project's history)

1. **No fabricated numbers.** Every number in the docs came from a real run;
   placeholders are marked. If something cannot be measured, say so (e.g.
   `nvidia-smi` snapshots were never captured, GPU utilization at job start is
   unmeasurable retroactively).
2. **All thresholds in `configs/default.yaml`**, overrides only via the CLI
   flags (documented per scene).
3. **Held-out protocol** (§6).
4. **Devlog everything**, including failures and root causes.
5. **GPU policy**: use idle GPUs, never wait on a busy one, never kill the
   user's other jobs; the `gpu-session` may be stopped temporarily to free the
   CPU quota, restart it afterwards (the waiters did this).
6. **Tags**: from `main`, after verification; format `v<major>.<minor>.<bugs>`.

---

## 6. Held-out protocol (do not violate)

- `docs/heldout.md` freezes the detection files (sha256) per held-out scene.
  **No fix may touch those files for that scene's result.**
- A held-out scene is run **exactly once**, cold, defaults, no overrides.
- After the run: commit the outputs as produced; write the per-change table,
  the extras, the unverified list, the verbatim summary and the failure
  analysis; build the checking sheet
  (`scripts/checking_sheet.py --out outputs/<name>`) for human review.
- Any later change is **post-hoc**: label it, keep it in a separate output
  directory, and report it separately from the held-out result.
- The scoring rule for scene5 is pre-registered in `docs/heldout.md`.

---

## 7. Commands you will actually use

```bash
# one command (from the repo root; submits + polls a PBS job on a login node)
~/miniconda3/envs/spatialdiff/bin/python detect.py --a data/a.mp4 --b data/b.mp4 --name demo

# re-render only (no GPU, reuses caches; stops if a cache is missing)
~/miniconda3/envs/spatialdiff/bin/python detect.py --a data/a.mp4 --b data/b.mp4 \
    --name demo --render-only

# a development scene with its documented working point
~/miniconda3/envs/spatialdiff/bin/python detect.py --a data/scene3/a.mp4 --b data/scene3/b.mp4 \
    --name scene3 --method tabletop --min_change_score 85

# tests (fast, no GPU)
~/miniconda3/envs/spatialdiff/bin/python tests/test_summary.py
~/miniconda3/envs/spatialdiff/bin/python tests/test_ma_contract.py

# inspect a run
scripts/inspect_cache.py --cache cache/scene5      # stage state
grep comment <(qstat -f <jobid>)                   # why a job is queued
```

---

## 8. Known pitfalls (all found the hard way; details in `docs/devlog.md`)

- **The VLM cache is keyed by cluster id + model**, not by crop content: a
  `--force` re-run that renumbers clusters triggers fresh VLM calls and can
  change labels (part of the scene2 variability).
- **`--force` in `detect.py`** passes through to the stages (full re-run) and
  allows overwriting `outputs/<name>/`; without it, existing outputs are
  protected and cached stages are skipped.
- **The auto mode's plane coverage** (50% threshold, ±2 cm) is sensitive to
  the exact input encoding (the scene5 re-encode flips it).
- **The summary validator** rejects anything outside the label + type actions
  + non-object words; the report's free-text description is deliberately not a
  vocabulary source (that is how "leaving only the cracked surface" slipped
  through once).
- **Chrome is not installed on the GPU node**, `visualize` falls back to a
  matplotlib GIF there; the Chrome GIF is rendered on the login node.
- **A fresh clone has no `logs/`** (git-ignored), `detect.py` creates it, but
  keep the pattern in mind for new scripts.
- **The job's exit code must be propagated** (`exit $rc`), otherwise a trailing
  `echo` masks failures as exit 0.
- **DINOv2 is loaded twice per describe run** (appearance check + move
  matching), ~1 s each, harmless but known.
- **scene5's shipped `b.mp4` is a re-encode** (4.6 MB, plain git); the original
  6.9 MB LFS object is in history (oid `1be79cea00`). The held-out result came
  from the original.

## Repository description and topics (owner actions)

Suggested one-line description for the GitHub repository:

> Two phone videos of a room in, a 3D change report out: moved, added and
> removed objects with labels, sentences and interactive 3D viewers.

Suggested topics: `3d-reconstruction`, `change-detection`, `scene-understanding`,
`vggt`, `qwen3-vl`, `dinov2`, `computer-vision`, `python`.
