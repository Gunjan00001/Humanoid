# Measured timings

Real numbers from the PBS job records (`qstat -xf`) and the job logs in
`logs/`. Nothing here is estimated unless the row says so.

**Clock bases.** The PBS client reports times on the login node's clock; the
compute node (the GPU node) writes its logs 3:30:00 behind that clock
(verified: job 10802 `stime` 02:39:35 ↔ its first log activity 23:09:35, and
`obittime` 02:44:52 ↔ the last log line 23:14:52). Durations inside one source
are exact; cross-source absolute times are quoted in one clock and the offset
is stated.

## One full scene prediction + 3D map, scene4, job 10802

| PBS record | value |
|---|---|
| submitted (`qtime`) | 02:38:02 |
| started (`stime`) | 02:39:35 → **queue wait 1:33** |
| ended (`obittime`) | 02:44:52 |
| walltime used | **00:05:14** |

The queue wait was **not** a GPU shortage: the `gpu` queue has a per-user
**16-CPU** quota and it was fully used; the queue comment on the queued job
reads `Queue gpu per-user limit reached on resource ncpus`. After stopping the
idle `gpu-session` (which held 4 CPUs) the job started immediately.

### Stage timeline (node clock, all from `logs/pbs_scene4.log`)

| stage | start → end | duration | notes |
|---|---|---|---|
| conda + imports | 23:09:35 → 23:09:42 | ~7 s | job start (PBS − 3:30) to first log line |
| frame extraction | 23:09:42 → 23:09:47 | **5 s** | 46 s videos, 3 fps, blur ≥ 100, 60 frames |
| reconstruction (VGGT + ICP + write) | 23:09:47 → 23:10:35 | **48 s** | incl. `model loaded in 12.5s` (logged) |
| tabletop diff | 23:10:38 → 23:10:48 | **10 s** | 16 clusters kept |
| semantics (VLM + DINOv2) | 23:10:48 → 23:14:17 | **3:29** | VLM load 26 s + 10 VLM calls at ~11 s |
| report | → 23:14:20 | 3 s | |
| viewer.html | → 23:14:40 | 20 s | 1.7 MB |
| GIF | → 23:14:46 | 6 s | matplotlib fallback, **no Chrome on the GPU node** |
| TSDF mesh | → 23:14:50 | 4 s | 199 131 triangles, 9.1 MB |
| job end | 23:14:52 | | matches `obittime` exactly |

**VLM calls in this run: 10** (8 verifications + 2 fresh sentences for
reclassified clusters), counted from the log. Measured rate: verifications
1→8 took 77 s (23:11:26 → 23:12:43) = **11.0 s per call**.

### Model load times (measured across runs)

| model | measured | source |
|---|---|---|
| VGGT-1B | **12.5 s** (scene4); 8.0–14.2 s across 6 logged runs | `model loaded in …` lines: pad_exp 8.0 s, scene2b 8.7 s, scene3 9.7 s, scene4 12.5 s, DA3-smoke 14.2 s |
| DA3-LARGE (different model) | 39.7 s | `pbs_da3_full.log` |
| Qwen3-VL-32B | **26 s** (scene4: `loading VLM` → first verification) | `pbs_scene4.log` |
| Qwen3-VL-32B (early run, cold/network FS) | 2:07 | job 10715, `logs/describe_real.log` |
| DINOv2-large | ~1 s per load, **loaded twice per run** | two `loading matcher` lines in 10802 (appearance check, then move matching) |

### Contention at job start (what can and cannot be measured)

- **GPU utilization (`nvidia-smi`): not measurable**, no snapshot was
  captured in any log. This number cannot be produced retroactively.
- From PBS records: 3 concurrent jobs of the same account were running during
  10802, each **1 GPU** (three jobs belonging to another user, 02:05→03:30, 02:09→03:10, 02:34→03:07)
  on an 8-GPU node. The job asked for 2 GPUs, so GPU capacity was not the
  blocker; the CPU quota was (see above).

## End-to-end wall clock: why scene4 took ~1 h in practice

All scene4 jobs (login clock):

| job | submitted | start delay | walltime | purpose |
|---|---|---|---|---|
| 10802 | 02:38:02 | 1:33 | 5:14 | full pipeline, first result |
| 10804 | 02:57:12 | 2 s | 0:33 | semantics fix round 1 |
| 10806 | 03:00:31 | 4 s | 1:04 | fix round 2 (2 fresh VLM calls) |
| 10808 | 03:04:23 | 3 s | 2:12 | scene4 + 3 regression scenes |
| 10810 | 03:10:58 | 2 s | 2:02 | scene4 + 3 regression scenes (final) |
| login-node renders | ~03:14 → 03:36 |, | ~22 min | Chrome GIF + mesh re-renders (see below) |

**First submission → final visual ≈ 58 min.** One prediction is 5:14 on the
node; the hour is five GPU rounds (the first run plus four semantic-fix /
regression rounds) plus two visual rounds plus the analysis between them , 
not one long run.

## v0.12 one-command runs (`detect.py`)

| run | queue wait | job runtime | total wall clock | notes |
|---|---|---|---|---|
| **scene5 held-out** (job 10824, cold) | 0 s | **7:07** | 7:22 | extract 0:03, reconstruct 0:23, align 0:24, diff 0:33, describe 3:54, report+visualize+summary ~1:45 |
| scene5 post-hoc tabletop (job 10829, cold) | 1 s | 6:33 | 6:41 | one variable changed (`--method tabletop`) |
| **4 development-scene regressions** (job 10830, `--force` with warm VLM caches) | 0 s | **10:25** |, | per scene: demo 2:27, scene2 2:16, scene3 2:20, scene4 3:12 (re-extract/reconstruct/diff re-ran; the cached VLM answers skipped the model load) |

The cold/warm split is the dominant factor: a cold scene pays the VLM load
(~26 s) and the fresh verifications (~11 s per candidate); a warm re-run of a
whole scene is 2-3 min even with `--force`.

## Warm re-runs (VLM cache warm)

- `describe` + `report` only: **0:33** (10804), **1:04** (10806, includes two
  fresh VLM calls), **2:02–2:12** (10808/10810, scene4 *plus* the three
  regression scenes in the same job).
- The three regression scenes' describe+report: ~15–25 s each (combined logs).

## Login-node render times (CPU only, Chrome available)

`--render-only` runs (2026-10-11, one per scene, measured wall clock):

| scene | wall clock | notes |
|---|---|---|
| demo | **191 s** (3:11) | 3 composites |
| scene2 | **204 s** (3:24) | 11 composites |
| scene2_geo | **129 s** (2:09) | 5 composites |
| scene3 | **346 s** (5:46) | 31 composites (27 unverified); its Chrome GIF alone took **4:44** (denser cloud + more candidates) |
| scene4 | **201 s** (3:21) | 10 composites |

Per-artifact measurements from earlier runs:

| artifact | measured | source |
|---|---|---|
| Chrome GIF (24 frames @ 1100×820, from `viewer.html`) | **82–284 s** (typically ~2 min) | GIF-only re-renders: demo 150 s, scene2 129 s, scene2-geo 147 s, scene3 129 s, scene4 109 s; one scene3 run took 284 s under load |
| TSDF mesh (60 frames, 0.01 m voxel) | **~40–89 s** | 03:16:59 → 03:18:28 |
| full `visualize` (composites + viewer + GIF + mesh) | **2:09–5:46** per scene | `--render-only` pass |

Note: the *same* mesh step took **4 s** inside job 10802 on the GPU node
(arrays in page cache, 96-core host) versus ~89 s on the shared login node.
Which factor dominates (page cache vs cores) was **not isolated**, listed
here as unverified.

## Old timing claims vs measurements (corrections)

| old claim (where) | measured | verdict |
|---|---|---|
| "~3.5 min (model load)" VGGT (`docs/results.md`) | 8–14 s (6 logged runs) | corrected |
| "~2–4 min VGGT / ~1.5–3 min Qwen3-VL loads" (`docs/debugging.md`) | 12.5 s / 26 s warm; 2:07 for the VLM on the early cold run | corrected to a cache-dependent range |
| "~15 min first semantics run" (`docs/results.md`) | demo run 10715: describe stage 5:25 total; scene4: 3:29 | **not reproducible from any log**; corrected |
| "Visualization + report ~10 s" (`docs/results.md`) | viewer 20 s, GIF 6 s (fallback) / ~1:51 (Chrome), mesh 4–89 s, report 3 s | corrected to 0:33–3:30 depending on artifacts |

## Not measurable from the logs

- `nvidia-smi` / per-GPU utilization at job start (no snapshot captured).
- The wall-clock time before the first submission (when the scene4 videos
  arrived), not recorded in any log I can read.

## Reproducibility runs (2026-10-11)

Twelve `detect.py --force` runs (scene2 working point, scene4, scene5, scene6 , 
3× each; fresh caches; 2 CPUs + 2 GPUs per job; 3 workers). Per-run wall clock
(submit → summary), which includes queue waits on the busy shared queue:

| scene | run 1 | run 2 | run 3 |
|---|---|---|---|
| scene4 | 7:02 | 9:41 | 7:22 |
| scene5 | 33:04* | 13:02 | 22:43* |
| scene6 | 11:42 | 11:22 | 11:02 |
| scene2 (working point) | 26:23* | 5:21 | (n/a) |

\* queued behind the per-user 16-CPU quota / other users' jobs; the node-side
stage time was ~5–12 min. scene6's three runs (~11–12 min) were faster than the
**cold held-out run (18:53)** because the model files were warm. Counts and
types were identical across all three runs of each scene (see
[`results.md`](results.md#reproducibility-frozen-pipeline-3x-per-scene)).
