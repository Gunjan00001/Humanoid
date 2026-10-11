# Troubleshooting

Cluster details are in [`../gpu.md`](../gpu.md); this page collects the failure
modes most likely to hit this project.

## Cluster / environment

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False`, but `nvidia-smi` lists GPUs | PBS sets `CUDA_VISIBLE_DEVICES` to an empty string | `unset CUDA_VISIBLE_DEVICES` (or `export CUDA_VISIBLE_DEVICES=0`) |
| `conda activate spatialdiff` "works" but `python`/`pip` still resolve to the shared Python 3.8 | the session PATH contains duplicated `/cm/shared/apps/anaconda3/bin` entries *before* the env; activation inserts the env at a later position | run `export PATH="$CONDA_PREFIX/bin:$PATH"` after activating (all scripts here already do), or call `~/miniconda3/envs/spatialdiff/bin/python` directly |
| pip installs land in `~/.local/lib/python3.8/site-packages` unexpectedly | same PATH problem: pip ran with the shared python | check `python -c "import sys; print(sys.executable)"` before installing; use the env interpreter explicitly |
| `CUDA initialization: The NVIDIA driver ... is too old (found version 12030)` | CUDA-13 torch build on the 525 driver | install `cu126` wheels; never `cu130` |
| `qsub -I` → "standard input and output must be a terminal" | no TTY | submit a batch job, or wrap with `script -qec` |
| Job killed with `Cgroup mem limit exceeded` | default 32 GB RAM cgroup | raise with `-l mem=` (max `64gb`) |
| `ModuleNotFoundError` on a compute node | env not activated / package only on login node | activate the conda env in the job; install into it, not into `base` |
| `huggingface-cli` can't reach the internet inside a job | compute nodes may have no internet | pre-download on the login node into `~/models/` |
| GPU shows ~100% / 60 GB used by someone else | shared node, PBS does not pin GPUs | pick an idle GPU (`nvidia-smi`), then `export CUDA_VISIBLE_DEVICES=<id>` (or set `reconstruction.gpu_index`) |
| Job queues although GPUs are free | the queue's **per-user 16-CPU quota** is full | `qstat -f <id> | grep comment`; stop the idle `gpu-session` to free 4 CPUs, restart it after |
| `git-lfs: command not found` on the compute node | git-lfs is login-node only | clone on the login node / shared FS, never `/tmp` (node-local) |
| PBS `-o` log stays empty during the run | PBS spools job output until the end | rely on `qstat` state; the log arrives at completion |
| A job "succeeded" (exit 0) but did nothing | a trailing `echo` masked the real exit code | propagate it (`exit $rc`) — `detect.py` does |
| `FileNotFoundError: logs/job_detect_*.pbs` | a fresh clone has no `logs/` (git-ignored) | `detect.py` creates it on demand |

## Model / pipeline

| Symptom | Likely cause | Mitigation (in order) |
|---|---|---|
| A and B clouds do not overlap | not enough overlap / frames | more frames, more overlap, ICP on static regions (`src/align.py`) |
| VGGT runs out of GPU memory | too many frames / too high resolution | reduce frames or resolution; chunk the input |
| Qwen3-VL-32B won't fit | ~64 GB in bf16 | request 2 GPUs + `device_map="auto"`; fall back to Qwen3-VL-8B |
| Host RAM exhausted while loading a model | model materialised in RAM first | `low_cpu_mem_usage=True`, load straight to GPU, bf16 |
| "removed" objects that are actually just occluded | naive diff | keep the visibility check enabled; verify poses/depth cached correctly |
| Many tiny noisy clusters | voxel/threshold too aggressive | tune `diff.*` in the config (voxel size, DBSCAN eps/min points) |
| Moved object split into removed + added | weak crop match, or the origin cluster was filtered out before matching | `match_similarity_threshold` (0.65 calibrated); move matching runs before the filters in v0.10+; a filtered origin still breaks the pair (scene5's clock) |
| A summary sentence names an object/action not in the report | the VLM phrasing invented it | rejected by `src/summary.py`'s validator -> deterministic template; the report's free-text description is deliberately not a vocabulary source |
| The auto diff mode flips between runs/encodings | the 50% plane-coverage rule is threshold-sensitive (scene5's re-encode flipped geometric -> tabletop) | pass `--method` explicitly when reproducing a documented working point; the choice + measured coverage are printed |
| Poor reconstruction of walls | textureless surfaces | add texture/more views; state as a limitation |
| False changes from lighting | different light between recordings | re-record under similar light; note as a limitation |

## Debugging workflow

1. Reproduce on the smallest input (2–3 frames) in an interactive GPU session.
2. Confirm the stage's cached artifact exists and is sane before the next stage.
3. Only then scale back up to 20–30 frames.
4. Keep thresholds in `configs/default.yaml` — never patch them inside scripts.
