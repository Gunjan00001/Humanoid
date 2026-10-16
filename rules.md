# Rules for Agents Working in This Folder

**Scope:** `$HOME/Desktop/Humanoid` (referred to below as
"this folder" / the workspace).

These rules apply to **every AI agent** operating here — including the agent that
wrote this file. By working in this folder you agree to follow them.

---

## 1. Stay inside this folder

- Treat `$HOME/Desktop/Humanoid` as your **working directory and
  boundary**.
- **Do not** read, create, modify, move, or delete files **outside** this folder
  unless the user explicitly asks for that specific action.
- Do not `cd` out of the workspace to do work; do not write tool output, temp
  files, caches, logs, or generated artifacts elsewhere.
- All deliverables (code, docs, data you generate, notebooks, scripts, outputs)
  must live **inside this folder**.

### What "outside" means
Anything not under the workspace path, including but not limited to:
`~` (home), `~/.config`, `~/.cache`, `/tmp`, `/opt`, other users' homes,
`/cm/shared`, and the root filesystem.

## 2. Only approved exceptions

A few things physically live outside the folder because they are cluster/system
tooling that was explicitly requested by the user. These are **pre-existing
exceptions**, not permission to roam:

| Path | Why |
| --- | --- |
| `~/notebook.sh` | cluster-standard Jupyter launcher (user-requested) |
| `~/bin/gpu-run`, `~/bin/gpu-session`, `~/bin/gpu-send` | GPU access tooling (user-requested) |
| `~/.gpu-session/` | state for the persistent GPU session (user-requested) |
| `~/models/` | shared model weights for the 3D change-detection project (approved) |
| `~/miniconda3/envs/spatialdiff` | conda env for the 3D change-detection project (approved) |

Rules for exceptions:
- Do **not** add new exceptions on your own; ask the user first.
- Do **not** expand the footprint of an existing exception beyond what it is for.
- Prefer project-local alternatives when one is possible (for example, install
  Python tooling into a `.venv/` **inside this folder** instead of globally).

## 3. Installing tools and dependencies

- Prefer installing into the workspace, e.g. `.venv/` here, over global or
  user-level locations.
- Never modify global/system configuration, dotfiles (`~/.bashrc`, `~/.gitconfig`,
  ...), or other users' files without explicit, specific approval.
- Record anything you install and where it went (add it to this file or a nearby
  README).

## 4. Handling existing work

- Treat unfamiliar files as the user's work. Investigate before deleting or
  overwriting. Ask when unsure.
- Do not delete files outside the workspace at all.

## 5. Git conventions

### 5.1 Tags: `major.minor.bugs`
- Every milestone/release is tagged as **`v<major>.<minor>.<bugs>`** — three
  numbers. Examples: `v0.5.1`, `v0.6.0`, `v1.0.0`.
- Increment **bugs** for fixes, documentation and tooling changes that do not
  change behaviour; increment **minor** for each completed stage/checkpoint;
  increment **major** for milestones that change the project's public
  behaviour or deliverable.
- Tags published before 2026-10-10 used the old two-number format
  (`v0.1` … `v0.5`); they stay as they are. Never move or re-point a published
  tag; make a new one instead.
- Tag from `main` only, after the checkpoint has been verified.
- Use annotated tags with a short message, and always push tags:
  ```bash
  git tag -a v0.6.0 -m "Extra scenes robustness results"
  git push origin v0.6.0
  ```

### 5.2 Large files: use Git LFS when needed
- Use **Git LFS** for any large binary asset that must be versioned: videos
  (`*.mp4`, `*.mov`, `*.avi`), point clouds (`*.ply`), model weights and
  checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.onnx`).
- Do **not** put large binaries in regular git. If an asset is reproducible,
  don't commit it at all (`cache/`, `outputs/*`, `models/` are gitignored).
- When a large asset genuinely needs versioning, track it via LFS:
  ```bash
  git lfs install            # once per machine
  git lfs track "*.mp4"      # writes/updates .gitattributes
  git add .gitattributes <file>
  ```
  `.gitattributes` is committed, so every clone uses the same filters.
- `.gitignore` excludes large media by default; a file that must be versioned
  despite that may need `git add -f` once it is LFS-tracked.
- Never rewrite, prune, or delete LFS objects without the user's approval, and
  never run `git lfs migrate` on history without explicit instruction.

## 6. Resource use (this is an HPC cluster)

- Be a good citizen: GPUs on the GPU node are shared and often busy. Check
  `nvidia-smi` and pick an idle GPU; don't hog all 8.
- **Never wait on a busy GPU.** If the GPU you would use is busy (e.g. another
  agent's job), pick another idle GPU instead of queueing behind it. Jobs that
  can be split across free GPUs should be submitted in parallel rather than
  serialised; when a job must queue, prefer resource shapes that fit the free
  capacity (fewer GPUs/CPUs) over waiting.
- **Report an estimated remaining time** for every GPU job when launching it
  and when reporting progress (use `qstat` elapsed times and the observed
  durations of comparable previous runs).
- Clean up after yourself: `qdel` jobs you no longer need (`gpu-session stop`),
  and remove scratch files you created.
- Respect the PBS limits (`gpu` queue: 5 running jobs/user, max 16 CPUs,
  max 64 GB, 24 h walltime). The **per-user 16-CPU quota is usually the real
  bottleneck**, not the GPUs: check `qstat -f <id> | grep comment` for the
  reason a job is queued. The persistent `gpu-session` holds 4 CPUs; the
  established pattern is to stop the idle session to free a slot, run, and
  restart it afterwards (`~/bin/gpu-session stop|start`).

## 7. Development log

- **Append an entry to [`docs/devlog.md`](docs/devlog.md) after every working
  step** — including failed attempts. Each entry: what was done, the failure
  symptom, the root cause, and the fix. This is the institutional memory for
  future developers and agents; the expensive debugging lessons (PATH/conda
  activation, the thin-object problem, the verification rule chain, the DA3
  integration chain) must stay discoverable.
- Keep entries short; reference commits/tags when useful.

## 8. Held-out scenes and post-hoc changes

- A **held-out scene** is run **exactly once**, cold, with the frozen detection
  code and default settings; its protocol (frozen commit + file digests, the
  pre-registered scoring rule) lives in [`docs/heldout.md`](docs/heldout.md).
- **No fix may touch the frozen detection files for that scene's result.**
- Any later change that would affect it is **post-hoc**: label it
  *"post-hoc, not part of the held-out result"*, keep its outputs in a
  separate directory, and report it separately.
- New scenes meant for evaluation must not be used for tuning first; record
  them as `data-v*` and follow the held-out protocol.

## 9. Reporting

- When a task requires touching anything outside this folder, **say so and get
  confirmation first**.
- Keep the user informed about where files were written.

---

*See `gpu.md` in this folder for how to access the GPUs, and `AGENT_PLAN.md` for the project plan.*
*The agent writing this file follows these rules too.*

## 10. Branching (standing rule, from v1.2.1)

**All work goes directly to main.** No feature branches, no docs branches, no long
lived experiment branches. Commit small, test every commit, and tag only a verified
release: a patch tag for documentation and fixes, a minor tag for a behaviour
change, always checked from a fresh clone.

History from before this rule is kept in archive tags (`archive/exp-accuracy`,
`archive/exp-semantics-v2`) and those branches are deleted on the remote, so the
remote shows main plus tags.

Never force push and never rewrite history. The held-out outputs of scene5 and
scene6 stay byte identical to `v1.0.0`.
