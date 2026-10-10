# Rules for Agents Working in This Folder

**Scope:** `/home/user/Desktop/Humanoid` (referred to below as
"this folder" / the workspace).

These rules apply to **every AI agent** operating here — including the agent that
wrote this file. By working in this folder you agree to follow them.

---

## 1. Stay inside this folder

- Treat `/home/user/Desktop/Humanoid` as your **working directory and
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

### 5.1 Tags: `major.minor`
- Every milestone/release is tagged as **`v<major>.<minor>`** — two numbers only,
  no patch component. Examples: `v0.1`, `v0.2`, `v1.0`, `v2.0`.
- Increment **minor** for each completed stage/checkpoint; increment **major**
  for milestones that change the project's public behaviour or deliverable.
- Tag from `main` only, after the checkpoint has been verified.
- Use annotated tags with a short message, and always push tags:
  ```bash
  git tag -a v0.1 -m "Docs and plan complete"
  git push origin v0.1
  ```
- Never move or re-point a published tag; make a new one instead.

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

- Be a good citizen: GPUs on `the GPU node` are shared and often busy. Check
  `nvidia-smi` and pick an idle GPU; don't hog all 8.
- Clean up after yourself: `qdel` jobs you no longer need (`gpu-session stop`),
  and remove scratch files you created.
- Respect the PBS limits (`gpu` queue: 5 running jobs/user, max 16 CPUs,
  max 64 GB, 24 h walltime).

## 7. Reporting

- When a task requires touching anything outside this folder, **say so and get
  confirmation first**.
- Keep the user informed about where files were written.

---

*See `gpu.md` in this folder for how to access the GPUs, and `AGENT_PLAN.md` for the project plan.*
*The agent writing this file follows these rules too.*
