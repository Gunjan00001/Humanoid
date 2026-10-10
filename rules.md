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

## 5. Resource use (this is an HPC cluster)

- Be a good citizen: GPUs on `the GPU node` are shared and often busy. Check
  `nvidia-smi` and pick an idle GPU; don't hog all 8.
- Clean up after yourself: `qdel` jobs you no longer need (`gpu-session stop`),
  and remove scratch files you created.
- Respect the PBS limits (`gpu` queue: 5 running jobs/user, max 16 CPUs,
  max 64 GB, 24 h walltime).

## 6. Reporting

- When a task requires touching anything outside this folder, **say so and get
  confirmation first**.
- Keep the user informed about where files were written.

---

*See `gpu.md` in this folder for how to access the GPUs, and `AGENT_PLAN.md` for the project plan.*
*The agent writing this file follows these rules too.*
