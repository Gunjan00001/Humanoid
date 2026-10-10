# AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)

> This file is the single source of truth for the AI agent building this project.
> Read it fully before writing any code. Follow the phases in order. Do not skip checkpoints.

---

## 0. Critical rules (read first)

1. **Use GPU nodes for all large-model work.** Never run VGGT, MASt3R, Qwen3-VL, SAM 2, or DINOv2 on the login node. Submit a PBS job to the `gpu` queue (or an interactive GPU session with `qsub -I`). The login node is only for editing files, git, light CPU scripts, and downloading model weights.
2. **Read `/home/jiit_992401210050/Desktop/Humanoid/gpu.md` first** for the exact PBS syntax and cluster details. If it conflicts with this file, `gpu.md` wins for cluster commands.
3. **Cache everything expensive** (reconstruction outputs, embeddings, VLM answers) to disk so that later stages can be re-run without the GPU.
4. **One working stage at a time.** Finish and verify each checkpoint before starting the next phase.
5. **Keep every model name, path, and threshold in `configs/default.yaml`**, never hard-coded.
6. **No made-up results.** Every number, image, and claim in the README must come from a real run.
7. **The final deliverable is a public GitHub repository** containing the completed challenge. Incomplete submissions are not considered.

---

## 1. Goal

Build a system that:

1. Takes **two phone recordings** of the **same small indoor area** (taken at different times),
2. Builds a **shared 3D representation** of the scene,
3. Detects and **visualizes in 3D** which objects/regions were **moved, added, or removed**,
4. **Describes** each change with a semantic label and a natural-language sentence,
5. Is submitted as a **public GitHub repo**.

No real-time requirement. The approach is open: any tools, models, or agentic workflows are allowed.

**One-command interface:**

```bash
python run.py --video_a data/a.mp4 --video_b data/b.mp4 --out outputs/
```

---

## 2. Environment constraints (cluster)

| Item | Value |
|---|---|
| Node | `jiit-gpu01` |
| GPUs | 8 x NVIDIA A100-SXM4-80GB (NVLink, fully connected) |
| Driver | 525.147.05 → **max CUDA 12.3** |
| MIG | Disabled, ECC on |
| CPU / RAM | 2 x Xeon Platinum 8468 (96 cores), ~2 TB RAM |
| PBS `gpu` queue limits | max 16 CPUs, max 64 GB RAM, default 1 GPU / 32 GB, 24 h walltime, 5 jobs per user |

**Consequences:**

- Install PyTorch wheels built for **CUDA 12.6** (`cu126`). This is verified working on the driver here (CUDA minor-version compatibility) and keeps recent torch versions that newer models may need. CUDA-13 builds (`cu130`) do **not** work. See `gpu.md`.
- Prefer plain Hugging Face `transformers` over stacks that need newer CUDA (e.g. vLLM) unless verified to work.
- Host RAM is capped at 64 GB per job. Load big models directly to GPU (`device_map="auto"`, `low_cpu_mem_usage=True`, bf16) so loading does not exhaust RAM.
- Compute nodes may have **no internet**. Download all weights on the login node into a shared folder (e.g. `~/models/`) and load from local paths.
- Keep jobs short and modular (max 5 jobs per user at once).

---

## 3. Model stack (decided)

| Role | Primary | Fallback | Notes |
|---|---|---|---|
| 3D reconstruction (poses, depth, point cloud) | **VGGT** | **MASt3R**, then COLMAP/hloc | Reconstruct both videos **together** in one run |
| Labels and descriptions | **Qwen3-VL-32B-Instruct** | Qwen3-VL-8B, Gemma (vision) | Needs about 64 GB in bf16, so use **2 GPUs** with `device_map="auto"` |
| Object matching across scenes ("moved") | **DINOv2** | SigLIP 2 | Compare crop embeddings |
| Object masks (stretch goal) | **SAM 2** | DBSCAN clusters only | Optional polish |
| Geometry and visualization | OpenCV, Open3D, NumPy, SciPy, scikit-learn | — | Classical, no GPU needed for the diff |

**GPU rule per model:**

| Model | Where to run | GPUs |
|---|---|---|
| VGGT / MASt3R | GPU node | 1 |
| Qwen3-VL-32B | GPU node | 2 (or 1 if memory allows, which is tight) |
| Qwen3-VL-8B | GPU node preferred | 1 |
| DINOv2 / SAM 2 | GPU node | 1 |
| Voxel diff, DBSCAN, visualization | Any machine (CPU) | 0 |

Check licenses of VGGT/MASt3R checkpoints (some are non-commercial). Note them in the README.

---

## 4. Repository structure

```
Humanoid/                       # repo root = this folder
├── README.md
├── AGENT_PLAN.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── run.py                      # one-command pipeline
├── configs/
│   └── default.yaml            # models, paths, thresholds, GPU settings
├── src/
│   ├── extract_frames.py       # video -> sharp frames
│   ├── reconstruct.py          # VGGT/MASt3R/COLMAP backend switch
│   ├── align.py                # optional ICP refinement
│   ├── diff.py                 # voxel diff, clustering, visibility check
│   ├── describe.py             # crops, Qwen3-VL labels, DINOv2 matching
│   ├── visualize.py            # colored PLY, arrows, viewer
│   └── report.py               # JSON + Markdown report
├── scripts/
│   ├── setup_env.sh            # conda env + pip installs (cu126)
│   ├── download_models.sh      # run on login node
│   ├── job_reconstruct.pbs
│   ├── job_describe.pbs
│   └── job_full.pbs
├── data/                       # input videos (short ones only in git)
├── cache/                      # reconstruction outputs, embeddings (gitignored)
├── outputs/                    # PLY, JSON, MD, images, GIF
└── docs/                       # pipeline diagram, screenshots
```

---

## 5. Pipeline overview

```
Video A ─┐                                         ┌─> colored 3D PLY
         ├─> frames ─> VGGT (joint) ─> point cloud ┤
Video B ─┘            shared frame       A / B tags └─> diff + visibility check
                                                         │
                                         clusters ─> crops ─> Qwen3-VL-32B ─> labels + sentences
                                                         │
                                         DINOv2 matching ─> moved / added / removed
                                                         │
                                                         └─> JSON + Markdown report
```

---

## 6. Development phases

### Phase 0 — Data collection (Day 1)

- Pick one small indoor area (desk, shelf, room corner).
- Record **Video A**. Then make 3 clear changes: **move** one object, **remove** one object, **add** one object.
- Record **Video B** with similar lighting and a similar path.
- Recording rules: 30–60 s each, slow pans, high overlap, textured surfaces, no heavy motion blur, consistent lighting, changes that are large and obvious.
- Optional: record a second scene for a robustness example.
- Store in `data/` (keep files small; compress if needed for GitHub).

**Checkpoint 0:** two videos exist, with the three changes documented in a short `data/ground_truth.md` (what moved, what was added, what was removed).

### Phase 1 — Setup (Day 1)

- Create the repo and folder structure above.
- On the cluster: create the conda env (Python 3.10) and install PyTorch for `cu126`, Open3D, OpenCV, transformers, accelerate, scikit-learn, scipy. Install VGGT from its GitHub repo following its README.
- On the **login node**, download weights (VGGT, Qwen3-VL-32B, DINOv2, optionally MASt3R/SAM 2) to a shared folder via `huggingface-cli download`.
- Write `configs/default.yaml` with model paths and thresholds.
- Verify inside a **GPU job** (not the login node): `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`.

**Checkpoint 1:** CUDA is visible in a GPU job and all weights load from local paths.

### Phase 2 — Reconstruction (Day 1–2) — most important

Script: `src/extract_frames.py`
- Sample about 2–3 fps, drop blurry frames (Laplacian variance), target **20–30 frames per video** to start. Tag each frame with its source video.

Script: `src/reconstruct.py` (**run on a GPU node**)
- Run **VGGT on all frames of both videos together** so everything lands in one coordinate frame.
- Output: camera poses, depth maps, a colored point cloud with a per-point source tag (A or B), saved to `cache/`.
- Backend selectable via config: `vggt` | `mast3r` | `colmap`.
- If memory is a problem, reduce frame count or image resolution.

**Checkpoint 2:** load the combined cloud and confirm static structure (walls, large furniture) from A and B overlaps closely. If not: add more frames/overlap, apply ICP (`src/align.py`) on static regions, or switch backend to MASt3R.

### Phase 3 — Change detection (Day 2)

Script: `src/diff.py` (CPU is fine)
1. Voxel-downsample both clouds.
2. Compute nearest-neighbor distances A→B and B→A; threshold to get change candidates.
3. Remove noise (statistical outlier removal) and cluster with DBSCAN into object-sized regions.
4. **Visibility check:** project each cluster into the other video's cameras (using the cached poses and depth). Mark a cluster as **removed** only if the other video actually observed that space as empty. Discard clusters that were merely occluded or unseen.
5. Initial classification: only in B → **added**, only in A → **removed**.

**Checkpoint 3:** the clusters correspond to the three known changes, with few false positives. Tune thresholds in the config, not in code.

### Phase 4 — Semantics (Day 3)

Script: `src/describe.py` (**run on a GPU node**)
1. For each cluster, choose the best camera view (largest visible area, least occlusion) and crop it (optionally with SAM 2 masks).
2. **Qwen3-VL-32B-Instruct** returns a short object label and a description. Request structured JSON output and parse it safely.
3. **DINOv2** embeds crops. A removed cluster and an added cluster with high embedding similarity (and a changed 3D position) become one **moved** object, with old and new positions.
4. Qwen generates the final one-sentence description per change, for example: "Blue backpack moved from the chair to the floor."
5. Cache VLM outputs to disk.

**Checkpoint 4:** every detected change has a type, label, 3D location, and sentence. Moved objects are matched correctly in the demo scene.

### Phase 5 — Visualization and report (Day 3–4)

Scripts: `src/visualize.py`, `src/report.py`
- Colored point cloud export (PLY): **red = removed, green = added, orange = moved**, plus an arrow from old to new position for moved objects. Static background in gray or true color.
- Interactive viewer (Open3D or Rerun) and static screenshots/GIF for the README.
- `outputs/report.json` (one entry per change: type, label, description, 3D centroid, bounding box, confidence) and `outputs/report.md` (human-readable summary).
- Wire everything into `run.py`.

**Checkpoint 5:** one command produces all outputs from the two videos.

### Phase 6 — Polish and submission (Day 4)

- README (see Section 9).
- PBS scripts and a laptop-friendly path documented.
- Optional ablations: occlusion-aware vs. naive diff, Qwen3-VL-32B vs. 8B, VGGT vs. MASt3R.
- Make the GitHub repo **public** and check that a fresh clone follows the README.

---

## 7. GPU job templates

Confirm exact `select` syntax in `gpu.md`; the following is a starting point.

**Reconstruction (1 GPU):**

```bash
#!/bin/bash
#PBS -N reconstruct
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
unset CUDA_VISIBLE_DEVICES   # required: PBS sets this empty, hiding all GPUs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatialdiff
python src/reconstruct.py --config configs/default.yaml
```

**Description with Qwen3-VL-32B (2 GPUs):**

```bash
#!/bin/bash
#PBS -N describe
#PBS -q gpu
#PBS -l select=1:ncpus=12:mem=60gb:ngpus=2
#PBS -l walltime=03:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
unset CUDA_VISIBLE_DEVICES   # required: PBS sets this empty, hiding all GPUs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatialdiff
python src/describe.py --config configs/default.yaml
```

**Interactive debugging:** `qsub -I` with a GPU request, then run the stage manually.

Remember: 24 h walltime cap, 5 jobs per user, and 64 GB RAM maximum.

---

## 8. Expected results

**On a clean, well-recorded scene:**

- Both videos align in a single 3D scene, with static structure overlapping closely.
- The three planned changes are detected and shown in the correct colors.
- Each change gets a plausible label and sentence, for example:
  - "Blue backpack moved from the chair to the floor."
  - "White mug removed from the desk."
  - "Water bottle added on the shelf."
- Large objects in a small room are labeled correctly in most cases.

**Known imperfections (document honestly in the README):**

- False positives from reflective surfaces, thin objects, or lighting changes.
- Small objects may be missed or merged into larger clusters.
- A moved object is sometimes reported as one removal plus one addition when the visual match is weak.
- Textureless walls can reduce reconstruction quality.
- Scale ambiguity between runs if alignment is imperfect.

**Deliverables:**

1. Public GitHub repo with working code and one-command run
2. 3D visualization (screenshots and a short GIF)
3. `report.json` and `report.md` for the example scene
4. Example input videos (or download instructions) and outputs
5. README with approach, setup, results, and limitations

---

## 9. README requirements

- One-paragraph problem summary and a result GIF at the top
- Pipeline diagram (use the one in Section 5 or a drawn version)
- Models used, why they were chosen, and their licenses
- Setup for both the laptop (CPU stages) and the GPU cluster (PBS)
- How to run: the one-command example and the per-stage commands
- Example results with the real report output
- Design decisions: joint reconstruction, occlusion-aware diff, and DINOv2-based move matching
- Limitations and failure cases (honest and specific)
- Optional: ablation table

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scenes misalign | More overlap and frames, ICP on static regions, switch to MASt3R, then COLMAP |
| GPU out of memory in VGGT | Fewer frames, lower resolution, process in chunks |
| Qwen3-VL-32B too heavy for the allocation | Use 2 GPUs, `device_map="auto"`, or fall back to 8B |
| Host RAM cap (64 GB) hit during model load | `low_cpu_mem_usage=True`, load straight to GPU |
| No internet on compute nodes | Pre-download all weights on the login node |
| CUDA mismatch (driver max 12.3) | Use `cu126` PyTorch wheels (verified working); avoid CUDA-13 stacks |
| False "removed" objects from occlusion | Visibility check using cached poses and depth |
| Noisy clusters | Tune voxel size and DBSCAN thresholds, add SAM 2 masks |
| Lighting differences between videos | Record under similar light; state it as a limitation |

---

## 11. Config skeleton (`configs/default.yaml`)

```yaml
paths:
  video_a: data/a.mp4
  video_b: data/b.mp4
  cache_dir: cache/
  out_dir: outputs/
  model_dir: ~/models/

frames:
  fps: 2
  max_frames_per_video: 30
  blur_threshold: 100.0

reconstruction:
  backend: vggt          # vggt | mast3r | colmap
  use_icp_refinement: false

diff:
  voxel_size: 0.02
  change_distance: 0.05
  dbscan_eps: 0.08
  dbscan_min_points: 50
  visibility_check: true

semantics:
  vlm: qwen3-vl-32b-instruct   # or qwen3-vl-8b-instruct
  matcher: dinov2
  match_similarity_threshold: 0.75
  use_sam2: false

gpu:
  vlm_gpus: 2
  dtype: bfloat16
  torch_cuda: cu126
```

All values are starting points; tune them against the example scene.

---

## 12. Definition of done

- [x] Two example videos with documented ground-truth changes — `data/a.mp4`,
  `data/b.mp4` + `data/ground_truth.md` (plus scene2/scene3 from `data-v0.2`)
- [x] Joint reconstruction aligns both scenes (Checkpoint 2 passed) — ICP
  static mean NN 15.2 → 10.7 mm; three backends compared in `docs/results.md`
- [x] Added, removed, and moved objects are detected with the visibility check
  (Checkpoint 3 passed) — 3/3 on the demo scene, 0 false positives
- [x] Each change has a label and a natural-language sentence (Checkpoint 4
  passed) — Qwen3-VL, verified against before/after crops
- [x] Colored 3D visualization and JSON/Markdown report generated by `run.py`
  (Checkpoint 5 passed) — `outputs/scene.ply`, `change.gif`, screenshots,
  `report.json/md`
- [x] README complete, with real results and honest limitations
- [x] Repository is **public** and reproducible from a fresh clone — verified
  2026-10-11: fresh clone pulled all LFS videos and ran `extract_frames`
  (identical 30+19 frames); GPU stages reproduce via `docs/setup.md`
