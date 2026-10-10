# Graph Report - Humanoid  (2026-10-11)

## Corpus Check
- 47 files · ~125,103 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 4, .pbs 3)

## Summary
- 488 nodes · 1049 edges · 29 communities (20 shown, 9 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ff7ccd53`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- common.py
- write_ply
- docs/README.md
- AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)
- GPU Access on This Cluster
- describe.py
- Rules for Agents Working in This Folder
- make_test_videos.py
- Stage detail
- build_change_list
- 3D Scene Change Detection
- numpy
- Setup
- inspect_cache.py
- download_models.sh
- gpu_smoke_test.sh
- AGENTS.md
- setup_env.sh
- tests/README.md
- The rules, in order
- Details
- ground_truth.md
- Timer
- run_mapanything.py
- Development log
- Backend comparison: VGGT vs MapAnything vs Depth Anything 3
- make_fake_reconstruction.py
- scene2/ground_truth.md
- scene3/ground_truth.md

## God Nodes (most connected - your core abstractions)
1. `repo_path()` - 32 edges
2. `load_config()` - 31 edges
3. `write_json()` - 25 edges
4. `read_json()` - 23 edges
5. `main()` - 22 edges
6. `cache_dir()` - 20 edges
7. `stamp()` - 19 edges
8. `apply_overrides()` - 19 edges
9. `add_common_args()` - 18 edges
10. `main()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `GPU debugging` --references--> `pin_idle_gpus()`  [INFERRED]
  docs/debugging.md → src/common.py
- `Semantics (crops, VLM, verification)` --references--> `best_crop_for_cluster()`  [INFERRED]
  docs/debugging.md → src/describe.py
- `Failure: cell-code collision attached wrong points to clusters` --references--> `cell_maps()`  [INFERRED]
  docs/devlog.md → src/tabletop.py
- `5. Static-object suppression (`src.tabletop.static_object_mask`)` --references--> `static_object_mask()`  [INFERRED]
  docs/verification.md → src/tabletop.py
- `Config reference` --references--> `cache_dir()`  [INFERRED]
  docs/architecture.md → src/common.py

## Import Cycles
- None detected.

## Communities (29 total, 9 thin omitted)

### Community 0 - "common.py"
Cohesion: 0.08
Nodes (30): main(), stage_enabled(), main(), main(), static_subset(), add_common_args(), apply_overrides(), cache_dir() (+22 more)

### Community 2 - "docs/README.md"
Cohesion: 0.08
Nodes (27): Architecture, Caches and outputs, Config reference, Conventions, Module responsibilities, Repository layout, Additional scenes, Checklist before starting the pipeline (+19 more)

### Community 3 - "AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)"
Cohesion: 0.10
Nodes (21): 0. Critical rules (read first), 10. Risks and mitigations, 11. Config skeleton (`configs/default.yaml`), 12. Definition of done, 1. Goal, 2. Environment constraints (cluster), 3. Model stack (decided), 4. Repository structure (+13 more)

### Community 4 - "GPU Access on This Cluster"
Cohesion: 0.10
Nodes (21): 1. TL;DR — the shortest path, 2. GPU hardware configuration, 3. Node / system configuration, 4. PBS queue configuration, 5. The three gotchas (this is what wastes everyone's time), 6. Ways to access the GPUs, 7. Helper tools installed for this, 8. Verified evidence (+13 more)

### Community 5 - "describe.py"
Cohesion: 0.10
Nodes (14): Failure: VLM verification over-confirmed (iterations), 1. VLM verification (`PROMPT_VERIFY`), resolve_model(), best_crop_for_cluster(), embed_crops(), load_matcher(), load_vlm(), main() (+6 more)

### Community 6 - "Rules for Agents Working in This Folder"
Cohesion: 0.17
Nodes (12): 1. Stay inside this folder, 2. Only approved exceptions, 3. Installing tools and dependencies, 4. Handling existing work, 5.1 Tags: `major.minor.bugs`, 5.2 Large files: use Git LFS when needed, 5. Git conventions, 6. Resource use (this is an HPC cluster) (+4 more)

### Community 7 - "make_test_videos.py"
Cohesion: 0.22
Nodes (5): _scan_video(), base_scene(), main(), make_video(), render_frame()

### Community 8 - "Stage detail"
Cohesion: 0.18
Nodes (11): 1. Frame extraction, 2. Joint reconstruction (most important), 3. Alignment (optional), 4. Change detection, 5. Semantics, 6. Visualization, 7. Report, Caching strategy (+3 more)

### Community 9 - "build_change_list"
Cohesion: 0.12
Nodes (13): Change detection: misses or spams candidates, Debugging, Frame extraction, GPU debugging, Logs, Playbook by symptom, Reconstruction, Report / visualization (+5 more)

### Community 10 - "3D Scene Change Detection"
Cohesion: 0.17
Nodes (12): 3D Scene Change Detection, Backend comparison (v0.5), Design decisions, Documentation, How to run, License, Limitations and failure cases, Models (+4 more)

### Community 11 - "numpy"
Cohesion: 0.06
Nodes (28): candidate_points(), cluster_entry(), cluster_points(), main(), remove_statistical_outliers(), visibility_check(), load_reconstruction(), project_points() (+20 more)

### Community 12 - "Setup"
Cohesion: 0.25
Nodes (8): Batch jobs, Cluster environment, Configuration, Laptop environment (CPU stages), Model weights, Setup, Verify CUDA inside a GPU job, Which stages run where

### Community 13 - "inspect_cache.py"
Cohesion: 0.48
Nodes (9): _fmt(), inspect_diff(), inspect_frames(), inspect_outputs(), inspect_reconstruction(), inspect_semantics(), main(), _read_json() (+1 more)

### Community 14 - "download_models.sh"
Cohesion: 0.50
Nodes (4): download(), HF_HOME, HF_HUB_ENABLE_HF_TRANSFER, download_models.sh script

### Community 19 - "The rules, in order"
Cohesion: 0.17
Nodes (12): 2. Self-consistency override (removed/added), 3. Mutual cancellation, 4. Score floor (`semantics.min_change_score`, default 80), 5. Static-object suppression (`src.tabletop.static_object_mask`), 6. Replaced pixel gate (`semantics.replaced_min_pixel_diff`, default 70), 7. Same-appearance rejection (`semantics.same_appearance_threshold`, 0.80), 8. Duplicate rules, 9. Fragment merging (+4 more)

### Community 20 - "Details"
Cohesion: 0.33
Nodes (5): 1. REMOVED - comb, 2. ADDED - pen and device, 3. REPLACED - remote control, Change report, Details

### Community 23 - "run_mapanything.py"
Cohesion: 0.10
Nodes (15): load_frames(), main(), pin_idle_gpus(), model_bbox_to_original(), model_to_original_transform(), prediction_to_arrays(), _squeeze_batch(), transform_for() (+7 more)

### Community 24 - "Development log"
Cohesion: 0.08
Nodes (26): 2026-10-10 — Backend comparison: MapAnything + DA3 (`v0.5`), 2026-10-10 — Debuggability + docs (`v0.4`), 2026-10-10 — Phase 1: environment + scaffold (`v0.2`), 2026-10-10 — Phase 2–4: the demo scene (checkpoints 2–4), 2026-10-10 — Phase 5: first end-to-end results (`v0.3`), 2026-10-10 — Process updates (`v0.5.1`), 2026-10-10 — Project start (docs-only repo), 2026-10-10 — Robustness runs on the extra scenes (`data-v0.2`) (+18 more)

### Community 25 - "Backend comparison: VGGT vs MapAnything vs Depth Anything 3"
Cohesion: 0.12
Nodes (17): Ablations (observed, not benchmarked), Backend comparison: VGGT vs MapAnything vs Depth Anything 3, Detection vs ground truth, End-to-end results (ground truth: comb removed, pen added, remote moved+flipped), How this run was produced, Limitations and failure cases (as observed on this scene), Metric scale, Outputs of the run (+9 more)

### Community 26 - "make_fake_reconstruction.py"
Cohesion: 0.43
Nodes (3): main(), make_frame(), unproject()

## Knowledge Gaps
- **3 isolated node(s):** `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 266 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config reference` connect `docs/README.md` to `common.py`, `describe.py`?**
  _High betweenness centrality (0.255) - this node is a cross-community bridge._
- **What connects `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `common.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07993827160493827 - nodes in this community are weakly interconnected._
- **Why does `cache_dir()` connect `common.py` to `docs/README.md`, `numpy`, `describe.py`?**
  _High betweenness centrality (0.152) - this node is a cross-community bridge._
- **Should `docs/README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.0753045404208195 - nodes in this community are weakly interconnected._
- **Should `AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._
- **Should `GPU Access on This Cluster` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._