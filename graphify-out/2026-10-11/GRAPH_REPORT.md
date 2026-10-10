# Graph Report - Humanoid  (2026-10-10)

## Corpus Check
- 39 files · ~114,925 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 4, .pbs 3)

## Summary
- 394 nodes · 836 edges · 23 communities (16 shown, 7 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d1002b39`
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

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 25 edges
2. `repo_path()` - 25 edges
3. `main()` - 22 edges
4. `write_json()` - 21 edges
5. `cache_dir()` - 19 edges
6. `main()` - 18 edges
7. `stamp()` - 17 edges
8. `apply_overrides()` - 17 edges
9. `build_change_list()` - 17 edges
10. `read_json()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `GPU debugging` --references--> `pin_idle_gpus()`  [INFERRED]
  docs/debugging.md → src/common.py
- `Semantics (crops, VLM, verification)` --references--> `best_crop_for_cluster()`  [INFERRED]
  docs/debugging.md → src/describe.py
- `5. Static-object suppression (`src.tabletop.static_object_mask`)` --references--> `static_object_mask()`  [INFERRED]
  docs/verification.md → src/tabletop.py
- `Config reference` --references--> `cache_dir()`  [INFERRED]
  docs/architecture.md → src/common.py
- `Config reference` --references--> `out_dir()`  [INFERRED]
  docs/architecture.md → src/common.py

## Import Cycles
- None detected.

## Communities (23 total, 7 thin omitted)

### Community 0 - "common.py"
Cohesion: 0.08
Nodes (31): main(), stage_enabled(), main(), static_subset(), add_common_args(), apply_overrides(), cache_dir(), config_hash() (+23 more)

### Community 2 - "docs/README.md"
Cohesion: 0.06
Nodes (32): Architecture, Caches and outputs, Config reference, Conventions, Module responsibilities, Repository layout, Checklist before starting the pipeline, Data Collection (+24 more)

### Community 3 - "AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)"
Cohesion: 0.10
Nodes (21): 0. Critical rules (read first), 10. Risks and mitigations, 11. Config skeleton (`configs/default.yaml`), 12. Definition of done, 1. Goal, 2. Environment constraints (cluster), 3. Model stack (decided), 4. Repository structure (+13 more)

### Community 4 - "GPU Access on This Cluster"
Cohesion: 0.10
Nodes (21): 1. TL;DR — the shortest path, 2. GPU hardware configuration, 3. Node / system configuration, 4. PBS queue configuration, 5. The three gotchas (this is what wastes everyone's time), 6. Ways to access the GPUs, 7. Helper tools installed for this, 8. Verified evidence (+13 more)

### Community 5 - "describe.py"
Cohesion: 0.06
Nodes (19): 1. VLM verification (`PROMPT_VERIFY`), pin_idle_gpus(), resolve_model(), best_crop_for_cluster(), embed_crops(), load_matcher(), load_vlm(), main() (+11 more)

### Community 6 - "Rules for Agents Working in This Folder"
Cohesion: 0.18
Nodes (11): 1. Stay inside this folder, 2. Only approved exceptions, 3. Installing tools and dependencies, 4. Handling existing work, 5.1 Tags: `major.minor`, 5.2 Large files: use Git LFS when needed, 5. Git conventions, 6. Resource use (this is an HPC cluster) (+3 more)

### Community 7 - "make_test_videos.py"
Cohesion: 0.18
Nodes (7): _evenly_spaced(), extract_one(), _scan_video(), base_scene(), main(), make_video(), render_frame()

### Community 8 - "Stage detail"
Cohesion: 0.18
Nodes (11): 1. Frame extraction, 2. Joint reconstruction (most important), 3. Alignment (optional), 4. Change detection, 5. Semantics, 6. Visualization, 7. Report, Caching strategy (+3 more)

### Community 9 - "build_change_list"
Cohesion: 0.12
Nodes (13): Change detection: misses or spams candidates, Debugging, Frame extraction, GPU debugging, Logs, Playbook by symptom, Reconstruction, Report / visualization (+5 more)

### Community 10 - "3D Scene Change Detection"
Cohesion: 0.20
Nodes (10): 3D Scene Change Detection, Design decisions, Documentation, How to run, License, Limitations and failure cases, Models, Pipeline (+2 more)

### Community 11 - "numpy"
Cohesion: 0.09
Nodes (16): tagged_points(), cell_maps(), _cells_to_points(), detect(), fit_color_field(), fit_dominant_plane(), grid_bounds(), objectness() (+8 more)

### Community 12 - "Setup"
Cohesion: 0.25
Nodes (8): Batch jobs, Cluster environment, Configuration, Laptop environment (CPU stages), Model weights, Setup, Verify CUDA inside a GPU job, Which stages run where

### Community 13 - "inspect_cache.py"
Cohesion: 0.26
Nodes (11): _fmt(), inspect_diff(), inspect_frames(), inspect_outputs(), inspect_reconstruction(), inspect_semantics(), main(), _read_json() (+3 more)

### Community 14 - "download_models.sh"
Cohesion: 0.50
Nodes (4): download(), HF_HOME, HF_HUB_ENABLE_HF_TRANSFER, download_models.sh script

### Community 19 - "The rules, in order"
Cohesion: 0.17
Nodes (12): 2. Self-consistency override (removed/added), 3. Mutual cancellation, 4. Score floor (`semantics.min_change_score`, default 80), 5. Static-object suppression (`src.tabletop.static_object_mask`), 6. Replaced pixel gate (`semantics.replaced_min_pixel_diff`, default 70), 7. Same-appearance rejection (`semantics.same_appearance_threshold`, 0.80), 8. Duplicate rules, 9. Fragment merging (+4 more)

### Community 20 - "Details"
Cohesion: 0.33
Nodes (5): 1. REMOVED - comb, 2. ADDED - pen and device, 3. REPLACED - remote control, Change report, Details

## Knowledge Gaps
- **3 isolated node(s):** `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 212 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config reference` connect `docs/README.md` to `common.py`, `describe.py`?**
  _High betweenness centrality (0.307) - this node is a cross-community bridge._
- **What connects `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `common.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08354430379746836 - nodes in this community are weakly interconnected._
- **Why does `cache_dir()` connect `common.py` to `docs/README.md`, `describe.py`, `make_test_videos.py`?**
  _High betweenness centrality (0.189) - this node is a cross-community bridge._
- **Should `docs/README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.06382978723404255 - nodes in this community are weakly interconnected._
- **Should `AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._
- **Should `GPU Access on This Cluster` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._