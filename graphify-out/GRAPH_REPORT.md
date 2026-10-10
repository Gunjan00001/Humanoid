# Graph Report - Humanoid  (2026-10-10)

## Corpus Check
- 36 files · ~111,754 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 4, .pbs 3)

## Summary
- 356 nodes · 755 edges · 22 communities (17 shown, 5 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ec566ae7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- common.py
- numpy
- docs/README.md
- AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)
- GPU Access on This Cluster
- describe.py
- Rules for Agents Working in This Folder
- make_test_videos.py
- Stage detail
- build_change_list
- 3D Scene Change Detection
- tabletop.py
- Setup
- Results
- download_models.sh
- gpu_smoke_test.sh
- AGENTS.md
- setup_env.sh
- tests/README.md
- make_fake_reconstruction.py
- Details
- ground_truth.md

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 23 edges
2. `repo_path()` - 23 edges
3. `main()` - 22 edges
4. `write_json()` - 21 edges
5. `cache_dir()` - 19 edges
6. `main()` - 18 edges
7. `stamp()` - 17 edges
8. `apply_overrides()` - 17 edges
9. `read_json()` - 16 edges
10. `add_common_args()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Config reference` --references--> `resolve_model()`  [INFERRED]
  docs/architecture.md → src/common.py
- `Config reference` --references--> `cache_dir()`  [INFERRED]
  docs/architecture.md → src/common.py
- `Config reference` --references--> `out_dir()`  [INFERRED]
  docs/architecture.md → src/common.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  tests/gpu_describe_smoke.py → src/common.py
- `main()` --calls--> `write_json()`  [EXTRACTED]
  tests/make_fake_reconstruction.py → src/common.py

## Import Cycles
- None detected.

## Communities (22 total, 5 thin omitted)

### Community 0 - "common.py"
Cohesion: 0.08
Nodes (23): Config reference, main(), stage_enabled(), apply_overrides(), cache_dir(), config_hash(), _json_default(), load_config() (+15 more)

### Community 1 - "numpy"
Cohesion: 0.11
Nodes (17): main(), static_subset(), add_common_args(), candidate_points(), cluster_entry(), cluster_points(), main(), remove_statistical_outliers() (+9 more)

### Community 2 - "docs/README.md"
Cohesion: 0.08
Nodes (24): Architecture, Caches and outputs, Conventions, Module responsibilities, Repository layout, Checklist before starting the pipeline, Data Collection, Files (+16 more)

### Community 3 - "AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)"
Cohesion: 0.10
Nodes (21): 0. Critical rules (read first), 10. Risks and mitigations, 11. Config skeleton (`configs/default.yaml`), 12. Definition of done, 1. Goal, 2. Environment constraints (cluster), 3. Model stack (decided), 4. Repository structure (+13 more)

### Community 4 - "GPU Access on This Cluster"
Cohesion: 0.10
Nodes (21): 1. TL;DR — the shortest path, 2. GPU hardware configuration, 3. Node / system configuration, 4. PBS queue configuration, 5. The three gotchas (this is what wastes everyone's time), 6. Ways to access the GPUs, 7. Helper tools installed for this, 8. Verified evidence (+13 more)

### Community 5 - "describe.py"
Cohesion: 0.09
Nodes (16): pin_idle_gpus(), resolve_model(), best_crop_for_cluster(), embed_crops(), load_matcher(), load_vlm(), main(), _objects_overlap() (+8 more)

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
Cohesion: 0.29
Nodes (3): build_change_list(), crop(), evidence()

### Community 10 - "3D Scene Change Detection"
Cohesion: 0.20
Nodes (10): 3D Scene Change Detection, Design decisions, Documentation, How to run, License, Limitations and failure cases, Models, Pipeline (+2 more)

### Community 11 - "tabletop.py"
Cohesion: 0.12
Nodes (11): cell_maps(), _cells_to_points(), detect(), fit_color_field(), fit_dominant_plane(), grid_bounds(), objectness(), plane_height() (+3 more)

### Community 12 - "Setup"
Cohesion: 0.25
Nodes (8): Batch jobs, Cluster environment, Configuration, Laptop environment (CPU stages), Model weights, Setup, Verify CUDA inside a GPU job, Which stages run where

### Community 13 - "Results"
Cohesion: 0.29
Nodes (7): Ablations (observed, not benchmarked), Detection vs ground truth, How this run was produced, Limitations and failure cases (as observed on this scene), Outputs of the run, Results, Timings (wall clock, this run)

### Community 14 - "download_models.sh"
Cohesion: 0.50
Nodes (4): download(), HF_HOME, HF_HUB_ENABLE_HF_TRANSFER, download_models.sh script

### Community 19 - "make_fake_reconstruction.py"
Cohesion: 0.43
Nodes (3): main(), make_frame(), unproject()

### Community 20 - "Details"
Cohesion: 0.33
Nodes (5): 1. REMOVED - comb, 2. ADDED - pen and device, 3. REPLACED - remote control, Change report, Details

## Knowledge Gaps
- **3 isolated node(s):** `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 196 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Architecture` connect `docs/README.md` to `common.py`?**
  _High betweenness centrality (0.422) - this node is a cross-community bridge._
- **What connects `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `common.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08298368298368299 - nodes in this community are weakly interconnected._
- **Why does `Config reference` connect `common.py` to `docs/README.md`, `describe.py`?**
  _High betweenness centrality (0.415) - this node is a cross-community bridge._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.10628019323671498 - nodes in this community are weakly interconnected._
- **Why does `cache_dir()` connect `common.py` to `numpy`, `describe.py`, `make_test_videos.py`?**
  _High betweenness centrality (0.242) - this node is a cross-community bridge._
- **Should `docs/README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.08258258258258258 - nodes in this community are weakly interconnected._