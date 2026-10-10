# Graph Report - Humanoid  (2026-10-10)

## Corpus Check
- 31 files · ~20,640 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 4, .pbs 3)

## Summary
- 296 nodes · 640 edges · 19 communities (15 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `849dca82`
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
- run.py
- Setup
- Architecture
- download_models.sh
- gpu_smoke_test.sh
- AGENTS.md
- setup_env.sh
- tests/README.md

## God Nodes (most connected - your core abstractions)
1. `repo_path()` - 23 edges
2. `load_config()` - 21 edges
3. `write_json()` - 21 edges
4. `cache_dir()` - 19 edges
5. `stamp()` - 17 edges
6. `apply_overrides()` - 17 edges
7. `main()` - 17 edges
8. `read_json()` - 16 edges
9. `add_common_args()` - 16 edges
10. `main()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Config reference` --references--> `cache_dir()`  [INFERRED]
  docs/architecture.md → src/common.py
- `Config reference` --references--> `out_dir()`  [INFERRED]
  docs/architecture.md → src/common.py
- `Config reference` --references--> `resolve_model()`  [INFERRED]
  docs/architecture.md → src/common.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  run.py → src/common.py
- `main()` --calls--> `repo_path()`  [EXTRACTED]
  run.py → src/common.py

## Import Cycles
- None detected.

## Communities (19 total, 4 thin omitted)

### Community 0 - "common.py"
Cohesion: 0.09
Nodes (22): cache_dir(), config_hash(), _json_default(), out_dir(), read_json(), repo_path(), setup_logging(), stage_outputs() (+14 more)

### Community 1 - "numpy"
Cohesion: 0.10
Nodes (19): main(), static_subset(), add_common_args(), apply_overrides(), load_config(), candidate_points(), cluster_entry(), cluster_points() (+11 more)

### Community 2 - "docs/README.md"
Cohesion: 0.09
Nodes (23): Checklist before starting the pipeline, Data Collection, Files, `ground_truth.md` template, Recording guide (from `AGENT_PLAN.md` §6, Phase 0), Downloads, Exact identifiers and licenses — CHECKLIST, Memory notes (+15 more)

### Community 3 - "AGENT_PLAN.md — 3D Scene Change Detection (Spatial AI & Navigation Challenge)"
Cohesion: 0.10
Nodes (21): 0. Critical rules (read first), 10. Risks and mitigations, 11. Config skeleton (`configs/default.yaml`), 12. Definition of done, 1. Goal, 2. Environment constraints (cluster), 3. Model stack (decided), 4. Repository structure (+13 more)

### Community 4 - "GPU Access on This Cluster"
Cohesion: 0.10
Nodes (21): 1. TL;DR — the shortest path, 2. GPU hardware configuration, 3. Node / system configuration, 4. PBS queue configuration, 5. The three gotchas (this is what wastes everyone's time), 6. Ways to access the GPUs, 7. Helper tools installed for this, 8. Verified evidence (+13 more)

### Community 5 - "describe.py"
Cohesion: 0.17
Nodes (10): resolve_model(), best_crop_for_cluster(), embed_crops(), load_matcher(), load_vlm(), main(), parse_json_reply(), vlm_answer() (+2 more)

### Community 6 - "Rules for Agents Working in This Folder"
Cohesion: 0.17
Nodes (11): 1. Stay inside this folder, 2. Only approved exceptions, 3. Installing tools and dependencies, 4. Handling existing work, 5.1 Tags: `major.minor`, 5.2 Large files: use Git LFS when needed, 5. Git conventions, 6. Resource use (this is an HPC cluster) (+3 more)

### Community 7 - "make_test_videos.py"
Cohesion: 0.31
Nodes (4): base_scene(), main(), make_video(), render_frame()

### Community 8 - "Stage detail"
Cohesion: 0.18
Nodes (11): 1. Frame extraction, 2. Joint reconstruction (most important), 3. Alignment (optional), 4. Change detection, 5. Semantics, 6. Visualization, 7. Report, Caching strategy (+3 more)

### Community 9 - "build_change_list"
Cohesion: 0.18
Nodes (3): build_change_list(), model_bbox_to_original(), recon_dir_for()

### Community 10 - "3D Scene Change Detection"
Cohesion: 0.20
Nodes (10): 3D Scene Change Detection, Design decisions, Documentation, How to run, License, Limitations and failure cases, Models, Pipeline (+2 more)

### Community 11 - "run.py"
Cohesion: 0.27
Nodes (3): main(), stage_enabled(), Timer

### Community 12 - "Setup"
Cohesion: 0.25
Nodes (8): Batch jobs, Cluster environment, Configuration, Laptop environment (CPU stages), Model weights, Setup, Verify CUDA inside a GPU job, Which stages run where

### Community 13 - "Architecture"
Cohesion: 0.33
Nodes (6): Architecture, Caches and outputs, Config reference, Conventions, Module responsibilities, Repository layout

### Community 14 - "download_models.sh"
Cohesion: 0.50
Nodes (4): download(), HF_HOME, HF_HUB_ENABLE_HF_TRANSFER, download_models.sh script

## Knowledge Gaps
- **3 isolated node(s):** `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 163 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Architecture` connect `Architecture` to `docs/README.md`?**
  _High betweenness centrality (0.458) - this node is a cross-community bridge._
- **What connects `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `common.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08711433756805807 - nodes in this community are weakly interconnected._
- **Why does `Config reference` connect `Architecture` to `common.py`, `describe.py`?**
  _High betweenness centrality (0.448) - this node is a cross-community bridge._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.10040816326530612 - nodes in this community are weakly interconnected._
- **Why does `cache_dir()` connect `common.py` to `numpy`, `describe.py`, `Architecture`?**
  _High betweenness centrality (0.272) - this node is a cross-community bridge._
- **Should `docs/README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.08739495798319327 - nodes in this community are weakly interconnected._