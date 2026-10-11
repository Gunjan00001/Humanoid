# Graph Report - Humanoid  (2026-10-11)

## Corpus Check
- 57 files · ~64,276 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 653 nodes · 1620 edges · 42 communities (31 shown, 11 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Geometry & alignment
- Architecture & layout
- detect.py runner
- Stage CLI entry points
- Tabletop diff & plane fit
- Common utilities
- Checking sheet & encoding
- Development log history
- Setup & environment
- MapAnything artifact contract
- Model backends & findings
- Change detection core
- Natural-language summary
- Documentation set
- Project plan
- Verification rule chain
- Debugging playbook
- Config & cache rules
- Frame extraction
- Held-out protocol & handoff
- Troubleshooting guide
- README overview
- Working rules
- Pipeline stage detail
- Results & ablations
- Environment setup
- GPU access methods
- Move & reconstruction decisions
- Data collection guide
- Demo change report
- Model downloads
- Roadmap & next steps
- Frame loading order
- GPU smoke test
- Graphify integration
- Demo ground truth
- Scene2 ground truth
- Scene3 ground truth
- Environment setup script
- Static subset helper
- Even spacing helper

## God Nodes (most connected - your core abstractions)
1. `repo_path()` - 47 edges
2. `load_config()` - 40 edges
3. `write_json()` - 29 edges
4. `main()` - 29 edges
5. `read_json()` - 28 edges
6. `load_reconstruction()` - 22 edges
7. `main()` - 22 edges
8. `stamp()` - 21 edges
9. `apply_overrides()` - 21 edges
10. `cache_dir()` - 20 edges

## Surprising Connections (you probably didn't know these)
- `GPU debugging` --references--> `pin_idle_gpus()`  [INFERRED]
  docs/debugging.md → src/common.py
- `Semantics (crops, VLM, verification)` --references--> `best_crop_for_cluster()`  [INFERRED]
  docs/debugging.md → src/describe.py
- `5. Static-object suppression` --references--> `static_object_mask()`  [INFERRED]
  docs/verification.md → src/tabletop.py
- `Config reference` --references--> `cache_dir()`  [INFERRED]
  docs/architecture.md → src/common.py
- `Config reference` --references--> `out_dir()`  [INFERRED]
  docs/architecture.md → src/common.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Held-out evaluation of scene5** — docs_heldout_heldout_protocol, docs_heldout_frozen_commit_e8c77d9, docs_heldout_preregistered_scoring_rule, docs_heldout_scene5_result, docs_heldout_scene5_failure_analysis, docs_heldout_posthoc_tabletop_diagnostic, rules_heldout_scenes_and_posthoc [EXTRACTED 1.00]
- **Verification rule chain (26 candidates to 3 changes)** — docs_verification_1_vlm_verification_prompt_verify, docs_verification_2_self_consistency_override_removed_added, docs_verification_3_mutual_cancellation, docs_verification_4_score_floor_semantics_min_change_score_default_80, docs_verification_5_static_object_suppression_src_tabletop_static_object_mask, docs_verification_6_replaced_pixel_gate_semantics_replaced_min_pixel_diff_default_70, docs_verification_7_same_appearance_rejection_semantics_same_appearance_threshold_0_80, docs_verification_8_duplicate_rules, docs_verification_9_fragment_merging, docs_verification_nothing_silently_dropped [EXTRACTED 1.00]
- **Change-detection modes and their selection** — docs_pipeline_geometric_diff, docs_pipeline_tabletop_diff, docs_pipeline_diff_mode_choice, configs_default_diff_config, agent_plan_visibility_check [EXTRACTED 1.00]

## Communities (42 total, 11 thin omitted)

### Community 0 - "Geometry & alignment"
Cohesion: 0.09
Nodes (18): load_reconstruction(), recon_dir_for(), tagged_points(), voxel_downsample(), export_mesh(), _camera_pos(), change_framing(), check_render_caches() (+10 more)

### Community 1 - "Architecture & layout"
Cohesion: 0.06
Nodes (26): Architecture, Caches and outputs, Config reference, Conventions, Module responsibilities, Repository layout, Failure: VLM verification over-confirmed (iterations), pin_idle_gpus() (+18 more)

### Community 2 - "detect.py runner"
Cohesion: 0.09
Nodes (22): build_pbs_script(), check_pair(), choose_mode(), fmt(), main(), on_gpu_node(), run_local(), run_stage() (+14 more)

### Community 3 - "Stage CLI entry points"
Cohesion: 0.14
Nodes (22): main(), main(), main(), add_common_args(), apply_overrides(), cache_dir(), _json_default(), load_config() (+14 more)

### Community 4 - "Tabletop diff & plane fit"
Cohesion: 0.10
Nodes (14): plane_coverage(), Failure: cell-code collision attached wrong points to clusters, cell_maps(), fit_color_field(), fit_dominant_plane(), grid_bounds(), objectness(), plane_height() (+6 more)

### Community 5 - "Common utilities"
Cohesion: 0.13
Nodes (5): config_hash(), setup_logging(), main(), make_frame(), unproject()

### Community 6 - "Checking sheet & encoding"
Cohesion: 0.16
Nodes (13): main(), _fmt(), inspect_diff(), inspect_frames(), inspect_outputs(), inspect_reconstruction(), inspect_semantics(), main() (+5 more)

### Community 7 - "Development log history"
Cohesion: 0.08
Nodes (25): 2026-10-10 — Backend comparison: MapAnything + DA3 (`v0.5`), 2026-10-10 — Debuggability + docs (`v0.4`), 2026-10-10 — Phase 1: environment + scaffold (`v0.2`), 2026-10-10 — Phase 2–4: the demo scene (checkpoints 2–4), 2026-10-10 — Phase 5: first end-to-end results (`v0.3`), 2026-10-10 — Process updates (`v0.5.1`), 2026-10-10 — Project start (docs-only repo), 2026-10-10 — Robustness runs on the extra scenes (`data-v0.2`) (+17 more)

### Community 8 - "Setup & environment"
Cohesion: 0.13
Nodes (17): Environment cheat-sheet, spatialdiff conda env, Clock bases (3:30:00 offset), Model load times (measured), One full scene prediction (job 10802, 5:14), Queue wait = per-user 16-CPU quota, spatialdiff conda env export, 1. TL;DR — the shortest path (+9 more)

### Community 9 - "MapAnything artifact contract"
Cohesion: 0.13
Nodes (11): model_bbox_to_original(), prediction_to_arrays(), _squeeze_batch(), transform_for(), tests, check(), _fake_pred(), main() (+3 more)

### Community 10 - "Model backends & findings"
Cohesion: 0.11
Nodes (18): Downloads, Model identifiers and license checklist, MapAnything (facebook/map-anything-apache), Memory notes, Models, Stack, VGGT (facebook/VGGT-1B), Why these models (+10 more)

### Community 11 - "Change detection core"
Cohesion: 0.15
Nodes (10): candidate_points(), cluster_entry(), cluster_points(), main(), remove_statistical_outliers(), visibility_check(), project_points(), sample_depth() (+2 more)

### Community 12 - "Natural-language summary"
Cohesion: 0.20
Nodes (13): build_summary(), sentence_for(), _significant(), _singular(), template_sentence(), validate_sentence(), _with_video_ref(), write_summaries() (+5 more)

### Community 13 - "Documentation set"
Cohesion: 0.26
Nodes (6): State v0.12.0 (2026-10-12), Documentation Index, Presentation + correctness pass (v0.9.0), Version history (v0.5 to v0.12.0), Old timing claims vs measurements, 7. Development log

### Community 14 - "Project plan"
Cohesion: 0.10
Nodes (20): 0. Critical rules (read first), 10. Risks and mitigations, 11. Config skeleton (`configs/default.yaml`), 12. Definition of done, 1. Goal, 2. Environment constraints (cluster), 4. Repository structure, 5. Pipeline overview (+12 more)

### Community 15 - "Verification rule chain"
Cohesion: 0.19
Nodes (10): semantics config (VLM, matcher, thresholds), 2. Self-consistency override (removed/added), 3. Mutual cancellation, 8. Duplicate rules, 9. Fragment merging, Final outcome on the demo scene, The rules, in order, Verification: from 26 candidates to 3 changes (+2 more)

### Community 16 - "Debugging playbook"
Cohesion: 0.12
Nodes (13): Change detection: misses or spams candidates, Debugging, Frame extraction, GPU debugging, Logs, Playbook by symptom, Reconstruction, Report / visualization (+5 more)

### Community 17 - "Config & cache rules"
Cohesion: 0.21
Nodes (11): Model stack (decided), diff config (geometric + tabletop parameters), src/summary.py natural-language summary, scripts/inspect_cache.py read-only inspector, Known pitfalls, Qwen3-VL-32B-Instruct, Caching strategy, Pipeline (+3 more)

### Community 18 - "Frame extraction"
Cohesion: 0.21
Nodes (6): extract_one(), _scan_video(), base_scene(), main(), make_video(), render_frame()

### Community 19 - "Held-out protocol & handoff"
Cohesion: 0.21
Nodes (11): detect.py one-command runner, Prioritized next steps, Frozen commit e8c77d9 + file digests, Post-hoc tabletop diagnostic (scene5), Re-encoded b.mp4 check, Scene5 failure analysis, Scene5 held-out result (1/2, type only), GitHub Pages hosting (+3 more)

### Community 20 - "Troubleshooting guide"
Cohesion: 0.22
Nodes (10): Cluster / environment, CUDA_VISIBLE_DEVICES gotcha, Debugging workflow, Model / pipeline, Troubleshooting, 5. The three gotchas (this is what wastes everyone's time), Bonus — all 8 GPUs are visible, and the node is busy, Gotcha 1: PBS hides GPUs via empty CUDA_VISIBLE_DEVICES (+2 more)

### Community 21 - "README overview"
Cohesion: 0.18
Nodes (11): 3D Scene Change Detection, Backend comparison (v0.5), Documentation, How to run, License, Limitations and failure cases, Models, Pipeline (+3 more)

### Community 22 - "Working rules"
Cohesion: 0.20
Nodes (9): 1. Stay inside this folder, 2. Only approved exceptions, 3. Installing tools and dependencies, 4. Handling existing work, 5.1 Tags: `major.minor.bugs`, 5. Git conventions, 8. Reporting, Rules for Agents Working in This Folder (+1 more)

### Community 23 - "Pipeline stage detail"
Cohesion: 0.25
Nodes (8): 1. Frame extraction, 2. Joint reconstruction (most important), 3. Alignment (optional), 4. Change detection, 5. Semantics, 6. Visualization, 7. Report, Stage detail

### Community 24 - "Results & ablations"
Cohesion: 0.25
Nodes (8): Ablations (observed, not benchmarked), Detection vs ground truth, How this run was produced, Limitations and failure cases (as observed on this scene), Outputs of the run, Results, Robustness on the extra scenes (data-v0.2), Timings (wall clock, this run)

### Community 25 - "Environment setup"
Cohesion: 0.25
Nodes (8): Batch jobs, Cluster environment, Configuration, Laptop environment (CPU stages), Model weights, Setup, Verify CUDA inside a GPU job, Which stages run where

### Community 26 - "GPU access methods"
Cohesion: 0.25
Nodes (8): 6. Ways to access the GPUs, A. Interactive shell (normal workflow), B. Batch job (recommended for training), C. PTY wrapper (for scripts/agents that have no TTY), D. Jupyter notebook, E. Persistent GPU session (best for interactive development / agents), F. One-shot runner, G. Run the agent itself on a GPU node (fully native)

### Community 29 - "Data collection guide"
Cohesion: 0.33
Nodes (6): Additional scenes, Checklist before starting the pipeline, Data Collection, Files, `ground_truth.md` template, Recording guide (from `AGENT_PLAN.md` §6, Phase 0)

### Community 30 - "Demo change report"
Cohesion: 0.33
Nodes (5): 1. REMOVED - comb, 2. ADDED - pen and device, 3. REPLACED - remote control, Change report, Details

### Community 31 - "Model downloads"
Cohesion: 0.50
Nodes (4): download(), HF_HOME, HF_HUB_ENABLE_HF_TRANSFER, download_models.sh script

### Community 32 - "Roadmap & next steps"
Cohesion: 0.50
Nodes (4): Backend comparison (v0.5), Immediate next steps, Infrastructure already done, Roadmap

## Knowledge Gaps
- **3 isolated node(s):** `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 271 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Backend comparison: VGGT vs MapAnything vs DA3` connect `Model backends & findings` to `Results & ablations`, `Geometry & alignment`, `Documentation set`?**
  _High betweenness centrality (0.131) - this node is a cross-community bridge._
- **What connects `HF_HOME`, `HF_HUB_ENABLE_HF_TRANSFER`, `CUDA_VISIBLE_DEVICES` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Geometry & alignment` be split into smaller, more focused modules?**
  _Cohesion score 0.08571428571428572 - nodes in this community are weakly interconnected._
- **Why does `repo_path()` connect `Stage CLI entry points` to `Geometry & alignment`, `Architecture & layout`, `detect.py runner`, `Common utilities`, `Checking sheet & encoding`, `Debugging playbook`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Should `Architecture & layout` be split into smaller, more focused modules?**
  _Cohesion score 0.06205673758865248 - nodes in this community are weakly interconnected._
- **Why does `Development log` connect `Development log history` to `Documentation set`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Should `detect.py runner` be split into smaller, more focused modules?**
  _Cohesion score 0.08599033816425121 - nodes in this community are weakly interconnected._