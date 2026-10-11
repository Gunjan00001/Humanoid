# Roadmap

Status board for the phases in [`../AGENT_PLAN.md`](../AGENT_PLAN.md) §6 and the
Definition of Done in §12. Update this file as checkpoints pass.

**Checkpoint discipline:** a checkpoint is only ✅ when it passes on real data.
Code that exists but has not been validated end-to-end is marked 🔶.

| Phase | Deliverable | Checkpoint | Status |
|---|---|---|---|
| 0 | Two videos + `data/ground_truth.md` | Checkpoint 0 | ✅ 2026-10-10 (`data/a.mp4`, `data/b.mp4`, 3 documented changes) |
| 1 | Repo scaffold, `spatialdiff` env (cu126), weights in `~/models`, `configs/default.yaml` | Checkpoint 1: CUDA visible in a GPU job + weights load | ✅ verified 2026-10-10: 8×A100 visible (torch 2.14.1+cu126), VGGT loaded offline from `~/models/VGGT-1B` |
| 2 | `src/extract_frames.py`, `src/reconstruct.py` (joint VGGT), `src/align.py` | Checkpoint 2: A/B static structure aligns | ✅ 49-frame joint reconstruction; ICP static mean NN 15.2 → 10.7 mm; 17 s inference |
| 3 | `src/diff.py` with visibility check | Checkpoint 3: 3 known changes found, few false positives | ✅ all 3 found; final report has 0 false positives (22 candidates rejected with reasons); required the photometric tabletop mode - see results.md |
| 4 | `src/describe.py` (Qwen3-VL + DINOv2) | Checkpoint 4: every change has type, label, location, sentence | ✅ 3/3 with labels + sentences; VLM verification rejects lighting artefacts |
| 5 | `src/visualize.py`, `src/report.py`, `run.py` | Checkpoint 5: one command produces all outputs | ✅ `run.py` produces scene.ply, change.gif, screenshots, report.json/md |
| 6 | README with real results, ablations, public repo | Definition of done | ✅ 2026-10-11: README + results.md with real numbers, three-way backend comparison, repo public, fresh-clone verified (LFS + CPU stage) |

## Backend comparison (v0.5)

Two more reconstruction backends were added and compared against VGGT on the
demo scene with identical frames: **MapAnything**
(`reconstruction.backend: mapanything`, isolated `spatialdiff-ma` env) and the
stretch goal **Depth Anything 3** (DA3-LARGE-1.1 through MapAnything's unified
interface). All numbers are measured; see
[`results.md`](results.md#backend-comparison-vggt-vs-mapanything-vs-depth-anything-3).

Headline: **VGGT + tabletop stays the best single configuration (3/3 changes,
0 false positives)**. MapAnything matches it with its threshold calibrated and
adds metric scale + a geometric diff that sees the 1 cm objects; DA3 is the
fastest and lightest backend with the best alignment and the strongest
thin-object response. Neither replaces the default; both are selectable.

## Infrastructure already done

- [x] GPU access solved and documented (`gpu.md`, `gpu-run`, persistent session)
- [x] `cu126` PyTorch verified on the A100s
- [x] Documentation set (`docs/`)
- [x] Git repo initialized and connected to GitHub
- [x] Full pipeline code + config + PBS templates + one-command `run.py`
- [x] Model weights download script; licenses verified (`docs/models.md`)
- [x] Synthetic smoke tests for the CPU stages (`tests/`, never claimed as results)

## After Phase 6 — what shipped since (all verified, all tagged)

| Version | What |
|---|---|
| v0.5 / v0.5.1 | MapAnything + Depth Anything 3 backends compared against VGGT (`docs/results.md`) |
| v0.6.x / v0.7.0 | extra scenes 2 and 3 (`data-v0.2`) end-to-end; per-scene working points documented |
| v0.8.0 | visualization rework: table-plane-up camera, focused 0.4 m crop, TSDF mesh, self-contained viewer, `input_mode` (crop kept), pad experiment |
| v0.9.0 | presentation + correctness pass: 2-row layouts, Chrome-rendered GIF from the viewer, occlusion-aware crops, geometric type correction (scene3's chair -> `removed`, apple/cup -> `replaced`) |
| v0.10.0 | scene4 (`data-v0.3`): move matching before the filters, arrivals at a move's origin, composite-replaced rule -> **3/3, 0 extras** |
| v0.11.0 | timing audit (`docs/timing.md`), `--render-only`, GIF palette fix, unverified hidden behind a toggle |
| **v0.12.0** | **`detect.py` one-command runner, natural-language summaries (`src/summary.py`), held-out scene5 run + protocol (`docs/heldout.md`), post-hoc diagnostics, regressions, b.mp4 re-encode, fresh-clone verified** |

## Current next steps

See **[`docs/handoff.md`](handoff.md)** for the full brief. In short:

1. **Enable GitHub Pages** (owner action) so the committed viewers go live.
2. **scene5 post-hoc work** (the missed clock move; the auto-mode encoding
   sensitivity; orientation normalization) — all labelled post-hoc, never
   touching the held-out result.
3. **scene2 reproducibility** (geometric-diff run-to-run spread).
4. Optional: Qwen3-VL-8B vs 32B ablation, MASt3R/COLMAP backends, a **new
   held-out scene6** (same orientation, landscape preferred).

Validation evidence so far (all from real runs; smoke tests are explicitly
*not* project results):
- `spatialdiff` env: torch 2.14.1+cu126, CUDA visible in a GPU job, 8×A100.
- Five scenes run end-to-end; demo 3/3 with 0 extras and scene4 3/3 with 0
  extras; the held-out scene5 scored 1/2 type-only (honest, documented).
- `tests/test_summary.py` + `tests/test_ma_contract.py` pass (fast, no GPU).

Note: the docs intentionally contain placeholders rather than invented numbers.
Fill them only from real runs.
