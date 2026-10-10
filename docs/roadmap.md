# Roadmap

Status board for the phases in [`../AGENT_PLAN.md`](../AGENT_PLAN.md) §6 and the
Definition of Done in §12. Update this file as checkpoints pass.

| Phase | Deliverable | Checkpoint | Status |
|---|---|---|---|
| 0 | Two videos + `data/ground_truth.md` | Checkpoint 0 | ⬜ not started |
| 1 | Repo scaffold, `spatialdiff` env (cu126), weights in `~/models`, `configs/default.yaml` | Checkpoint 1: CUDA visible in a GPU job + weights load | ⬜ not started |
| 2 | `src/extract_frames.py`, `src/reconstruct.py` (joint VGGT), `src/align.py` | Checkpoint 2: A/B static structure aligns | ⬜ not started |
| 3 | `src/diff.py` with visibility check | Checkpoint 3: 3 known changes found, few false positives | ⬜ not started |
| 4 | `src/describe.py` (Qwen3-VL + DINOv2) | Checkpoint 4: every change has type, label, location, sentence | ⬜ not started |
| 5 | `src/visualize.py`, `src/report.py`, `run.py` | Checkpoint 5: one command produces all outputs | ⬜ not started |
| 6 | README with real results, ablations, public repo | Definition of done | ⬜ not started |

## Infrastructure already done

- [x] GPU access solved and documented (`gpu.md`, `gpu-run`, persistent session)
- [x] `cu126` PyTorch verified on the A100s
- [x] Documentation set (`docs/`)
- [x] Git repo initialized and connected to GitHub

## Immediate next steps

1. **Phase 0** — record the two videos and write `data/ground_truth.md`
   (or obtain sample videos).
2. **Phase 1** — create the `spatialdiff` conda env, download weights to
   `~/models/`, write `configs/default.yaml`, scaffold `src/`, `scripts/`,
   `run.py`; verify CUDA in a GPU job.
3. **Phase 2** — frame extraction + joint reconstruction.

Note: the docs intentionally contain placeholders rather than invented numbers.
Fill them only from real runs.
