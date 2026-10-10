# Roadmap

Status board for the phases in [`../AGENT_PLAN.md`](../AGENT_PLAN.md) §6 and the
Definition of Done in §12. Update this file as checkpoints pass.

**Checkpoint discipline:** a checkpoint is only ✅ when it passes on real data.
Code that exists but has not been validated end-to-end is marked 🔶.

| Phase | Deliverable | Checkpoint | Status |
|---|---|---|---|
| 0 | Two videos + `data/ground_truth.md` | Checkpoint 0 | ⬜ waiting on recordings |
| 1 | Repo scaffold, `spatialdiff` env (cu126), weights in `~/models`, `configs/default.yaml` | Checkpoint 1: CUDA visible in a GPU job + weights load | 🔶 code + scripts done; env/weights installing; GPU verification pending |
| 2 | `src/extract_frames.py`, `src/reconstruct.py` (joint VGGT), `src/align.py` | Checkpoint 2: A/B static structure aligns | 🔶 extract_frames verified (synthetic); reconstruct/align written, GPU validation pending |
| 3 | `src/diff.py` with visibility check | Checkpoint 3: 3 known changes found, few false positives | 🔶 verified on a synthetic reconstruction (2/2 objects, boundary false positives correctly rejected); real-data tuning pending |
| 4 | `src/describe.py` (Qwen3-VL + DINOv2) | Checkpoint 4: every change has type, label, location, sentence | 🔶 written; GPU validation pending |
| 5 | `src/visualize.py`, `src/report.py`, `run.py` | Checkpoint 5: one command produces all outputs | 🔶 verified end-to-end on the synthetic cache; real-data run pending |
| 6 | README with real results, ablations, public repo | Definition of done | ⬜ |

## Infrastructure already done

- [x] GPU access solved and documented (`gpu.md`, `gpu-run`, persistent session)
- [x] `cu126` PyTorch verified on the A100s
- [x] Documentation set (`docs/`)
- [x] Git repo initialized and connected to GitHub
- [x] Full pipeline code + config + PBS templates + one-command `run.py`
- [x] Model weights download script; licenses verified (`docs/models.md`)
- [x] Synthetic smoke tests for the CPU stages (`tests/`, never claimed as results)

## Immediate next steps

1. **Finish Phase 1** — let `scripts/setup_env.sh` + `scripts/download_models.sh`
   finish; verify CUDA + VGGT import inside a GPU session.
2. **Phase 0 (user)** — record `data/a.mp4`, `data/b.mp4` + `data/ground_truth.md`.
3. **Phase 2** — run joint reconstruction on the real videos; check A/B alignment
   (Checkpoint 2), tune `diff.*` thresholds on the real scene (Checkpoint 3).
4. **Phase 4** — GPU validation of `src/describe.py` on the real clusters.

Note: the docs intentionally contain placeholders rather than invented numbers.
Fill them only from real runs.
