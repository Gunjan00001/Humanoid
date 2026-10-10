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
| 6 | README with real results, ablations, public repo | Definition of done | 🔶 README + results.md filled with the real run; repo public check pending |

## Infrastructure already done

- [x] GPU access solved and documented (`gpu.md`, `gpu-run`, persistent session)
- [x] `cu126` PyTorch verified on the A100s
- [x] Documentation set (`docs/`)
- [x] Git repo initialized and connected to GitHub
- [x] Full pipeline code + config + PBS templates + one-command `run.py`
- [x] Model weights download script; licenses verified (`docs/models.md`)
- [x] Synthetic smoke tests for the CPU stages (`tests/`, never claimed as results)

## Immediate next steps

1. **Phase 0 (user)** — record `data/a.mp4`, `data/b.mp4` + `data/ground_truth.md`.
2. **Phase 2** — run joint reconstruction on the real videos; check A/B alignment
   (Checkpoint 2), tune `diff.*` thresholds on the real scene (Checkpoint 3).
3. **Phase 4** — GPU validation of `src/describe.py` on the real clusters.

Validation evidence so far (2026-10-10, all from real runs; smoke tests are
explicitly *not* project results):
- `spatialdiff` env: torch 2.14.1+cu126, CUDA visible in a GPU job, 8×A100.
- VGGT smoke run on 6 static kitchen frames: 544k confident points, scene bbox
  ≈ 4.4 × 1.8 × 3.6 m, inference 1.2 s/frame, **0 changes** detected (all
  noise candidates rejected by the visibility check) - `logs/gpu_smoke.log`.

Note: the docs intentionally contain placeholders rather than invented numbers.
Fill them only from real runs.
