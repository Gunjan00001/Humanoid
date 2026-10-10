# tests

Development fixtures for smoke-testing the CPU stages **before real recordings
exist**. Nothing here is part of the pipeline, and nothing produced from these
scripts counts as a project result (see the honesty rule in `docs/README.md`).

| Script | Produces | Used by |
|---|---|---|
| `make_test_videos.py` | `cache/test_videos/{a,b}.mp4` — two tiny synthetic videos (blur stretch included) | `src/extract_frames.py` smoke test |
| `make_fake_reconstruction.py` | `cache/fake_run/reconstruction/*` — analytic scene with one A-only box and one B-only box | `src/diff.py`, `src/visualize.py`, `src/report.py` smoke test |
| `prepare_gpu_smoke.py` | `cache/gpu_smoke/frames/` — VGGT example kitchen frames split into pseudo A/B | `src/reconstruct.py` GPU smoke test |
| `gpu_smoke_test.sh` | Phase-1 CUDA verification + static-scene null test | `gpu-run -f tests/gpu_smoke_test.sh` |
| `gpu_describe_smoke.py` | VLM + DINOv2 model-path validation on real crops | `gpu-run 'python tests/gpu_describe_smoke.py'` |
| `test_ma_contract.py` | CPU fixture tests for the MapAnything artifact contract (transform maths, pose inversion, colour fallback) | `python tests/test_ma_contract.py` |
| `compare_backends.py` | VGGT vs MapAnything measurements (alignment, thin-object heights, scale) | `python tests/compare_backends.py` |

Quick CPU chain on the synthetic cache:

```bash
python tests/make_fake_reconstruction.py
python src/diff.py       --config configs/default.yaml --cache cache/fake_run
python src/visualize.py  --config configs/default.yaml --cache cache/fake_run --out cache/fake_run/outputs
python src/report.py     --config configs/default.yaml --cache cache/fake_run --out cache/fake_run/outputs
```

Expected: one removed cluster at (-0.4, 0.1, 1.5), one added cluster at
(0.4, -0.05, 1.6); boundary-strip false candidates are rejected by the
visibility check. The fake reconstruction has no real frames on disk, so
`src/describe.py` cannot run on it (it needs JPEGs to crop).
