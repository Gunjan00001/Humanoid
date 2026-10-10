# tests

Development fixtures for smoke-testing the CPU stages **before real recordings
exist**. Nothing here is part of the pipeline, and nothing produced from these
scripts counts as a project result (see the honesty rule in `docs/README.md`).

| Script | Produces | Used by |
|---|---|---|
| `make_test_videos.py` | `cache/test_videos/{a,b}.mp4` — two tiny synthetic videos (blur stretch included) | `src/extract_frames.py` smoke test |
| `make_fake_reconstruction.py` | `cache/fake_run/reconstruction/*` — analytic scene with one A-only box and one B-only box | `src/diff.py`, `src/visualize.py`, `src/report.py` smoke test |

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
