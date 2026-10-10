# Results

> **Status: first successful end-to-end run — 2026-10-10.** The demo scene
> (`data/a.mp4`, `data/b.mp4`, ground truth: comb removed, pen added, AC remote
> moved/flipped) was processed with the full pipeline. All numbers below come
> from that run; nothing is estimated.

## Detection vs ground truth

| # | Ground truth | Detected type | Label (Qwen3-VL) | Correct? | Confidence |
|---|---|---|---|---|---|
| 1 | moved: AC remote (flipped) | `replaced` | remote control | ✅ (see note) | 0.84 |
| 2 | removed: comb | `removed` | comb | ✅ | 0.87 |
| 3 | added: pen | `added` | pen and device | ✅ | 0.38 |

All three planned changes were found; **no false positives in the final report**
(the change detector proposed 26 candidates, 22 were rejected with documented
reasons in `cache/semantics/rejected.json` — mostly lighting/blur differences on
static objects, see limitations).

Notes:
- The remote **stayed in place and flipped over** (black back up in A, white
  face up in B). The pipeline classifies this as `replaced` (state change)
  rather than `moved`; both readings match the ground truth note.
- The pen's label includes the neighbouring timer ("pen and device") because
  the verification crop is wider than the pen itself.
- The pen sits at the edge of video A's coverage (portrait center-crop), which
  is why its confidence (0.38) is lower than the comb's (0.87).

## Outputs of the run

- `outputs/change.gif` — rotating view of the colored cloud
  (red = removed, green = added, orange = moved, magenta = replaced).
- `outputs/changes/change_00{1,2,3}.png` — one framed screenshot per change
  with its sentence.
- `outputs/scene.ply` — the colored point cloud (red/green/magenta changes on a
  dimmed background).
- `outputs/report.json` / `outputs/report.md` — the machine- and
  human-readable reports.

`report.md` (excerpt):

```
**3 change(s) detected**: 1 added, 1 removed, 1 replaced

| # | type     | label           | centroid (x, y, z)      | confidence |
| 1 | removed  | comb            | (-0.05, 0.13, 0.91)     | 0.87       |
| 2 | added    | pen and device  | (-0.11, 0.19, 0.91)     | 0.38       |
| 3 | replaced | remote control  | (-0.22, -0.34, 0.95)    | 0.84       |
```

## How this run was produced

```bash
# Phase 2 (GPU, 1 A100): joint VGGT over 49 frames (30 A + 19 B)
python src/reconstruct.py --config configs/default.yaml
# Phase 3 (CPU): ICP refinement + tabletop photometric diff
python src/align.py    --config configs/default.yaml
python src/diff.py     --config configs/default.yaml
# Phase 4 (GPU, 2 A100): VLM verification + labels + DINOv2 matching
python src/describe.py --config configs/default.yaml
# Phase 5 (CPU): visualization + report
python src/visualize.py --config configs/default.yaml
python src/report.py    --config configs/default.yaml
```

## Timings (wall clock, this run)

| Stage | Where | Time |
|---|---|---|
| Frame extraction | login node (CPU) | ~1 min (49 frames kept of 77 sampled) |
| Reconstruction (VGGT) | 1× A100 | ~3.5 min (model load) + **17.2 s inference** |
| ICP alignment | login node (CPU) | ~10 s |
| Tabletop diff | login node (CPU) | ~10 s |
| Semantics (Qwen3-VL-32B + DINOv2) | 2× A100 | ~15 min first run (model load + 26 VLM calls); re-runs reuse the VLM cache |
| Visualization + report | login node (CPU) | ~10 s |

## Ablations (observed, not benchmarked)

- **Geometric diff alone fails on this scene**: the comb and pen are 1 cm
  thick, below VGGT's patch scale — they are *absent from the depth maps*
  (their z equals the table's). The plan-skeleton thresholds (2 cm voxels /
  5 cm distance) found nothing; even retuned (6 mm / 12 mm) the thin objects
  stayed invisible. The photometric tabletop mode (`diff.method: tabletop`)
  detects them from appearance over the fitted table plane.
- **ICP refinement matters**: static-region mean NN distance between the two
  clouds dropped 15.2 mm → 10.7 mm, which visibly reduced false candidates.
- **VLM verification is load-bearing**: the raw photometric candidate list has
  26 clusters (mostly lighting/blur fragments of the timer and highlighter);
  the Qwen3-VL confirmation + structural filters reduce it to the 3 real
  changes. Rejections are logged, not silently dropped.

## Limitations and failure cases (as observed on this scene)

- Static objects whose appearance differs between recordings (blur, exposure,
  viewing angle) generate candidate fragments; they are rejected by the VLM
  and the structural filters, but a few survived intermediate stages before
  the final rules (`rejected.json` documents each one).
- The pipeline cannot distinguish a *replacement at the same spot* from a
  state change of the same object purely geometrically — the pen was placed
  where the comb had been; the detection worked because the pen lies within
  video A's observed area and its appearance differs strongly.
- Areas one video never observed (portrait crop, occlusion) can only yield
  *unverified* candidates; they are reported separately and never counted as
  changes.
- "Moved" (DINOv2 matching) did not trigger on this scene: the remote flipped
  in place, which the `replaced` category captures instead.
