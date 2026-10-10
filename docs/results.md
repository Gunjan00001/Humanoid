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

## Backend comparison: VGGT vs MapAnything

Added 2026-10-10 (`v0.5`): MapAnything (Apache-2.0 checkpoint
`facebook/map-anything-apache`) as a second backend
(`reconstruction.backend: mapanything`) on the **same 49 frames** of the demo
scene. All numbers below are measured; raw logs and the machine-readable
summary are in `logs/` (`ma_full.log`, `describe_ma.log`, `pbs_*.log`,
`backend_comparison.json`).

### Reconstruction cost

| | VGGT (CC-BY-NC) | MapAnything (Apache-2.0) |
|---|---|---|
| Model input per frame | 518×518 (centre-cropped) | 518×294 (full portrait frame) |
| Model load (network FS) | ~3.0 min | ~4.3 min |
| Inference, 49 frames | **17.2 s** (0.35 s/frame) | 20.0 s (0.41 s/frame) |
| Peak GPU memory | **11.3 GiB** | 20.1 GiB |
| Confidence working point | p50 (`p50` ≈ 1.01) | **p25**: MA's confidence head sits at its floor (≤1.0) for all of the blurrier video B (mean 0.97); p50 discards 99% of B |
| Colours | correct from the model | `img_no_norm` absent → rebuilt from the JPEGs (`scripts/rebuild_colors.py`); the worker now always crop-resizes the originals as fallback |

### Static-structure alignment (A↔B, 6 mm voxels)

| | VGGT | MapAnything |
|---|---|---|
| A→B NN p50 / p75 / p90 | 4.1 / 7.9 / 16.3 mm | 4.1 / 6.5 / 23.5 mm |
| ICP static mean NN (p25 working point) | 15.2 → 10.7 mm | 17.9 → 10.6 mm |
| ICP static mean NN at p50 (MA B-sparse) | — | 17.5 → 4.4 mm |

After ICP both backends align their two videos equally well (~10.6 vs
10.7 mm). At MA's *clean* p50 subsets the residual drops to 4.4 mm, which
suggests MA's geometry is more rigidly consistent (VGGT's residual is
dominated by local surface warp), but that subset only contains 1% of B.

### Thin objects: are the comb and pen visible in the depth maps?

Height of each backend's own detected clusters above a *local* table plane
(ring fit around the cluster, so surface warp is not mistaken for object
height). A 1 cm object should read ≈ +10 mm if resolved.

| Cluster | VGGT | MapAnything |
|---|---|---|
| comb (removed) | median −5.5 mm, p90 **+0.6 mm** (absorbed) | median −6.6 mm, p90 **+7.2 mm** (partial) |
| comb handle (removed) | — | median −5.8 mm, p90 +10.7 mm |
| pen (added) | median +0.6 mm, p90 **+8.8 mm** | median −1.5 mm, p90 **+14.0 mm** (partial) |

Neither backend cleanly resolves the 1 cm objects, but MA shows a clearly
stronger positive tail — and its *geometric* diff actually detects the comb
and pen (see below), which VGGT's geometric diff cannot.

### Metric scale

- MapAnything applies metric scaling (factor ≈ 0.847 on this scene) and its
  world frame is room-scale (z ∈ [0, 2.3 m]); VGGT's frame is camera-anchored
  and only approximately metric.
- Measured comb length: **VGGT 160.5 mm vs MapAnything 161.3 mm** (0.5 %
  agreement; plausibility-consistent with a ~16 cm pocket comb — not
  independently verified with a ruler, stated as such).

### End-to-end results (ground truth: comb removed, pen added, remote moved+flipped)

| Backend | diff.method | Detected | False positives | Notes |
|---|---|---|---|---|
| VGGT | `geometric` | 0/3 | 2 (timer ring, table blob) | thin objects absent from the depth maps |
| VGGT | `tabletop` | **3/3** | **0** | the production default (see above) |
| MapAnything | `geometric` | 2/3 (comb, pen) | 0 | **remote missed**; thin objects found by geometry alone |
| MapAnything | `tabletop` | 2/3 (comb, remote) | 0 | pen VLM-confirmed but scored 62.7 < the VGGT-calibrated floor 80 |
| MapAnything | `tabletop`, floor recalibrated to 60 | **3/3** | **0** | same method, per-backend threshold |

### Verdict: did MapAnything help?

- **VGGT + `tabletop` remains the best single configuration** (3/3, 0 false
  positives, fastest, half the GPU memory, no extra env).
- MapAnything does **not** beat it overall on this scene, but it brings three
  real strengths: **metric scale by construction** (Apache-2.0 too), a
  **geometric diff that sees 1 cm objects** (its depth partially resolves them;
  VGGT's cannot), and equal post-ICP alignment.
- Its costs: 1.8× peak GPU memory, a confidence head that needs a different
  working point (p25) on the blurrier video, missing denormalised colours, and
  photometric scores on a different scale (needs its own `min_change_score`).
- The clear next step suggested by the data: a **hybrid** (MapAnything geometry
  + the appearance-based tabletop comparison) could combine both strengths —
  MA's geometric run already covers what VGGT's misses, and vice versa.

### Stretch goal: Depth Anything 3 through MapAnything's unified interface

Attempted. `--model-key da3` is implemented (external wrappers via
`init_model_from_config` + `model(views)`); MapAnything's hydra configs are not
part of its pip package, so `scripts/setup_env.sh` clones the source into
`third_party/` (gitignored) and the worker prepends it before the first
import. The DA3 smoke job is in `logs/pbs_da3_smoke.log`; the last attempt's
result is recorded there. This is a stretch item and is **not** part of the
comparison table above.

### Pipeline robustness fixes that came out of this comparison

All structural, not scene-specific (VGGT demo regression: identical clusters):
- tabletop mode: one **joint plane** refined on the dense surface band (MA's
  room-scale cloud made per-video fits disagree by 10°), plus a ±15 cm
  **height band** (walls share the x/y footprint in room-scale clouds);
- `--conf_percentile`, `--method`, `--backend`, `--min_change_score` CLI
  overrides for per-backend working points;
- `scripts/rebuild_colors.py` (rebuild `colors.npz` from the JPEGs, no GPU);
- the score floor no longer rejects geometric clusters (they carry visibility
  evidence, not a photometric score);
- fragment merging covers `replaced` clusters and geometric clusters.

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
