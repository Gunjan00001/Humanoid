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

## Backend comparison: VGGT vs MapAnything vs Depth Anything 3

Added 2026-10-10 (`v0.5`): two additional backends on the **same 49 frames** of
the demo scene - **MapAnything** (Apache-2.0 checkpoint
`facebook/map-anything-apache`, `reconstruction.backend: mapanything`) and the
stretch goal **Depth Anything 3** (DA3-LARGE-1.1, Apache-2.0, through
MapAnything's unified external-model interface, `--model-key da3_apache`).
All numbers below are measured; raw logs and the machine-readable summary are
in `logs/` (`ma_full.log`, `pbs_da3_full.log`, `describe_ma.log`,
`pbs_da3_describe.log`, `backend_comparison.json`).

### Reconstruction cost

| | VGGT (CC-BY-NC) | MapAnything (Apache-2.0) | DA3-LARGE-1.1 (Apache-2.0) |
|---|---|---|---|
| Model input per frame | 518×518 (centre-cropped) | 518×294 (full portrait frame) | 518×294 |
| Model load (network FS) | ~3.0 min | ~4.3 min | **~0.7 min** |
| Inference, 49 frames | 17.2 s (0.35 s/frame) | 20.0 s (0.41 s/frame) | **4.7 s (0.10 s/frame)** |
| Peak GPU memory | 11.3 GiB | 20.1 GiB | **6.2 GiB** |
| Confidence working point | p50 | **p25** (floor-level for video B) | p50 (healthy for both videos) |
| Colours | correct | `img_no_norm` absent → rebuilt from JPEGs | correct |

### Static-structure alignment (A↔B, 6 mm voxels)

| | VGGT | MapAnything | DA3-LARGE |
|---|---|---|---|
| A→B NN p50 / p75 / p90 | 4.1 / 7.9 / 16.3 mm | 4.1 / 6.5 / 23.5 mm | **4.1 / 7.2 / 11.8 mm** |
| ICP static mean NN | 15.2 → 10.7 mm | 17.9 → 10.6 mm | 15.1 → **10.2 mm** |

### Thin objects: are the comb and pen visible in the depth maps?

Height of each backend's own detected clusters above a *local* table plane
(ring fit around the cluster). A 1 cm object should read ≈ +10 mm if resolved.

| Backend | object cluster | height (median / p90) |
|---|---|---|
| VGGT | comb | −5.5 / **+0.6 mm** (absorbed) |
| VGGT | pen | +0.6 / +8.8 mm |
| MapAnything | comb | −6.6 / +7.2 mm (partial) |
| MapAnything | pen | −1.5 / +14.0 mm (partial) |
| DA3-LARGE | cluster at (0.03, −0.15) | **+10.2 / +19.9 mm (resolved)** |
| DA3-LARGE | cluster at (0.18, −0.03) | −1.8 / **+19.6 mm** |

DA3's depth shows the strongest object response of the three - a median of
+10.2 mm is exactly a 1 cm object; VGGT's geometry cannot see them at all.

### Metric scale

- MapAnything applies metric scaling (factor ≈ 0.847 on this scene); its world
  frame is room-scale (z ∈ [0, 2.3 m]).
- Measured comb length: **VGGT 160.5 mm / MapAnything 161.3 mm / DA3 161.3 mm**
  (0.5 % agreement; plausibility-consistent with a ~16 cm pocket comb - not
  independently verified with a ruler, stated as such).

### End-to-end results (ground truth: comb removed, pen added, remote moved+flipped)

| Backend | diff.method | Detected | False positives | Notes |
|---|---|---|---|---|
| VGGT | `geometric` | 0/3 | 2 (timer ring, table blob) | thin objects absent from the depth maps |
| VGGT | `tabletop` | **3/3** | **0** | the production default |
| MapAnything | `geometric` | 2/3 (comb, pen) | 0 | remote missed; thin objects found by geometry alone |
| MapAnything | `tabletop` | 2/3 (comb, remote) | 0 | pen VLM-confirmed but scored 62.7 < the VGGT-calibrated floor 80 |
| MapAnything | `tabletop`, floor recalibrated to 60 | **3/3** | **0** | same method, per-backend threshold |
| DA3-LARGE | `geometric` | 1/3 (comb fragment) | 1 (timer) | fastest backend; thin objects visible in its depth |
| DA3-LARGE | `tabletop` | 2/3 (pen, remote) | 1 (timer) | comb fragments fell below the VGGT-calibrated score floor; one timer false positive |

### Verdict: did MapAnything / DA3 help?

- **VGGT + `tabletop` remains the best single configuration** (3/3, 0 false
  positives) and needs no extra env.
- **MapAnything** matches that end-to-end result with its threshold calibrated
  (`3/3, 0 FP`) and brings **metric scale by construction** plus a geometric
  diff that sees the 1 cm objects. Costs: 1.8× peak memory, isolated env, a
  confidence head that needs a lower working point on the blurrier video.
- **DA3-LARGE-1.1** is the **fastest and lightest** backend (0.10 s/frame,
  6.2 GiB), aligns best (p90 11.8 mm) and shows the **strongest thin-object
  response**; its photometric scores run on a different scale (comb fragments
  fell below the VGGT-calibrated floor) and it produced one timer false
  positive in tabletop mode. As a drop-in geometry engine it is very
  attractive; its appearance-based scores need calibration.
- The data suggests the same next step for both: a **hybrid** (DA3 or
  MapAnything geometry + the tabletop appearance comparison with per-backend
  score calibration) could cover all three changes with zero false positives
  and less compute than VGGT.

### Stretch goal: Depth Anything 3 through MapAnything's unified interface

**Working** (`--model-key da3_apache`). Integration notes, each found and fixed
while bringing it up (all in `scripts/run_mapanything.py`):
- MapAnything's hydra configs are **not part of its pip package** → the source
  checkout under `third_party/` must shadow the installed package *before the
  first import* (`setup_env.sh` clones it and generates
  `configs/model/da3_apache.yaml` for the Apache-2.0 LARGE checkpoint).
- External wrappers use `forward()` (not `infer()`), expect the views already
  on the device, and return the **raw** unified format (`cam_quats`,
  `cam_trans`, `depth_along_ray`); MapAnything's own
  `postprocess_model_outputs_for_inference` derives the final fields
  (`depth_z`, `intrinsics`, `camera_poses`, `img_no_norm`, `mask`) that the
  artifact contract needs.
- The wrapper's default checkpoint (GIANT-1.1) is CC-BY-NC-4.0; the generated
  `da3_apache` config selects the Apache-2.0 LARGE-1.1 instead.

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

## Results across all scenes

Updated 2026-10-10/11 (`v0.7.0`): all uploaded video sets were run through the
full pipeline. The demo scene remains the primary result; the two extra scenes
(`data-v0.2`) needed a short per-scene tuning pass, documented here with the
exact knobs so the choices are reproducible.

| Scene | Ground truth | Mode | Tuning used | Detected | Extra entries |
|---|---|---|---|---|---|
| demo | comb removed, pen added, remote moved+flipped | `tabletop` | none (defaults) | **3/3** | 0 |
| scene2 | coconut removed | `geometric` | `--blur_threshold 25` | **1/1** (coconut, 5896 pts seen empty) | 4 (coverage artifacts) |
| scene2 | coconut removed | `tabletop` | `--objectness_margin 80` | 0/1 (missed) | 1 (apple, false positive) + 11 unverified |
| scene3 | chair removed, apple→cup | `tabletop` | `--min_change_score 85` | 2/2 objects, **types now correct** (chair `removed`, apple→cup `replaced` - v0.9.0) | 2 extra entries + unreliable naming (see the correctness pass below) |

### scene2 — why the mode matters

The two recordings differ strongly in lighting/angle, and video B covers part
of the room that A never sees. Consequences, all measured:

- **`tabletop`** (appearance): the coconut's removal merges with a broad
  table-appearance change; with `--objectness_margin 80` the merge shrinks but
  the surviving cluster sits at the apple's area (a false positive: the apple
  is present in both videos; the VLM's B-side crop missed it). The coconut is
  not reported in this mode.
- **`geometric`** (3D): the coconut is a ~15 cm object — clearly present in
  VGGT's depth — and the diff finds it as a 5896-point "removed" cluster; the
  visibility check confirms it ("5896 points seen empty" in video B). This is
  the mode's intended strength: it is immune to the lighting difference. Its
  4 extra entries are coverage-difference artifacts (regions only B saw), the
  known weakness the plan predicted for one-sided coverage.

### scene3 — clean after one threshold

With `--min_change_score 85` the pipeline reports exactly the two ground-truth
changes and no false positives. Notes: the chair's removal is typed `replaced`
(the VLM's before/after crops see the chair's absence as a change of state),
and the apple→cup replacement is labeled generically ("tabletop objects
removed") — both are type/label nuances, not detection misses.

### Pipeline fixes that came out of these runs

- `report.py` takes the input video paths from the frames index (provenance);
- `--blur_threshold`, `--objectness_margin`, `--min_change_score` overrides for
  per-scene working points;
- **single-view fallback fix**: a candidate whose other-video crop is
  unavailable used to be counted as a *confirmed* change from a label-only
  answer; it is now downgraded to the unverified list (found on scene2, where
  it had produced two bogus "confirmed" entries);
- the VLM loads lazily: re-runs with all answers cached skip the model load.

Logs: `logs/pbs_scene2b.log`, `logs/pbs_scene2dbg.log`,
`logs/pbs_scene2geo.log`, `logs/pbs_scene3t.log`; outputs in
`outputs/scene{2,2_geo,3}/`; failure/fix history in
[`devlog.md`](devlog.md#2026-10-10--robustness-runs-on-the-extra-scenes-datav02).

## Visualization rework (Steps 0-5)

The original `change.gif`s rendered the scene as dark slivers. **Step 0
diagnosis** (`tests/diagnose_projection.py`, numbers before any change):

- The **reconstruction is faithful**: reprojecting the fused cloud into the
  first camera reproduces the table and every object with true colours, with
  100% of the frame covered by points. **The problem was the rendering** —
  `static_dim 0.35` (dark cloud), 1-pixel markers, and whole-scene framing
  (the table became a sliver inside a room-sized 3D box).
- Model-input coverage: the demo and scene2 are **portrait** (480×85x) and
  VGGT's crop mode delivers **56.1%** of each frame (top/bottom discarded);
  **scene3 is landscape** (860×480) and loses nothing.

What changed (all structural, detection untouched by default):

| Step | Change | Result |
|---|---|---|
| 1 | `reconstruction.input_mode: crop \| pad` (default `crop`) | pad-mode padding masked from confidence; see the comparison below |
| 2 | `visualize.conf_percentile` 5 / `voxel_size` 8 mm / `static_dim` 0.8 | display cloud separate from the detection cutoff, true colours |
| 3 | `src/mesh.py` — colored TSDF mesh | `scene_mesh.ply`, 85–115k triangles, 3.5–4.4 MB (≤ 10 MB) per scene |
| 4 | `src/visualize.py` rewritten | dark background, distance-sized splats, framing on the changed region, bbox + label + legend per change, unverified dim cyan, **before \| after \| 3D** composites and GIF layout |
| 5 | `src/viewer.py` — self-contained HTML | `viewer.html` 1.7 MB (three.js r128 pinned from cdnjs, inline orbit controls, quantized embedded cloud, A/B/change toggles, labels); validated by screenshotting it in headless Chrome |

### Step 6 — regression and the `input_mode=pad` experiment

- **Regression (crop, the default): unchanged** — demo 3/3 changes, 0 false
  positives after the rework (`logs/pbs_demo_reg.log`).
- **`input_mode=pad` on the portrait demo: worse, with evidence** — 5 entries
  vs the crop baseline's 3: the pen is **missed**, and three extra entries
  appear ("brown object corner" at the frame edge, "marker replaced" +
  "white object added" at the highlighter's area, vague labels). Reason: pad
  mode renders the portrait content at 294 px wide inside the 518 px canvas
  (~1.76× coarser pixels), and the newly-included frame borders carry
  unreliable geometry.
- **Decision: `crop` remains the default.** The option stays available for
  scenes where the important content sits at the frame edges
  (`reconstruction.input_mode: pad`).

Logs: `logs/pbs_pad_exp.log`, `logs/pbs_demo_reg.log`; pad artifacts in
`cache/pad_run` + `outputs/pad_run` (not committed).

## Presentation + correctness pass (v0.9.0)

### Layout and 3D view (steps 1-2)

- Composites and GIF are now **two rows**: `before | after` on top (larger),
  a large 3D panel below; the legend sits **outside** the 3D panel and labels
  are drawn unclipped.
- The 3D view **crops the display geometry to 0.4 m around the changed
  region** and orbits it; the camera elevation is 40° with the **fitted
  table-plane normal as the up axis** (`table_up_rotation`), so the scene
  reads like the real tabletop instead of an arbitrary model frame.
- Bug found and fixed on the way: rotating the two bbox corners does not give
  a rotated AABB (negative extents made the viewer draw giant boxes); all
  eight corners are rotated now.

### GIF rendered from the interactive viewer (step 3)

`src/render_gif.py` screenshots `viewer.html` with **headless Chrome**
(SwiftShader; no GPU needed) at 24 orbit angles and assembles the two-row GIF,
so the GIF is literally the interactive viewer's rendering. The viewer takes
URL parameters (`?shot=1&az=..&elev=..`) for deterministic frames and embeds
the full cloud sorted by distance to the focus, so its "full scene" toggle is
a cheap draw range. A matplotlib fallback remains for machines without Chrome.

### GitHub Pages (step 4)

[`docs/hosting.md`](hosting.md) documents the one-time enablement
(Settings → Pages → `main` / root) and lists the live URLs; the README links
them. No workflow file is needed - the viewers are committed static files.

### Stale screenshots (step 5)

`visualize.py` clears `outputs/<scene>/changes/` on every run and, as a
defence in depth, deletes any `change_*.png` whose id is not in the current
change list (the ids the report references).

### Scene3 change-type logic (step 6)

Two structural changes to `describe.py`:

1. **Occlusion-aware crop framing** - a point counts as in-view when it is
   not *behind* the observed surface (previously: only when the object was
   present, which made a removed object's frames look empty and let crops
   land on neighbouring objects).
2. **Geometric type correction** - the pipeline's own principle decides the
   type: run the empty-space test on every `removed`/`replaced` cluster.
   Calibration measured on the scenes: comb 0.73 empty vs apple/cup 0.01
   (occupied) → `removed` → `replaced`; remote 0.00 (occupied) vs chair 0.21
   (empty) → `replaced` → `removed`. Corrected clusters get a fresh sentence
   from the VLM naming the actual event.

Results after the pass (see the per-scene table above for the full picture):

- **demo: 3/3, 0 extra - unchanged** (verified after the rework; the
  reclassified timer/highlighter fragments were removed by the post-filters).
- **scene3** (with its tuned floor 85): **4 entries vs 2 ground-truth
  changes** - the chair is now correctly **`removed`** (was `replaced`; the
  empty-space evidence says 22%) and the apple→cup region is a
  **`replacement`** (occupied, 2%), but the apple region also appears as a
  second `removed` entry ("red apple"), and a "green and yellow object" entry
  appears for a region the ground truth does not list. The chair entry's label
  reads "black trash can" (a naming error from a distant crop).

### Remaining mismatches (honest)

- **scene3's naming**: the apple/cup region's crop in video B is partly
  occluded by the *remaining chair*, so the VLM's replacement sentence can
  name the wrong object; the chair's own label came out as "black trash can".
  The **types are correct** (geometric evidence), the **naming is not
  reliable** on this scene - a recording-angle limitation, not a rule failure.
- **scene3's extra entries**: the apple region appears twice (as the
  `replaced` apple→cup entry and as a `removed` "red apple" entry); the
  deduplication rules (bbox overlap, mutual cancellation) do not catch this
  pair because their clusters sit ~0.2 m apart in the reconstruction. One more
  per-scene tuning pass on the diff thresholds would be needed.
- **scene2** keeps its documented behaviour (coconut found by the geometric
  mode; the tabletop mode's apple-area false positive remains).

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
