# 3D Scene Change Detection

System that takes two phone recordings of the same small indoor space, builds a
shared 3D reconstruction, detects which objects were **moved**, **added**, or
**removed**, and describes each change in natural language.

Part of the **Spatial AI & Navigation Challenge**.

> **Status: first successful end-to-end run (2026-10-10).** The three planned
> changes in the demo scene were detected and described. Full numbers, timings
> and honest limitations: [`docs/results.md`](docs/results.md). The plan and
> checkpoints live in [`AGENT_PLAN.md`](AGENT_PLAN.md).

## Quick start

```bash
git clone git@github.com:Gunjan00001/Humanoid.git && cd Humanoid
conda create -n spatialdiff python=3.10 -y && conda activate spatialdiff
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt          # model weights: docs/setup.md
python detect.py --a data/a.mp4 --b data/b.mp4 --name demo
```

`detect.py` validates both videos and runs the whole pipeline (on a login node
it submits and polls the PBS gpu job itself), writing `outputs/demo/`:
`report.md`, `summary.*`, `change.gif`, `viewer.html`, `scene_mesh.ply`.
A real run printed:

```text
Compared a.mp4 and b.mp4. I found 3 change(s):
1. Removed - comb: The comb was removed from the table in video B.
2. Added - pen and device: Possibly a blue pen and a white device were added to the surface in video B.
3. Replaced - remote control: The remote control was replaced with a different one in video B.

Notes: Object names come from a vision-language model and can be wrong. Detection thresholds were tuned on the development scenes (demo, scene2, scene3, scene4). No absolute distances are reported (the reconstruction scale is not metric-validated); directions refer to the before/after videos as stated.
Files: viewer.html, change.gif, report.json
```

## Results

On the demo scene (two 8–12 s phone recordings of a table; ground truth: comb
**removed**, pen **added**, AC remote **moved and flipped**), the pipeline
detected and described all three changes with **no false positives** in the
final report:

![3D change visualization](outputs/change.gif)

| # | Ground truth | Detected | Description (Qwen3-VL) | Confidence |
|---|---|---|---|---|
| 1 | comb removed | `removed` | "The comb was removed from the table." | 0.87 |
| 2 | pen added | `added` | "A blue pen and a white device were added to the surface." | 0.38 |
| 3 | remote moved + flipped | `replaced` | "The remote control was replaced with a different one." | 0.84 |

- Visualization: `outputs/change.gif` (rotating colored cloud — red = removed,
  green = added, magenta = replaced) and one framed screenshot per change in
  `outputs/changes/`.
- Reports: `outputs/report.json`, `outputs/report.md`.
- The change detector proposed 26 candidates; 22 were rejected with documented
  reasons (`cache/semantics/rejected.json`) — mostly lighting/blur differences
  on static objects. Nothing was silently dropped.

Known limitations of this run (details in [`docs/results.md`](docs/results.md)):
the remote's in-place flip is reported as `replaced` rather than `moved`; the
pen's label includes the neighbouring timer; one-video-only areas can only
produce *unverified* candidates.

### Backend comparison (v0.5)

Two more reconstruction backends were added and compared against VGGT on the
same frames ([full numbers](docs/results.md#backend-comparison-vggt-vs-mapanything-vs-depth-anything-3)):

- **VGGT + `tabletop` remains the best single configuration** (3/3 changes,
  0 false positives).
- **MapAnything** (Apache-2.0) matches it with its threshold calibrated
  (3/3, 0 FP) and brings **metric scale by construction** plus a geometric
  diff that sees the 1 cm-thick comb and pen (VGGT's depth maps cannot).
- **Depth Anything 3** (DA3-LARGE-1.1, Apache-2.0, stretch goal, via
  MapAnything's unified interface) is the **fastest and lightest** backend
  (0.10 s/frame, 6.2 GiB), aligns best, and shows the strongest thin-object
  response; its appearance scores need calibration and it kept one timer false
  positive in tabletop mode.
- Backend choice is `reconstruction.backend: vggt | mapanything` (DA3 via
  `--model-key da3_apache`); the default stays VGGT + `tabletop`.

### Results on the development scenes (v0.10.0)

**These are development scenes, not a held-out test.** The pipeline's defaults
and the per-scene thresholds (scene2, scene3) were tuned on these same
recordings, so the numbers below measure fit on development data. All five
runs use the same pipeline; full details in
[`docs/results.md`](docs/results.md#results-across-all-scenes):

| Scene | Ground truth | Detected | False positives | Mode / tuning |
|---|---|---|---|---|
| **demo** (`data/`) | comb removed, pen added, remote moved+flipped | **3/3** | **0** | `tabletop` (default) |
| **scene2** (`data/scene2/`) | coconut removed | **1/1** (coconut ✓) — the committed report; a full v0.12 re-run produces 2 entries (coconut correctly named + the apple false positive), see the reproducibility note | 4 extra entries | `geometric` (big object; lighting-immune) + `--blur_threshold 25` |
| **scene3** (`data/scene3/`) | chair removed, apple→cup | **2/2** objects, types now correct (chair `removed`, apple→cup `replaced` — v0.9.0) | 2 extra entries + unreliable naming (see results.md) | `tabletop` + `--min_change_score 85` |
| **scene4** (`data/scene4/`) | apple moved, scissor removed, bottle added | **3/3** (`moved` apple ✓, `removed` scissors ✓, `added` bottle ✓) | **0 extra entries** | `tabletop` (default) |

**scene2** — coconut removed (`geometric` mode; the visibility check confirmed
it with 5896 points seen empty). The `tabletop` mode misses it because video
B's lighting differs strongly; the geometric mode adds 4 coverage-difference
entries (its known weakness with limited A coverage).

![scene2 result](outputs/scene2_geo/change.gif)

**scene3** — both changes found: the chair is now correctly typed `removed`
(empty-space evidence 22%) and the apple→cup region `replaced` (occupied 2%);
2 extra entries and an unreliable replacement sentence remain (video B's crop
of the table is partly occluded by the remaining chair) — see
[the correctness pass](docs/results.md#presentation--correctness-pass-v090).

![scene3 result](outputs/scene3/change.gif)

**scene4** — all three ground-truth changes, no tuning and no extras: the
apple is reported as **`moved`** (from → to shown in the report and drawn as a
dashed line in the 3D view), the scissors as **`removed`**, and the bottle as
**`added`** at the apple's old spot. This scene drove the v0.10.0 semantics
work: move matching now runs before the structural filters, objects arriving
at a moved object's origin are reported, and "replaced" claims whose two sides
are independently confirmed changes are dropped as composites — see
[scene4 in results](docs/results.md#scene4).

![scene4 result](outputs/scene4/change.gif)

Honest summary, **on these development scenes**: 9/9 ground-truth changes
found overall (demo 3/3, scene2 1/1, scene3 2/2, scene4 3/3). The demo and
scene4 reach their totals with 0 extra entries using the default mode; scene2
and scene3 each needed a short per-scene tuning pass and both carry documented
extra entries. That is the expected behaviour of a scene-tuned photometric
pipeline — and because the thresholds were tuned on these same recordings, the
9/9 should not be read as a held-out benchmark result.

**Faithful 3D rendering (v0.8.0)** — each scene now ships a dark-background
render with per-change bounding boxes + labels and a before | after | 3D
layout, a colored **TSDF mesh** (`scene_mesh.ply`, ≤ 10 MB) and a
**self-contained interactive viewer** (`viewer.html` — open it in any browser:
orbit, zoom, toggle video A / B / changes / unverified / full scene).
Diagnosis and the `input_mode=pad` experiment:
[visualization rework](docs/results.md#visualization-rework-steps-05).

### Held-out result (scene5, v0.12)

Scene5 (`data/scene5/`: clock moved, chair removed; never used for tuning) was
run **exactly once** with the frozen code and default settings
(`python detect.py --a data/scene5/a.mp4 --b data/scene5/b.mp4 --name scene5`).
Honest result, confirmed by overlay review:

| GT change | detected | correct? | confidence |
|---|---|---|---|
| clock **moved** | — (no `moved` entry) | **missed** | — |
| chair **removed** | `removed`, 3 fragments (labels "wooden leg" / "chair leg" / "table leg") | **type only**; labels wrong | 0.74 / 0.29 / 0.45 |

**Scoring (pre-registered rule): found 1/2 (type only) · labels 0/2 · missed
1 · extras ≥ 2.** The runner flagged a mixed-orientation recording (A
landscape 860x480, B portrait 480x860) and the auto mode chose `geometric`
(dominant plane 11% of confident points); both are *likely* contributors (not
proven). The verbatim summary, runtimes, the failure analysis, a post-hoc
tabletop diagnostic and the checking sheet are in
[`docs/heldout.md`](docs/heldout.md).

### Known limitations

- **mixed orientation / resolution between the two recordings** — detect.py
  warns loudly and recommends recording both videos in the same orientation
  (landscape preferred); scene5's A/B mismatch likely weakened the
  reconstruction and the crops;
- **floor reflections and lighting** produce candidate fragments — scene5's
  "small round object" was a floor reflection in video B;
- **large objects fragment** — scene5's chair removal was reported as three
  separate "leg" entries instead of one chair;
- **object names are unreliable** (VLM-generated): scene5's chair fragments,
  scene3's "black trash can" for the chair, scene4's scissor sentence before
  the summary-stage fix.

**Runtime (measured, not estimated)** — full numbers and clock bases in
[`docs/timing.md`](docs/timing.md):

| Case | Wall clock | Conditions |
|---|---|---|
| One new scene, full pipeline + 3D map | **5–6 min on the node** | 2 A100s, model files warm, 3 concurrent 1-GPU jobs from the same account running; excludes queue wait |
| Queue wait | 0–1:33 in the measured runs | caused by the queue's **per-user 16-CPU quota**, not by GPUs; 0 s once a slot was free |
| Semantics re-run (`describe` + `report`), VLM cache warm | **0:33–2:12** | no model load; fresh VLM calls add ~11 s each |
| Re-render only (composites + Chrome GIF + mesh, no GPU) | **2:09–5:46 per scene** (measured: demo 3:11, scene2 3:24, scene2-geo 2:09, scene3 5:46, scene4 3:21) | login node, Chrome available; scene3's time is dominated by its 31 change candidates (27 unverified composites) and a denser cloud in the Chrome GIF |
| Model loads | VGGT 8–14 s · Qwen3-VL-32B 26 s (warm) to 2:07 (cold) | measured across runs |

Re-render only, without touching any pipeline stage:

```bash
python src/visualize.py --config configs/default.yaml \
    --cache cache/scene4 --out outputs/scene4 --render-only
```

It reuses the cached reconstruction/diff/VLM outputs, needs no GPU, and stops
with a list of missing files if anything required is absent — it never
recomputes a stage.

An earlier claim of "~15 min per scene" was not reproducible from the logs and
has been corrected in [`docs/results.md`](docs/results.md#timings-wall-clock-this-run).

**Live interactive viewers** (GitHub Pages — enable once, see
[`docs/hosting.md`](docs/hosting.md)):
[demo](https://gunjan00001.github.io/Humanoid/outputs/viewer.html) ·
[scene2](https://gunjan00001.github.io/Humanoid/outputs/scene2/viewer.html) ·
[scene2 geometric](https://gunjan00001.github.io/Humanoid/outputs/scene2_geo/viewer.html) ·
[scene3](https://gunjan00001.github.io/Humanoid/outputs/scene3/viewer.html) ·
[scene4](https://gunjan00001.github.io/Humanoid/outputs/scene4/viewer.html)

## Pipeline

```
Video A ─┐                                         ┌─> colored 3D PLY
         ├─> frames ─> VGGT (joint) ─> point cloud ┤
Video B ─┘            shared frame       A / B tags └─> diff + visibility check
                                                         │
                                         clusters ─> crops ─> Qwen3-VL ─> labels + sentences
                                                         │
                                         DINOv2 matching ─> moved / added / removed
                                                         │
                                                         └─> JSON + Markdown report
```

Stage-by-stage detail: [`docs/pipeline.md`](docs/pipeline.md).

## Models

| Role | Primary | Fallback |
|---|---|---|
| 3D reconstruction (poses, depth, point cloud) | VGGT | MapAnything → MASt3R → COLMAP/hloc |
| Labels and descriptions | Qwen3-VL-32B-Instruct | Qwen3-VL-8B, Gemma (vision) |
| Cross-scene object matching ("moved") | DINOv2 | SigLIP 2 |
| Object masks (stretch goal) | SAM 2 | DBSCAN clusters only |

Two reconstruction backends are implemented and write the same artifact
contract, so the rest of the pipeline is backend-agnostic
(`reconstruction.backend: vggt | mapanything`). The measured comparison
(alignment, metric scale, thin-object visibility, end-to-end results) is in
[`docs/results.md`](docs/results.md#backend-comparison-vggt-vs-mapanything).

Model licenses — **verified 2026-10-10**: Qwen3-VL, DINOv2, SigLIP 2, SAM 2 and
the MapAnything checkpoint we use are Apache-2.0; **VGGT is CC-BY-NC-4.0
(non-commercial)**; MASt3R's repo is gated.
See [`docs/models.md`](docs/models.md) for IDs, commits, and the checklist.

## Setup

- **GPU cluster (PBS):** [`docs/setup.md`](docs/setup.md) and [`gpu.md`](gpu.md)
- **Laptop (CPU-only stages):** [`docs/setup.md`](docs/setup.md)

Cluster essentials: PyTorch wheels for **CUDA 12.6 (`cu126`)**, and
**`unset CUDA_VISIBLE_DEVICES`** before any CUDA work (PBS sets it empty, which
hides all GPUs). Per-job limits: 16 CPUs, 64 GB RAM, 24 h, 5 jobs.

## How to run

```bash
python run.py --video_a data/a.mp4 --video_b data/b.mp4 --out outputs/
```

`detect.py` is the one-command runner (validation, automatic diff-mode
choice, GPU/PBS handling, natural-language summary); `run.py` remains the
low-level orchestrator. Every stage also runs alone and reads/writes only the
cache:

```bash
python src/extract_frames.py --config configs/default.yaml
python src/reconstruct.py    --config configs/default.yaml   # GPU
python src/diff.py           --config configs/default.yaml
python src/describe.py       --config configs/default.yaml   # GPU
python src/visualize.py      --config configs/default.yaml
python src/report.py         --config configs/default.yaml
```

Status: end-to-end on five recordings; the demo, scene3 and scene4 reports
reproduce identically under a full `detect.py --force` re-run, scene2 does not
(see [`docs/heldout.md`](docs/heldout.md) for the held-out protocol and
[`docs/devlog.md`](docs/devlog.md) for the reproducibility note). Cluster
templates live in `scripts/`.

## Design decisions

1. **Joint reconstruction** — both videos are reconstructed in a single VGGT
   run so every frame lands in one coordinate frame, avoiding fragile
   cross-run registration.
2. **Occlusion-aware diff** — a region counts as *removed* only if the other
   video actually observed that space and saw it empty (visibility check using
   cached poses and depth). This suppresses false positives from occlusion.
3. **DINOv2 move matching** — a removal and an addition with similar crop
   embeddings but different 3D positions are reported as one *moved* object
   instead of two separate changes.

## Limitations and failure cases

Observed on the demo scene (quantified in [`docs/results.md`](docs/results.md)):

- objects thinner than VGGT's patch scale (~1.4 cm here) are **absent from the
  depth maps** — geometric diffing cannot see them; the photometric tabletop
  mode is required;
- static objects whose appearance differs between recordings (blur, exposure,
  viewing angle) generate candidate fragments that the VLM and structural
  filters must reject;
- a replacement at the same spot (comb → pen) is geometrically
  indistinguishable from "nothing changed"; appearance comparison is what
  separates them;
- areas one video never observed (portrait crop, occlusion) can only yield
  *unverified* candidates, reported separately and never counted as changes;
- reflective surfaces, textureless walls and scale ambiguity remain expected
  failure modes on other scenes (not all reproduced here).

## Documentation

| File | Contents |
|---|---|
| [`docs/handoff.md`](docs/handoff.md) | **Next-agent brief: state, next steps, environment, pitfalls** |
| [`docs/heldout.md`](docs/heldout.md) | Held-out protocol + scene5's honest result and post-hoc runs |
| [`AGENT_PLAN.md`](AGENT_PLAN.md) | Plan, phases, checkpoints (source of truth) |
| [`docs/pipeline.md`](docs/pipeline.md) | Stage-by-stage pipeline and inputs/outputs |
| [`docs/verification.md`](docs/verification.md) | Every verification rule with the demo's real numbers |
| [`docs/debugging.md`](docs/debugging.md) | Cache map, inspector tool, per-symptom playbooks |
| [`docs/setup.md`](docs/setup.md) | Environment setup and run instructions |
| [`docs/models.md`](docs/models.md) | Model stack, VRAM plan, license checklist |
| [`docs/data.md`](docs/data.md) | Recording guide and ground-truth template |
| [`docs/architecture.md`](docs/architecture.md) | Repo layout, modules, config reference |
| [`docs/results.md`](docs/results.md) | Real results of the first end-to-end run |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Cluster and model failure modes |
| [`docs/roadmap.md`](docs/roadmap.md) | Phase status checklist + per-version history |
| [`docs/timing.md`](docs/timing.md) | Measured runtimes (queue vs job, cold vs warm) |
| [`docs/hosting.md`](docs/hosting.md) | GitHub Pages for the interactive viewers |
| [`gpu.md`](gpu.md) | Cluster / GPU access reference |
| [`rules.md`](rules.md) | Working rules for agents |

## License

MIT — see [`LICENSE`](LICENSE). Model checkpoints keep their own licenses.
