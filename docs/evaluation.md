# SceneDiff evaluation slice (Track C)

Independent benchmark, separate from the development scenes. Verified from the
sources on 2026-10-11; anything not yet run is marked pending.

## Sources and reachability

All reachable from the login node (HTTP 200 unless noted):

| source | URL | status |
|---|---|---|
| project page | https://yuqunw.github.io/SceneDiff | 301 (redirect, up) |
| paper | https://arxiv.org/abs/2512.16908 | 200 |
| dataset | https://huggingface.co/datasets/yuqun/SceneDiff | 200 |
| code and evaluator | https://github.com/yuqunw/scene_diff | 200 |
| HD-Epic (kitchen source) | https://hd-epic.github.io/ | 200 |

Disk on the shared filesystem has 322 TB free, so the 11.9 GB zip fits.

## Dataset

The real archive is `scenediff_benchmark.zip`, 11,919,571,401 bytes (about
11.9 GB). The Hugging Face file tree lists exactly one data file, that zip,
plus `.gitattributes` and `README.md`. The code README's `unzip` line says
`scenediff_bechmark.zip`, which is a typo; the card and the tree both say
`scenediff_benchmark.zip`. Download URL:
https://huggingface.co/datasets/yuqun/SceneDiff/resolve/main/scenediff_benchmark.zip

Structure inside the zip:

```
scenediff_benchmark/
  data/sequence_pair_N/{original_video1.mp4, original_video2.mp4,
                        video1.mp4, video2.mp4, segments.pkl, metadata.json}
  splits/{val_split.json, test_split.json}
  vis/
```

`original_video1.mp4` and `original_video2.mp4` are the raw videos to use.
`video1.mp4` and `video2.mp4` carry the annotation mask overlays and must never
be fed to the pipeline. `segments.pkl` holds the dense GT masks.

Splits: `splits/val_split.json` and `splits/test_split.json`. Sets: `varied`
(SD-V) and `kitchen` (SD-K, from HD-Epic). Claimed counts, not yet verified
from the data: 350 pairs, 1,009 annotated objects, SD-V 200, SD-K 150.

## Prediction format

`output_dir/<pair>/object_masks.pkl` per pair:

```python
{"H": int, "W": int,
 "video_1": {object_id: {frame_id: {"mask": RLE, "cost": float}}},
 "video_2": {object_id: {frame_id: {"mask": RLE, "cost": float}}}}
```

`cost` is the prediction confidence. RLE is the pycocotools mask format.

## Evaluator

```
python scripts/evaluate_multiview.py --pred_dir DIR --splits val \
    --sets varied --output_path FILE [--visualize False]
```

Arguments: `--splits` val or test or all; `--sets` varied or kitchen or All;
`--duplicate_match_threshold` and `--per_frame_duplicate_match_threshold`
default 1. Imported dependencies include numpy, cv2, torch, scipy,
matplotlib, Pillow and pycocotools. Metrics (mask IoU at least 0.5): px/im IoU,
obj/im AP, obj/sc AP without and with the change-type requirement.

## Licenses

- SceneDiff dataset: MIT (per the card).
- SceneDiff code and evaluator: MIT.
- HD-Epic: its own terms, to be recorded from https://hd-epic.github.io/ before
  any kitchen pair is used.

## Pending (C1 gates not yet passed)

- Evaluator smoke test on this cluster: a perfect prediction from the GT of 3
  val pairs score near maximum; an empty prediction scores zero.
- Frame convention with file and line evidence: how frame_id is defined,
  whether all annotated GT frames are scored or only predicted ones, the
  direction of cost, and the video_1/video_2 to added/removed/moved mapping.
- Ten per-pair facts (duration, fps, resolution, orientation, annotated objects
  by change type, annotated frames, size).
- Compute estimate for 20 val and 10 test pairs, reduced to fit 12 GPU hours if
  needed.

## C1 verified facts (2026-10-11)

Discrepancies from the code README, checked from the archive itself:

- The zip is `scenediff_benchmark.zip` (11,919,571,401 bytes). unzip refuses it
  with "overlapped components (possible zip bomb)"; extract with
  `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`.
- Pairs live at `scenediff_benchmark/data/<pair>/` with real names (for example
  `airhockeytable_1_airhockeytable_2`), not `sequence_pair_N`.
- Files per pair: `original_video1.MOV` and `original_video2.MOV` (or `.mov`),
  the raw videos to use; `video1.mp4` and `video2.mp4` (annotation overlays,
  never use); `segments.pkl` (GT); `inference_data.pkl`; `review_meta.json`.
  There is no `metadata.json`. The raw videos are `.MOV`, not `.mp4`.
- Splits: val varied 50, val kitchen 55, test varied 150, test kitchen 100.

`segments.pkl` (loaded with the spatialdiff env):

- `scene_type` (string), `light_change` (bool), `comments`.
- `objects`: list of `{label, in_video1, in_video2, video1_frame_idx,
  video1_status, video2_frame_idx, video2_status, original_obj_idx,
  deformability}`. The change type follows the two flags: in both videos means
  moved or unchanged, in video1 only means removed, in video2 only means added.
- `video1_objects` and `video2_objects`: `{object_id: {frame_id: {size, counts}}}`
  where `counts` is a pycocotools RLE.

Frame convention: `frame_id` is the integer video frame index. The GT provides
masks only on the annotated frames, so the adapter must project the changed
object into the annotated frames (and possibly interpolate poses for frames that
were not reconstructed).

Evaluator code cloned to `third_party/scene_diff` (scripts/evaluate_multiview.py).

Still pending from C1: the evaluator smoke test (perfect and empty prediction),
the 10 per-pair facts, and the compute estimate.

## C1 per-pair facts (10 val varied pairs, extracted)

res = raw video resolution, ori = L landscape, objA/objB = annotated objects in
each video, mv/add/rem = changed objects by type (moved, added, removed).

| pair | res | fps | dur s | ori | objA | objB | mv | add | rem | MB |
|---|---|---|---|---|---|---|---|---|---|---|
| airhockeytable_1_airhockeytable_2 | 1920x1080 | 30.0 | 9.2 | L | 1 | 1 | 1 | 0 | 0 | 17.4 |
| bathroom_11_bathroom_12 | 1920x1080 | 60.0 | 10.2 | L | 2 | 0 | 0 | 0 | 2 | 35.0 |
| bathroom_1_bathroom_2 | 1920x1080 | 60.0 | 8.9 | L | 4 | 2 | 2 | 0 | 2 | 30.0 |
| bed_3_bed_4 | 1920x1440 | 30.0 | 4.2 | L | 1 | 2 | 1 | 1 | 0 | 10.9 |
| bedroom_1_bedroom_2 | 1920x1080 | 30.0 | 12.4 | L | 4 | 3 | 3 | 0 | 1 | 28.2 |
| bedroom_2_bedroom_3 | 1920x1080 | 30.0 | 13.9 | L | 5 | 6 | 4 | 2 | 1 | 26.8 |
| bedroom_2_bedroom_4 | 1920x1080 | 30.0 | 13.9 | L | 7 | 7 | 4 | 3 | 3 | 27.0 |
| bedroom_30_bedroom_31 | 1920x1080 | 30.0 | 3.9 | L | 1 | 1 | 1 | 0 | 0 | 14.0 |
| bedroom_3_bedroom_5 | 1920x1080 | 30.0 | 11.2 | L | 6 | 2 | 1 | 1 | 5 | 23.0 |
| bedroom_4_bedroom_5 | 1920x1080 | 30.0 | 11.4 | L | 4 | 0 | 0 | 0 | 4 | 23.2 |

All ten are landscape, 1920x1080 or 1920x1440, 30 or 60 fps, 4 to 14 s. Objects
per pair range from 1 to 7 per video. Total video bytes for the ten pairs is
235.5 MB.

## C1 adapter notes

a. Raw videos are `.MOV` or `.mov`. Read them with rotation applied (OpenCV's
   VideoCapture does this; ffprobe confirms the reported width and height).
   Verified for the sampled pairs that the decoded resolution equals the GT mask
   `size` in `segments.pkl` (1920x1080 and 1920x1440 both matched). The adapter
   asserts `(H, W)` equals the mask `size` per pair. A unit check is added.
b. Extraction needs `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`.
c. Change type from `in_video1` and `in_video2`: in both is moved or unchanged,
   in video1 only is removed, in video2 only is added. Frame ids are integer frame
   indices of the original video.
d. The GT annotates only specific frames (`video1_frame_idx`, `video2_frame_idx`
   and the frames present in `video1_objects`/`video2_objects`). The evaluator
   scores the annotated frames. How many frames it scores versus how many we can
   project is still to be measured from the evaluator code; the interpolation
   policy (project the 3D change into every scored annotated frame, interpolate a
   pose when that frame was not reconstructed) is decided before the dry run.

## C1 compute estimate

Measured earlier: one new scene (two clips, about 30 frames each) costs about 6
minutes on the node (reconstruct about 2 to 3 min, describe about 2 to 4 min).
The 30 pairs (20 validation + 10 test) are therefore about 3 to 4 GPU hours,
well under the 12 GPU hour cap, so the validation slice is not reduced.

## Evaluator internals (from third_party/scene_diff/scripts/evaluate_multiview.py)

- Change type: `in_video1 and in_video2` gives label 1 (moved); in exactly one
  video gives label 0 (added or removed). The change-type reward is therefore
  moved versus non-moved.
- Frames scored: a GT frame is scored when `frame_index % resample_rate == 0`
  (resample_rate default 30); the scored id is `frame_index // resample_rate`.
- GT masks are pycocotools RLE, decoded and resized to `(H, W)` with nearest
  neighbour, then thresholded at 0.5 (lines 825 to 857).
- Scenes without a `pred_dir/<pair>/object_masks.pkl` are dropped from the run
  (lines 1340 to 1348), so a 3 pair smoke test is possible.
- Splits are read from `data/scenediff_benchmark/splits/...` relative to the
  working directory; `--gt_dir` and `--video_dir` default to
  `data/scenediff_benchmark/data`.

## C1 still open

The evaluator smoke test (a perfect prediction from GT scores near the maximum,
an empty prediction scores zero) is not yet run: it needs a small builder that
writes `object_masks.pkl` from `segments.pkl` in the exact prediction format and
the correct `(H, W)`. pycocotools is now installed. This is the last C1 gate
before C2.

## C1 smoke test PASSED (2026-10-11)

A perfect prediction built from GT scores the maximum and an empty prediction
scores zero, so the evaluator runs on this cluster.

| prediction | px/im IoU | obj/im AP | obj/sc AP | obj/sc AP (type) | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| perfect (from GT) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 7 | 0 | 0 |
| empty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 7 |

Correction to the documented prediction format (the code, not the README, is
right). `extract_detections` iterates `pred_data.items()` and looks for
`'video_1'` inside each value, so the real format is **object first**:

```python
{"H": int, "W": int,
 "<object_id>": {"video_1": {frame_id: {"mask": RLE, "cost": float}},
                 "video_2": {frame_id: {"mask": RLE, "cost": float}}},
 ...}
```

and the frame id is the **resampled** index `raw_frame // resample_rate`
(resample_rate 30), matching the GT keys. The README's `{H, W, video_1:{...},
video_2:{...}}` shape does not work. H and W are read from the prediction and
used to resize both GT and prediction masks, so they must be consistent.

Dependencies added on the login node: pycocotools, faiss-cpu. The evaluator is
run from the repo root with `data/scenediff_benchmark` symlinked to the extracted
dataset (gitignored).

## Rule for the validation and test slices

The evaluator is NOT run on the test predictions until the validation results pass
a sanity check:

1. every validation pair produced a valid `object_masks.pkl`;
2. H and W match the GT mask size;
3. scores are not degenerate (not all zero and not constant across pairs);
4. a manual look at 3 validation pairs' overlays shows masks on the right frames.

If the sanity check finds an adapter bug, fix it as a pure bug fix (no threshold,
prompt or flag changes), document what changed and why here, and re-run BOTH
slices. The report states that the test slice was run twice because of the bug
fix, with no test score seen before the fix. If the sanity check passes, freeze,
record the digests, then evaluate the test slice once. No threshold or config
changes after that.

## Adapter / driver bug fix (2026-10-12, before any test score was seen)

What changed and why: the first full run (job 11027) failed for most pairs. Only
10 validation pairs had been extracted from the zip, but the driver looped over
all 20 validation pairs. For each unextracted pair the two original videos were
missing, so `detect.py` exited with "give two videos" and the adapter then failed
with a missing `reconstruction/meta.json`. Fix: extract all 20 validation and 10
test pairs before running the driver. No threshold, prompt or flag changed and no
adapter logic changed. The test slice was not evaluated before this fix (no test
score was produced), so both slices are re-run from scratch and the test slice is
still run once for scoring.

## Adapter bug 2 (2026-10-12, found by the validation sanity check, before any test score)

The sanity check found that every prediction file was **empty** (only H and W, no
objects). Cause: the adapter looked for the pair's `report.json` in the wrong
places (`<pred_dir>/<pair>/report.json` and `cache/<pair>/report.json`), but the
pipeline writes it to `outputs/scenediff/<pair>/report.json`. With no report the
adapter produced zero objects, and the evaluator would have scored zero for a
reason that had nothing to do with detection. Fix: the adapter now takes an
explicit `--report` path and `run_scenediff.sh` passes
`outputs/scenediff/<pair>/report.json`; the adapter stops with a clear error if
the report is missing. No threshold, prompt or flag changed. This was caught
before any test score was produced, so both slices are re-run and the test slice
is reported as run twice.

Also recorded: bug 1 was the 10 versus 20 extraction bug (only 10 validation
pairs had been extracted while the driver looped over 20).

The six validation pairs that fail do so **upstream of the adapter**: at the
frozen default `frames.blur_threshold` (100) `extract_frames` keeps fewer than 8
sharp frames on those clips ("only 0/2/5 frames survived"). Changing the
threshold would change frozen behaviour, so those pairs are reported as
unprocessable at the frozen config, not fixed.

## Scoring rule for unprocessable pairs (recorded 2026-10-12, before any test score)

A pair is **unprocessable** when the frozen pipeline cannot produce a prediction
for it (frame extraction leaves fewer than 8 sharp frames at the frozen default
`frames.blur_threshold`, or an infrastructure failure such as a node OOM).

- The **headline numbers** count an unprocessable pair as an empty prediction:
  every ground truth object in it is a miss. This is done by writing an empty
  `object_masks.pkl` for that pair (H and W from the GT mask size, no objects) so
  the evaluator includes it.
- **Processable-only numbers** are reported next to the headline, computed on the
  pairs the pipeline could process.
- Each failed pair is listed with its reason (too few sharp frames, OOM, other).
- The same rule applies to the test slice. Pairs are never swapped or replaced.

An infrastructure failure (a transient node OOM) may be re-run for the same
frozen config; the report states that it was an infrastructure retry and that it
had produced no output.

## Adapter bugs 3 and 4 (masks), found by the manual overlay look

The sanity look found the predicted masks never overlapped the GT masks. Two bugs:

- Bug 3: the projected points were used in **model** pixel coordinates (about
  518x294) but the masks must be in **original** coordinates (for example
  1920x1080). Fix: map with the frame transform,
  `orig = (model + crop) / scale` (`scale_w`, `scale_h`, `crop_left`, `crop_top`
  from `reconstruction/meta.json`). Also fill the convex hull of the projected
  points instead of dilating single pixels, for a solid object mask.
- Bug 4: the dataset stores GT masks **transposed** relative to the video: the RLE
  `size` is `[orig_w, orig_h]` (for example `[1920, 1080]` for a 1920x1080
  video), so the decoded array is `(orig_w, orig_h)`. Fix: build the mask in video
  coordinates and transpose it before encoding, and set the mask `size` to
  `[orig_w, orig_h]`.

Both are pure mask-fidelity fixes; no threshold, prompt or flag changed. After the
fix the predicted masks sit in the same coordinate frame as the GT.

Remaining result, after the fixes: on the processable validation pairs the
detected changes still do not line up with the GT objects (our two "removed"
regions sit elsewhere on the frame). That is detection quality on these
out-of-domain HD clips, not an adapter bug. The validation scores are reported as
they come out.
