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
