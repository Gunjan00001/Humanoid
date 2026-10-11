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
