# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/living_room_15_living_room_16/original_video1.mov`
- video B: `data/external/scenediff/scenediff_benchmark/data/living_room_15_living_room_16/original_video2.MOV`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 53 | semantics: vlm-verified
- generated: 2026-10-15T14:18:35Z | config `530c43b66d5f`

**5 change(s) detected**: 2 added, 3 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | colorful box | (-0.22, 0.16, 0.96) | 0.75 |
| 2 | removed | plastic object | (-0.21, 0.42, 0.74) | 0.75 |
| 3 | removed | cork coaster | (-0.01, -0.16, 0.76) | 0.50 |
| 4 | added | tissue box | (-0.01, 0.01, 0.65) | 0.75 |
| 5 | added | pineapple | (0.03, -0.20, 0.63) | 0.75 |

## Details

### 1. REMOVED - colorful box

> A colorful box appeared on the white table in the second image.

- centroid: (-0.22, 0.16, 0.96) | confidence: 0.75
- cluster ids: [2]
- visible in: a/frame_00250.jpg, b/frame_00170.jpg
- confirmed by: b/frame_00190.jpg (218 points seen empty)

### 2. REMOVED - plastic object

> The white plastic object was removed from the table.

- centroid: (-0.21, 0.42, 0.74) | confidence: 0.75
- cluster ids: [3]
- visible in: a/frame_00210.jpg, b/frame_00250.jpg
- confirmed by: b/frame_00320.jpg (1097 points seen empty)

### 3. REMOVED - cork coaster

> The cork coaster and oranges were removed and replaced with a pineapple and tissue box.

- centroid: (-0.01, -0.16, 0.76) | confidence: 0.50
- cluster ids: [4]
- visible in: a/frame_00040.jpg, b/frame_00140.jpg
- confirmed by: b/frame_00140.jpg (810 points seen empty)

### 4. ADDED - tissue box

> A tissue box was added to the table.

- centroid: (-0.01, 0.01, 0.65) | confidence: 0.75
- cluster ids: [13]
- visible in: a/frame_00050.jpg, b/frame_00060.jpg
- confirmed by: a/frame_00010.jpg (2426 points seen empty)

### 5. ADDED - pineapple

> A pineapple was added to the table.

- centroid: (0.03, -0.20, 0.63) | confidence: 0.75
- cluster ids: [14]
- visible in: a/frame_00010.jpg, b/frame_00320.jpg
- confirmed by: a/frame_00090.jpg (693 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
