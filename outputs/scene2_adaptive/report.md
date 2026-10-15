# Change report

- video A: `data/scene2/a.mp4`
- video B: `data/scene2/b.mp4`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 35 | semantics: vlm-verified
- generated: 2026-10-15T13:34:20Z | config `798e1549834b`

**5 change(s) detected**: 2 added, 3 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | coconut | (-0.09, -0.33, 1.08) | 0.75 |
| 2 | removed | apple | (0.19, -0.06, 0.99) | 0.50 |
| 3 | removed | small dark speck | (0.40, 0.12, 0.90) | 0.75 |
| 4 | added | round object | (-0.09, -0.25, 1.09) | 0.36 |
| 5 | added | finger | (0.28, 0.06, 0.92) | 0.33 |

## Details

### 1. REMOVED - coconut

> The coconut was removed from the table.

- centroid: (-0.09, -0.33, 1.08) | confidence: 0.75
- cluster ids: [1]
- visible in: a/frame_00072.jpg, b/frame_00040.jpg
- confirmed by: b/frame_00120.jpg (6137 points seen empty)

### 2. REMOVED - apple

> The apple was removed from the table.

- centroid: (0.19, -0.06, 0.99) | confidence: 0.50
- cluster ids: [2]
- visible in: a/frame_00000.jpg, b/frame_00088.jpg
- confirmed by: b/frame_00000.jpg (569 points seen empty)

### 3. REMOVED - small dark speck

> The small dark speck on the surface was removed between the two images.

- centroid: (0.40, 0.12, 0.90) | confidence: 0.75
- cluster ids: [4]
- visible in: a/frame_00040.jpg, b/frame_00032.jpg
- confirmed by: b/frame_00016.jpg (656 points seen empty)

### 4. ADDED - round object

> The round object was removed from the table surface.

- centroid: (-0.09, -0.25, 1.09) | confidence: 0.36
- cluster ids: [6]
- visible in: a/frame_00088.jpg, b/frame_00000.jpg
- confirmed by: a/frame_00000.jpg (67 points seen empty)

### 5. ADDED - finger

> A finger was added to the table surface in the second image.

- centroid: (0.28, 0.06, 0.92) | confidence: 0.33
- cluster ids: [8]
- visible in: a/frame_00088.jpg, b/frame_00000.jpg
- confirmed by: a/frame_00016.jpg (48 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
