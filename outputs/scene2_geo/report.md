# Change report

- video A: `data/scene2/a.mp4`
- video B: `data/scene2/b.mp4`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 24 | semantics: vlm-verified
- generated: 2026-10-10T18:56:24Z | config `e6e15c30467a`

**5 change(s) detected**: 3 added, 2 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | coconut | (-0.09, -0.33, 1.08) | 0.75 |
| 2 | removed | apple | (0.15, -0.04, 0.95) | 0.75 |
| 3 | added | round object | (-0.07, -0.23, 1.07) | 0.45 |
| 4 | added | apple | (-0.01, 0.01, 0.90) | 0.36 |
| 5 | added | apple | (0.22, -0.14, 1.04) | 0.75 |

## Details

### 1. REMOVED - coconut

> The coconut was removed from the table.

- centroid: (-0.09, -0.33, 1.08) | confidence: 0.75
- cluster ids: [1]
- visible in: a/frame_00072.jpg, b/frame_00040.jpg
- confirmed by: b/frame_00120.jpg (5896 points seen empty)

### 2. REMOVED - apple

> The apple was removed from the table.

- centroid: (0.15, -0.04, 0.95) | confidence: 0.75
- cluster ids: [2]
- visible in: a/frame_00016.jpg, b/frame_00088.jpg
- confirmed by: b/frame_00000.jpg (562 points seen empty)

### 3. ADDED - round object

> The round object was removed from the table surface.

- centroid: (-0.07, -0.23, 1.07) | confidence: 0.45
- cluster ids: [5]
- visible in: a/frame_00088.jpg, b/frame_00000.jpg
- confirmed by: a/frame_00000.jpg (119 points seen empty)

### 4. ADDED - apple

> An apple appeared in the second image.

- centroid: (-0.01, 0.01, 0.90) | confidence: 0.36
- cluster ids: [6]
- visible in: a/frame_00088.jpg, b/frame_00016.jpg
- confirmed by: a/frame_00000.jpg (67 points seen empty)

### 5. ADDED - apple

> The apple was removed from the table.

- centroid: (0.22, -0.14, 1.04) | confidence: 0.75
- cluster ids: [8]
- visible in: a/frame_00088.jpg, b/frame_00120.jpg
- confirmed by: a/frame_00032.jpg (1420 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
