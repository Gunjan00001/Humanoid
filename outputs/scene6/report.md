# Change report

- video A: `data/scene6/a.mp4`
- video B: `data/scene6/b.mp4`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 60 | semantics: vlm-verified
- generated: 2026-10-11T10:16:33Z | config `a93e2d60be34`

**2 change(s) detected**: 1 added, 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | remote control | (-0.22, 0.04, 0.72) | 0.50 |
| 2 | added | electrical outlet | (-0.32, -0.34, 0.93) | 0.33 |

## Details

### 1. REMOVED - remote control

> The remote control was removed from the table.

- centroid: (-0.22, 0.04, 0.72) | confidence: 0.50
- cluster ids: [3]
- visible in: a/frame_00000.jpg, b/frame_01430.jpg
- confirmed by: b/frame_00120.jpg (276 points seen empty)

### 2. ADDED - electrical outlet

> An electrical outlet was added to the wall.

- centroid: (-0.32, -0.34, 0.93) | confidence: 0.33
- cluster ids: [20]
- visible in: a/frame_01610.jpg, b/frame_00570.jpg
- confirmed by: a/frame_00000.jpg (12 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
