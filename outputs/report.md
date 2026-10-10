# Change report

- video A: `/home/jiit_992401210050/Desktop/Humanoid/data/a.mp4`
- video B: `/home/jiit_992401210050/Desktop/Humanoid/data/b.mp4`
- reconstruction: `vggt` (/home/jiit_992401210050/models/VGGT-1B)
- frames: 49 | semantics: vlm-verified
- generated: 2026-10-10T20:31:01Z | config `2b487e1c26d7`

**3 change(s) detected**: 1 added, 1 removed, 1 replaced

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | comb | (-0.05, 0.13, 0.91) | 0.87 |
| 2 | added | pen and device | (-0.11, 0.19, 0.91) | 0.38 |
| 3 | replaced | remote control | (-0.22, -0.34, 0.95) | 0.84 |

## Details

### 1. REMOVED - comb

> The comb was removed from the table.

- centroid: (-0.05, 0.13, 0.91) | confidence: 0.87
- cluster ids: [3]
- visible in: a/frame_00000.jpg, b/frame_00030.jpg

### 2. ADDED - pen and device

> A blue pen and a white device were added to the surface.

- centroid: (-0.11, 0.19, 0.91) | confidence: 0.38
- cluster ids: [12]
- visible in: a/frame_00432.jpg, b/frame_00090.jpg

### 3. REPLACED - remote control

> The remote control was replaced with a different one.

- centroid: (-0.22, -0.34, 0.95) | confidence: 0.84
- cluster ids: [18]
- visible in: a/frame_00234.jpg, b/frame_00030.jpg

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
