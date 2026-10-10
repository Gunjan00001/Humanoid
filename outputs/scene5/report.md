# Change report

- video A: `/home/user/Desktop/Humanoid/data/scene5/a.mp4`
- video B: `/home/user/Desktop/Humanoid/data/scene5/b.mp4`
- reconstruction: `vggt` (/home/user/models/VGGT-1B)
- frames: 60 | semantics: vlm-verified
- generated: 2026-10-10T23:43:55Z | config `a93e2d60be34`

**5 change(s) detected**: 2 added, 3 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | wooden leg | (-0.15, 0.41, 0.76) | 0.74 |
| 2 | removed | chair leg | (-0.04, 0.16, 1.17) | 0.29 |
| 3 | removed | table leg | (0.21, 0.48, 0.61) | 0.45 |
| 4 | added | small round object | (-0.01, -0.21, 1.08) | 0.34 |
| 5 | added | bottle and clock | (0.11, -0.18, 0.97) | 0.30 |

## Details

### 1. REMOVED - wooden leg

> The wooden leg was removed from the table surface.

- centroid: (-0.15, 0.41, 0.76) | confidence: 0.74
- cluster ids: [6]
- visible in: a/frame_00000.jpg, b/frame_01270.jpg
- confirmed by: b/frame_00320.jpg (293 points seen empty)

### 2. REMOVED - chair leg

> The chair leg and black object were removed from the scene.

- centroid: (-0.04, 0.16, 1.17) | confidence: 0.29
- cluster ids: [16]
- visible in: a/frame_00000.jpg, b/frame_01270.jpg
- confirmed by: b/frame_00000.jpg (26 points seen empty)

### 3. REMOVED - table leg

> The table leg and seat were removed, leaving only the cracked surface.

- centroid: (0.21, 0.48, 0.61) | confidence: 0.45
- cluster ids: [18]
- visible in: a/frame_00250.jpg, b/frame_01230.jpg
- confirmed by: b/frame_00290.jpg (29 points seen empty)

### 4. ADDED - small round object

> A small round object appeared on the floor in the second image.

- centroid: (-0.01, -0.21, 1.08) | confidence: 0.34
- cluster ids: [38]
- visible in: a/frame_00800.jpg, b/frame_00440.jpg
- confirmed by: a/frame_00000.jpg (54 points seen empty)

### 5. ADDED - bottle and clock

> A plastic water bottle and a digital clock were added to the table.

- centroid: (0.11, -0.18, 0.97) | confidence: 0.30
- cluster ids: [44]
- visible in: a/frame_00800.jpg, b/frame_00520.jpg
- confirmed by: a/frame_00280.jpg (30 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
