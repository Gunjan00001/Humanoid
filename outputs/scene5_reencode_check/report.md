# Change report

- video A: `data/scene5/a.mp4`
- video B: `data/scene5/b.mp4`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 60 | semantics: vlm-verified
- generated: 2026-10-11T00:25:10Z | config `a93e2d60be34`

**7 change(s) detected**: 1 added, 6 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | chair | (-0.18, -0.13, 1.08) | 0.75 |
| 2 | removed | chair leg | (-0.03, 0.17, 1.17) | 0.75 |
| 3 | removed | chair leg | (0.00, 0.30, 0.92) | 0.44 |
| 4 | removed | chair leg | (0.27, 0.36, 0.81) | 0.50 |
| 5 | removed | chair backrest | (0.31, 0.01, 1.03) | 0.34 |
| 6 | removed | stacked bowls | (0.31, 0.20, 0.89) | 0.34 |
| 7 | added | small brown object | (0.03, -0.25, 1.08) | 0.50 |

## Details

### 1. REMOVED - chair

> The chair was removed from the scene.

- centroid: (-0.18, -0.13, 1.08) | confidence: 0.75
- cluster ids: [1]
- visible in: a/frame_00000.jpg, b/frame_01270.jpg
- confirmed by: b/frame_00000.jpg (34699 points seen empty)

### 2. REMOVED - chair leg

> The chair leg was removed from the scene.

- centroid: (-0.03, 0.17, 1.17) | confidence: 0.75
- cluster ids: [8]
- visible in: a/frame_00000.jpg, b/frame_01270.jpg
- confirmed by: b/frame_00000.jpg (601 points seen empty)

### 3. REMOVED - chair leg

> The chair leg and seat were removed from the scene.

- centroid: (0.00, 0.30, 0.92) | confidence: 0.44
- cluster ids: [9]
- visible in: a/frame_00000.jpg, b/frame_01270.jpg
- confirmed by: b/frame_00000.jpg (96 points seen empty)

### 4. REMOVED - chair leg

> The chair leg and black object were removed from the scene.

- centroid: (0.27, 0.36, 0.81) | confidence: 0.50
- cluster ids: [19]
- visible in: a/frame_00000.jpg, b/frame_01240.jpg
- confirmed by: b/frame_00260.jpg (197 points seen empty)

### 5. REMOVED - chair backrest

> The metal chair backrest was removed, leaving only the wooden table leg visible.

- centroid: (0.31, 0.01, 1.03) | confidence: 0.34
- cluster ids: [24]
- visible in: a/frame_00410.jpg, b/frame_01140.jpg
- confirmed by: b/frame_01140.jpg (54 points seen empty)

### 6. REMOVED - stacked bowls

> The stacked metal bowls were removed from the table.

- centroid: (0.31, 0.20, 0.89) | confidence: 0.34
- cluster ids: [25]
- visible in: a/frame_00000.jpg, b/frame_01270.jpg
- confirmed by: b/frame_00530.jpg (52 points seen empty)

### 7. ADDED - small brown object

> A small brown object appeared on the surface in the second image.

- centroid: (0.03, -0.25, 1.08) | confidence: 0.50
- cluster ids: [34]
- visible in: a/frame_00800.jpg, b/frame_00200.jpg
- confirmed by: a/frame_00000.jpg (40 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
