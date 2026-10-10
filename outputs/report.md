# Change report

- video A: `/home/user/Desktop/Humanoid/data/a.mp4`
- video B: `/home/user/Desktop/Humanoid/data/b.mp4`
- reconstruction: `vggt` (/home/user/models/VGGT-1B)
- frames: 49 | semantics: vlm-verified
- generated: 2026-10-10T13:11:15Z | config `eb78e286bd44`

**17 change(s) detected**: 3 added, 9 removed, 5 replaced

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | small object | (-0.29, -0.07, 0.92) | 0.30 |
| 2 | removed | black remote | (-0.25, -0.34, 0.95) | 0.47 |
| 3 | removed | comb | (-0.13, 0.10, 0.91) | 0.87 |
| 4 | removed | digital timer | (-0.16, -0.16, 0.94) | 0.33 |
| 5 | removed | timer | (-0.08, -0.17, 0.95) | 0.77 |
| 6 | removed | digital timer | (-0.10, -0.06, 0.92) | 0.79 |
| 7 | removed | comb | (-0.09, 0.03, 0.94) | 0.66 |
| 8 | removed | digital timer | (-0.04, -0.10, 0.92) | 0.88 |
| 9 | removed | wooden comb | (0.02, 0.15, 0.91) | 0.42 |
| 10 | added | remote control | (-0.11, -0.08, 0.94) | 0.38 |
| 11 | added | white electronic device | (0.08, -0.03, 0.95) | 0.23 |
| 12 | added | highlighter | (0.12, -0.10, 0.96) | 0.27 |
| 13 | replaced | remote control | (-0.22, -0.34, 0.95) | 0.84 |
| 14 | replaced | timer | (-0.11, -0.04, 0.93) | 0.63 |
| 15 | replaced | digital timer | (-0.07, -0.08, 0.92) | 0.56 |
| 16 | replaced | digital timer | (-0.04, -0.03, 0.93) | 0.66 |
| 17 | replaced | highlighter | (0.10, -0.19, 0.96) | 0.75 |

## Details

### 1. REMOVED - small object

> the small object was removed

- centroid: (-0.29, -0.07, 0.92) | confidence: 0.30
- cluster ids: [1]
- visible in: a/frame_00072.jpg, b/frame_00030.jpg

### 2. REMOVED - black remote

> the black remote was removed

- centroid: (-0.25, -0.34, 0.95) | confidence: 0.47
- cluster ids: [2]
- visible in: a/frame_00234.jpg, b/frame_00000.jpg

### 3. REMOVED - comb

> the comb was removed

- centroid: (-0.13, 0.10, 0.91) | confidence: 0.87
- cluster ids: [3]
- visible in: a/frame_00000.jpg, b/frame_00030.jpg

### 4. REMOVED - digital timer

> the digital timer was removed

- centroid: (-0.16, -0.16, 0.94) | confidence: 0.33
- cluster ids: [4]
- visible in: a/frame_00000.jpg, b/frame_00000.jpg

### 5. REMOVED - timer

> the timer was removed

- centroid: (-0.08, -0.17, 0.95) | confidence: 0.77
- cluster ids: [5]
- visible in: a/frame_00504.jpg, b/frame_00030.jpg

### 6. REMOVED - digital timer

> the digital timer was removed

- centroid: (-0.10, -0.06, 0.92) | confidence: 0.79
- cluster ids: [6]
- visible in: a/frame_00000.jpg, b/frame_00000.jpg

### 7. REMOVED - comb

> the comb was removed

- centroid: (-0.09, 0.03, 0.94) | confidence: 0.66
- cluster ids: [7]
- visible in: a/frame_00576.jpg, b/frame_00030.jpg

### 8. REMOVED - digital timer

> the digital timer was removed

- centroid: (-0.04, -0.10, 0.92) | confidence: 0.88
- cluster ids: [8]
- visible in: a/frame_00000.jpg, b/frame_00000.jpg

### 9. REMOVED - wooden comb

> the wooden comb was removed

- centroid: (0.02, 0.15, 0.91) | confidence: 0.42
- cluster ids: [9]
- visible in: a/frame_00000.jpg, b/frame_00030.jpg

### 10. ADDED - remote control

> a remote control was added to the table

- centroid: (-0.11, -0.08, 0.94) | confidence: 0.38
- cluster ids: [12]
- visible in: a/frame_00000.jpg, b/frame_00060.jpg

### 11. ADDED - white electronic device

> a white electronic device was added to the table

- centroid: (0.08, -0.03, 0.95) | confidence: 0.23
- cluster ids: [16]
- visible in: a/frame_00000.jpg, b/frame_00030.jpg

### 12. ADDED - highlighter

> a highlighter was added to the table

- centroid: (0.12, -0.10, 0.96) | confidence: 0.27
- cluster ids: [17]
- visible in: a/frame_00000.jpg, b/frame_00000.jpg

### 13. REPLACED - remote control

> the remote control was flipped over

- centroid: (-0.22, -0.34, 0.95) | confidence: 0.84
- cluster ids: [18]
- visible in: a/frame_00234.jpg, b/frame_00000.jpg

### 14. REPLACED - timer

> the timer was moved and its display changed

- centroid: (-0.11, -0.04, 0.93) | confidence: 0.63
- cluster ids: [19]
- visible in: a/frame_00000.jpg, b/frame_00130.jpg

### 15. REPLACED - digital timer

> the digital timer was flipped over

- centroid: (-0.07, -0.08, 0.92) | confidence: 0.56
- cluster ids: [20]
- visible in: a/frame_00000.jpg, b/frame_00000.jpg

### 16. REPLACED - digital timer

> the digital timer was moved and rotated

- centroid: (-0.04, -0.03, 0.93) | confidence: 0.66
- cluster ids: [21]
- visible in: a/frame_00000.jpg, b/frame_00030.jpg

### 17. REPLACED - highlighter

> the highlighter was moved

- centroid: (0.10, -0.19, 0.96) | confidence: 0.75
- cluster ids: [22]
- visible in: a/frame_00000.jpg, b/frame_00000.jpg

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
