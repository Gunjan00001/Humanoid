# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/bedroom_30_bedroom_31/original_video1.mov`
- video B: `data/external/scenediff/scenediff_benchmark/data/bedroom_30_bedroom_31/original_video2.mov`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 23 | semantics: vlm-verified
- generated: 2026-10-15T10:32:22Z | config `798e1549834b`

**19 change(s) detected**: 11 added, 2 moved, 6 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | moved | wall corner | (-0.74, 0.00, 0.83) | 0.75 |
| 2 | moved | wall | (-0.24, -0.57, 0.58) | 0.51 |
| 3 | removed | plush toy | (-0.04, 0.15, 0.26) | 0.29 |
| 4 | removed | small dark object | (0.38, -0.67, 0.73) | 0.68 |
| 5 | removed | paper document | (0.34, -0.69, 0.24) | 0.50 |
| 6 | removed | surface color | (0.43, 0.22, 0.42) | 0.61 |
| 7 | removed | black rectangular object | (0.40, 0.14, 1.49) | 0.29 |
| 8 | removed | paper | (0.54, 0.05, 1.11) | 0.41 |
| 9 | added | table edge | (-0.68, 0.44, 0.84) | 0.40 |
| 10 | added | edge of object | (-0.66, 0.16, 0.73) | 0.35 |
| 11 | added | thin dark line | (-0.31, -0.27, 0.79) | 0.50 |
| 12 | added | shadow | (-0.31, -0.10, 0.56) | 0.32 |
| 13 | added | vertical mark | (-0.30, 0.25, 0.47) | 0.31 |
| 14 | added | table edge | (-0.05, -0.14, 0.68) | 0.50 |
| 15 | added | black object | (-0.01, -0.77, 0.50) | 0.35 |
| 16 | added | white box | (0.43, -0.08, 1.28) | 0.44 |
| 17 | added | panda card | (0.49, -0.08, 1.13) | 0.34 |
| 18 | added | a plain surface with no visible object | (-0.74, 0.00, 0.83) | 0.75 |
| 19 | added | a plain light surface with no visible objects | (-0.24, -0.57, 0.58) | 0.40 |

## Details

### 1. MOVED - wall corner

> the wall corner moved slightly to the right

- from: (-0.74, 0.00, 0.83)
- to:   (-0.76, 0.37, 1.08)

- centroid: (-0.74, 0.00, 0.83) | confidence: 0.75
- cluster ids: [1, 70]
- visible in: a/frame_00000.jpg, b/frame_00100.jpg
- confirmed by: b/frame_00090.jpg (239 points seen empty)

### 2. MOVED - wall

> the wall remained in the same position with no visible object movement

- from: (-0.24, -0.57, 0.58)
- to:   (-0.56, 0.12, 1.04)

- centroid: (-0.24, -0.57, 0.58) | confidence: 0.51
- cluster ids: [33, 75]
- visible in: a/frame_00040.jpg, b/frame_00070.jpg
- confirmed by: b/frame_00000.jpg (88 points seen empty)

### 3. REMOVED - plush toy

> A plush toy appeared on the surface in the second image.

- centroid: (-0.04, 0.15, 0.26) | confidence: 0.29
- cluster ids: [44]
- visible in: a/frame_00000.jpg, b/frame_00100.jpg
- confirmed by: b/frame_00080.jpg (25 points seen empty)

### 4. REMOVED - small dark object

> The small dark object in the bottom right corner was removed.

- centroid: (0.38, -0.67, 0.73) | confidence: 0.68
- cluster ids: [54]
- visible in: a/frame_00040.jpg, b/frame_00070.jpg
- confirmed by: b/frame_00000.jpg (258 points seen empty)

### 5. REMOVED - paper document

> The paper document was removed from the surface.

- centroid: (0.34, -0.69, 0.24) | confidence: 0.50
- cluster ids: [58]
- visible in: a/frame_00080.jpg, b/frame_00000.jpg
- confirmed by: b/frame_00000.jpg (136 points seen empty)

### 6. REMOVED - surface color

> The surface color changed from beige to white.

- centroid: (0.43, 0.22, 0.42) | confidence: 0.61
- cluster ids: [61]
- visible in: a/frame_00010.jpg, b/frame_00100.jpg
- confirmed by: b/frame_00090.jpg (215 points seen empty)

### 7. REMOVED - black rectangular object

> The black rectangular object was removed from the table.

- centroid: (0.40, 0.14, 1.49) | confidence: 0.29
- cluster ids: [63]
- visible in: a/frame_00000.jpg, b/frame_00100.jpg
- confirmed by: b/frame_00050.jpg (27 points seen empty)

### 8. REMOVED - paper

> A piece of paper appeared on the table.

- centroid: (0.54, 0.05, 1.11) | confidence: 0.41
- cluster ids: [66]
- visible in: a/frame_00000.jpg, b/frame_00100.jpg
- confirmed by: b/frame_00040.jpg (96 points seen empty)

### 9. ADDED - table edge

> A table edge appeared in the second image.

- centroid: (-0.68, 0.44, 0.84) | confidence: 0.40
- cluster ids: [73]
- visible in: a/frame_00010.jpg, b/frame_00100.jpg
- confirmed by: a/frame_00000.jpg (91 points seen empty)

### 10. ADDED - edge of object

> An edge of an object appeared in the second image that was not present in the first.

- centroid: (-0.66, 0.16, 0.73) | confidence: 0.35
- cluster ids: [74]
- visible in: a/frame_00040.jpg, b/frame_00060.jpg
- confirmed by: a/frame_00000.jpg (63 points seen empty)

### 11. ADDED - thin dark line

> A thin dark line appeared on the table surface in the second image.

- centroid: (-0.31, -0.27, 0.79) | confidence: 0.50
- cluster ids: [79]
- visible in: a/frame_00080.jpg, b/frame_00000.jpg
- confirmed by: a/frame_00000.jpg (62 points seen empty)

### 12. ADDED - shadow

> A shadow appeared in the second image that was not present in the first.

- centroid: (-0.31, -0.10, 0.56) | confidence: 0.32
- cluster ids: [92]
- visible in: a/frame_00060.jpg, b/frame_00040.jpg
- confirmed by: a/frame_00000.jpg (43 points seen empty)

### 13. ADDED - vertical mark

> A faint vertical mark appeared on the surface in the second image.

- centroid: (-0.30, 0.25, 0.47) | confidence: 0.31
- cluster ids: [94]
- visible in: a/frame_00010.jpg, b/frame_00090.jpg
- confirmed by: a/frame_00000.jpg (38 points seen empty)

### 14. ADDED - table edge

> A table edge became visible in the second image.

- centroid: (-0.05, -0.14, 0.68) | confidence: 0.50
- cluster ids: [102]
- visible in: a/frame_00070.jpg, b/frame_00010.jpg
- confirmed by: a/frame_00000.jpg (31 points seen empty)

### 15. ADDED - black object

> A black object appeared in the second image.

- centroid: (-0.01, -0.77, 0.50) | confidence: 0.35
- cluster ids: [110]
- visible in: a/frame_00080.jpg, b/frame_00000.jpg
- confirmed by: a/frame_00060.jpg (59 points seen empty)

### 16. ADDED - white box

> A white box-like object appeared in the second image.

- centroid: (0.43, -0.08, 1.28) | confidence: 0.44
- cluster ids: [121]
- visible in: a/frame_00060.jpg, b/frame_00050.jpg
- confirmed by: a/frame_00000.jpg (112 points seen empty)

### 17. ADDED - panda card

> A panda card was added to the surface.

- centroid: (0.49, -0.08, 1.13) | confidence: 0.34
- cluster ids: [124]
- visible in: a/frame_00060.jpg, b/frame_00050.jpg
- confirmed by: a/frame_00000.jpg (56 points seen empty)

### 18. ADDED - a plain surface with no visible object

> a a plain surface with no visible object appeared where the wall corner was before it moved

- centroid: (-0.74, 0.00, 0.83) | confidence: 0.75
- cluster ids: [900]

### 19. ADDED - a plain light surface with no visible objects

> a a plain light surface with no visible objects appeared where the wall was before it moved

- centroid: (-0.24, -0.57, 0.58) | confidence: 0.40
- cluster ids: [901]

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
