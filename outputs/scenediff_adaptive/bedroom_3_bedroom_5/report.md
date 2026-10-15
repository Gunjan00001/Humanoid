# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/bedroom_3_bedroom_5/original_video1.MOV`
- video B: `data/external/scenediff/scenediff_benchmark/data/bedroom_3_bedroom_5/original_video2.MOV`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 26 | semantics: vlm-verified
- generated: 2026-10-15T10:36:47Z | config `798e1549834b`

**1 change(s) detected**: 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | fabric surface | (0.46, 0.20, 0.48) | 0.37 |

## Details

### 1. REMOVED - fabric surface

> The white smooth surface was replaced by a gray textured fabric.

- centroid: (0.46, 0.20, 0.48) | confidence: 0.37
- cluster ids: [2]
- visible in: a/frame_00210.jpg, b/frame_00290.jpg
- confirmed by: b/frame_00300.jpg (70 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
