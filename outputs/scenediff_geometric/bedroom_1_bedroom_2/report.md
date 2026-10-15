# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/bedroom_1_bedroom_2/original_video1.MOV`
- video B: `data/external/scenediff/scenediff_benchmark/data/bedroom_1_bedroom_2/original_video2.MOV`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 45 | semantics: vlm-verified
- generated: 2026-10-15T14:20:03Z | config `530c43b66d5f`

**1 change(s) detected**: 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | white rectangular object | (0.07, -0.04, 1.17) | 0.29 |

## Details

### 1. REMOVED - white rectangular object

> The white rectangular object was removed from the surface.

- centroid: (0.07, -0.04, 1.17) | confidence: 0.29
- cluster ids: [2]
- visible in: a/frame_00070.jpg, b/frame_00370.jpg
- confirmed by: b/frame_00370.jpg (14 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
