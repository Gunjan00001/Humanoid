# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/living_room_17_living_room_18/original_video1.MOV`
- video B: `data/external/scenediff/scenediff_benchmark/data/living_room_17_living_room_18/original_video2.MOV`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 57 | semantics: vlm-verified
- generated: 2026-10-15T00:55:38Z | config `798e1549834b`

**1 change(s) detected**: 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | cushion | (0.38, -0.30, 1.00) | 0.75 |

## Details

### 1. REMOVED - cushion

> The patterned cushion was removed from the sofa.

- centroid: (0.38, -0.30, 1.00) | confidence: 0.75
- cluster ids: [2]
- visible in: a/frame_00100.jpg, b/frame_00020.jpg
- confirmed by: b/frame_00080.jpg (2619 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
