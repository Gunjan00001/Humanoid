# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/living_room_29_living_room_30/original_video1.MOV`
- video B: `data/external/scenediff/scenediff_benchmark/data/living_room_29_living_room_30/original_video2.MOV`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 19 | semantics: vlm-verified
- generated: 2026-10-15T13:32:13Z | config `798e1549834b`

**1 change(s) detected**: 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | speckled object | (-0.17, -0.30, 0.80) | 0.69 |

## Details

### 1. REMOVED - speckled object

> The speckled object was removed from the table.

- centroid: (-0.17, -0.30, 0.80) | confidence: 0.69
- cluster ids: [1]
- visible in: a/frame_00020.jpg, b/frame_00640.jpg
- confirmed by: b/frame_00640.jpg (73 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
