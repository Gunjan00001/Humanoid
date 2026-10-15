# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/living_room_1_living_room_2/original_video1.mp4`
- video B: `data/external/scenediff/scenediff_benchmark/data/living_room_1_living_room_2/original_video2.mp4`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 47 | semantics: vlm-verified
- generated: 2026-10-15T04:23:14Z | config `798e1549834b`

**1 change(s) detected**: 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | brown bag | (-0.33, 0.22, 1.03) | 0.75 |

## Details

### 1. REMOVED - brown bag

> The brown bag was removed from the table.

- centroid: (-0.33, 0.22, 1.03) | confidence: 0.75
- cluster ids: [1]
- visible in: a/frame_00090.jpg, b/frame_00270.jpg
- confirmed by: b/frame_00270.jpg (4129 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
