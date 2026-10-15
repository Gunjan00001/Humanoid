# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/kitchen_6_kitchen_7/original_video1.mp4`
- video B: `data/external/scenediff/scenediff_benchmark/data/kitchen_6_kitchen_7/original_video2.mp4`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 52 | semantics: vlm-verified
- generated: 2026-10-15T11:44:59Z | config `798e1549834b`

**1 change(s) detected**: 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | jar with black lid | (0.98, 0.05, 0.54) | 0.40 |

## Details

### 1. REMOVED - jar with black lid

> The small jar with a black lid was removed from the table.

- centroid: (0.98, 0.05, 0.54) | confidence: 0.40
- cluster ids: [12]
- visible in: a/frame_00120.jpg, b/frame_00180.jpg
- confirmed by: b/frame_00080.jpg (90 points seen empty)

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
