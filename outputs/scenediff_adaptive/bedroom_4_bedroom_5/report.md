# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/bedroom_4_bedroom_5/original_video1.MOV`
- video B: `data/external/scenediff/scenediff_benchmark/data/bedroom_4_bedroom_5/original_video2.MOV`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 27 | semantics: vlm-verified
- generated: 2026-10-14T22:04:34Z | config `798e1549834b`

**0 change(s) detected**: 

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|

## Details

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
