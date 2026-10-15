# Change report

- video A: `data/external/scenediff/scenediff_benchmark/data/gas_station_3_gas_station_4/original_video1.mp4`
- video B: `data/external/scenediff/scenediff_benchmark/data/gas_station_3_gas_station_4/original_video2.mp4`
- reconstruction: `vggt` (~/models/VGGT-1B)
- frames: 53 | semantics: vlm-verified
- generated: 2026-10-14T23:15:16Z | config `798e1549834b`

**0 change(s) detected**: 

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|

## Details

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
