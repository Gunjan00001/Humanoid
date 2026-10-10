# Change report

- video A: `/home/jiit_992401210050/Desktop/Humanoid/data/scene2/a.mp4`
- video B: `/home/jiit_992401210050/Desktop/Humanoid/data/scene2/b.mp4`
- reconstruction: `vggt` (/home/jiit_992401210050/models/VGGT-1B)
- frames: 24 | semantics: vlm-verified
- generated: 2026-10-10T20:26:03Z | config `af2c7bd058ca`

**1 change(s) detected**: 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | removed | apple | (0.06, -0.25, 1.05) | 0.86 |

## Details

### 1. REMOVED - apple

> The apple was removed from the table.

- centroid: (0.06, -0.25, 1.05) | confidence: 0.86
- cluster ids: [1]
- visible in: a/frame_00000.jpg, b/frame_00032.jpg


## Unverified candidates

These regions contain object-like appearance in ONE video while the other video never observed the area (out of view, cropped or occluded). They are **not confirmed changes** - listed for transparency so nothing detected is silently dropped.

- **added?** concrete floor at (-0.48, -0.03, 0.88) - a gray concrete floor meeting a white wall baseboard.
- **added?** crumpled paper at (-0.44, -0.38, 1.13) - a small, crumpled piece of white paper lying on a dark floor near a white wall or baseboard.
- **added?** small metallic object at (-0.35, -0.19, 1.04) - a shiny, circular object with a reflective surface, possibly a coin or small metal piece, lying on a dark, textured surface.
- **added?** white wall at (0.36, 0.27, 0.83) - a plain, light-colored wall surface with subtle texture and minor imperfections.
- **added?** wooden beam at (0.38, -0.55, 1.35) - a diagonal wooden beam against a light-colored wall.
- **added?** wooden frame at (0.38, -0.61, 1.38) - a diagonal wooden frame or trim against a light-colored wall.
- **added?** floor corner at (0.39, -0.50, 1.31) - a close-up of a floor corner with a dark baseboard and light-colored flooring.
- **added?** metal ball at (0.41, -0.27, 1.18) - a small, shiny metallic ball resting on a dark surface near a wooden edge.
- **added?** wooden beam at (0.40, -0.47, 1.28) - a diagonal wooden beam against a light-colored wall, possibly part of a ceiling or frame.
- **added?** wooden beam at (0.44, -0.05, 1.04) - a vertical wooden beam with visible grain and a rough texture, partially framing the edge of the image.

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
