# Change report

- video A: `/home/jiit_992401210050/Desktop/Humanoid/data/scene3/a.mp4`
- video B: `/home/jiit_992401210050/Desktop/Humanoid/data/scene3/b.mp4`
- reconstruction: `vggt` (/home/jiit_992401210050/models/VGGT-1B)
- frames: 30 | semantics: vlm-verified
- generated: 2026-10-10T21:43:03Z | config `2514ba94c2d8`

**4 change(s) detected**: 3 removed, 1 replaced

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | replaced | black chair | (0.02, -0.18, 0.86) | 0.56 |
| 2 | removed | red apple | (-0.04, -0.40, 1.11) | 1.00 |
| 3 | removed | green and yellow object | (0.29, -0.38, 1.04) | 0.98 |
| 4 | removed | black trash can | (0.32, 0.12, 0.73) | 0.57 |

## Details

### 1. REPLACED - black chair

> the red object was replaced by a black chair

- centroid: (0.02, -0.18, 0.86) | confidence: 0.56
- cluster ids: [5]
- visible in: a/frame_00020.jpg, b/frame_00010.jpg

### 2. REMOVED - red apple

> the red apple was removed from the table

- centroid: (-0.04, -0.40, 1.11) | confidence: 1.00
- cluster ids: [26]
- visible in: a/frame_00000.jpg, b/frame_00010.jpg

### 3. REMOVED - green and yellow object

> the green and yellow object was removed from the scene

- centroid: (0.29, -0.38, 1.04) | confidence: 0.98
- cluster ids: [27]
- visible in: a/frame_00000.jpg, b/frame_00010.jpg

### 4. REMOVED - black trash can

> the black trash can was removed from the corner

- centroid: (0.32, 0.12, 0.73) | confidence: 0.57
- cluster ids: [28]
- visible in: a/frame_00000.jpg, b/frame_00010.jpg


## Unverified candidates

These regions contain object-like appearance in ONE video while the other video never observed the area (out of view, cropped or occluded). They are **not confirmed changes** - listed for transparency so nothing detected is silently dropped.

- **removed?** cracked floor at (-0.70, -0.02, 1.19) - a white tiled floor with visible cracks running across the surface.
- **removed?** wall corner at (-0.79, -0.51, 1.27) - a close-up of an indoor wall corner with a textured surface and a dark vertical edge.
- **removed?** window frame at (-0.69, -0.40, 1.29) - a close-up of a dark window frame with a white wall beside it, showing part of a glass pane.
- **removed?** metal chair leg at (-0.58, -0.33, 1.32) - a curved metal chair leg with a shiny surface, partially visible next to a blue curtain and dark wall.
- **removed?** marble surface at (-0.58, 0.01, 1.17) - a close-up of a white marble surface with gray veining patterns.
- **removed?** cracked surface at (-0.54, 0.07, 1.10) - a close-up of a white surface with dark, irregular cracks resembling marble or dried paint.
- **removed?** cracked surface at (-0.48, 0.08, 1.08) - a close-up of a white surface with dark, irregular cracks resembling marble or dried paint.
- **removed?** metal chair back at (-0.50, -0.35, 1.34) - a curved, shiny metal chair back with a dark blue curtain in the background.
- **removed?** metal chair back at (-0.43, -0.37, 1.32) - a curved, shiny metal chair back with a decorative top, positioned near a blue curtain and wooden surface.
- **removed?** metal railing at (-0.35, -0.40, 1.29) - a curved, shiny metal railing next to a wooden surface and dark curtain.
- **removed?** metal table leg at (-0.26, 0.20, 0.94) - A shiny metal table leg standing on a tiled floor with visible cracks.
- **removed?** metal chair at (-0.24, -0.19, 0.99) - a chrome-finished metal chair with curved backrest and armrests, partially visible next to a wooden table.
- **removed?** wooden table corner at (-0.24, -0.28, 0.97) - a close-up of the corner of a wooden table with a dark finish, partially visible next to a blue curtain.
- **removed?** apple on table at (-0.23, -0.42, 1.19) - A red apple sits on a wooden table in a dimly lit indoor setting.
- **removed?** bar stool at (-0.10, 0.01, 0.80) - A black, round, leather-upholstered bar stool with a chrome metal frame and legs, placed on a tiled floor.
- **removed?** bar stool at (-0.14, 0.10, 0.77) - a metal bar stool with a black seat and chrome legs, partially visible in a cropped indoor setting.
- **removed?** metal chair leg at (-0.02, 0.17, 0.93) - A shiny, cylindrical metal leg of a chair standing on a cracked white surface.
- **removed?** metal chair leg at (0.13, 0.27, 0.81) - A close-up of a metal chair leg on a white floor with visible cracks and a black cable nearby.
- **removed?** stacked metal bowls at (0.28, -0.01, 0.79) - a close-up of shiny, reflective metal bowls stacked on top of each other.
- **removed?** metal bowl at (0.33, -0.07, 0.89) - a shiny, curved edge of a metal bowl reflecting light.
- **removed?** metal chair frame at (0.38, -0.08, 0.86) - a close-up of a shiny chrome chair frame with curved armrests and a hinge mechanism.
- **removed?** doorframe at (0.51, 0.25, 0.74) - a vertical edge of a door or wall with a visible seam and adjacent flooring.
- **added?** wooden table leg at (-0.22, -0.24, 0.95) - A close-up of a wooden table leg with a visible grain pattern and a metal tabletop above it.
- **added?** dark vertical object at (-0.20, -0.51, 1.14) - a blurry, dark, vertical object with a light-colored edge, possibly a door frame or furniture leg.
- **added?** cracked wall at (-0.18, -0.25, 1.17) - a white wall with multiple dark cracks spreading across its surface.
- **added?** metal chair leg at (0.15, 0.32, 0.81) - a close-up of a metal chair leg with a circular base, casting a shadow on a light-colored floor.
- **added?** door frame at (0.41, 0.50, 0.48) - a close-up of a wooden door frame with a light-colored wall and a metal threshold below.

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
