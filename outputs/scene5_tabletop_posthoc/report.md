# Change report

- video A: `/home/jiit_992401210050/Desktop/Humanoid/data/scene5/a.mp4`
- video B: `/home/jiit_992401210050/Desktop/Humanoid/data/scene5/b.mp4`
- reconstruction: `vggt` (/home/jiit_992401210050/models/VGGT-1B)
- frames: 60 | semantics: vlm-verified
- generated: 2026-10-11T00:04:38Z | config `a93e2d60be34`

**4 change(s) detected**: 3 removed, 1 replaced

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | replaced | wooden table leg | (-0.29, -0.24, 1.13) | 0.56 |
| 2 | removed | wooden chair leg | (-0.23, -0.13, 0.96) | 0.50 |
| 3 | removed | black chair | (-0.03, 0.03, 1.10) | 0.47 |
| 4 | removed | cracked shelf | (0.05, -0.30, 1.10) | 0.84 |

## Details

### 1. REPLACED - wooden table leg

> A close-up of a wooden table leg with a visible crack in the tiled floor beneath it.

- centroid: (-0.29, -0.24, 1.13) | confidence: 0.56
- cluster ids: [19]
- visible in: a/frame_00000.jpg

### 2. REMOVED - wooden chair leg

> the wooden chair leg was removed from the scene

- centroid: (-0.23, -0.13, 0.96) | confidence: 0.50
- cluster ids: [20]
- visible in: a/frame_00000.jpg, b/frame_01070.jpg

### 3. REMOVED - black chair

> the black chair was removed from the room

- centroid: (-0.03, 0.03, 1.10) | confidence: 0.47
- cluster ids: [21]
- visible in: a/frame_00000.jpg, b/frame_01270.jpg

### 4. REMOVED - cracked shelf

> the cracked shelf was removed from the room

- centroid: (0.05, -0.30, 1.10) | confidence: 0.84
- cluster ids: [22]
- visible in: a/frame_00000.jpg, b/frame_01040.jpg


## Unverified candidates

These regions contain object-like appearance in ONE video while the other video never observed the area (out of view, cropped or occluded). They are **not confirmed changes** - listed for transparency so nothing detected is silently dropped.

- **removed?** chair back at (-0.60, -0.35, 1.18) - dark wooden chair back with horizontal slats, partially visible next to a blue curtain.
- **removed?** door frame at (-0.61, -0.32, 1.18) - a dark wooden door frame with a light-colored baseboard and tiled floor visible at the bottom.
- **removed?** marble floor at (-0.44, 0.34, 0.90) - a white floor with black veining resembling marble or a similar stone pattern.
- **removed?** cracked ceiling at (-0.43, 0.19, 1.10) - a white ceiling with multiple dark cracks spreading across its surface.
- **removed?** cracked tile at (0.50, 0.28, 1.02) - a white tile with a visible crack near the baseboard.
- **removed?** door frame at (0.54, -0.15, 1.02) - a wooden door frame with a white door and a metal handle, set against a tiled floor.
- **removed?** cracked tile floor at (0.52, 0.40, 0.79) - a white tiled floor with visible cracks running across the surface near a dark baseboard.
- **removed?** wooden door at (0.57, -0.27, 1.05) - a close-up of a brown wooden door with vertical grain texture, partially visible next to a white cabinet.
- **added?** wooden cabinet door at (0.50, -0.02, 0.99) - a dark brown wooden cabinet door with a metallic handle, partially visible next to a white cabinet.

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
