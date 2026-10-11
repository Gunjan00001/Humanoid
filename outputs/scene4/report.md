# Change report

- video A: `/home/user/Desktop/Humanoid/data/scene4/a.mp4`
- video B: `/home/user/Desktop/Humanoid/data/scene4/b.mp4`
- reconstruction: `vggt` (/home/user/models/VGGT-1B)
- frames: 60 | semantics: vlm-verified
- generated: 2026-10-11T00:16:19Z | config `a93e2d60be34`

**3 change(s) detected**: 1 added, 1 moved, 1 removed

| # | type | label | centroid (x, y, z) | confidence |
|---|---|---|---|---|
| 1 | moved | apple | (-0.01, -0.36, 1.06) | 0.76 |
| 2 | removed | scissors | (0.17, -0.19, 0.97) | 0.60 |
| 3 | added | plastic bottle | (-0.01, -0.36, 1.06) | 0.75 |

## Details

### 1. MOVED - apple

> the apple moved slightly to the right and upward

- from: (-0.01, -0.36, 1.06)
- to:   (0.19, -0.14, 0.95)

- centroid: (-0.01, -0.36, 1.06) | confidence: 0.76
- cluster ids: [2, 6]
- visible in: a/frame_00690.jpg, b/frame_00210.jpg

### 2. REMOVED - scissors

> The scissors were removed and replaced with an apple.

- centroid: (0.17, -0.19, 0.97) | confidence: 0.60
- cluster ids: [3]
- visible in: a/frame_00000.jpg, b/frame_00210.jpg

### 3. ADDED - plastic bottle

> a plastic bottle appeared where the apple was before it moved

- centroid: (-0.01, -0.36, 1.06) | confidence: 0.75
- cluster ids: [900]


## Unverified candidates

These regions contain object-like appearance in ONE video while the other video never observed the area (out of view, cropped or occluded). They are **not confirmed changes** - listed for transparency so nothing detected is silently dropped.

- **removed?** concrete floor at (-0.33, -0.05, 0.86) - dark, textured surface of a concrete floor with minor debris and discoloration.
- **removed?** hole in floor at (-0.32, -0.11, 0.91) - a circular hole in a dark floor surface, possibly from damage or wear.
- **removed?** concrete wall at (0.34, 0.09, 0.81) - a textured, gray concrete wall surface with minor imperfections and a vertical edge on the right.
- **added?** metal surface at (-0.33, -0.29, 1.02) - a close-up of a smooth, gray metallic surface with a small circular indentation at the bottom.
- **added?** metal door frame at (0.32, -0.51, 1.18) - a close-up of a metallic door frame with a glass panel, showing part of a hinge and surrounding wall.
- **added?** metal railing at (0.32, -0.44, 1.14) - a curved metal railing with a dark finish, likely part of a staircase or balcony, visible against a blurred background.
- **added?** metal chair leg at (0.41, -0.49, 1.18) - a curved metal leg of a chair, partially visible with a shiny surface.

---

Notes:
- confidence is a heuristic in [0,1]: cluster size and change-score (photometric candidates), DINOv2 crop similarity for moved objects; unverified candidates (one video never saw the area) are capped at 0.3. It is not a calibrated probability.
- candidates are produced photometrically over the table plane (`diff.method: tabletop`, see src/tabletop.py) or geometrically (`geometric`); verified candidates are confirmed by the VLM against before/after crops, and moved objects come from DINOv2 crop matching.
- see docs/pipeline.md for the full description.
