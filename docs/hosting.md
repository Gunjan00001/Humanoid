# Hosting the interactive viewer (GitHub Pages)

Each scene ships a self-contained `outputs/<scene>/viewer.html` (≤ 5 MB,
three.js pinned from cdnjs, no build step, no server code). The simplest way to
make them publicly reachable is **GitHub Pages served from the repository
root**, the files are already committed, so no copying or CI is needed.

## Status: **ENABLED** (2026-10-11)

Build and deployment: **Deploy from a branch**, branch **`main`**, folder
**`/` (root)**. Configured by the repository owner in **Settings → Pages**; any
push to `main` re-deploys automatically and **there is no workflow file**.
Verified live: every URL in the table below returns HTTP 200.

> If the settings require a `gh` CLI instead of the web UI:
> ```bash
> gh api -X POST repos/Gunjan00001/Humanoid/pages \
>   -f 'source[branch]=main' -f 'source[path]=/'
> ```
> (Any push to `main` re-deploys automatically; there is no workflow file.)

## Live links

| Scene | Viewer |
|---|---|
| demo | https://gunjan00001.github.io/Humanoid/outputs/viewer.html |
| scene2 (tabletop) | https://gunjan00001.github.io/Humanoid/outputs/scene2/viewer.html |
| scene2 (geometric) | https://gunjan00001.github.io/Humanoid/outputs/scene2_geo/viewer.html |
| scene3 | https://gunjan00001.github.io/Humanoid/outputs/scene3/viewer.html |
| scene4 | https://gunjan00001.github.io/Humanoid/outputs/scene4/viewer.html |
| scene5 (held-out) | https://gunjan00001.github.io/Humanoid/outputs/scene5/viewer.html |
| scene5 post-hoc (tabletop) | https://gunjan00001.github.io/Humanoid/outputs/scene5_tabletop_posthoc/viewer.html |
| scene6 (held-out) | https://gunjan00001.github.io/Humanoid/outputs/scene6/viewer.html |

## Notes

- The viewers are plain static files; GitHub Pages' 1 GB site / 100 GB-month
  soft limits are far above what these use (≈ 1.8 MB each).
- They need **no network access except cdnjs** (the pinned three.js r128).
  On an air-gapped machine, download `three.min.js` once and replace the
  `<script src>` line with a local path.
- The 3D meshes (`scene_mesh.ply`) are **not** used by the viewer (it renders
  the point cloud directly); they are for offline use in MeshLab/Blender/
  Open3D.
- Local preview without Pages: `google-chrome outputs/viewer.html` or any
  static server (`python -m http.server` from the repo root).
