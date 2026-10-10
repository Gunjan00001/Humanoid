"""Colored TSDF mesh export (CPU, Open3D).

Fuses the cached depth maps into a colored triangle mesh with Open3D's
ScalableTSDFVolume and exports a small file (<= `visualize.mesh_max_mb` MB,
default 10) next to the point-cloud visualizations:

    outputs/<scene>/scene_mesh.ply      (vertex-coloured triangles)

Notes
-----
- The reconstruction's scale is approximately metric (VGGT) / metric
  (MapAnything), so the TSDF voxel size in config (`visualize.mesh_voxel`,
  default 0.01) is in metres.
- Depth maps are integrated from every frame with the cached poses and
  intrinsics; the confidence filter is NOT applied here (the TSDF's own
  truncation handles outliers), but the depths are clamped to a sane range to
  keep padding/garbage out of the volume.
- Decimation (quadric) targets the size budget; the mesh is also cropped to a
  percentile-based bounding box so stray points do not blow up the volume.
- Runs entirely on CPU - no GPU job needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("mesh")


def export_mesh(cfg: dict, out_dir: Path) -> Path | None:
    import open3d as o3d

    from src.geometry import load_reconstruction

    vcfg = cfg["visualize"]
    recon = load_reconstruction(cfg)
    depth, conf = recon["depth"], recon["depth_conf"]
    intr, extr = recon["intrinsic"], recon["extrinsic"]
    colors = recon["colors"]
    s, h, w = depth.shape

    # keep only reasonably confident, plausible depths
    q = float(np.percentile(conf, float(vcfg.get("mesh_conf_percentile", 20))))
    valid = (conf >= q) & np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)
    log.info("mesh: integrating %d frames (%d valid pixels/frame median)",
             s, int(np.median(valid.sum(axis=(1, 2)))))

    voxel = float(vcfg.get("mesh_voxel", 0.01))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=4.0 * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for i in range(s):
        d = np.where(valid[i], depth[i], 0.0).astype(np.float32)
        c = np.ascontiguousarray(colors[i])
        e4 = np.eye(4)
        e4[:3, :4] = extr[i]
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(c), o3d.geometry.Image(d),
            depth_scale=1.0, depth_trunc=20.0, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, intrinsic_matrix=intr[i].astype(np.float64)), e4)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    log.info("mesh: %d triangles after TSDF (voxel %sm)", len(mesh.triangles), voxel)

    # crop to the robust scene bbox (percentiles keep stray geometry out)
    verts = np.asarray(mesh.vertices)
    lo = np.percentile(verts, 0.5, axis=0)
    hi = np.percentile(verts, 99.5, axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
    mesh = mesh.crop(bbox)

    # decimate to the size budget (roughly: ~200k triangles ~ 10 MB PLY)
    max_mb = float(vcfg.get("mesh_max_mb", 10))
    tri_budget = int(max_mb * 1024 * 1024 / 50)      # ~50 bytes/triangle in PLY
    if len(mesh.triangles) > tri_budget:
        mesh = mesh.simplify_quadric_decimation(tri_budget)
        mesh.compute_vertex_normals()
        log.info("mesh: decimated to %d triangles (budget %d)", len(mesh.triangles), tri_budget)

    path = out_dir / "scene_mesh.ply"
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    size_mb = path.stat().st_size / 1024 / 1024
    log.info("mesh: wrote %s (%d triangles, %.1f MB)", path, len(mesh.triangles), size_mb)
    return path
