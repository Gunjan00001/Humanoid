"""Stage 6 - visualization (CPU).

Colored point cloud + screenshots + a rotating GIF:

    red    = removed
    green  = added
    orange = moved (old and new position, arrow between)

    python src/visualize.py --config configs/default.yaml
    python src/visualize.py --config configs/default.yaml --view   # interactive (needs display)

Writes outputs/scene.ply, outputs/changes/change_XXX.png, outputs/change.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    add_common_args,
    apply_overrides,
    load_config,
    out_dir,
    repo_path,
    setup_logging,
)
from src.geometry import build_change_list, load_reconstruction, tagged_points, voxel_downsample  # noqa: E402

log = setup_logging("visualize")

COLORS = {
    "removed": (0.87, 0.15, 0.15),
    "added": (0.20, 0.80, 0.25),
    "moved": (1.00, 0.55, 0.05),
    "replaced": (0.85, 0.12, 0.85),
}
UNVERIFIED_COLOR = (0.10, 0.75, 0.90)   # cyan: one video never saw this area


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Minimal binary little-endian PLY writer (points + uchar rgb)."""
    arr = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (f"ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(arr.tobytes())


def scene_arrays(cfg: dict):
    """Background cloud (dim/voxelized) + per-change colored point sets."""
    vcfg = cfg["visualize"]
    recon = load_reconstruction(cfg)
    pts, rgb, tags, _ = tagged_points(recon, cfg["reconstruction"]["conf_percentile"])
    bg_pts, bg_rgb = voxel_downsample(pts, rgb, voxel=float(vcfg["background_voxel"]))
    if len(bg_pts) > int(vcfg["max_background_points"]):
        idx = np.linspace(0, len(bg_pts) - 1, int(vcfg["max_background_points"])).round().astype(int)
        bg_pts, bg_rgb = bg_pts[idx], bg_rgb[idx]
    bg_rgb = (bg_rgb.astype(np.float32) * float(vcfg["static_dim"])).astype(np.uint8)

    diff_dir = repo_path(cfg["paths"]["cache_dir"]) / "diff"
    points = np.load(diff_dir / "points.npz")
    cand = {"a": (points["a_points"], points["a_labels"]),
            "b": (points["b_points"], points["b_labels"])}

    changes = build_change_list(cfg)
    change_pts: list[dict] = []
    for ch in changes:
        color = np.array(COLORS.get(ch["type"], (0.6, 0.6, 0.6)), dtype=np.float32)
        if not ch.get("verified", True):
            color = np.array(UNVERIFIED_COLOR, dtype=np.float32)
        arrays = []
        for cid in ch["cluster_ids"]:
            for tag in ("a", "b"):
                p, lab = cand[tag]
                if lab is not None:
                    sel = p[lab == cid]
                    if len(sel):
                        arrays.append(sel)
        pts_c = np.concatenate(arrays) if arrays else np.zeros((0, 3), dtype=np.float32)
        change_pts.append({"change": ch, "points": pts_c.astype(np.float32),
                           "colors": np.tile(color, (len(pts_c), 1))})

    # remove background points that coincide with change points (radius covers
    # the lattice offset between the two downsample grids)
    if len(change_pts):
        all_c = np.concatenate([c["points"] for c in change_pts if len(c["points"])])
        if len(all_c):
            dist, _ = cKDTree(all_c).query(bg_pts, k=1, workers=-1)
            keep = dist > float(cfg["visualize"]["background_voxel"]) * 2.0
            bg_pts, bg_rgb = bg_pts[keep], bg_rgb[keep]
    return bg_pts, bg_rgb, change_pts


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--view", action="store_true", help="open the interactive Open3D viewer")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    out = out_dir(cfg)
    bg_pts, bg_rgb, change_pts = scene_arrays(cfg)
    log.info("background: %d points, %d change groups", len(bg_pts), len(change_pts))

    # ---- PLY ------------------------------------------------------------
    all_pts = [bg_pts] + [c["points"] for c in change_pts]
    all_rgb = [(bg_rgb / 255.0).astype(np.float32)] + [c["colors"] for c in change_pts]
    pts = np.concatenate(all_pts)
    cols = (np.concatenate(all_rgb) * 255).astype(np.uint8)
    write_ply(out / "scene.ply", pts, cols)
    log.info("wrote %s (%d points)", out / "scene.ply", len(pts))

    if len(pts) == 0:
        log.warning("empty scene - nothing to render")
        return 0

    scene_center = bg_pts.mean(axis=0) if len(bg_pts) else pts.mean(axis=0)
    scene_radius = float(np.linalg.norm(bg_pts - scene_center, axis=1).max()) if len(bg_pts) else 1.0

    def draw_scene(ax, highlight: dict | None = None, azimuth: float = -60):
        ax.scatter(bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2], s=1,
                   c=bg_rgb / 255.0, linewidths=0, depthshade=False)
        for c in change_pts:
            if highlight is not None and c["change"]["id"] != highlight["change"]["id"]:
                continue
            p = c["points"]
            if len(p):
                ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=6, c=c["colors"], linewidths=0)
            ch = c["change"]
            if ch["type"] == "moved":
                f, t = np.array(ch["from"]), np.array(ch["to"])
                ax.plot([f[0], t[0]], [f[1], t[1]], [f[2], t[2]], c=COLORS["moved"], lw=2)
        ax.set_box_aspect((1, 1, 1))
        r = scene_radius * 1.05
        ax.set_xlim(scene_center[0] - r, scene_center[0] + r)
        ax.set_ylim(scene_center[1] - r, scene_center[1] + r)
        ax.set_zlim(scene_center[2] - r, scene_center[2] + r)
        ax.view_init(elev=22, azim=azimuth)
        ax.set_axis_off()

    # ---- per-change screenshots ----------------------------------------
    changes_dir = out / "changes"
    changes_dir.mkdir(parents=True, exist_ok=True)
    for c in change_pts:
        ch = c["change"]
        p = c["points"]
        if ch["type"] == "moved":
            f, t = np.array(ch["from"]), np.array(ch["to"])
            focus = (f + t) / 2
            span = np.linalg.norm(t - f)
            reach = np.linalg.norm(p - focus, axis=1).max() if len(p) else 0.0
            radius = max(span * 0.75 + reach, 0.35)
        else:
            focus = np.array(ch["centroid"])
            radius = max((np.linalg.norm(p - focus, axis=1).max() if len(p) else 0.0) * 2.0, 0.35)
        near = np.linalg.norm(bg_pts - focus, axis=1) < radius * 2 if len(bg_pts) else np.zeros(0, bool)
        fig = plt.figure(figsize=(7, 6), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        if near.any():
            q = bg_pts[near]
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=1, c=bg_rgb[near] / 255.0,
                       linewidths=0, depthshade=False)
        if len(p):
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=8, c=c["colors"],
                       linewidths=0, depthshade=False)
        if ch["type"] == "moved":
            f, t = np.array(ch["from"]), np.array(ch["to"])
            ax.plot([f[0], t[0]], [f[1], t[1]], [f[2], t[2]], c=COLORS["moved"], lw=2.5)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(focus[0] - radius, focus[0] + radius)
        ax.set_ylim(focus[1] - radius, focus[1] + radius)
        ax.set_zlim(focus[2] - radius, focus[2] + radius)
        ax.view_init(elev=22, azim=-60)
        ax.set_axis_off()
        title = f"{ch['id']:02d} - {ch['type'].upper()}"
        if not ch.get("verified", True):
            title += " (UNVERIFIED)"
        if ch.get("label"):
            title += f": {ch['label']}"
        ax.set_title(title, fontsize=11)
        if ch.get("description"):
            fig.text(0.5, 0.02, ch["description"], ha="center", fontsize=9, wrap=True)
        path = changes_dir / f"change_{ch['id']:03d}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        log.info("wrote %s", path)

    # ---- rotating GIF ---------------------------------------------------
    gif_path = out / "change.gif"
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    n = int(cfg["visualize"]["gif_frames"])
    for k in range(n):
        ax.clear()
        draw_scene(ax, azimuth=-60 + 360 * k / n)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
    plt.close(fig)
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=120, loop=0)
    log.info("wrote %s (%d frames)", gif_path, n)

    if args.view:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
            o3d.visualization.draw_geometries([pcd])
        except Exception as exc:  # noqa: BLE001
            log.warning("interactive viewer unavailable (%s); scene.ply is written anyway", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
