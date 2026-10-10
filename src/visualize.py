"""Stage 6 - visualization (CPU).

Renders the scene the way the Step-0 diagnosis showed it should be: the
reconstruction is faithful, so the display just has to stop hiding it.

Design (see docs/results.md, "Visualization rework"):
  - a *display* cloud with its own settings (`visualize.*`): lower confidence
    cutoff, finer voxels, true colours barely dimmed - the detection pipeline
    keeps its own (stricter) settings and is unaffected;
  - dark neutral background, point splats sized by distance to the camera;
  - the camera is framed on the changed region, not the whole scene;
  - per confirmed change: a 3D bounding box + text label; a legend; unverified
    candidates only as a dim cyan style;
  - per-change PNG: before frame | after frame | 3D view with highlights;
  - scene GIF: the same three-panel layout with the 3D panel orbiting;
  - optional colored TSDF mesh (`src/mesh.py`) and a self-contained
    interactive HTML viewer (`src/viewer.py`).

    python src/visualize.py --config configs/default.yaml
    python src/visualize.py --config configs/default.yaml --view   # Open3D viewer
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
    read_json,
    repo_path,
    setup_logging,
)
from src.geometry import build_change_list, load_reconstruction, tagged_points, voxel_downsample  # noqa: E402

log = setup_logging("visualize")

COLORS = {
    "removed": (0.95, 0.20, 0.20),
    "added": (0.25, 0.90, 0.35),
    "moved": (1.00, 0.60, 0.10),
    "replaced": (0.95, 0.25, 0.95),
}
UNVERIFIED_COLOR = (0.35, 0.75, 0.85)
BG_COLOR = "#101418"
LABEL_COLOR = "#f2f2f2"


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


# ---------------------------------------------------------------------------
# scene data


def scene_arrays(cfg: dict):
    """Display cloud + per-change point sets.

    Uses the *display* settings (visualize.conf_percentile / voxel_size /
    static_dim), never the detection ones.
    """
    vcfg = cfg["visualize"]
    recon = load_reconstruction(cfg)
    pts, rgb, tags, _ = tagged_points(recon, float(vcfg["conf_percentile"]))
    log.info("display cloud: %d points at conf>p%s", len(pts), vcfg["conf_percentile"])
    bg_pts, bg_rgb, bg_tags = voxel_downsample(pts, rgb, tags,
                                               voxel=float(vcfg["voxel_size"]))
    if len(bg_pts) > int(vcfg["max_background_points"]):
        idx = np.linspace(0, len(bg_pts) - 1,
                          int(vcfg["max_background_points"])).round().astype(int)
        bg_pts, bg_rgb, bg_tags = bg_pts[idx], bg_rgb[idx], bg_tags[idx]
    bg_rgb = (bg_rgb.astype(np.float32) * float(vcfg["static_dim"])).astype(np.uint8)
    log.info("display cloud after %sm voxels: %d points",
             vcfg["voxel_size"], len(bg_pts))

    diff_dir = repo_path(cfg["paths"]["cache_dir"]) / "diff"
    points = np.load(diff_dir / "points.npz")
    cand = {"a": (points["a_points"], points["a_labels"]),
            "b": (points["b_points"], points["b_labels"])}

    changes = build_change_list(cfg)
    change_pts: list[dict] = []
    for ch in changes:
        verified = ch.get("verified", True)
        color = np.array(COLORS.get(ch["type"], (0.6, 0.6, 0.6)), dtype=np.float32)
        if not verified:
            color = np.array(UNVERIFIED_COLOR, dtype=np.float32) * 0.6
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
                           "colors": np.tile(color, (len(pts_c), 1)),
                           "verified": verified})

    # remove display-cloud points that coincide with change points
    if len(change_pts):
        all_c = np.concatenate([c["points"] for c in change_pts if len(c["points"])])
        if len(all_c):
            dist, _ = cKDTree(all_c).query(bg_pts, k=1, workers=-1)
            keep = dist > float(vcfg["voxel_size"]) * 2.0
            bg_pts, bg_rgb, bg_tags = bg_pts[keep], bg_rgb[keep], bg_tags[keep]
    return bg_pts, bg_rgb, bg_tags, change_pts


def change_framing(change_pts: list[dict], bg_pts: np.ndarray):
    """Camera focus + radius on the union of the confirmed changes' extents."""
    verified = [c for c in change_pts if c["verified"] and len(c["points"])]
    if verified:
        p = np.concatenate([c["points"] for c in verified])
        lo, hi = p.min(axis=0), p.max(axis=0)
        focus = (lo + hi) / 2
        radius = float(np.linalg.norm(hi - lo)) / 2 + 0.25
        return focus, max(radius, 0.35)
    if len(bg_pts):                                   # no changes: whole scene
        lo, hi = bg_pts.min(axis=0), bg_pts.max(axis=0)
        focus = (lo + hi) / 2
        return focus, float(np.linalg.norm(hi - lo)) / 2 + 0.05
    return np.zeros(3), 1.0


# ---------------------------------------------------------------------------
# 3D rendering


def _camera_pos(focus: np.ndarray, radius: float, azimuth: float, elev: float) -> np.ndarray:
    a, e = np.radians(azimuth), np.radians(elev)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return focus - d * (2.6 * radius)


def draw_3d(ax, bg_pts, bg_rgb, change_pts, focus, radius, azimuth, elev,
            splat: float, show_labels: bool = True,
            show_unverified: bool = False) -> None:
    """Dark background, distance-sized splats, change boxes + labels, legend."""
    ax.set_facecolor(BG_COLOR)
    cam = _camera_pos(focus, radius, azimuth, elev)

    def splat_sizes(points: np.ndarray, base: float) -> np.ndarray:
        """Marker size proportional to 1/distance (nearer points look bigger)."""
        if not len(points):
            return np.zeros(0)
        d = np.linalg.norm(points - cam, axis=1)
        return np.clip(base * (2.2 * radius) / np.maximum(d, 1e-6), 1.0, 4.0 * base)

    if len(bg_pts):
        ax.scatter(bg_pts[:, 0], bg_pts[:, 1], bg_pts[:, 2],
                   s=splat_sizes(bg_pts, splat), c=bg_rgb / 255.0,
                   linewidths=0, depthshade=False)

    for c in change_pts:
        p, ch = c["points"], c["change"]
        if len(p):
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=splat_sizes(p, splat * 1.6),
                       c=c["colors"], linewidths=0, depthshade=False)
        unverified = not c["verified"]
        if unverified and not show_unverified:
            continue
        # bounding box wireframe
        lo = np.asarray(ch["bbox_min"], dtype=float)
        hi = np.asarray(ch["bbox_max"], dtype=float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
                 (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        color = UNVERIFIED_COLOR if unverified else COLORS.get(ch["type"], (0.8, 0.8, 0.8))
        for e in edges:
            a, b = corners[e[0]], corners[e[1]]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], c=color, lw=1.2, alpha=0.9)
        if ch["type"] == "moved" and ch.get("from") and ch.get("to"):
            # show the destination of a moved object
            fr = np.asarray(ch["from"], dtype=float)
            to = np.asarray(ch["to"], dtype=float)
            ax.plot([fr[0], to[0]], [fr[1], to[1]], [fr[2], to[2]], c=color,
                    lw=1.6, ls="--", alpha=0.95)
            ax.scatter([to[0]], [to[1]], [to[2]], s=splat * 2.2, c=[color],
                       marker="o", linewidths=0, depthshade=False)
        if show_labels and not unverified:
            label = f"{ch['id']}: {ch['type']} - {ch.get('label') or '?'}"
            ax.text(lo[0], lo[1], hi[2] + 0.02 * max(radius, 0.3), label,
                    color=LABEL_COLOR, fontsize=8, zorder=10, clip_on=False,
                    bbox=dict(facecolor=color, alpha=0.55, edgecolor="none", pad=1.5))

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(focus[0] - radius, focus[0] + radius)
    ax.set_ylim(focus[1] - radius, focus[1] + radius)
    ax.set_zlim(focus[2] - radius, focus[2] + radius)
    ax.view_init(elev=elev, azim=azimuth)
    ax.set_axis_off()
    ax.grid(False)


def legend_handles(change_pts: list[dict]):
    """Proxy artists for the figure-level legend (outside the 3D panel)."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[t], label=t)
               for t in ("removed", "added", "moved", "replaced")]
    if any(not c["verified"] for c in change_pts):
        handles.append(Line2D([], [], marker="o", ls="", color=UNVERIFIED_COLOR,
                              label="unverified"))
    return handles


def table_up_rotation(bg_pts: np.ndarray) -> np.ndarray:
    """Rotation aligning the dominant (table) plane normal with +z.

    The camera then orbits with the table's own up axis as 'up', which makes
    the scene read like the real tabletop instead of an arbitrary model frame.
    """
    from src.tabletop import fit_dominant_plane
    sub = bg_pts[:: max(1, len(bg_pts) // 50000)] if len(bg_pts) else bg_pts
    coef = fit_dominant_plane(sub)
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = float(np.linalg.norm(v))
    c = float(n @ z)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s ** 2)


def check_render_caches(cfg: dict) -> list[str]:
    """Required cache paths for --render-only that are missing (empty = ready).

    Render-only never recomputes a pipeline stage: if anything the rendering
    needs is absent, the run stops and lists what is missing.
    """
    from src.geometry import recon_dir_for
    cache = repo_path(cfg["paths"]["cache_dir"])
    rdir = recon_dir_for(cfg, prefer_aligned=True)
    required = [
        ("reconstruction (meta)", rdir / "meta.json"),
        ("reconstruction (cameras)", rdir / "cameras.npz"),
        ("reconstruction (depth/conf)", rdir / "dense.npz"),
        ("reconstruction (colors)", rdir / "colors.npz"),
        ("diff points", cache / "diff" / "points.npz"),
        ("diff clusters", cache / "diff" / "clusters.json"),
        ("VLM confirmations", cache / "semantics" / "confirmed.json"),
        ("VLM labels", cache / "semantics" / "vlm.json"),
        ("crop evidence", cache / "semantics" / "crops.json"),
    ]
    return [f"{name}: {path}" for name, path in required if not path.exists()]


def load_evidence_frame(cfg: dict, file: str | None):
    """Load an evidence JPEG (relative to cache/frames) for the side panels."""
    if not file:
        return None
    from PIL import Image
    path = repo_path(cfg["paths"]["cache_dir"]) / "frames" / file
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--view", action="store_true", help="open the interactive Open3D viewer")
    parser.add_argument("--render-only", action="store_true",
                        help="reuse the cached reconstruction/diff/VLM outputs and "
                             "only re-render figures, GIF, viewer and mesh; stops "
                             "with a list when a required cache is missing (never "
                             "recomputes a pipeline stage)")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    if args.render_only:
        missing = check_render_caches(cfg)
        if missing:
            log.error("--render-only: %d required cache(s) missing - nothing was "
                      "recomputed:", len(missing))
            for m in missing:
                log.error("  missing %s", m)
            log.error("run the pipeline stages first "
                      "(extract_frames / reconstruct / align / diff / describe)")
            return 2
        log.info("--render-only: caches present (reconstruction, diff, VLM) - "
                 "rendering from cache only")

    vcfg = cfg["visualize"]
    out = out_dir(cfg)
    bg_pts, bg_rgb, bg_tags, change_pts = scene_arrays(cfg)

    # table-plane-up frame: rotate everything so the dominant (table) normal
    # is +z; the orbit then has the table's own up axis as "up"
    rot = table_up_rotation(bg_pts)
    bg_pts = bg_pts @ rot.T
    for c in change_pts:
        c["points"] = c["points"] @ rot.T
        ch = c["change"]
        for key in ("centroid", "from", "to"):
            if ch.get(key) is not None:
                ch[key] = (np.asarray(ch[key], dtype=float) @ rot.T).round(4).tolist()
        if ch.get("bbox_min") is not None:
            # rotate the AABB properly: all 8 corners, then min/max - rotating
            # just the two corners swaps the component-wise order (negative
            # extents broke the viewer's Box3Helper)
            lo, hi = np.asarray(ch["bbox_min"], float), np.asarray(ch["bbox_max"], float)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ rot.T
            ch["bbox_min"] = corners.min(axis=0).round(4).tolist()
            ch["bbox_max"] = corners.max(axis=0).round(4).tolist()

    focus, radius = change_framing(change_pts, bg_pts)
    # the viewer embeds the FULL rotated cloud (its "full scene" toggle and
    # its own crop-by-drawRange need the whole geometry)
    bg_full = (bg_pts, bg_rgb, bg_tags)

    # focused display crop for the renders: ~0.4 m around the changed region
    crop_r = float(vcfg.get("display_crop_radius", 0.4))
    if any(c["verified"] for c in change_pts):
        d = np.linalg.norm(bg_pts - focus, axis=1)
        keep = d <= crop_r
        log.info("display crop: keeping %d/%d points within %.2f m of the changes",
                 int(keep.sum()), len(bg_pts), crop_r)
        bg_pts, bg_rgb, bg_tags = bg_pts[keep], bg_rgb[keep], bg_tags[keep]
    log.info("framing on %s (radius %.2f m, table-up)", np.round(focus, 2).tolist(), radius)

    # ---- PLY ------------------------------------------------------------
    all_pts = [bg_pts] + [c["points"] for c in change_pts]
    all_rgb = [(bg_rgb / 255.0).astype(np.float32)] + [c["colors"] for c in change_pts]
    pts = np.concatenate(all_pts) if all_pts else np.zeros((0, 3), np.float32)
    cols = (np.concatenate(all_rgb) * 255).astype(np.uint8) if all_rgb else \
        np.zeros((0, 3), np.uint8)
    write_ply(out / "scene.ply", pts, cols)
    log.info("wrote %s (%d points)", out / "scene.ply", len(pts))

    if len(pts) == 0:
        log.warning("empty scene - nothing to render")
        return 0

    splat = float(vcfg.get("splat_size", 14))
    elev = float(vcfg.get("orbit_elev", 22))

    # ---- per-change composites: before | after | 3D ----------------------
    changes_dir = out / "changes"
    changes_dir.mkdir(parents=True, exist_ok=True)
    for stale in changes_dir.glob("change_*.png"):
        stale.unlink()
    crops = {}
    crops_path = repo_path(cfg["paths"]["cache_dir"]) / "semantics" / "crops.json"
    if crops_path.exists():
        crops = read_json(crops_path)
    for c in change_pts:
        ch = c["change"]
        ci = crops.get(str(ch["cluster_ids"][0]), {})
        before = load_evidence_frame(cfg, ci.get("frame"))
        after = load_evidence_frame(cfg, ci.get("frame_other"))
        # two rows: [before | after] on top (larger), a large 3D panel below
        fig = plt.figure(figsize=(12, 11), dpi=110, facecolor=BG_COLOR)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.5],
                              hspace=0.06, wspace=0.02)
        for k, (img, title) in enumerate(((before, "before (video A)"),
                                          (after, "after (video B)"))):
            ax = fig.add_subplot(gs[0, k])
            ax.set_facecolor(BG_COLOR)
            if img is not None:
                ax.imshow(img)
            ax.set_title(title, color=LABEL_COLOR, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#333")
        ax3 = fig.add_subplot(gs[1, :], projection="3d")
        ch_focus = np.asarray(ch["centroid"], dtype=float)
        ch_radius = max(float(np.linalg.norm(
            np.asarray(ch["bbox_max"]) - np.asarray(ch["bbox_min"]))) * 1.4, 0.3)
        draw_3d(ax3, bg_pts, bg_rgb, [c], ch_focus, ch_radius, -52, elev, splat,
                show_unverified=True)
        fig.legend(handles=legend_handles([c]), loc="lower center",
                   ncol=5, framealpha=0.15, labelcolor=LABEL_COLOR, fontsize=9)
        title = f"{ch['id']:02d} - {ch['type'].upper()}"
        if not c["verified"]:
            title += " (UNVERIFIED)"
        if ch.get("label"):
            title += f": {ch['label']}"
        fig.suptitle(title + (f"   |   {ch['description']}" if ch.get("description") else ""),
                     color=LABEL_COLOR, fontsize=12)
        fig.subplots_adjust(top=0.92, bottom=0.07, left=0.02, right=0.98)
        path = changes_dir / f"change_{ch['id']:03d}.png"
        fig.savefig(path, facecolor=BG_COLOR)
        plt.close(fig)
        log.info("wrote %s", path)
    # stale-screenshot cleanup: keep only files whose id is in report.json;
    # when the report is from an older run (ids differ), the current change
    # list is authoritative so fresh screenshots are never deleted
    valid_ids = {c["change"]["id"] for c in change_pts}
    keep_ids, source = valid_ids, "the current change list"
    report_path = out / "report.json"
    if report_path.exists():
        try:
            rep = read_json(report_path)
            rep_ids = {ch["id"] for ch in rep.get("changes", [])}
            rep_ids |= {ch["id"] for ch in rep.get("unverified_changes", [])}
            if rep_ids and rep_ids == valid_ids:
                keep_ids, source = rep_ids, "report.json"
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read %s (%s) - using the current list",
                        report_path, exc)
    removed = 0
    for png in changes_dir.glob("change_*.png"):
        try:
            cid = int(png.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if cid not in keep_ids:
            png.unlink()
            removed += 1
    if removed:
        log.info("stale-screenshot cleanup: removed %d file(s) not in %s",
                 removed, source)

    if vcfg.get("export_viewer", False):
        try:
            from src.viewer import write_viewer
            write_viewer(cfg, out, bg_full[0], bg_full[1], bg_full[2], change_pts,
                         focus=focus, radius=radius)
        except Exception as exc:  # noqa: BLE001
            log.warning("viewer export skipped: %s", exc)

    # ---- scene GIF: rendered from viewer.html via headless Chrome ---------
    gif_path = out / "change.gif"
    headline = next((c for c in change_pts if c["verified"]),
                    change_pts[0] if change_pts else None)
    ci = crops.get(str(headline["change"]["cluster_ids"][0]), {}) if headline else {}
    before_file, after_file = ci.get("frame"), ci.get("frame_other")

    def _abs(file):
        return str(repo_path(cfg["paths"]["cache_dir"]) / "frames" / file) if file else None

    made = None
    if vcfg.get("chrome_gif", True):
        try:
            from src.render_gif import render_gif
            made = render_gif(cfg, out, _abs(before_file), _abs(after_file),
                              frames=int(vcfg["gif_frames"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("chrome GIF failed (%s) - falling back to matplotlib", exc)
    if made is None:
        before = load_evidence_frame(cfg, before_file)
        after = load_evidence_frame(cfg, after_file)
        fig = plt.figure(figsize=(12, 10.5), dpi=80, facecolor=BG_COLOR)
        frames = []
        n = int(vcfg["gif_frames"])
        for k in range(n):
            fig.clf()
            gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5],
                                  hspace=0.06, wspace=0.02)
            for j, (img, title) in enumerate(((before, "before (video A)"),
                                              (after, "after (video B)"))):
                ax = fig.add_subplot(gs[0, j])
                ax.set_facecolor(BG_COLOR)
                if img is not None:
                    ax.imshow(img)
                ax.set_title(title, color=LABEL_COLOR, fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#333")
            ax3 = fig.add_subplot(gs[1, :], projection="3d")
            draw_3d(ax3, bg_pts, bg_rgb, change_pts, focus, radius,
                    -52 + 360 * k / n, elev, splat)
            fig.legend(handles=legend_handles(change_pts), loc="lower center",
                       ncol=5, framealpha=0.15, labelcolor=LABEL_COLOR, fontsize=9)
            fig.subplots_adjust(top=0.94, bottom=0.07, left=0.02, right=0.98)
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)
        from PIL import Image

        from src.render_gif import gif_quantize
        imgs = [gif_quantize(Image.fromarray(f)) for f in frames]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=150, loop=0, optimize=False)
        log.info("wrote %s (%d frames, matplotlib fallback)", gif_path, n)

    # ---- optional mesh + interactive viewer ------------------------------
    if vcfg.get("export_mesh", False):
        try:
            from src.mesh import export_mesh
            export_mesh(cfg, out)
        except Exception as exc:  # noqa: BLE001 - never fail the stage over extras
            log.warning("mesh export skipped: %s", exc)
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
