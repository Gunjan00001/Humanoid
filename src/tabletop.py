"""Tabletop change detection (CPU) - the appearance-based diff mode.

Why this exists: VGGT's geometry absorbs objects that are thinner than its
patch scale (~1.4 cm on the surface at this distance). On the demo scene the
comb and the pen are 1 cm thick and simply do not exist in the depth maps -
their z equals the table's. What *does* differ is appearance: every object is
clearly visible in the point colours.

This module therefore compares the two videos photometrically over the
dominant (table) plane:

1. fit the dominant plane of each cloud (iterative least squares),
2. rasterise both clouds into 5mm cells: height, mean colour, point count,
3. per cell, "objectness" = colour distance from the *local* table colour
   (median filter) - robust to the global colour cast between recordings,
4. per cell, compare the two videos:
     object in A only        -> removed
     object in B only        -> added
     different objects       -> replaced  (colour ratios differ, both present)
5. cluster changed cells (DBSCAN), drop rings (static objects show as edge
   rings), drop clusters touching the coverage boundary, keep compact blobs.

The output uses the same cluster format as the geometric diff, so
describe/visualize/report work unchanged. Coverage of *both* videos is
required per cell, which replaces the geometric visibility check here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes
from sklearn.cluster import DBSCAN

log = logging.getLogger("tabletop")


# ---------------------------------------------------------------------------
# plane + grid


def fit_dominant_plane(points: np.ndarray, thresh: float = 0.03, iters: int = 6) -> np.ndarray:
    """z = a*x + b*y + c for the dominant surface, via iterative reweighting."""
    p = points
    coef = np.array([0.0, 0.0, float(np.median(points[:, 2]))])
    for _ in range(iters):
        A = np.c_[p[:, 0], p[:, 1], np.ones(len(p))]
        coef, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
        resid = np.abs(p[:, 2] - (coef[0] * p[:, 0] + coef[1] * p[:, 1] + coef[2]))
        keep = resid < thresh
        if keep.sum() < 100:
            break
        p = p[keep]
    return coef


def plane_height(points: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return points[:, 2] - (coef[0] * points[:, 0] + coef[1] * points[:, 1] + coef[2])


def grid_bounds(points: np.ndarray, coef: np.ndarray, cell: float, margin_cells: int = 2):
    """In-plane bounding box (x, y) around the points near the plane."""
    near = np.abs(plane_height(points, coef)) < 0.05
    p = points[near] if near.sum() > 100 else points
    x0 = np.floor(np.percentile(p[:, 0], 0.5) / cell) * cell - margin_cells * cell
    x1 = np.ceil(np.percentile(p[:, 0], 99.5) / cell) * cell + margin_cells * cell
    y0 = np.floor(np.percentile(p[:, 1], 0.5) / cell) * cell - margin_cells * cell
    y1 = np.ceil(np.percentile(p[:, 1], 99.5) / cell) * cell + margin_cells * cell
    return x0, x1, y0, y1


def cell_maps(points: np.ndarray, colors: np.ndarray, coef: np.ndarray,
              cell: float, bounds: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise into (height [nx,ny], colour [nx,ny,3], count [nx,ny], point_codes [N]).

    point_codes[i] is the flat cell index of point i, or -1 for points outside
    the grid. Codes are only computed for inside points so that out-of-grid
    indices can never collide with valid cells (a raw ix*ny+iy does collide for
    negative ix/iy - that bug attached wrong points to clusters).
    """
    x0, x1, y0, y1 = bounds
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    ix = ((points[:, 0] - x0) / cell).astype(np.int64)
    iy = ((points[:, 1] - y0) / cell).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = ix[inside] * ny + iy[inside]
    n = np.bincount(flat, minlength=nx * ny).astype(np.float64)
    h = plane_height(points[inside], coef)
    H = np.bincount(flat, weights=h, minlength=nx * ny) / np.maximum(n, 1)
    C = np.zeros((nx * ny, 3))
    for ch in range(3):
        C[:, ch] = np.bincount(flat, weights=colors[inside, ch].astype(np.float64),
                               minlength=nx * ny) / np.maximum(n, 1)
    H = np.where(n > 0, H, np.nan).reshape(nx, ny)
    C = np.where(n.reshape(nx, ny, 1) > 0, C.reshape(nx, ny, 3), np.nan)
    codes = np.full(len(points), -1, dtype=np.int64)
    codes[inside] = flat
    return H, C, n.reshape(nx, ny), codes


def fit_color_field(color: np.ndarray, valid: np.ndarray, iters: int = 4,
                    thresh: float = 30.0) -> np.ndarray:
    """Per-channel robust plane fit (a + b*x + c*y) over valid cells.

    Models the table's own colour gradient so 'objectness' is insensitive to
    object size (a local median fails for objects larger than its window) and
    to illumination gradients (a single global colour would not).
    Returns the fitted colour field [nx,ny,3].
    """
    nx, ny = valid.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    sel = valid & np.isfinite(color[..., 0])
    X = np.c_[ii[sel], jj[sel], np.ones(int(sel.sum()))]
    field = np.zeros((nx, ny, 3))
    for ch in range(3):
        y = color[..., ch][sel]
        w = np.ones(len(y), dtype=bool)
        coef = np.zeros(3)
        for _ in range(iters):
            if w.sum() < 50:
                break
            coef, *_ = np.linalg.lstsq(X[w], y[w], rcond=None)
            resid = np.abs(X @ coef - y)
            w = resid < thresh
        field[..., ch] = (np.c_[ii.ravel(), jj.ravel(), np.ones(nx * ny)] @ coef).reshape(nx, ny)
    return field


def objectness(color: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Per-cell distance from the fitted table colour (0-255 RGB units)."""
    return np.linalg.norm(color - field, axis=2)


def ratio_distance(color_a: np.ndarray, color_b: np.ndarray,
                   field_a: np.ndarray, field_b: np.ndarray, floor: float = 12.0) -> np.ndarray:
    """Cast-robust comparison: how different are the two colours *relative* to
    their own fitted table colours? (used for 'replaced')."""
    ra = color_a / np.maximum(field_a, floor)
    rb = color_b / np.maximum(field_b, floor)
    return np.linalg.norm(ra - rb, axis=2) * 255.0 / 3.0


# ---------------------------------------------------------------------------
# cluster assembly


def _cells_to_points(cell_ij: np.ndarray, cloud_codes: np.ndarray, points: np.ndarray,
                     cell: float, bounds: tuple, max_per_cell: int,
                     grid_shape: tuple) -> np.ndarray:
    """Gather the actual cloud points that fall into the given cells (vectorised)."""
    x0, _, y0, _ = bounds
    ny = grid_shape[1]
    want = np.unique(cell_ij[:, 0] * ny + cell_ij[:, 1])
    pts = points[np.isin(cloud_codes, want)]     # codes are -1 for out-of-grid points
    if len(pts) == 0:
        # fall back to cell centres
        return np.c_[(cell_ij[:, 0] + 0.5) * cell + x0, (cell_ij[:, 1] + 0.5) * cell + y0,
                     np.zeros(len(cell_ij))]
    if max_per_cell and len(pts) > max_per_cell * len(cell_ij):
        idx = np.random.default_rng(0).choice(len(pts), max_per_cell * len(cell_ij), replace=False)
        pts = pts[idx]
    return pts


def _prepare(recon: dict[str, Any], cfg: dict[str, Any]):
    """Shared preprocessing for detect() and static_object_mask().

    Steps (all CPU, all vectorised):
      1. load the tagged cloud with the confidence filter applied;
      2. fit the dominant plane of EACH video separately (the table) - this
         absorbs any global tilt/offset between the two reconstructions;
      3. rasterise both clouds into a common cell grid (5mm default):
         height above the plane, mean colour, point count per cell;
      4. fit a smooth per-channel colour field for each video (the table's own
         colour gradient) and measure per-cell *objectness* = colour distance
         from that field.  An object is 'colourful compared to the local table'
         no matter whether it is 2cm or 20cm wide;
      5. cast-robust ratio distance: colour relative to each video's own table
         colour - this is what tells a real appearance change (black->white)
         from the global colour-cast difference between the recordings.
    """
    from src.geometry import tagged_points

    tcfg = cfg["diff"]["tabletop"]
    cell = float(tcfg["cell_size"])
    conf_q = cfg["reconstruction"]["conf_percentile"]

    pts, rgb, tags, _ = tagged_points(recon, conf_q)
    a_mask, b_mask = tags == "a", tags == "b"
    A, CA = pts[a_mask], rgb[a_mask]
    B, CB = pts[b_mask], rgb[b_mask]

    # --- one JOINT plane for both videos -------------------------------
    # The cells of A and B must correspond spatially, so a single plane is the
    # correct choice. Two-stage fit: coarse fit on the joint cloud, then a
    # refit on the dense surface band only. This keeps the plane on the TABLE
    # even in room-scale clouds (MapAnything), where a plain iterative fit can
    # converge to a wall/floor blend (observed: A and B fits disagreed by 10
    # degrees on the demo scene before this refinement).
    joint = np.concatenate([A, B])
    coef = fit_dominant_plane(joint)
    near = np.abs(plane_height(joint, coef)) < 0.08
    if near.sum() > 10_000:
        coef = fit_dominant_plane(joint[near])
    log.debug("joint plane: %s (refined on %d surface points)",
              np.round(coef, 4).tolist(), int(near.sum()))

    # Restrict to a band around the fitted plane. The tabletop mode analyses
    # ONE surface: without the band, points from walls/floor that share the
    # x/y footprint (common with room-scale reconstructions like MapAnything's)
    # would pollute the per-cell colours. Objects up to `height_band` tall
    # stay included; taller objects are documented as clipped (their base and
    # edges still contribute).
    band = float(tcfg.get("height_band", 0.15))
    ha, hb = plane_height(A, coef), plane_height(B, coef)
    A, CA = A[np.abs(ha) <= band], CA[np.abs(ha) <= band]
    B, CB = B[np.abs(hb) <= band], CB[np.abs(hb) <= band]
    log.debug("height band %.2fm: A=%d B=%d points remain", band, len(A), len(B))

    bounds = grid_bounds(np.concatenate([A, B]), coef, cell)
    log.debug("grid bounds: %s (cell %.0fmm)", np.round(bounds, 3).tolist(), cell * 1000)
    HA, CMA, NA, codesA = cell_maps(A, CA, coef, cell, bounds)
    HB, CMB, NB, codesB = cell_maps(B, CB, coef, cell, bounds)

    min_pts = int(tcfg["min_cell_points"])
    obsA = NA >= min_pts
    obsB = NB >= min_pts
    fieldA = fit_color_field(CMA, obsA)
    fieldB = fit_color_field(CMB, obsB)
    objA = objectness(CMA, fieldA)
    objB = objectness(CMB, fieldB)
    ratio = ratio_distance(CMA, CMB, fieldA, fieldB)
    return (A, CA, B, CB, HA, CMA, NA, codesA, HB, CMB, NB, codesB,
            objA, objB, ratio, bounds, cell)


def static_object_mask(recon: dict[str, Any], cfg: dict[str, Any]):
    """Cells where BOTH videos show an object (objA > T and objB > T).

    These are locations of objects that exist in both recordings (timer,
    highlighter, ...). Candidate fragments adjacent to them are edges/lighting
    differences of a static object, not changes; describe.py suppresses those.
    Returns (mask, bounds, cell_size).
    """
    (_, _, _, _, _, _, _, _, _, _, _, _, objA, objB, _, bounds, cell) = _prepare(recon, cfg)
    T = float(cfg["diff"]["tabletop"]["objectness_threshold"])
    return (objA > T) & (objB > T), bounds, cell


def detect(recon: dict[str, Any], cfg: dict[str, Any]) -> list[dict]:
    """Run the tabletop photometric diff. Returns kept clusters (pipeline format)."""
    (A, CA, B, CB, HA, CMA, NA, codesA, HB, CMB, NB, codesB,
     objA, objB, ratio, bounds, cell) = _prepare(recon, cfg)
    tcfg = cfg["diff"]["tabletop"]
    log.info("tabletop: A=%d B=%d points | grid %d x %d cells (%.0fmm)",
             len(A), len(B), HA.shape[0], HA.shape[1], cell * 1000)

    min_pts = int(tcfg["min_cell_points"])
    obsA = NA >= min_pts
    obsB = NB >= min_pts
    obs = obsA & obsB

    T_obj = float(tcfg["objectness_threshold"])
    D = float(tcfg["objectness_margin"])
    T_ratio = float(tcfg["ratio_threshold"])
    # change = one video's objectness clearly exceeds the other's, not merely
    # "high in absolute terms" (a lighting patch can be high in both videos)
    m_removed = (objA - objB > D) & (objA > T_obj)
    m_added = (objB - objA > D) & (objB > T_obj)
    # an object in both videos whose appearance differs strongly -> replaced
    # (e.g. the remote flipped over: dark back in A, white face in B)
    m_replaced = (objA > T_obj) & (objB > T_obj) & (ratio > T_ratio)

    # verified: both videos observed the cell, away from coverage boundaries
    erode = int(tcfg["boundary_erode_cells"])

    def _erode(mask: np.ndarray, k: int) -> np.ndarray:
        # note: scipy treats iterations<1 as "repeat until stable", so 0 must
        # be handled explicitly
        return mask if k <= 0 else binary_erosion(mask, iterations=k)

    common = _erode(obs, erode)
    # unverified: an object-like cell in one video where the other video has no
    # coverage at all (portrait crop / occluded). Kept separately and clearly
    # flagged - never mixed into the verified change list.
    obsA_core = _erode(obsA, erode)
    obsB_core = _erode(obsB, erode)
    m_removed_unv = (objA > T_obj) & obsA_core & ~obs
    m_added_unv = (objB > T_obj) & obsB_core & ~obs
    log.info("observed-both cells: %d | removed=%d added=%d replaced=%d | unverified r=%d a=%d",
             int(common.sum()), int((m_removed & common).sum()),
             int((m_added & common).sum()), int((m_replaced & common).sum()),
             int(m_removed_unv.sum()), int(m_added_unv.sum()))

    clusters: list[dict] = []
    next_id = 1
    for mask, ctype, verified, reason in (
        (m_removed & common, "removed", True, "photometric mode: cell observed in both videos"),
        (m_added & common, "added", True, "photometric mode: cell observed in both videos"),
        (m_replaced & common, "replaced", True, "photometric mode: cell observed in both videos"),
        (m_removed_unv, "removed", False, "video B never observed this area (occluded or out of view)"),
        (m_added_unv, "added", False, "video A never observed this area (occluded or out of view)"),
    ):
        cells = np.argwhere(mask)
        if len(cells) == 0:
            continue
        labels = DBSCAN(eps=float(tcfg["dbscan_eps_cells"]),
                        min_samples=int(tcfg["dbscan_min_samples"]),
                        ).fit_predict(cells)
        for lab in sorted(set(labels.tolist())):
            if lab == -1:
                continue
            cell_ij = np.argwhere(mask)[labels == lab]
            grid = np.zeros_like(mask, dtype=bool)
            grid[cell_ij[:, 0], cell_ij[:, 1]] = True
            filled = binary_fill_holes(grid)
            hole_ratio = 1.0 - grid.sum() / max(filled.sum(), 1)
            min_cells = int(tcfg["min_cells"] if verified else tcfg["min_cells_unverified"])
            log.debug("  candidate %s: %d cells, hole_ratio %.2f (min_cells %d)",
                      ctype, grid.sum(), hole_ratio, min_cells)
            if grid.sum() < min_cells:
                log.info("  drop %s cluster: %d cells < min_cells", ctype, grid.sum())
                continue
            # a *ring* of changed cells is how a static object looks in the XOR
            # masks (its interior is unchanged, only its edges differ because
            # the two videos blur/expose it differently) - drop those
            if hole_ratio > float(tcfg["ring_hole_ratio"]):
                log.info("  drop %s cluster: ring shape (hole ratio %.2f)", ctype, hole_ratio)
                continue

            if ctype == "added":
                src_pts, src_codes = B, codesB
            else:  # removed, replaced and unverified-removed are anchored in video A
                src_pts, src_codes = A, codesA
            sample = _cells_to_points(cell_ij, src_codes, src_pts, cell, bounds,
                                      int(tcfg["max_points_per_cell"]), HA.shape)

            score = np.abs(objA - objB)[cell_ij[:, 0], cell_ij[:, 1]].mean() \
                if ctype != "replaced" else ratio[cell_ij[:, 0], cell_ij[:, 1]].mean()
            if not np.isfinite(score):   # unverified: the other video has no data here
                own = objA if ctype == "removed" else objB
                score = float(own[cell_ij[:, 0], cell_ij[:, 1]].mean())
            entry = {
                "id": next_id,
                "type": ctype,
                "source_tag": "both" if ctype == "replaced" else ("a" if ctype == "removed" else "b"),
                "n_points": int(len(sample)),
                "n_cells": int(grid.sum()),
                "centroid": [round(float(v), 4) for v in sample.mean(axis=0)],
                "bbox_min": [round(float(v), 4) for v in sample.min(axis=0)],
                "bbox_max": [round(float(v), 4) for v in sample.max(axis=0)],
                "extent": [round(float(v), 4) for v in (sample.max(axis=0) - sample.min(axis=0))],
                "mean_change_score": round(float(score), 1),
                "verified": bool(verified),
                "visibility": {"checked": bool(verified), "kept": True, "reason": reason},
                "sample_points": [[round(float(v), 4) for v in p]
                                  for p in sample[:200]],
            }
            if ctype == "replaced":
                other = _cells_to_points(cell_ij, codesB, B, cell, bounds,
                                         int(tcfg["max_points_per_cell"]), HA.shape)
                entry["sample_points_other"] = [[round(float(v), 4) for v in p]
                                                for p in other[:200]]
            log.info("  keep %s #%d: %d cells, score %.0f, centroid %s (cells at %.3f, %.3f)",
                     ctype, next_id, grid.sum(), score, entry["centroid"],
                     bounds[0] + (cell_ij[:, 0].mean() + 0.5) * cell,
                     bounds[2] + (cell_ij[:, 1].mean() + 0.5) * cell)
            clusters.append(entry)
            next_id += 1

    log.info("tabletop diff kept %d cluster(s)", len(clusters))
    return clusters
