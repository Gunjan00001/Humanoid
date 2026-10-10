#!/usr/bin/env python
"""Inspect every cached artifact of the pipeline - the first debugging tool.

    python scripts/inspect_cache.py                          # default cache/
    python scripts/inspect_cache.py --cache cache/gpu_smoke  # any cache dir

Prints, for each stage that has run:
  - what the artifact contains (counts, shapes, statistics),
  - a per-cluster table with the change-detector's and the VLM's decisions,
  - consistency checks (e.g. does every kept cluster have a crop on disk?),
  - the exact command that produces the stage if it has not run yet.

This script only READS. It never modifies the cache, so it is always safe to
run - see docs/debugging.md for the full debugging workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, repo_path  # noqa: E402


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - inspection must never crash
        return {"__error__": f"cannot read {path.name}: {exc}"}


def _fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def inspect_frames(cache: Path) -> None:
    section("1. frames  (cache/frames/index.json)")
    index = cache / "frames" / "index.json"
    if not index.exists():
        print("not produced yet -> python src/extract_frames.py --config configs/default.yaml")
        return
    doc = _read_json(index)
    for tag in ("a", "b"):
        v = doc.get("videos", {}).get(tag, {})
        frames = v.get("frames", [])
        blurs = [f.get("blur", 0) for f in frames]
        on_disk = len(list((cache / "frames" / tag).glob("frame_*.jpg")))
        print(f"  video {tag}: {v.get('path')}")
        print(f"    sampled candidates {v.get('candidates')} -> kept {len(frames)}"
              f"  (on disk: {on_disk})")
        if blurs:
            print(f"    blur  min {min(blurs):.0f}  median {np.median(blurs):.0f}"
                  f"  max {max(blurs):.0f}   (threshold in config: frames.blur_threshold)")
        if on_disk != len(frames):
            print("    !! disk/index mismatch - re-run with --force (stale files)")


def inspect_reconstruction(cache: Path, cfg: dict) -> None:
    section("2. reconstruction  (cache/reconstruction/)")
    rdir = cache / "reconstruction"
    if not (rdir / "meta.json").exists():
        print("not produced yet -> python src/reconstruct.py --config configs/default.yaml")
        return
    meta = _read_json(rdir / "meta.json")
    print(f"  backend {meta.get('backend')} | model {meta.get('model')}")
    print(f"  frames {meta.get('num_frames')} | image {meta.get('image_hw')} | "
          f"device {meta.get('device')} | dtype {meta.get('dtype')}")
    print(f"  inference {meta.get('inference_seconds')}s | "
          f"conf_percentile {meta.get('conf_percentile')}")
    try:
        dense = np.load(rdir / "dense.npz")
        c = dense["depth_conf"].reshape(-1)
        pts = dense["world_points"].reshape(-1, 3)
        print(f"  depth conf p10/p50/p90: "
              f"{[round(float(np.percentile(c, q)), 2) for q in (10, 50, 90)]}"
              f"   (values at 1.0 = the model's floor -> low confidence)")
        print(f"  world bbox min {np.round(pts.min(0), 2).tolist()} "
              f"max {np.round(pts.max(0), 2).tolist()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! dense.npz problem: {exc}")
    if (rdir / "aligned" / "transform.json").exists():
        t = _read_json(rdir / "aligned" / "transform.json")
        print(f"  ICP aligned: fitness {t.get('fitness')} | static mean NN "
              f"{t.get('static_mean_nn_before')} -> {t.get('static_mean_nn_after')}"
              f"  (use_icp_refinement={cfg['reconstruction'].get('use_icp_refinement')})")


def inspect_diff(cache: Path, cfg: dict) -> None:
    section("3. change detection  (cache/diff/)")
    ddir = cache / "diff"
    clusters_doc = ddir / "clusters.json"
    if not clusters_doc.exists():
        print("not produced yet -> python src/diff.py --config configs/default.yaml")
        return
    doc = _read_json(clusters_doc)
    method = doc.get("method", "geometric")
    print(f"  method: {method} (config diff.method: {cfg['diff'].get('method')})")
    clusters = doc.get("clusters", [])
    print(f"  kept clusters: {len(clusters)}")
    print(f"    {'id':>3} {'type':<9} {'cells':>6} {'score':>6} {'centroid'}")
    for c in clusters:
        print(f"    {c['id']:>3} {c['type']:<9} {c.get('n_cells', '-'):>6} "
              f"{_fmt(c.get('mean_change_score'), 0):>6} {c['centroid']}")
    unc = ddir / "uncertain.json"
    if unc.exists():
        u = _read_json(unc).get("clusters", [])
        print(f"  unverified (one video never saw the area): {len(u)}")
        for c in u:
            print(f"    {c['id']:>3} {c['type']:<9} {c.get('n_cells', '-'):>6} {c['centroid']}")
    if method == "tabletop":
        print("  note: tabletop thresholds live in config diff.tabletop.* - see")
        print("        docs/debugging.md#change-detection-misses-or-spams-candidates")


def inspect_semantics(cache: Path) -> None:
    section("4. semantics  (cache/semantics/)")
    sdir = cache / "semantics"
    if not sdir.exists():
        print("not produced yet -> python src/describe.py --config configs/default.yaml  (GPU)")
        return
    vlm = _read_json(sdir / "vlm.json") if (sdir / "vlm.json").exists() else {}
    crops = _read_json(sdir / "crops.json") if (sdir / "crops.json").exists() else {}
    rejected = _read_json(sdir / "rejected.json").get("rejected", {}) \
        if (sdir / "rejected.json").exists() else {}
    confirmed = _read_json(sdir / "confirmed.json") if (sdir / "confirmed.json").exists() else {}
    matches = _read_json(sdir / "matches.json") if (sdir / "matches.json").exists() else []

    print(f"  crops on disk: {len(list((sdir / 'crops').glob('*.png')))}"
          f" | vlm answers: {len(vlm)} | confirmed: {len(confirmed.get('clusters', []))}"
          f" | rejected: {len(rejected)} | moved matches: {len(matches)}")
    print(f"    {'id':>3} {'type':<9} {'changed':>7} {'label':<26} first -> second")
    for cid, e in sorted(vlm.items(), key=lambda kv: int(kv[0])):
        first = (e.get("first") or e.get("description") or "")[:30]
        second = (e.get("second") or "")[:30]
        print(f"    {cid:>3} {str(e.get('changed')):>7} {str(e.get('label'))[:26]:<26} "
              f"{first!r} -> {second!r}")
    if rejected:
        print("\n  rejected candidates (why they are NOT changes):")
        for cid, e in sorted(rejected.items(), key=lambda kv: int(kv[0])):
            print(f"    {cid:>3} {e.get('type'):<9} {e.get('reason')}")
    for m in matches:
        print(f"  moved: {m['removed_id']} -> {m['added_id']} "
              f"(sim {m.get('similarity')}): {m.get('sentence')}")
    # consistency: every confirmed cluster should have both crops on disk
    for c in confirmed.get("clusters", []):
        ci = crops.get(str(c["id"]), {})
        for key in ("crop", "crop_other"):
            p = ci.get(key)
            if p and not Path(p).exists():
                print(f"  !! confirmed cluster {c['id']}: {key} missing on disk: {p}")


def inspect_outputs(out: Path) -> None:
    section("5. outputs  (outputs/)")
    if not out.exists() or not any(out.iterdir()):
        print("not produced yet -> python src/visualize.py && python src/report.py")
        return
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size / 1024:.0f} KB)")
    rj = out / "report.json"
    if rj.exists():
        r = _read_json(rj)
        print(f"\n  report.json: {len(r.get('changes', []))} change(s), "
              f"{len(r.get('unverified_changes', []))} unverified")
        for c in r.get("changes", []):
            print(f"    {c['type']:<9} {str(c.get('label')):<24} "
                  f"conf {c.get('confidence')}  {str(c.get('description'))[:60]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cache", default=None, help="override paths.cache_dir")
    parser.add_argument("--out", dest="out_dir", default=None, help="override paths.out_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cache:
        cfg["paths"]["cache_dir"] = args.cache
    if args.out_dir:
        cfg["paths"]["out_dir"] = args.out_dir
    cache = repo_path(cfg["paths"]["cache_dir"])
    out = repo_path(cfg["paths"]["out_dir"])
    print(f"inspecting cache: {cache}\n           outputs: {out}")

    inspect_frames(cache)
    inspect_reconstruction(cache, cfg)
    inspect_diff(cache, cfg)
    inspect_semantics(cache)
    inspect_outputs(out)
    print("\ndone - see docs/debugging.md for what to do when something looks wrong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
