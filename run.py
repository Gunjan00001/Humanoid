"""One-command pipeline (see AGENT_PLAN.md section 1).

    python run.py --video_a data/a.mp4 --video_b data/b.mp4 --out outputs/

Runs the stages in order, each as its own process (so a GPU stage only sees
CUDA when it actually needs it). Stages whose cached outputs already exist are
skipped unless --force is given. Per-stage wall-clock timings land in
outputs/timings.json. This file contains no model logic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.common import Timer, load_config, repo_path, stage_outputs, write_json  # noqa: E402

STAGES = ["extract_frames", "reconstruct", "align", "diff", "describe", "visualize", "report"]
GPU_STAGES = {"reconstruct", "describe"}


def stage_enabled(stage: str, cfg: dict) -> bool:
    if stage == "align":
        return bool(cfg["reconstruction"].get("use_icp_refinement"))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video_a", default=None)
    parser.add_argument("--video_b", default=None)
    parser.add_argument("--out", dest="out_dir", default=None)
    parser.add_argument("--cache", dest="cache_dir", default=None)
    parser.add_argument("--stages", default="all",
                        help=f"comma list from {STAGES} (default: all)")
    parser.add_argument("--from_stage", default=None, choices=STAGES,
                        help="start from this stage (rerun a suffix)")
    parser.add_argument("--force", action="store_true",
                        help="redo stages even if their outputs exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.video_a:
        cfg["paths"]["video_a"] = args.video_a
    if args.video_b:
        cfg["paths"]["video_b"] = args.video_b
    if args.out_dir:
        cfg["paths"]["out_dir"] = args.out_dir
    if args.cache_dir:
        cfg["paths"]["cache_dir"] = args.cache_dir

    if args.stages == "all":
        wanted = list(STAGES)
    else:
        wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in STAGES]
        if unknown:
            raise SystemExit(f"unknown stage(s): {unknown}; valid: {STAGES}")
        wanted = [s for s in STAGES if s in wanted]  # enforce pipeline order
    if args.from_stage:
        wanted = wanted[wanted.index(args.from_stage):]

    overrides = []
    for flag, key in (("--video_a", "video_a"), ("--video_b", "video_b"),
                      ("--out", "out_dir"), ("--cache", "cache_dir")):
        value = getattr(args, key)
        if value:
            overrides += [flag, str(value)]

    expected = stage_outputs(cfg)
    timings: dict[str, float] = {}
    print(f"pipeline: {' -> '.join(s for s in wanted if stage_enabled(s, cfg))}")
    for stage in wanted:
        if not stage_enabled(stage, cfg):
            print(f"[skip] {stage}: disabled in config")
            continue
        missing = [p for p in expected[stage] if not p.exists()]
        if not missing and not args.force:
            print(f"[skip] {stage}: outputs already exist ({expected[stage][0]})")
            continue
        cmd = [sys.executable, str(ROOT / "src" / f"{stage}.py"),
               "--config", args.config] + overrides
        print(f"[run ] {stage}: {' '.join(cmd)}", flush=True)
        with Timer() as t:
            proc = subprocess.run(cmd, cwd=ROOT)
        timings[stage] = t.elapsed_s
        if proc.returncode != 0:
            print(f"\nstage '{stage}' failed after {t.elapsed_s:.1f}s "
                  f"(see output above). Fix and rerun with --from_stage {stage}.")
            return proc.returncode

    out = repo_path(cfg["paths"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "timings.json",
               {"stages": timings, "total_s": round(sum(timings.values()), 1)})
    print(f"\ndone in {sum(timings.values()) / 60:.1f} min (stages run: {len(timings)})")
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
