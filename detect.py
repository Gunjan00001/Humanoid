"""One-command change detection between two videos (v0.12, Step 1).

    python detect.py --a before.mp4 --b after.mp4 [--name NAME]
    python detect.py before.mp4 after.mp4 [--name NAME]

Validates the inputs, then runs the pipeline (extract -> reconstruct -> align
-> diff -> describe -> report -> visualize -> summary). Model stages run on a
GPU node only: on a GPU node (or with --local) the stages run here, on the
login node the job is submitted to the PBS gpu queue automatically and this
process polls it, printing short progress lines.

The diff mode defaults to `auto`: `tabletop` when a dominant plane covers at
least AUTO_PLANE_COVERAGE of the confident points, `geometric` otherwise;
the choice and the measured coverage are printed. No per-scene tuning is
applied unless the corresponding override flags are given explicitly (they
exist to reproduce the development scenes' documented working points; the
held-out scene5 run used none).

Detection logic, thresholds and prompts are NOT touched by this file: it only
orchestrates the existing stages.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.common import load_config, repo_path, stage_outputs  # noqa: E402

STAGES = ["extract_frames", "reconstruct", "align", "diff", "describe",
          "report", "visualize", "summary"]
AUTO_PLANE_COVERAGE = 0.50   # tabletop when the dominant plane covers >= this
PLANE_TOL = 0.02             # metres; a point counts as on the plane within this
POLL_S = 20


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# --- input validation -------------------------------------------------------

def validate_video(path: Path, role: str) -> dict:
    import cv2

    if not path.exists():
        raise SystemExit(f"[error] {role} not found: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"[error] {role} is empty: {path}")
    head = path.open("rb").read(64)
    if head.startswith(b"version https://git-lfs"):
        raise SystemExit(f"[error] {role} is a Git LFS pointer, not the video "
                         f"itself ({path}).\n        run: git lfs pull")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"[error] {role}: cannot decode as a video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"[error] {role}: no readable video stream: {path}")
    duration = frames / fps if fps > 0 else 0.0
    if duration < 3.0:
        print(f"[warn ] {role}: very short clip ({duration:.1f} s) - the "
              f"reconstruction may be unstable", flush=True)
    if min(w, h) < 240:
        print(f"[warn ] {role}: low resolution ({w}x{h})", flush=True)
    return {"path": path, "fps": fps, "frames": frames, "w": w, "h": h,
            "duration": duration}


def check_pair(info_a: dict, info_b: dict) -> None:
    if (info_a["w"], info_a["h"]) != (info_b["w"], info_b["h"]):
        print(f"[warn ] A/B resolutions differ (A {info_a['w']}x{info_a['h']}, "
              f"B {info_b['w']}x{info_b['h']}) - frames are used at their own "
              f"size; check that the crops look aligned", flush=True)
    ori = lambda i: "portrait" if i["h"] > i["w"] else "landscape"
    if ori(info_a) != ori(info_b):
        print(f"[warn ] A is {ori(info_a)}, B is {ori(info_b)} - unusual for "
              f"the same space", flush=True)


# --- GPU / PBS --------------------------------------------------------------

def on_gpu_node() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True,
                             timeout=20).stdout
        return "GPU 0" in out
    except Exception:  # noqa: BLE001
        return False


def build_pbs_script(args, name: str, a: Path, b: Path) -> Path:
    log = ROOT / "logs" / f"pbs_detect_{name}.log"
    script = ROOT / "logs" / f"job_detect_{name}.pbs"
    passthrough = []
    for flag, key in (("--method", "method"),
                      ("--blur_threshold", "blur_threshold"),
                      ("--min_change_score", "min_change_score"),
                      ("--objectness_margin", "objectness_margin"),
                      ("--conf_percentile", "conf_percentile"),
                      ("--backend", "backend")):
        value = getattr(args, key, None)
        if value is not None:
            passthrough += [flag, str(value)]
    if args.force:
        passthrough.append("--force")
    if args.debug:
        passthrough.append("--debug")
    py = '"$HOME/miniconda3/envs/spatialdiff/bin/python"'
    script.write_text(f"""#!/bin/bash
#PBS -N detect_{name}
#PBS -q gpu
#PBS -l select=1:ncpus=2:mem=60gb:ngpus=2
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -o {log}
cd "{ROOT}" || exit 1
unset CUDA_VISIBLE_DEVICES
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate spatialdiff
export PATH="$CONDA_PREFIX/bin:$PATH"
{py} -u detect.py --local --a "{a}" --b "{b}" --name {name} {' '.join(passthrough)}
echo DETECT_JOB_DONE
""")
    return script


def submit_and_poll(args, name: str, a: Path, b: Path) -> int:
    script = build_pbs_script(args, name, a, b)
    res = subprocess.run(["qsub", str(script)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[error] qsub failed: {res.stderr.strip()}", flush=True)
        return 2
    job_id = res.stdout.strip().split(".")[0]
    log_path = ROOT / "logs" / f"pbs_detect_{name}.log"
    print(f"[queue] submitted job {job_id} (gpu queue, 2 GPUs); "
          f"log: logs/pbs_detect_{name}.log", flush=True)
    if args.no_wait:
        print(f"[queue] --no-wait: not polling. Check: qstat -u {os.environ.get('USER', '')}", flush=True)
        return 0

    start = time.time()
    last = ""
    while True:
        r = subprocess.run(["qstat", "-f", job_id], capture_output=True, text=True)
        if r.returncode != 0 or "job_state" not in r.stdout:
            x = subprocess.run(["qstat", "-xf", job_id], capture_output=True, text=True)
            m = re.search(r"Exit_status = (\d+)", x.stdout)
            code = int(m.group(1)) if m else None
            print(f"[done ] job {job_id} finished (exit {code}) after "
                  f"{fmt(time.time() - start)}", flush=True)
            if log_path.exists():
                tail = [l for l in log_path.read_text(errors="replace").splitlines()
                        if l.strip()]
                for line in tail[-6:]:
                    print(f"        {line[:140]}", flush=True)
            summary_path = repo_path(f"outputs/{name}/summary.txt")
            if summary_path.exists() and (code in (0, None)):
                print("\n" + summary_path.read_text().rstrip() + "\n", flush=True)
            return 0 if code in (0, None) else (code or 1)
        m = re.search(r"job_state = (\w)", r.stdout)
        state = {"Q": "queued", "R": "running", "H": "held"}.get(
            m.group(1) if m else "?", "?")
        if log_path.exists():
            lines = [l for l in log_path.read_text(errors="replace").splitlines()
                     if l.strip()]
            if lines:
                last = lines[-1][:110]
        print(f"[{state:>7}] job {job_id} ({fmt(time.time() - start)})"
              + (f" - {last}" if last else ""), flush=True)
        time.sleep(POLL_S)


# --- local run --------------------------------------------------------------

def plane_coverage(cfg: dict) -> float | None:
    """Fraction of confident points within PLANE_TOL of the dominant plane."""
    import numpy as np

    from src.geometry import load_reconstruction, tagged_points
    from src.tabletop import fit_dominant_plane, plane_height

    recon = load_reconstruction(cfg)
    pts, _, _, _ = tagged_points(recon, float(cfg["reconstruction"]["conf_percentile"]))
    if not len(pts):
        return None
    sub = pts[:: max(1, len(pts) // 100000)]
    coef = fit_dominant_plane(sub)
    height = np.abs(plane_height(sub, coef))
    return float((height <= PLANE_TOL).mean())


def choose_mode(args, cfg: dict) -> str:
    if args.method:
        print(f"[mode ] {args.method} (explicit --method override)", flush=True)
        return args.method
    try:
        cov = plane_coverage(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[mode ] tabletop (auto: could not measure the plane coverage "
              f"({exc}) - defaulting to tabletop)", flush=True)
        return "tabletop"
    if cov is None:
        print("[mode ] tabletop (auto: no points to measure)", flush=True)
        return "tabletop"
    mode = "tabletop" if cov >= AUTO_PLANE_COVERAGE else "geometric"
    reason = (f"dominant plane covers {100 * cov:.0f}% of confident points "
              f"(threshold {100 * AUTO_PLANE_COVERAGE:.0f}%)")
    print(f"[mode ] {mode} (auto: {reason})", flush=True)
    return mode


def stage_expected(cfg: dict, stage: str) -> list[Path]:
    if stage == "summary":
        return [repo_path(cfg["paths"]["out_dir"]) / "summary.json"]
    return stage_outputs(cfg).get(stage, [])


def run_stage(stage: str, args, cfg: dict, mode: str, a: Path, b: Path) -> int:
    py = sys.executable
    cmd = [py, "-u", str(ROOT / "src" / f"{stage}.py"), "--config", args.config,
           "--video_a", str(a), "--video_b", str(b),
           "--cache", str(repo_path(cfg["paths"]["cache_dir"])),
           "--out", str(repo_path(cfg["paths"]["out_dir"]))]
    if args.force:
        cmd.append("--force")
    if args.debug:
        cmd.append("--debug")
    if args.conf_percentile is not None:
        cmd += ["--conf_percentile", str(args.conf_percentile)]
    if stage == "extract_frames" and args.blur_threshold is not None:
        cmd += ["--blur_threshold", str(args.blur_threshold)]
    if stage == "reconstruct" and args.backend:
        cmd += ["--backend", args.backend]
    if stage == "diff":
        cmd += ["--method", mode]
        if args.objectness_margin is not None:
            cmd += ["--objectness_margin", str(args.objectness_margin)]
    if stage == "describe" and args.min_change_score is not None:
        cmd += ["--min_change_score", str(args.min_change_score)]
    if stage == "summary" and getattr(args, "no_vlm", False):
        cmd.append("--no-vlm")

    start = time.time()
    print(f"[run  ] {stage}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - start
    if proc.returncode != 0:
        print(f"[fail ] stage '{stage}' failed (exit {proc.returncode}) after "
              f"{fmt(dt)}", flush=True)
        print(f"        likely cause: see the lines above; per-decision numbers "
              f"with --debug. Fix and re-run: the caches of finished stages are "
              f"reused.", flush=True)
        return proc.returncode
    print(f"[ok   ] {stage} ({fmt(dt)})", flush=True)
    return 0


def run_local(args, cfg: dict, name: str, a: Path, b: Path) -> int:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    total = time.time()
    mode = "tabletop"
    for stage in STAGES:
        expected = stage_expected(cfg, stage)
        if expected and all(p.exists() for p in expected) and not args.force:
            print(f"[skip ] {stage}: outputs exist ({expected[0]})", flush=True)
            if stage == "align":
                mode = choose_mode(args, cfg)   # needed for the diff decision
            continue
        if stage == "diff":
            mode = choose_mode(args, cfg)
        rc = run_stage(stage, args, cfg, mode, a, b)
        if rc != 0:
            return rc
    out = repo_path(cfg["paths"]["out_dir"])
    summary = out / "summary.txt"
    print(f"\n[done ] total {fmt(time.time() - total)} | outputs: {out}",
          flush=True)
    if summary.exists():
        print("\n" + summary.read_text().rstrip() + "\n", flush=True)
    return 0


# --- main -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="*", metavar="BEFORE AFTER",
                        help="the two videos (or use --a/--b)")
    parser.add_argument("--a", dest="a", default=None, help="before video")
    parser.add_argument("--b", dest="b", default=None, help="after video")
    parser.add_argument("--name", default=None,
                        help="output/cache name (default: the videos' folder name)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--local", action="store_true",
                        help="run here even if this is not a GPU node")
    parser.add_argument("--no-wait", action="store_true",
                        help="submit the PBS job and return immediately")
    parser.add_argument("--render-only", action="store_true",
                        help="skip the pipeline stages; re-render from caches "
                             "(no GPU needed)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite outputs/<name>/ and redo the stages")
    parser.add_argument("--debug", action="store_true",
                        help="verbose per-decision logging")
    parser.add_argument("--method", choices=["tabletop", "geometric"], default=None,
                        help="override the automatic diff-mode choice")
    parser.add_argument("--blur_threshold", type=float, default=None)
    parser.add_argument("--min_change_score", type=float, default=None)
    parser.add_argument("--objectness_margin", type=float, default=None)
    parser.add_argument("--conf_percentile", type=float, default=None)
    parser.add_argument("--backend", default=None,
                        help="reconstruction backend (vggt | mapanything | da3)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="summary: skip the Qwen3-VL phrasing call")
    args = parser.parse_args()

    a_raw = args.a or (args.videos[0] if len(args.videos) > 0 else None)
    b_raw = args.b or (args.videos[1] if len(args.videos) > 1 else None)
    if not a_raw or not b_raw:
        parser.error("give two videos: --a before.mp4 --b after.mp4 "
                     "(or positionally)")
    a, b = Path(a_raw).resolve(), Path(b_raw).resolve()

    name = args.name
    if not name:
        name = a.parent.name if a.parent == b.parent else f"{a.stem}_vs_{b.stem}"

    cfg = load_config(args.config)
    cfg["paths"]["video_a"] = str(a)
    cfg["paths"]["video_b"] = str(b)
    cfg["paths"]["cache_dir"] = str(ROOT / "cache" / name)
    cfg["paths"]["out_dir"] = str(ROOT / "outputs" / name)

    print(f"detect: A={a.name} B={b.name} name={name}", flush=True)
    info_a = validate_video(a, "video A")
    info_b = validate_video(b, "video B")
    check_pair(info_a, info_b)
    print(f"[check] A {info_a['w']}x{info_a['h']} {info_a['fps']:.1f}fps "
          f"{info_a['duration']:.1f}s | B {info_b['w']}x{info_b['h']} "
          f"{info_b['fps']:.1f}fps {info_b['duration']:.1f}s", flush=True)

    out = repo_path(cfg["paths"]["out_dir"])
    if (out / "report.json").exists() and not args.force:
        raise SystemExit(
            f"[error] {out}/report.json already exists - use --force to "
            f"overwrite, or --render-only to re-render from the caches")

    if args.render_only:
        # CPU-only: re-render from the existing caches, then refresh the summary
        cmd = [sys.executable, "-u", str(ROOT / "src" / "visualize.py"),
               "--config", args.config, "--cache", cfg["paths"]["cache_dir"],
               "--out", cfg["paths"]["out_dir"], "--render-only"]
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            print("[fail ] render-only failed (see above)", flush=True)
            return rc
        if not (out / "summary.json").exists():
            cmd = [sys.executable, "-u", str(ROOT / "src" / "summary.py"),
                   "--config", args.config, "--cache", cfg["paths"]["cache_dir"],
                   "--out", cfg["paths"]["out_dir"]]
            subprocess.run(cmd, cwd=ROOT)
        return 0

    if args.local or on_gpu_node():
        where = "GPU node" if on_gpu_node() else "--local"
        print(f"[node ] running the stages here ({where})", flush=True)
        return run_local(args, cfg, name, a, b)

    print("[node ] login node: submitting the model stages to the PBS gpu "
          "queue", flush=True)
    return submit_and_poll(args, name, a, b)


if __name__ == "__main__":
    raise SystemExit(main())
