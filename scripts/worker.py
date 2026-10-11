#!/usr/bin/env python3
"""Semantics-only experiment worker (branch exp/accuracy).

Runs a queue of (config-diff, scene) SEMANTICS-ONLY experiments from each
scene's cached reconstruction/diff/VLM - no VGGT, no frame re-extraction. Each
experiment gets an isolated cache dir whose reconstruction/diff/frames are
SYMLINKS to the scene's cache and whose semantics/vlm.json (+crops) are copied
in, so pure post-filter experiments (Phase D) reuse every cached VLM answer and
never load the 32B model at all. Written to run INSIDE a PBS gpu job (2 GPUs).

GPU safety: before each experiment it checks nvidia-smi and picks GPUs with
>= min_free_gb free; if too few are free it waits with backoff instead of
OOMing. Resumable: a run whose report.json exists is skipped. Deletes nothing
shared; just writes outputs/sprint/<exp>/<scene> + logs/sprint/.

Usage (inside the node): python scripts/worker.py --phase D
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SPRINT = ROOT / "logs" / "sprint"
(SPRINT / "configs").mkdir(parents=True, exist_ok=True)

SCENES = ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]
SCENE_DIR = {s: (ROOT / "data" if s == "demo" else ROOT / "data" / s) for s in SCENES}
GT = {s: SCENE_DIR[s] / "ground_truth.json" for s in SCENES}
MATCHES = {s: SCENE_DIR[s] / "matches.json" for s in SCENES}
# scene cache: demo lives at cache/, sceneN at cache/sceneN
SCENE_CACHE = {s: (ROOT / "cache" if s == "demo" else ROOT / "cache" / s) for s in SCENES}

EXPERIMENTS = {
    "baseline":   {},
    "d_softveto": {"semantics.soft_veto": True},
    "d_coverage": {"semantics.coverage_added": True},
    "d_both":     {"semantics.soft_veto": True, "semantics.coverage_added": True},
}
PHASES = {"D": ["baseline", "d_softveto", "d_coverage", "d_both"]}


def apply_overrides(cfg, dotted):
    for k, v in dotted.items():
        parts = k.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return cfg


def exp_config(name) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    apply_overrides(cfg, EXPERIMENTS[name])
    p = SPRINT / "configs" / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def free_gpus(min_free_gb=70.0):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    free = []
    for line in out.strip().splitlines():
        idx, total, used = (int(x) for x in line.split(","))
        if (total - used) / 1024.0 >= min_free_gb:
            free.append(idx)
    return free


def pick_gpus(n=2, min_free_gb=70.0, tries=6):
    for t in range(tries):
        g = free_gpus(min_free_gb)
        if len(g) >= n:
            return g[:n]
        wait = min(60 * 2 ** t, 600)
        print(f"[worker] only {len(g)} GPU(s) with >={min_free_gb}GB free; "
              f"waiting {wait}s (try {t+1}/{tries})", flush=True)
        time.sleep(wait)
    return None


def prepare(exp, scene):
    d = ROOT / "cache" / "sprint" / exp / scene
    d.mkdir(parents=True, exist_ok=True)
    src = SCENE_CACHE[scene]
    for part in ("reconstruction", "diff", "frames", "aligned"):
        link, target = d / part, src / part
        if target.exists() and not link.exists():
            link.symlink_to(target)
    sem_dst = d / "semantics"
    sem_dst.mkdir(exist_ok=True)
    sem_src = src / "semantics"
    for f in ("vlm.json", "crops.json"):
        if (sem_src / f).exists() and not (sem_dst / f).exists():
            shutil.copy2(sem_src / f, sem_dst / f)
    if (sem_src / "crops").exists() and not (sem_dst / "crops").exists():
        (sem_dst / "crops").symlink_to(sem_src / "crops")
    return d


def run_stage(script, cfg, cache, out):
    return subprocess.run([PY, "-u", str(ROOT / "src" / script), "--config", str(cfg),
                           "--cache", str(cache), "--out", str(out), "--force"],
                          cwd=ROOT).returncode


def run_one(exp, scene, cfg_path):
    out = ROOT / "outputs" / "sprint" / exp / scene
    if (out / "report.json").exists():
        return "cached"
    cache = prepare(exp, scene)
    log = SPRINT / f"{exp}__{scene}.log"
    for attempt in (1, 2):
        with log.open("a") as fh:
            subprocess.run(["bash", "-lc", f"echo '=== {exp}/{scene} try {attempt} "
                            f"{time.strftime('%H:%M:%S')} ==='"], stdout=fh)
        rc = run_stage("describe.py", cfg_path, cache, out)
        if rc == 0:
            run_stage("report.py", cfg_path, cache, out)
            if (out / "report.json").exists():
                return "ok"
        if rc != 0 and "OutOfMemory" in log.read_text(errors="replace")[-2000:]:
            pick_gpus(2)  # re-check GPUs and retry
    return "failed"


def score_one(exp, scene):
    rp = ROOT / "outputs" / "sprint" / exp / scene / "report.json"
    if not rp.exists():
        return None
    r = subprocess.run([PY, "scripts/score.py", "--report", str(rp), "--gt", str(GT[scene]),
                        "--matches", str(MATCHES[scene])], cwd=ROOT, capture_output=True, text=True)
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="D")
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--repeats-final", type=int, default=0)
    args = ap.parse_args()

    if not shutil.which("nvidia-smi"):
        print("[worker] no nvidia-smi - must run on a GPU node", flush=True)
        return 2
    picked = pick_gpus(2)
    if picked is None:
        print("[worker] could not find 2 idle GPUs - aborting (no OOM)", flush=True)
        return 3
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in picked)
    print(f"[worker] pinned CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    exps = args.experiments or PHASES[args.phase]
    cfg = {e: exp_config(e) for e in exps}
    todo = [(e, s) for e in exps for s in args.scenes
            if not (ROOT / "outputs" / "sprint" / e / s / "report.json").exists()]
    print(f"[worker] {len(todo)} semantics-only run(s)", flush=True)
    for i, (e, s) in enumerate(todo, 1):
        st = run_one(e, s, cfg[e])
        print(f"[worker] [{i}/{len(todo)}] {e}/{s}: {st}", flush=True)
        (SPRINT / "heartbeat.json").write_text(json.dumps(
            {"phase": args.phase, "todo": len(todo), "done": i, "last": f"{e}/{s}:{st}",
             "t": time.time()}, indent=2))
    print("\n===== SCORES =====", flush=True)
    for e in exps:
        for s in args.scenes:
            out = score_one(e, s)
            if out:
                print(f"\n##### {e} / {s} #####\n{out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
