#!/usr/bin/env python3
"""Accuracy-sprint orchestrator (branch exp/accuracy).

Runs a queue of (experiment, scene) detection jobs, at most N concurrently
(default 4 - leaves a PBS slot free), each `detect.py` submitting its own 2-CPU
gpu-queue job. Resumable: a run whose report.json exists is skipped. A failed
job is retried once. Everything logs under logs/sprint/ and a heartbeat file is
written. Then scores each finished run with scripts/score.py.

Usage:
  python scripts/sprint.py --phase D            # run the D experiments
  python scripts/sprint.py --experiments baseline soft_veto --scenes scene6
  python scripts/sprint.py --status             # just print what's done
"""
# EXPERIMENTAL, not used by detect.py. Part of the accuracy and naming
# experiments; kept as evidence. See docs/accuracy.md and docs/naming_v4.md.
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# Interpreter used for the experiment jobs. Override with SPATIALDIFF_PYTHON
# when the environment lives somewhere else; the default is the conda env that
# docs/setup.md creates.
PY = os.environ.get(
    "SPATIALDIFF_PYTHON",
    os.path.expanduser("~/miniconda3/envs/spatialdiff/bin/python"),
)
SPRINT = ROOT / "logs" / "sprint"
(SPRINT / "configs").mkdir(parents=True, exist_ok=True)

SCENES = {s: f"data/{s}" for s in ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]}
# demo videos live at data/a.mp4, data/b.mp4
SCENE_VIDEOS = {
    "demo": ("data/a.mp4", "data/b.mp4"),
    "scene2": ("data/scene2/a.mp4", "data/scene2/b.mp4"),
    "scene3": ("data/scene3/a.mp4", "data/scene3/b.mp4"),
    "scene4": ("data/scene4/a.mp4", "data/scene4/b.mp4"),
    "scene5": ("data/scene5/a.mp4", "data/scene5/b.mp4"),
    "scene6": ("data/scene6/a.mp4", "data/scene6/b.mp4"),
}
GT = {s: f"data/{'ground_truth' if s=='demo' else s+'/ground_truth'}.json" for s in SCENE_VIDEOS}
MATCHES = {s: f"data/{'matches' if s=='demo' else s+'/matches'}.json" for s in SCENE_VIDEOS}

# experiment name -> dotted config overrides on top of configs/default.yaml
EXPERIMENTS = {
    "baseline":        {},                                   # v1.0.0 behaviour
    "e0_matchflag":    {"semantics.match_before_filter": True},
    "d_softveto":      {"semantics.soft_veto": True},
    "d_coverage":      {"semantics.coverage_added": True},
    "d_both":          {"semantics.soft_veto": True, "semantics.coverage_added": True},
    "e_objmatch":      {"diff.objmatch": True},
    "d_both_e":        {"semantics.soft_veto": True, "semantics.coverage_added": True,
                        "diff.objmatch": True},
}

PHASES = {
    "D": ["baseline", "d_softveto", "d_coverage", "d_both"],
    "E0": ["baseline", "e0_matchflag"],
    "E": ["baseline", "e_objmatch", "d_both_e"],
}


def apply_overrides(cfg: dict, dotted: dict) -> dict:
    for k, v in dotted.items():
        parts = k.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return cfg


def write_exp_config(name: str) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    apply_overrides(cfg, EXPERIMENTS[name])
    p = SPRINT / "configs" / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def run_dir(exp: str, scene: str) -> Path:
    return ROOT / "outputs" / "sprint" / exp / scene


def cache_dir(exp: str, scene: str) -> Path:
    return ROOT / "cache" / "sprint" / exp / scene


def heartbeat(state: dict) -> None:
    (SPRINT / "heartbeat.json").write_text(json.dumps({**state, "t": time.time()}, indent=2))


def run_one(exp: str, scene: str, cfg_path: Path) -> dict:
    out, cache = run_dir(exp, scene), cache_dir(exp, scene)
    report = out / "report.json"
    if report.exists():
        return {"exp": exp, "scene": scene, "status": "cached"}
    a, b = SCENE_VIDEOS[scene]
    log = SPRINT / f"{exp}__{scene}.log"
    cmd = [PY, "-u", "detect.py", "--config", str(cfg_path), "--a", a, "--b", b,
           "--name", f"{exp}__{scene}", "--out", str(out), "--cache", str(cache), "--force"]
    for attempt in (1, 2):
        with log.open("a") as fh:
            fh.write(f"\n=== {exp}/{scene} attempt {attempt} {time.strftime('%H:%M:%S')} ===\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=60 * 90).returncode
        if rc == 0 and report.exists():
            return {"exp": exp, "scene": scene, "status": "ok", "rc": rc}
    return {"exp": exp, "scene": scene, "status": "failed", "rc": rc}


def score_all(exps: list[str], scenes: list[str]) -> None:
    rows = []
    for exp in exps:
        for scene in scenes:
            rp = run_dir(exp, scene) / "report.json"
            if not rp.exists():
                continue
            r = subprocess.run([PY, "scripts/score.py", "--report", str(rp),
                                "--gt", GT[scene], "--matches", MATCHES[scene]],
                               cwd=ROOT, capture_output=True, text=True)
            rows.append((exp, scene, r.stdout.strip().splitlines()))
    (SPRINT / "scores.txt").write_text(json.dumps(
        {f"{e}/{s}": out for e, s, out in rows}, indent=2))
    for e, s, out in rows:
        print(f"\n########## {e} / {s} ##########")
        print("\n".join(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES))
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--scenes", nargs="*", default=list(SCENE_VIDEOS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    exps = args.experiments or (PHASES[args.phase] if args.phase else ["baseline"])
    cfg_paths = {e: write_exp_config(e) for e in exps}

    if args.status or args.score_only:
        for e in exps:
            for s in args.scenes:
                print(f"{e}/{s}: {'DONE' if (run_dir(e,s)/'report.json').exists() else 'pending'}")
        if args.score_only:
            score_all(exps, args.scenes)
        return 0

    todo = [(e, s) for e in exps for s in args.scenes
            if not (run_dir(e, s) / "report.json").exists()]
    print(f"sprint: {len(todo)} run(s) to do, {args.jobs} concurrent")
    heartbeat({"todo": len(todo), "done": 0, "running": []})
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_one, e, s, cfg_paths[e]): (e, s) for e, s in todo}
        for fut in cf.as_completed(futs):
            res = fut.result()
            done += 1
            print(f"  [{done}/{len(todo)}] {res['exp']}/{res['scene']}: {res['status']}")
            heartbeat({"todo": len(todo), "done": done, "last": res})
    score_all(exps, args.scenes)
    heartbeat({"todo": len(todo), "done": done, "state": "finished"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
