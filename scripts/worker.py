#!/usr/bin/env python3
"""Semantics-only experiment worker (branch exp/accuracy). Runs on a GPU node.

Queue of (config-diff, scene) SEMANTICS-ONLY experiments from each scene's
cached reconstruction/diff/VLM (no VGGT, no re-extraction). Isolated cache per
experiment: reconstruction/diff/frames SYMLINKED, semantics/vlm.json + crops
copied in - so post-filter experiments reuse every cached VLM answer (no 32B).

GPU strategy (configurable, no fixed indices):
  * rank ALL GPUs by free memory (nvidia-smi);
  * small-model experiments (DINOv2 / cached-VLM post-filters): 1 GPU >= SMALL_GB;
  * 32B experiments (need fresh VLM answers): shard Qwen3-VL-32B bf16 across the
    2-4 GPUs with the most free memory via device_map="auto" + explicit
    max_memory (each >= BIG_GB, total >= BIG_TOTAL_GB); no per-GPU 70GB demand.
  * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  * OOM / busy: free, re-rank, retry with backoff (up to TRIES over ~1 h), then
    keep waiting rather than exit. Chosen GPUs + free memory logged.
Small-model work never shares a GPU with the 32B worker. No quantization.
Resumable; deletes nothing shared.
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

SMALL_GB = 25.0
BIG_GB = 40.0
BIG_TOTAL_GB = 80.0

SCENES = ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]
SCENE_DIR = {s: (ROOT / "data" if s == "demo" else ROOT / "data" / s) for s in SCENES}
GT = {s: SCENE_DIR[s] / "ground_truth.json" for s in SCENES}
MATCHES = {s: SCENE_DIR[s] / "matches.json" for s in SCENES}
SCENE_CACHE = {s: (ROOT / "cache" if s == "demo" else ROOT / "cache" / s) for s in SCENES}

# experiment name -> (dotted overrides, needs_fresh_vlm)
EXPERIMENTS = {
    "baseline":    ({}, False),
    "d_softveto":  ({"semantics.soft_veto": True}, False),
    "d_coverage":  ({"semantics.coverage_added": True}, False),
    "d_both":      ({"semantics.soft_veto": True, "semantics.coverage_added": True}, False),
    "e_objmatch":  ({"diff.objmatch": True}, False),
    "d_both_e":    ({"semantics.soft_veto": True, "semantics.coverage_added": True,
                     "diff.objmatch": True}, False),
}
PHASES = {
    "D": ["baseline", "d_softveto", "d_coverage", "d_both"],
    "E": ["baseline", "e_objmatch", "d_both_e"],
}


def apply_overrides(cfg, dotted):
    for k, v in dotted.items():
        parts = k.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return cfg


def exp_config(name, max_memory=None) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    apply_overrides(cfg, EXPERIMENTS[name][0])
    if max_memory:
        cfg["semantics"]["vlm_max_memory"] = max_memory
    p = SPRINT / "configs" / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def gpu_free():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        i, tot, used = (int(x) for x in line.split(","))
        rows.append((i, (tot - used) / 1024.0))
    return sorted(rows, key=lambda r: -r[1])


def log_gpu(msg: str):
    with (SPRINT / "gpu.log").open("a") as fh:
        fh.write(f"{time.strftime('%F %T')} {msg}\n")
    print(f"[worker] {msg}", flush=True)


def choose_small():
    for i, free in gpu_free():
        if free >= SMALL_GB:
            return [i], {0: f"{int(free) - 2}GiB"}
    return None, None


def choose_32b():
    cand = [(i, f) for i, f in gpu_free() if f >= BIG_GB][:4]
    for k in range(len(cand), 1, -1):
        top = cand[:k]
        if sum(f for _, f in top) >= BIG_TOTAL_GB:
            vmax = {vi: f"{int(f) - 2}GiB" for vi, (_, f) in enumerate(top)}
            return [i for i, _ in top], vmax
    return None, None


def prepare(exp, scene):
    d = ROOT / "cache" / "sprint" / exp / scene
    src = SCENE_CACHE[scene]
    # purge a stale REAL cache dir left by an earlier full-pipeline attempt, so
    # the symlinks below are (re)created and describe reads the scene's cache
    if d.exists() and (d / "reconstruction").exists() and not (d / "reconstruction").is_symlink():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    for part in ("reconstruction", "diff", "frames", "aligned"):
        link, target = d / part, src / part
        if target.exists() and not link.exists():
            link.symlink_to(target)
    sem_dst, sem_src = d / "semantics", src / "semantics"
    sem_dst.mkdir(exist_ok=True)
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


def oom_tail(log: Path) -> bool:
    try:
        return "OutOfMemory" in log.read_text(errors="replace")[-3000:]
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="D")
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--tries", type=int, default=6)
    args = ap.parse_args()

    if not shutil.which("nvidia-smi"):
        print("[worker] no nvidia-smi - must run on a GPU node", flush=True)
        return 2
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    exps = args.experiments or PHASES[args.phase]
    needs_vlm = any(EXPERIMENTS[e][1] for e in exps)
    todo = [(e, s) for e in exps for s in args.scenes
            if not (ROOT / "outputs" / "sprint" / e / s / "report.json").exists()]

    once_ok = False
    for attempt in range(1, args.tries + 1):
        free = gpu_free()
        log_gpu(f"attempt {attempt}: free GB = {free}")
        gpus, vmax = choose_32b() if needs_vlm else choose_small()
        if gpus is None:
            wait = min(120 * attempt, 600)
            log_gpu(f"no suitable GPUs (need {'2-4 big' if needs_vlm else '1 small'}); "
                    f"waiting {wait}s")
            time.sleep(wait)
            continue
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus)
        log_gpu(f"pinned GPUs {gpus} vmax={vmax}")
        cfg = {e: exp_config(e, vmax if EXPERIMENTS[e][1] else None) for e in exps}
        failed = 0
        for i, (e, s) in enumerate(todo, 1):
            if (ROOT / "outputs" / "sprint" / e / s / "report.json").exists():
                continue
            cache = prepare(e, s)
            log = SPRINT / f"{e}__{s}.log"
            rc = run_stage("describe.py", cfg[e], cache, ROOT / "outputs" / "sprint" / e / s)
            if rc == 0:
                run_stage("report.py", cfg[e], cache, ROOT / "outputs" / "sprint" / e / s)
                st = "ok" if (ROOT / "outputs" / "sprint" / e / s / "report.json").exists() else "no-report"
            else:
                st = "oom" if oom_tail(log) else "fail"
                failed += 1
            log_gpu(f"[{i}/{len(todo)}] {e}/{s}: {st}")
            (SPRINT / "heartbeat.json").write_text(json.dumps(
                {"phase": args.phase, "todo": len(todo), "done": i, "last": f"{e}/{s}:{st}",
                 "gpus": gpus, "t": time.time()}, indent=2))
            if st == "oom":
                break  # race: re-rank GPUs and restart the loop
        if failed == 0:
            once_ok = True
            break
        time.sleep(min(120 * attempt, 600))
    if not once_ok:
        log_gpu("could not complete under GPU pressure; leaving partial outputs")

    print("\n===== SCORES =====", flush=True)
    for e in exps:
        for s in args.scenes:
            rp = ROOT / "outputs" / "sprint" / e / s / "report.json"
            if not rp.exists():
                continue
            r = subprocess.run([PY, "scripts/score.py", "--report", str(rp), "--gt",
                                str(GT[s]), "--matches", str(MATCHES[s])],
                               cwd=ROOT, capture_output=True, text=True)
            print(f"\n##### {e} / {s} #####\n{r.stdout}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
