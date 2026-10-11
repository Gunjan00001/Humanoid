#!/usr/bin/env python3
"""Semantics-only experiment worker (branch exp/accuracy). Runs on a GPU node.

Runs a queue of (config-diff, scene) SEMANTICS-ONLY experiments from each
scene's cached reconstruction/diff (no VGGT, no re-extraction), **in-process**,
so Qwen3-VL-32B and DINOv2 are loaded ONCE and reused across the whole queue
(the subprocess-per-experiment design reloaded the 32B for every scene that
needs a move/reclassify sentence). Isolated cache per experiment:
reconstruction/diff/frames symlinked, semantics/vlm.json + crops copied in, so
cached VLM answers are reused too.

GPU strategy (no fixed indices): rank all GPUs by free memory; shard
Qwen3-VL-32B bf16 across the 2-4 GPUs with the most free memory via
device_map="auto" + explicit max_memory (each >= BIG_GB, total >= BIG_TOTAL_GB);
small models (DINOv2) on the same visible set. PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True. OOM/busy -> re-rank + backoff, never exit. No
quantization. Resumable. Deletes nothing shared.
"""
from __future__ import annotations

import argparse
import importlib
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
sys.path.insert(0, str(ROOT))
SPRINT = ROOT / "logs" / "sprint"
(SPRINT / "configs").mkdir(parents=True, exist_ok=True)

BIG_GB = 40.0
BIG_TOTAL_GB = 80.0

SCENES = ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]
SCENE_DIR = {s: (ROOT / "data" if s == "demo" else ROOT / "data" / s) for s in SCENES}
GT = {s: SCENE_DIR[s] / "ground_truth.json" for s in SCENES}
MATCHES = {s: SCENE_DIR[s] / "matches.json" for s in SCENES}
SCENE_CACHE = {s: (ROOT / "cache" if s == "demo" else ROOT / "cache" / s) for s in SCENES}

EXPERIMENTS = {
    "baseline":   {"semantics.match_before_filter": True},
    "d_softveto": {"semantics.soft_veto": True},
    "d_coverage": {"semantics.coverage_added": True},
    "d_both":     {"semantics.soft_veto": True, "semantics.coverage_added": True},
    "e_objmatch": {"diff.objmatch": True},
    "f_merge":    {"semantics.fragment_merge": True},
    "d_both_ef":  {"semantics.soft_veto": True, "semantics.coverage_added": True,
                   "diff.objmatch": True, "semantics.fragment_merge": True},
}
PHASES = {
    "D": ["baseline", "d_softveto", "d_coverage", "d_both"],
    "E": ["baseline", "e_objmatch", "d_both_ef"],
    "F": ["baseline", "f_merge"],
    "EF": ["baseline", "e_objmatch", "f_merge", "d_both_ef"],
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
    apply_overrides(cfg, EXPERIMENTS[name])
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
    rows = [(int(i), (int(t) - int(u)) / 1024.0) for i, t, u in
            (ln.split(",") for ln in out.strip().splitlines())]
    return sorted(rows, key=lambda r: -r[1])


def log_gpu(msg):
    with (SPRINT / "gpu.log").open("a") as fh:
        fh.write(f"{time.strftime('%F %T')} {msg}\n")
    print(f"[worker] {msg}", flush=True)


def choose_32b():
    cand = [(i, f) for i, f in gpu_free() if f >= BIG_GB][:4]
    for k in range(len(cand), 1, -1):
        top = cand[:k]
        if sum(f for _, f in top) >= BIG_TOTAL_GB:
            return [i for i, _ in top], {vi: f"{int(f) - 2}GiB" for vi, (_, f) in enumerate(top)}
    return None, None


def prepare(exp, scene):
    d = ROOT / "cache" / "sprint" / exp / scene
    src = SCENE_CACHE[scene]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default="EF")
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
    todo = [(e, s) for e in exps for s in args.scenes
            if not (ROOT / "outputs" / "sprint" / e / s / "report.json").exists()]

    D = importlib.import_module("src.describe")
    Rm = importlib.import_module("src.report")

    _vlm = {"m": None}
    _matcher = {"m": None}
    real_load_vlm, real_load_matcher = D.load_vlm, D.load_matcher

    def cached_vlm(cfg):
        if _vlm["m"] is None:
            _vlm["m"] = real_load_vlm(cfg)
            log_gpu("32B VLM loaded once")
        return _vlm["m"]

    def cached_matcher(cfg):
        if _matcher["m"] is None:
            _matcher["m"] = real_load_matcher(cfg)
            log_gpu("DINOv2 matcher loaded once")
        return _matcher["m"]

    D.load_vlm, D.load_matcher = cached_vlm, cached_matcher

    for attempt in range(1, args.tries + 1):
        gpus, vmax = choose_32b()
        if gpus is None:
            wait = min(120 * attempt, 600)
            log_gpu(f"free={gpu_free()}; need 2-4 GPUs >=40GB (sum>=80). wait {wait}s")
            time.sleep(wait)
            continue
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus)
        log_gpu(f"attempt {attempt}: pinned GPUs {gpus} max_memory={vmax}")
        cfgs = {e: exp_config(e, vmax) for e in exps}
        failed = False
        for i, (e, s) in enumerate(todo, 1):
            out = ROOT / "outputs" / "sprint" / e / s
            if (out / "report.json").exists():
                continue
            cache = prepare(e, s)
            argv = ["describe.py", "--config", str(cfgs[e]), "--cache", str(cache),
                    "--out", str(out), "--force"]
            old, sys.argv = sys.argv, argv
            try:
                rc = D.main()
                if rc == 0:
                    Rm.main()
                st = "ok" if (out / "report.json").exists() else "no-report"
            except Exception as exc:  # noqa: BLE001
                st = f"error:{type(exc).__name__}"
                if "OutOfMemory" in str(exc):
                    failed = True
            finally:
                sys.argv = old
            log_gpu(f"[{i}/{len(todo)}] {e}/{s}: {st}")
            (SPRINT / "heartbeat.json").write_text(json.dumps(
                {"phase": args.phase, "todo": len(todo), "done": i, "last": f"{e}/{s}:{st}",
                 "gpus": gpus, "t": time.time()}, indent=2))
            if failed:
                break
        if not failed:
            break
        # free the sharded model and retry on a fresh GPU set
        _vlm["m"] = None
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(min(120 * attempt, 600))

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
