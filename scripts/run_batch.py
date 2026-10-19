#!/usr/bin/env python3
"""Unattended task runner: progress must not depend on any session's budget.

Reads data/batch_manifest.json in order and runs each task on the cluster:

  * a task whose logs/batch/<id>.done exists is skipped;
  * the GPU with the most free memory is picked for every task; when none has the
    task's minimum, the runner backs off (60s, doubling to 300s) and retries, so
    a busy node delays but never kills the batch;
  * per-task stdout and stderr go to logs/batch/<id>_<pbsjobid>.out/.err, named
    with the job id so nothing is ever overwritten;
  * the task's own JSON is written incrementally by the task (geometry_eval does);
  * success writes <id>.done, failure writes <id>.failed with the error and the
    batch continues;
  * failed tasks are retried once at the end;
  * if tasks are still pending when the loop stops (walltime), the runner
    resubmits its own PBS script, so the batch survives the job boundary.

  python scripts/run_batch.py            # inside the PBS job
  python scripts/run_batch.py --dry-run  # print the plan
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "batch_manifest.json"
LOGS = ROOT / "logs" / "batch"
PY = sys.executable


def free_gpus() -> list[tuple[int, int]]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    gpus = []
    for line in out.splitlines():
        try:
            idx, mem = (int(x) for x in line.split(","))
            gpus.append((idx, mem))
        except ValueError:
            continue
    return sorted(gpus, key=lambda g: -g[1])


def pick_gpu(min_free_mb: int) -> int | None:
    for idx, mem in free_gpus():
        if mem >= min_free_mb:
            return idx
    return None


def task_command(task: dict) -> list[str]:
    kind = task["type"]
    if kind == "geometry":
        return [PY, "scripts/geometry_eval.py", "--scene", task.get("scene", "scene4"),
                "--backend", task["backend"], "--frames", task.get("frames", "1x"),
                "--out", str(LOGS / f"{task['id']}.json")]
    if kind == "shell":
        return task["command"]
    raise ValueError(f"unknown task type {kind!r}")


def run_task(task: dict, attempt: int = 1) -> bool:
    tid = task["id"]
    done, failed = LOGS / f"{tid}.done", LOGS / f"{tid}.failed"
    if done.exists():
        print(f"skip {tid} (done)")
        return True
    LOGS.mkdir(parents=True, exist_ok=True)
    min_free = int(task.get("min_free_mb", 20000))
    delay = 60
    while True:
        gpu = pick_gpu(min_free)
        if gpu is not None:
            break
        print(f"{tid}: no GPU with {min_free} MB free; backing off {delay}s")
        time.sleep(delay)
        delay = min(300, delay * 2)
    jobid = os.environ.get("PBS_JOBID", "manual").split(".")[0]
    out = LOGS / f"{tid}_{jobid}_a{attempt}.out"
    err = LOGS / f"{tid}_{jobid}_a{attempt}.err"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1")
    print(f"run  {tid} on GPU {gpu} (attempt {attempt}) -> {out.name}")
    with open(out, "w") as fo, open(err, "w") as fe:
        code = subprocess.call(task_command(task), cwd=ROOT, env=env,
                               stdout=fo, stderr=fe)
    if code == 0:
        done.write_text(f"gpu={gpu} attempt={attempt} at={time.strftime('%F %T')} job={jobid}\n")
        failed.unlink(missing_ok=True)
        print(f"done {tid}")
        return True
    tail = "\n".join(err.read_text(errors="ignore").splitlines()[-20:])
    failed.write_text(f"exit={code} gpu={gpu} attempt={attempt} at={time.strftime('%F %T')} "
                      f"job={jobid}\n--- stderr tail ---\n{tail}\n")
    print(f"FAIL {tid} (exit {code}); see {err.name}")
    return False


def pending(tasks: list[dict]) -> list[dict]:
    return [t for t in tasks if not (LOGS / f"{t['id']}.done").exists()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tasks = json.loads(Path(args.manifest).read_text())["tasks"]
    if args.dry_run:
        for t in tasks:
            state = "done" if (LOGS / f"{t['id']}.done").exists() else "pending"
            print(f"{state:8} {t['id']:24} {t['type']}")
        return 0

    for t in tasks:
        run_task(t)
    # one retry for everything that failed
    for t in pending(tasks):
        if (LOGS / f"{t['id']}.failed").exists():
            run_task(t, attempt=2)

    left = pending(tasks)
    print(f"batch state: {len(tasks) - len(left)} done, {len(left)} pending")
    if left:
        # survive the job boundary: resubmit the PBS wrapper once more
        script = os.environ.get("BATCH_RESUBMIT_SCRIPT")
        if script and os.environ.get("BATCH_ALREADY_RESUBMITTED") != "1":
            subprocess.call(["qsub", "-v", "BATCH_ALREADY_RESUBMITTED=1", script], cwd=ROOT)
            print(f"resubmitted {script} for the remaining {len(left)} task(s)")
        else:
            print("pending tasks remain; the next session should resubmit the job")
    return 0 if not left else 0


if __name__ == "__main__":
    raise SystemExit(main())
