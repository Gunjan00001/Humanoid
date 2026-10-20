#!/usr/bin/env python3
"""Redo the six uploaded scenes into fresh *_redo folders.

Frozen outputs are never touched: every run writes to outputs/<name>_redo and
cache/<name>_redo. The held-out reruns (scene5, scene6) are post-hoc only and are
labelled as such in logs/redo_status.json. Each scene is skipped when its report
already exists, a failure is recorded and the run continues, and the final status
file carries timings and the exact command used.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = [
    ("demo", "data/a.mp4", "data/b.mp4", "development"),
    ("scene2", "data/scene2/a.mp4", "data/scene2/b.mp4", "development"),
    ("scene3", "data/scene3/a.mp4", "data/scene3/b.mp4", "development"),
    ("scene4", "data/scene4/a.mp4", "data/scene4/b.mp4", "development"),
    ("scene5", "data/scene5/a.mp4", "data/scene5/b.mp4", "held-out, post-hoc redo"),
    ("scene6", "data/scene6/a.mp4", "data/scene6/b.mp4", "held-out, post-hoc redo"),
]


def main() -> int:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    status = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "scenes": {}}
    for name, a, b, kind in SCENES:
        out = ROOT / "outputs" / f"{name}_redo"
        cache = ROOT / "cache" / f"{name}_redo"
        entry = {"kind": kind, "out": str(out.relative_to(ROOT)),
                 "cache": str(cache.relative_to(ROOT)), "status": "running"}
        status["scenes"][name] = entry
        if (out / "report.json").exists():
            entry["status"] = "already-done"
            print(f"[skip] {name}: {out} already has a report")
            continue
        cmd = [sys.executable, "detect.py", "--a", a, "--b", b,
               "--name", f"{name}_redo", "--out", str(out), "--cache", str(cache)]
        entry["command"] = " ".join(cmd)
        start = time.time()
        log_path = logs / f"redo_{name}.log"
        with log_path.open("w") as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1",
                                        "TRANSFORMERS_OFFLINE": "1"})
        entry["seconds"] = round(time.time() - start, 1)
        entry["exit_code"] = code
        entry["log"] = str(log_path.relative_to(ROOT))
        if code == 0 and (out / "report.json").exists():
            entry["status"] = "done"
            print(f"[done] {name} in {entry['seconds']} s")
        else:
            entry["status"] = "failed"
            tail = subprocess.run(["tail", "-n", "20", str(log_path)],
                                  capture_output=True, text=True).stdout
            entry["error_tail"] = tail
            print(f"[fail] {name} exit {code}; see {log_path}")
        (logs / "redo_status.json").write_text(json.dumps(status, indent=2))
    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (logs / "redo_status.json").write_text(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
