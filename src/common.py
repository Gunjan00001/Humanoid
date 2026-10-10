"""Shared helpers for all pipeline stages.

Conventions (see docs/architecture.md):
- one interface:  ``python src/<stage>.py --config configs/default.yaml``
- paths come from the config or the CLI, never hard-coded;
- every artifact carries provenance (config hash, source video, frame files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the YAML config. Only stdlib + pyyaml are required."""
    import yaml  # local import keeps `--help` fast

    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable short hash of the config, for provenance in every artifact."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# paths


def repo_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root; expand ~."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def cache_dir(cfg: dict[str, Any], *parts: str) -> Path:
    out = repo_path(cfg["paths"]["cache_dir"]).joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def out_dir(cfg: dict[str, Any], *parts: str) -> Path:
    out = repo_path(cfg["paths"]["out_dir"]).joinpath(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_model(ref: str, model_dir: str | Path) -> str:
    """Turn a config model reference into something loadable offline.

    - an absolute path or ``~/...`` path is expanded and returned as str;
    - an HF repo id is first looked up as ``<model_dir>/<basename>`` (the layout
      produced by scripts/download_models.sh) and returned as a local path;
    - otherwise the repo id is returned unchanged (requires internet).
    """
    if ref.startswith(("~", "/", "./")) or Path(ref).is_absolute():
        return str(Path(ref).expanduser())
    candidate = Path(model_dir).expanduser() / ref.split("/")[-1]
    if candidate.is_dir():
        return str(candidate)
    logging.getLogger("common").warning(
        "model '%s' not found under %s - falling back to the HF repo id "
        "(needs internet; compute nodes may not have it)", ref, model_dir
    )
    return ref


# ---------------------------------------------------------------------------
# json helpers


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)


def _json_default(obj: Any):
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON-serializable: {type(obj)}")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# CLI plumbing shared by all stage scripts


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml",
                        help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--video_a", default=None, help="override paths.video_a")
    parser.add_argument("--video_b", default=None, help="override paths.video_b")
    parser.add_argument("--out", dest="out_dir", default=None,
                        help="override paths.out_dir")
    parser.add_argument("--cache", dest="cache_dir", default=None,
                        help="override paths.cache_dir")
    parser.add_argument("--force", action="store_true",
                        help="redo this stage even if its cache exists")


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, cfgpath in (
        ("video_a", ("paths", "video_a")),
        ("video_b", ("paths", "video_b")),
        ("out_dir", ("paths", "out_dir")),
        ("cache_dir", ("paths", "cache_dir")),
    ):
        value = getattr(args, key, None)
        if value:
            cfg[cfgpath[0]][cfgpath[1]] = value
    return cfg


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    return logging.getLogger(name)


class Timer:
    """Wall-clock context manager used by run.py for per-stage timings."""

    def __init__(self) -> None:
        self.elapsed_s: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_s = round(time.perf_counter() - self._t0, 2)


# ---------------------------------------------------------------------------
# artifact expectations (used by run.py to skip finished stages)


def stage_outputs(cfg: dict[str, Any]) -> dict[str, list[Path]]:
    """Map stage name -> files that prove the stage finished."""
    cache = repo_path(cfg["paths"]["cache_dir"])
    out = repo_path(cfg["paths"]["out_dir"])
    recon_dir = cache / "reconstruction"
    if cfg.get("reconstruction", {}).get("use_icp_refinement"):
        recon_dir = recon_dir / "aligned"
    return {
        "extract_frames": [cache / "frames" / "index.json"],
        "reconstruct": [recon_dir / "meta.json"],
        "align": [cache / "reconstruction" / "aligned" / "transform.json"],
        "diff": [cache / "diff" / "clusters.json"],
        "describe": [cache / "semantics" / "matches.json"],
        "visualize": [out / "scene.ply"],
        "report": [out / "report.json", out / "report.md"],
    }
