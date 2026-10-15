"""Stage 1 - frame extraction (CPU).

Video -> sharp, evenly spaced frames tagged with their source video.

    python src/extract_frames.py --config configs/default.yaml

Where it sits in the pipeline: stage 1 of nine, the only stage that touches
the recordings themselves; everything downstream consumes the JPEGs and the
index, so a change here ripples through reconstruction quality, which is why
the sharpness rule is a config value and not a hardcoded number. See
docs/code_map.md for the order and the files each stage reads and writes.

Inputs: the two recordings named on the command line. Outputs (under
cfg["paths"]["cache_dir"]): cache/frames/a/*.jpg and b/*.jpg plus
cache/frames/index.json (frame file, source frame index, timestamp and the
Blur score that kept it). The blur score is the variance of the Laplacian
computed on the grayscale frame, in arbitrary units; it is the only sharpness
measure in the pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    add_common_args,
    apply_overrides,
    cache_dir,
    load_config,
    read_json,
    repo_path,
    setup_logging,
    stamp,
    write_json,
)

log = setup_logging("extract_frames")


def _scan_video(video: Path, fps: float, blur_threshold: float,
                jpeg_quality: int = 92) -> tuple[float, list[dict]]:
    """Sample every ``src_fps/fps``-th frame, keep sharp ones, JPEG-encode them.

    Frames are kept encoded to bound RAM (a 1080p BGR frame is ~6 MB).
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    step = max(1, int(round(src_fps / fps)))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    candidates: list[dict] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur >= blur_threshold:
                ok_enc, buf = cv2.imencode(".jpg", frame, encode_params)
                if ok_enc:
                    candidates.append({
                        "source_frame": idx,
                        "time_s": round(idx / src_fps, 3),
                        "blur": round(blur, 1),
                        "jpeg": buf.tobytes(),
                    })
        idx += 1
    cap.release()
    return src_fps, candidates


def _scan_blurs(video: Path, fps: float) -> tuple[float, list[tuple[int, float, float]]]:
    """Blur-only scan of the sampled frames (no encoding): (idx, time, blur)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    step = max(1, int(round(src_fps / fps)))
    out: list[tuple[int, float, float]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.append((idx, round(idx / src_fps, 3),
                        float(cv2.Laplacian(gray, cv2.CV_64F).var())))
        idx += 1
    cap.release()
    return src_fps, out


def _evenly_spaced(items: list[dict], k: int) -> list[dict]:
    """Keep at most k items, evenly spaced over the sequence.

    Coverage first on purpose: taking the sharpest k frames instead would
    cluster them where the recording was still (typically the first second) and
    leave the end of the clip unrepresented, which produces a reconstruction
    that covers only part of the table. Spacing keeps k points across the whole
    take even when the blur scores are unevenly distributed.
    """
    if len(items) <= k:
        return items
    picks = np.linspace(0, len(items) - 1, num=k).round().astype(int)
    return [items[i] for i in sorted(set(picks.tolist()))]


def extract_one(video: Path, tag: str, cfg: dict) -> dict:
    """Extract one video (tag 'a' or 'b') into its cache directory.

    Order matters: _scan_video samples and keeps the frames that pass the
    absolute blur threshold, _evenly_spaced picks k of them for coverage, and
    only then, if the fallback flag is set AND too few frames came through,
    the adaptive_sharpness path rescans the whole video for blur scores and
    re-runs _scan_video with the threshold that picks the sharpest N by rank.
    The fallback is deliberately after the spacing step: the two overrides
    (threshold by rank, spacing for coverage) are independent knobs that can
    mix, which is what v1.1.0 verified byte-identical on the working scenes.

    The stale-frame cleanup before writing keeps --force reruns honest: without
    it, frames left by an earlier run with different settings would survive in
    the cache and the index would lie about what the reconstruction saw.
    """
    fcfg = cfg["frames"]
    log.info("[%s] scanning %s (fps=%.2f, blur>=%.0f)",
             tag, video, fcfg["fps"], fcfg["blur_threshold"])
    src_fps, candidates = _scan_video(video, fcfg["fps"], fcfg["blur_threshold"],
                                      fcfg.get("jpeg_quality", 92))
    log.info("[%s] %d sharp candidates after blur filtering", tag, len(candidates))

    keep = _evenly_spaced(candidates, int(fcfg["max_frames_per_video"]))
    if len(keep) < int(fcfg["min_frames"]) and bool(fcfg.get("adaptive_sharpness", False)):
        # Usability fallback: the absolute blur threshold left too few frames, so
        # take the sharpest N by rank instead (a global rule, no per scene value).
        _fps, blurs = _scan_blurs(video, fcfg["fps"])
        if blurs:
            n = int(fcfg["max_frames_per_video"])
            ranked = sorted(blurs, key=lambda b: -b[2])[:max(n, int(fcfg["min_frames"]))]
            thr = ranked[-1][2]
            log.warning("[%s] adaptive sharpness fallback: the absolute threshold left "
                        "%d frame(s), using the sharpest %d instead (blur >= %.1f)",
                        tag, len(keep), len(ranked), thr)
            src_fps, candidates = _scan_video(video, fcfg["fps"], thr - 1e-6,
                                              fcfg.get("jpeg_quality", 92))
            keep = _evenly_spaced(candidates, n)
    if len(keep) < int(fcfg["min_frames"]):
        raise RuntimeError(
            f"[{tag}] only {len(keep)} frames survived (min_frames="
            f"{fcfg['min_frames']}) - lower frames.blur_threshold or re-record"
        )

    tag_dir = cache_dir(cfg, "frames", tag)
    for old in tag_dir.glob("frame_*.jpg"):    # --force reruns: drop stale frames
        old.unlink()
    frames = []
    for i, item in enumerate(keep):
        name = f"frame_{item['source_frame']:05d}.jpg"
        (tag_dir / name).write_bytes(item["jpeg"])
        frames.append({
            "file": f"{tag}/{name}",
            "source_frame": item["source_frame"],
            "time_s": item["time_s"],
            "blur": item["blur"],
        })
        log.info("[%s] %2d/%d  %s  t=%.2fs  blur=%.0f",
                 tag, i + 1, len(keep), name, item["time_s"], item["blur"])

    return {
        "path": str(video),
        "src_fps": round(src_fps, 3),
        "candidates": len(candidates),
        "frames": frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--blur_threshold", type=float, default=None,
                        help="override frames.blur_threshold (per-scene tuning: "
                             "a soft recording may need 25 instead of 100)")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    if args.blur_threshold is not None:
        cfg["frames"]["blur_threshold"] = args.blur_threshold

    index_path = cache_dir(cfg, "frames") / "index.json"
    if index_path.exists() and not args.force:
        old = read_json(index_path)
        if (old.get("videos", {}).get("a", {}).get("path"),
                old.get("videos", {}).get("b", {}).get("path")) == (
                str(repo_path(cfg["paths"]["video_a"])),
                str(repo_path(cfg["paths"]["video_b"]))):
            log.info("frames already extracted for these videos (use --force to redo)")
            return 0

    video_a = repo_path(cfg["paths"]["video_a"])
    video_b = repo_path(cfg["paths"]["video_b"])
    for video in (video_a, video_b):
        if not video.is_file():
            raise FileNotFoundError(
                f"input video missing: {video}\n"
                "record the two videos per docs/data.md and place them in data/"
            )

    index = {
        "created": stamp(),
        "config_hash": cfg["_config_hash"],
        "videos": {
            "a": extract_one(video_a, "a", cfg),
            "b": extract_one(video_b, "b", cfg),
        },
    }
    write_json(index_path, index)
    log.info("wrote %s (%d + %d frames)",
             index_path, len(index["videos"]["a"]["frames"]),
             len(index["videos"]["b"]["frames"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
