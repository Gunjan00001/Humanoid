"""Generate two tiny synthetic videos for smoke-testing the CPU stages.

NOT part of the pipeline and NOT a source of results - this only exercises
frame extraction (and the diff plumbing) before real recordings exist.

    python tests/make_test_videos.py            # -> cache/test_videos/{a,b}.mp4

Both videos show the same "scene" (fixed noise texture + coloured blocks);
video B has one block moved and one added, so the diff stages have something
to chew on. A blurred segment around t=4s is injected to test the blur filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "cache" / "test_videos"

W, H, FPS, SECONDS = 960, 540, 30, 12


def base_scene(rng: np.random.Generator) -> np.ndarray:
    """Textured background so frame statistics are not degenerate."""
    noise = rng.integers(40, 215, size=(H, W, 3), dtype=np.uint8)
    bg = noise.copy()
    # fine diagonal stripes add high-frequency detail (like real footage)
    for x in range(0, W, 7):
        cv2.line(bg, (x, 0), (x + 40, H), (255, 255, 255), 1)
    # a few static rectangles as "furniture"
    cv2.rectangle(bg, (80, 300), (420, 480), (120, 90, 60), -1)
    cv2.rectangle(bg, (520, 220), (860, 330), (70, 110, 70), -1)
    # keep some texture on the rectangles too
    patch = rng.integers(80, 160, size=(180, 340, 3), dtype=np.uint8)
    bg[300:480, 80:420] = cv2.addWeighted(bg[300:480, 80:420], 0.5, patch, 0.5, 0)
    return bg


def render_frame(base: np.ndarray, objects: list[tuple], t: float) -> np.ndarray:
    """Pan+wobble the scene; draw coloured blocks."""
    dx = 60 * np.sin(2 * np.pi * t / 6.0)
    dy = 18 * np.sin(2 * np.pi * t / 3.0)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    frame = cv2.warpAffine(base, m, (W, H), borderMode=cv2.BORDER_REFLECT)
    for (x, y, w, h, color) in objects:
        cv2.rectangle(frame, (int(x + dx), int(y + dy)),
                      (int(x + dx + w), int(y + dy + h)), color, -1)
    return frame


def make_video(path: Path, objects: list[tuple], seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = base_scene(rng)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = FPS * SECONDS
    for i in range(n):
        t = i / FPS
        frame = render_frame(base, objects, t)
        if 110 <= i < 140:                      # blurred stretch (dropped by filter)
            frame = cv2.GaussianBlur(frame, (31, 31), 12)
        writer.write(frame)
    writer.release()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # object tuple: x, y, w, h, BGR colour
    red = (40, 40, 220)
    blue = (220, 80, 40)
    green = (60, 200, 60)

    # A: red block left, blue block centre            (green absent)
    a_objects = [(150, 360, 70, 70, red), (600, 240, 80, 60, blue)]
    # B: red moved right; green added                (same blue)
    b_objects = [(300, 380, 70, 70, red), (600, 240, 80, 60, blue), (760, 400, 60, 60, green)]

    make_video(OUT_DIR / "a.mp4", a_objects, seed=1)
    make_video(OUT_DIR / "b.mp4", b_objects, seed=2)
    print(f"wrote {OUT_DIR}/a.mp4 and {OUT_DIR}/b.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
