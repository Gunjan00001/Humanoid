"""Render the scene GIF from viewer.html with headless Chrome (CPU).

The GIF must match the interactive viewer, so its 3D panel is rendered by the
same three.js code (shaded, distance-sized splats, change boxes + labels),
screenshotted at a series of orbit angles. Layout per frame:

    [ before (video A) | after (video B) ]      <- larger, top row
    [        3D view from viewer.html    ]      <- large panel, orbiting

Chrome runs with SwiftShader (no GPU needed); if Chrome is unavailable the
caller falls back to the matplotlib renderer. Returns the GIF path, or None.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger("render_gif")

CHROME_FLAGS = ["--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
                "--use-gl=swiftshader", "--hide-scrollbars"]


def _chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _top_panel(before_path: Path | None, after_path: Path | None, width: int,
               height: int):
    """[before | after] strip with titles, as a PIL image."""
    from PIL import Image, ImageDraw

    gap = 8
    panel_w = (width - gap) // 2
    strip = Image.new("RGB", (width, height), (16, 20, 24))
    draw = ImageDraw.Draw(strip)
    for k, (path, title) in enumerate(((before_path, "before (video A)"),
                                       (after_path, "after (video B)"))):
        x0 = k * (panel_w + gap)
        if path and Path(path).exists():
            img = Image.open(path).convert("RGB")
            scale = min(panel_w / img.width, (height - 24) / img.height)
            img = img.resize((max(1, int(img.width * scale)),
                              max(1, int(img.height * scale))))
            strip.paste(img, (x0 + (panel_w - img.width) // 2, 24))
        draw.text((x0 + 8, 6), title, fill=(235, 235, 235))
    return strip


def render_gif(cfg: dict, out_dir: Path, before_file: str | None,
               after_file: str | None, frames: int = 24) -> Path | None:
    """Screenshot viewer.html at `frames` orbit angles and assemble the GIF."""
    chrome = _chrome()
    viewer = out_dir / "viewer.html"
    if chrome is None or not viewer.exists():
        log.warning("chrome_gif: %s missing - falling back to matplotlib",
                    "chrome" if chrome is None else viewer)
        return None

    vcfg = cfg["visualize"]
    width, height = int(vcfg.get("gif_width", 1400)), int(vcfg.get("gif_height", 520))
    panel_h = height // 2
    top = _top_panel(Path(before_file) if before_file else None,
                     Path(after_file) if after_file else None, width, panel_h)

    from PIL import Image
    gif_frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(frames):
            az = -52 + 360.0 * k / frames
            shot = Path(tmp) / f"az_{k:02d}.png"
            url = f"file://{viewer}?shot=1&az={az:.1f}&elev=40"
            cmd = [chrome, *CHROME_FLAGS, f"--window-size={width},{height}",
                   "--virtual-time-budget=6000", f"--screenshot={shot}", url]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if not shot.exists():
                log.warning("chrome_gif: screenshot %d failed (%s)", k,
                            (proc.stderr or "")[-200:])
                return None
            panel = Image.open(shot).convert("RGB").resize((width, height - panel_h))
            frame = Image.new("RGB", (width, height), (16, 20, 24))
            frame.paste(top, (0, 0))
            frame.paste(panel, (0, panel_h))
            gif_frames.append(frame)

    gif_path = out_dir / "change.gif"
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=150, loop=0)
    log.info("chrome_gif: wrote %s (%d frames from viewer.html)",
             gif_path, len(gif_frames))
    return gif_path
