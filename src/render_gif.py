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

import json
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


LEGEND = [("removed", (242, 51, 51)), ("added", (64, 230, 89)),
          ("moved", (255, 153, 26)), ("replaced", (242, 63, 242)),
          ("unverified", (89, 191, 217))]

# colours the GIF must keep exact: change colours, the dark background and the
# label white. A grey-dominated scene makes the adaptive 256-colour palette
# drop/mangle these isolated colours (measured: magenta rendered as grey), so
# they are quantized together with every frame.
KEY_COLORS = [c for _, c in LEGEND] + [(16, 20, 24), (242, 242, 242)]


def gif_quantize(frame):
    """Quantize one RGB frame for GIF while keeping the change colours exact.

    A strip of the key colours is attached below the frame before the adaptive
    quantization, so the palette reserves slots for them; the strip is cropped
    off afterwards.
    """
    from PIL import Image

    w, h = frame.size
    keys = Image.new("RGB", (w, 8 * len(KEY_COLORS)))
    for i, color in enumerate(KEY_COLORS):
        keys.paste(Image.new("RGB", (w, 8), color), (0, 8 * i))
    combo = Image.new("RGB", (w, h + keys.height))
    combo.paste(frame, (0, 0))
    combo.paste(keys, (0, h))
    return combo.quantize(colors=256).crop((0, 0, w, h))


def _legend_strip(width: int, height: int, present: list[str]):
    """Colour legend drawn OUTSIDE the 3D panel (its own strip)."""
    from PIL import Image, ImageDraw

    strip = Image.new("RGB", (width, height), (16, 20, 24))
    draw = ImageDraw.Draw(strip)
    x = 12
    for name, color in LEGEND:
        if name not in present:
            continue
        draw.rectangle([x, height // 2 - 5, x + 10, height // 2 + 5], fill=color)
        draw.text((x + 15, height // 2 - 6), name, fill=(225, 225, 225))
        x += 15 + 9 * (len(name) + 2)
    return strip


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
    width, height = int(vcfg.get("gif_width", 1100)), int(vcfg.get("gif_height", 780))
    panel_h = int(height * 0.42)
    legend_h = 22
    top = _top_panel(Path(before_file) if before_file else None,
                     Path(after_file) if after_file else None, width, panel_h)
    # legend reflects the change types actually present (from report.json)
    present = ["removed", "added", "moved", "replaced"]
    try:
        rep = json.loads((out_dir / "report.json").read_text())
        # the GIF shows the verified changes only (unverified are hidden by
        # default in the viewer), so the legend lists exactly those types
        types = {c["type"] for c in rep.get("changes", [])}
        present = [t for t in present if t in types]
    except Exception:  # noqa: BLE001
        pass
    legend = _legend_strip(width, legend_h, present)

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
            panel = Image.open(shot).convert("RGB").resize(
                (width, height - panel_h - legend_h))
            frame = Image.new("RGB", (width, height), (16, 20, 24))
            frame.paste(top, (0, 0))
            frame.paste(legend, (0, panel_h))
            frame.paste(panel, (0, panel_h + legend_h))
            gif_frames.append(gif_quantize(frame))

    gif_path = out_dir / "change.gif"
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=150, loop=0, optimize=False)
    log.info("chrome_gif: wrote %s (%d frames from viewer.html)",
             gif_path, len(gif_frames))
    return gif_path


if __name__ == "__main__":
    import argparse
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from src.common import load_config, repo_path
    from src.geometry import build_change_list

    ap = argparse.ArgumentParser(description="re-render change.gif from viewer.html")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg["paths"]["cache_dir"] = a.cache
    out = repo_path(a.out)
    # evidence frames of the first confirmed change (from the crops cache)
    crops_path = repo_path(a.cache) / "semantics" / "crops.json"
    before_file = after_file = None
    if crops_path.exists():
        crops = json.loads(crops_path.read_text())
        changes = build_change_list(cfg)
        if changes:
            ci = crops.get(str(changes[0]["cluster_ids"][0]), {})
            for f in (ci.get("frame"),):
                before_file = str(repo_path(a.cache) / "frames" / f) if f else None
            f = ci.get("frame_other")
            after_file = str(repo_path(a.cache) / "frames" / f) if f else None
    logging.basicConfig(level=logging.INFO)
    render_gif(cfg, out, before_file, after_file, frames=int(cfg["visualize"]["gif_frames"]))
