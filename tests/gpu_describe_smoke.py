"""GPU smoke test for src/describe.py's model paths (validation, not results).

Loads the fallback VLM (Qwen3-VL-8B-Instruct), labels a real kitchen crop with
both prompts, then embeds two views with DINOv2 and prints their similarity.
This validates the transformers-5 API usage, the chat template, generation and
the embedding pipeline - with no dependence on the demo videos.

    gpu-run -t 00:40:00 -g 1 '~/miniconda3/envs/spatialdiff/bin/python tests/gpu_describe_smoke.py'
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src import describe  # noqa: E402
from src.common import load_config, setup_logging  # noqa: E402

log = setup_logging("describe_smoke")


def main() -> int:
    cfg = load_config("configs/default.yaml")
    cfg["semantics"]["vlm"] = "fallback"          # 8B fits one GPU
    frame_a = ROOT / "cache/gpu_smoke/frames/a/frame_00000.png"
    frame_b = ROOT / "cache/gpu_smoke/frames/b/frame_00002.png"
    for p in (frame_a, frame_b):
        if not p.exists():
            raise FileNotFoundError(f"run tests/prepare_gpu_smoke.py first: {p}")

    t0 = time.time()
    model, processor, vlm_ref = describe.load_vlm(cfg)
    log.info("VLM loaded in %.1fs (%s)", time.time() - t0, vlm_ref)

    crop_a = Image.open(frame_a).convert("RGB")
    t0 = time.time()
    reply = describe.vlm_answer(model, processor, [crop_a], describe.PROMPT_SINGLE)
    log.info("single-image reply in %.1fs: %s", time.time() - t0, reply)
    parsed = describe.parse_json_reply(reply)
    print("PARSED single:", parsed)
    assert parsed.get("label") or parsed.get("description"), "JSON parser produced nothing"

    crop_b = Image.open(frame_b).convert("RGB")
    t0 = time.time()
    reply = describe.vlm_answer(model, processor, [crop_a, crop_b], describe.PROMPT_MOVE)
    log.info("two-image reply in %.1fs: %s", time.time() - t0, reply)
    print("PARSED move:", describe.parse_json_reply(reply))

    del model, processor
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    matcher, mproc = describe.load_matcher(cfg)
    t0 = time.time()
    emb = describe.embed_crops(matcher, mproc, [str(frame_a), str(frame_b)])
    log.info("embedded %s in %.1fs", emb.shape, time.time() - t0)
    sim = float(emb[0] @ emb[1])
    print("cosine similarity between two kitchen views: %.3f" % sim)
    assert emb.shape[0] == 2 and abs(float((emb ** 2).sum(axis=1).mean()) - 1.0) < 1e-3, \
        "embeddings are not L2-normalized"
    print("DESCRIBE SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
