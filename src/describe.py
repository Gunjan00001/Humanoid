"""Stage 5 - semantics (GPU, 1-2 devices).

Per cluster:
  1. pick the best frame of its own video (most points visible, least occlusion),
  2. crop it (mapped back to the original JPEG resolution),
  3. ask Qwen3-VL for a short label + one-sentence description (structured JSON),
  4. embed every crop with DINOv2 and merge a removed+added pair with a
     plausible match at a different 3D position into one *moved* object
     (Qwen3-VL then writes the final move sentence from the two crops).

    python src/describe.py --config configs/default.yaml
    python src/describe.py --config configs/default.yaml --skip_vlm      # crops only
    python src/describe.py --config configs/default.yaml --limit_clusters 4

Reads  cache/diff/... + cache/reconstruction/... + cache/frames/
Writes cache/semantics/{crops/*.png, crops.json, vlm.json, embeddings.npz, matches.json}

All VLM answers are cached incrementally: re-runs never re-spend GPU time on
the same crop. Model lifecycle: the VLM stays loaded for labels *and* final
move sentences; DINOv2 is loaded on the least-loaded GPU and freed before the
sentences are generated.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import (  # noqa: E402
    add_common_args,
    apply_overrides,
    cache_dir,
    load_config,
    pin_idle_gpus,
    read_json,
    repo_path,
    resolve_model,
    setup_logging,
    stamp,
    write_json,
)
from src.geometry import (  # noqa: E402
    load_reconstruction,
    model_bbox_to_original,
    project_points,
    sample_depth,
)

log = setup_logging("describe")

PROMPT_SINGLE = (
    "This image is a crop from a small indoor scene. A 3D change-detection "
    "pipeline flagged this region as an object that changed between two "
    "recordings. Identify the object.\n"
    'Answer with JSON only, no other text: {"label": "<2-4 word object name>", '
    '"description": "<one short sentence about the object>"}'
)
PROMPT_MOVE = (
    "These two crops come from the same small indoor scene at two different "
    "times. The same object appears in both, moved to a different place. "
    "The first image is the old position, the second image is the new position.\n"
    'Answer with JSON only, no other text: {"label": "<2-4 word object name>", '
    '"sentence": "<one sentence describing the move, e.g. the blue mug moved '
    'from the desk to the floor>"}'
)


# ---------------------------------------------------------------------------
# crop selection


def best_crop_for_cluster(cid: int, points: np.ndarray, tag: str, recon: dict, cfg: dict) -> dict | None:
    """Pick the frame of the cluster's own video that sees it best, crop it."""
    scfg = cfg["semantics"]
    conf_q = float(np.percentile(recon["depth_conf"], cfg["reconstruction"]["conf_percentile"]))
    own_frames = [i for i, t in enumerate(recon["tags"]) if t == tag]
    if len(own_frames) > int(scfg["max_crop_frames"]):
        sel = np.linspace(0, len(own_frames) - 1, int(scfg["max_crop_frames"])).round().astype(int)
        own_frames = [own_frames[i] for i in sorted(set(sel.tolist()))]

    best = None
    for i in own_frames:
        uv, z = project_points(points, recon["extrinsic"][i], recon["intrinsic"][i])
        observed, in_bounds = sample_depth(recon["depth"][i], uv)
        conf_ok = sample_depth(recon["depth_conf"][i], uv)[0] >= conf_q
        visible = in_bounds & conf_ok & (np.abs(z - observed) <= 0.05)
        n_vis = int(visible.sum())
        if best is None or n_vis > best["n_visible"]:
            best = {"frame_index": i, "n_visible": n_vis, "visible": visible, "uv": uv}

    if best is None or best["n_visible"] < 5:
        log.warning("cluster %d: no good view found (%s visible points)",
                    cid, None if best is None else best["n_visible"])
        return None

    uv = best["uv"][best["visible"]]
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    margin = float(scfg["crop_margin"])
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    frame_idx = best["frame_index"]
    tr = recon["meta"]["frames"][frame_idx]
    bbox_orig = model_bbox_to_original((x1 - mx, y1 - my, x2 + mx, y2 + my), tr)

    # enforce a minimum crop size around its centre
    min_px = int(scfg["min_crop_pixels"])
    cx, cy = (bbox_orig[0] + bbox_orig[2]) // 2, (bbox_orig[1] + bbox_orig[3]) // 2
    bx1 = max(0, min(cx - min_px // 2, max(0, tr["orig_w"] - min_px)))
    by1 = max(0, min(cy - min_px // 2, max(0, tr["orig_h"] - min_px)))
    bx2 = min(tr["orig_w"], max(bbox_orig[2], bx1 + min_px))
    by2 = min(tr["orig_h"], max(bbox_orig[3], by1 + min_px))
    bbox_orig = (bx1, by1, bx2, by2)

    import cv2
    frame_path = repo_path(cfg["paths"]["cache_dir"]) / "frames" / tr["file"]
    img = cv2.imread(str(frame_path))
    if img is None:
        log.warning("cluster %d: cannot read frame %s", cid, frame_path)
        return None
    crop = img[bbox_orig[1]:bbox_orig[3], bbox_orig[0]:bbox_orig[2]]
    if crop.size == 0:
        log.warning("cluster %d: empty crop %s", cid, bbox_orig)
        return None
    crop_path = cache_dir(cfg, "semantics", "crops") / f"cluster_{cid:03d}.png"
    cv2.imwrite(str(crop_path), crop)
    log.info("cluster %d: best view %s (%d visible pts), crop %s",
             cid, tr["file"], best["n_visible"], crop_path.name)
    return {"cluster_id": cid, "frame": tr["file"], "frame_index": int(frame_idx),
            "bbox_orig": list(bbox_orig), "n_visible": best["n_visible"],
            "crop": str(crop_path)}


# ---------------------------------------------------------------------------
# Qwen3-VL


def load_vlm(cfg: dict):
    import torch
    from transformers import AutoProcessor

    # pin the N idlest GPUs BEFORE torch touches CUDA (device_map="auto" would
    # otherwise spread the model over every visible A100 - see gpu.md)
    pin_idle_gpus(int(cfg["gpu"]["vlm_gpus"]), log)

    key = cfg["semantics"]["vlm"]
    ref = cfg["models"]["vlm_primary" if key == "primary" else "vlm_fallback"]
    ref = resolve_model(ref, cfg["paths"]["model_dir"])
    log.info("loading VLM: %s (device_map=auto, %s)", ref, cfg["gpu"]["dtype"])
    try:
        from transformers import AutoModelForImageTextToText as ModelCls
    except ImportError:  # older transformers
        from transformers import Qwen3VLForConditionalGeneration as ModelCls  # type: ignore
    dtype = getattr(torch, cfg["gpu"]["dtype"])
    model = ModelCls.from_pretrained(ref, dtype=dtype, device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(ref)
    log.info("VLM ready on %s", getattr(model, "hf_device_map", "?"))
    return model, processor


def vlm_answer(model, processor, images: list, prompt: str) -> str:
    import torch

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    return processor.batch_decode(
        [out[0][inputs["input_ids"].shape[1]:]], skip_special_tokens=True
    )[0].strip()


def parse_json_reply(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    return {"label": first[:60] or None, "description": text[:300] or None}


# ---------------------------------------------------------------------------
# DINOv2


def load_matcher(cfg: dict):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    ref = resolve_model(cfg["models"]["matcher"], cfg["paths"]["model_dir"])
    # place on the last GPU: with device_map="auto" the VLM leaves it least used
    device = f"cuda:{max(0, torch.cuda.device_count() - 1)}"
    log.info("loading matcher: %s -> %s", ref, device)
    processor = AutoImageProcessor.from_pretrained(ref)
    model = AutoModel.from_pretrained(ref, dtype=torch.float16).to(device).eval()
    return model, processor


def embed_crops(model, processor, paths: list[str]) -> np.ndarray:
    import torch
    from PIL import Image

    vecs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(model.device, torch.float16)
        with torch.no_grad():
            out = model(**inputs)
        v = out.last_hidden_state[:, 0]          # CLS token
        v = torch.nn.functional.normalize(v, dim=-1)
        vecs.append(v[0].float().cpu().numpy())
    return np.stack(vecs) if vecs else np.zeros((0, 1024), dtype=np.float32)


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--skip_vlm", action="store_true",
                        help="crops only; no labels, embeddings or matching")
    parser.add_argument("--limit_clusters", type=int, default=None,
                        help="debug: only process the first N clusters")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    sem_dir = cache_dir(cfg, "semantics")
    matches_path = sem_dir / "matches.json"
    if matches_path.exists() and not args.force:
        log.info("semantics already cached (%s); use --force to redo", matches_path)
        return 0

    recon = load_reconstruction(cfg)
    clusters = read_json(repo_path(cfg["paths"]["cache_dir"]) / "diff" / "clusters.json")["clusters"]
    points_npz = np.load(repo_path(cfg["paths"]["cache_dir"]) / "diff" / "points.npz")
    if args.limit_clusters:
        clusters = clusters[:args.limit_clusters]
    keep_ids = {c["id"] for c in clusters}
    cluster_points: dict[int, np.ndarray] = {}
    for c in clusters:
        tag = c["source_tag"]
        arr = points_npz[f"{tag}_points"]
        labels = points_npz[f"{tag}_labels"]
        cluster_points[c["id"]] = arr[labels == c["id"]].astype(np.float32)
    log.info("describing %d cluster(s)", len(clusters))

    # ---- 1) crops -------------------------------------------------------
    crops_path = sem_dir / "crops.json"
    crops = read_json(crops_path) if crops_path.exists() and not args.force else {}
    crops = {k: v for k, v in crops.items() if int(k) in keep_ids}
    for c in clusters:
        cid = c["id"]
        if str(cid) in crops:
            continue
        info = best_crop_for_cluster(cid, cluster_points[cid], c["source_tag"], recon, cfg)
        if info:
            crops[str(cid)] = info
    write_json(crops_path, crops)

    # ---- 2) VLM labels ---------------------------------------------------
    vlm_path = sem_dir / "vlm.json"
    vlm_cache = read_json(vlm_path) if vlm_path.exists() and not args.force else {}
    vlm_cache = {k: v for k, v in vlm_cache.items() if int(k) in keep_ids}
    vlm_model = vlm_processor = None
    if not args.skip_vlm and crops:
        from PIL import Image
        vlm_model, vlm_processor = load_vlm(cfg)
        for c in clusters:
            cid = str(c["id"])
            if cid not in crops or cid in vlm_cache:
                continue
            img = Image.open(crops[cid]["crop"]).convert("RGB")
            reply = vlm_answer(vlm_model, vlm_processor, [img], PROMPT_SINGLE)
            parsed = parse_json_reply(reply)
            vlm_cache[cid] = {"label": parsed.get("label"),
                              "description": parsed.get("description"),
                              "raw": reply[:500], "created": stamp()}
            log.info("cluster %s -> %s", cid, vlm_cache[cid]["label"])
            write_json(vlm_path, vlm_cache)   # incremental: crash-safe cache
    write_json(vlm_path, vlm_cache)

    # ---- 3) DINOv2 embeddings + move matching ---------------------------
    matches: list[dict] = []
    if not args.skip_vlm and crops:
        import torch
        from PIL import Image

        matcher, matcher_processor = load_matcher(cfg)
        ids = [c["id"] for c in clusters if str(c["id"]) in crops]
        paths = [crops[str(i)]["crop"] for i in ids]
        emb = embed_crops(matcher, matcher_processor, paths)
        np.savez(sem_dir / "embeddings.npz",
                 ids=np.array(ids), vectors=emb,
                 files=np.array([crops[str(i)]["frame"] for i in ids]))
        del matcher, matcher_processor
        gc.collect()
        torch.cuda.empty_cache()

        scfg = cfg["semantics"]
        by_id = {c["id"]: c for c in clusters}
        pairs = []
        for i, ri in enumerate(ids):
            if by_id[ri]["type"] != "removed":
                continue
            for j, aj in enumerate(ids):
                if by_id[aj]["type"] != "added":
                    continue
                sim = float(np.dot(emb[i], emb[j]))
                dist = float(np.linalg.norm(
                    np.asarray(by_id[ri]["centroid"]) - np.asarray(by_id[aj]["centroid"])))
                if sim >= float(scfg["match_similarity_threshold"]) and dist >= float(scfg["match_min_distance"]):
                    pairs.append({"removed_id": ri, "added_id": aj,
                                  "similarity": round(sim, 4), "distance": round(dist, 4)})
        pairs.sort(key=lambda p: -p["similarity"])
        used: set[int] = set()
        for p in pairs:
            if p["removed_id"] in used or p["added_id"] in used:
                continue
            used.update({p["removed_id"], p["added_id"]})
            entry = dict(p)
            entry["label"] = (vlm_cache.get(str(p["removed_id"]), {}).get("label")
                              or vlm_cache.get(str(p["added_id"]), {}).get("label"))
            imgs = [Image.open(crops[str(p["removed_id"])]["crop"]).convert("RGB"),
                    Image.open(crops[str(p["added_id"])]["crop"]).convert("RGB")]
            reply = vlm_answer(vlm_model, vlm_processor, imgs, PROMPT_MOVE)
            parsed = parse_json_reply(reply)
            entry["label"] = parsed.get("label") or entry["label"]
            entry["sentence"] = parsed.get("sentence") or parsed.get("description")
            entry["raw"] = reply[:500]
            log.info("moved %s: %s (sim %.3f)", [p["removed_id"], p["added_id"]],
                     entry["sentence"], p["similarity"])
            matches.append(entry)
        log.info("move matches: %d", len(matches))

    write_json(matches_path, matches)
    log.info("wrote %s", matches_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
