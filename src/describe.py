"""Stage 5 - semantics + VLM verification (GPU, 1-2 devices).

For every candidate cluster from src/diff.py:
  1. pick the best frame of its own video (most points visible, least occlusion)
     and of the *other* video (for verified candidates),
  2. crop both at the original JPEG resolution,
  3. ask Qwen3-VL to confirm or deny the change hypothesis and to label the
     object ("changed": true/false) - photometric candidates include lighting
     and blur artefacts, so this verification step is what keeps the report
     clean. Unverified clusters (one video has no coverage) get a label only.
  4. embed confirmed crops with DINOv2; a confirmed removed+added pair with a
     plausible match at a different 3D position becomes one *moved* object
     (Qwen3-VL writes the final move sentence from the two crops).

    python src/describe.py --config configs/default.yaml
    python src/describe.py --config configs/default.yaml --skip_vlm      # crops only
    python src/describe.py --config configs/default.yaml --limit_clusters 4

Reads  cache/diff/... + cache/reconstruction/... + cache/frames/
Writes cache/semantics/{crops/*.png, crops.json, vlm.json, rejected.json,
                       embeddings.npz, matches.json}

All VLM answers are cached incrementally: re-runs never re-spend GPU time on
the same crop. Model lifecycle: the VLM stays loaded for verification, labels
and final move sentences; DINOv2 is loaded on the least-used GPU and freed
before the sentences are generated.
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

CLAIMS = {
    "removed": "the object was REMOVED (it is present in the first image and gone in the second)",
    "added": "an object was ADDED (it is absent in the first image and present in the second)",
    "replaced": "the object changed state or appearance (e.g. it was moved, flipped or replaced)",
}
PROMPT_VERIFY = (
    "These two photos show the same small region of a table at two different "
    "times. The first image is from the earlier recording, the second from the "
    "later one.\n"
    "A change detector makes this claim: {claim}\n"
    "Step 1: name the main object(s) visible in the FIRST image.\n"
    "Step 2: name the main object(s) visible in the SECOND image.\n"
    "Step 3: decide whether the claim is true. Ignore lighting, exposure, "
    "shadow and blur differences - they do not count as changes.\n"
    'Answer with JSON only: {{"first": "<objects in image 1>", '
    '"second": "<objects in image 2>", "changed": true or false, '
    '"label": "<2-4 word object name>", '
    '"sentence": "<one short sentence about what changed>"}}'
)
PROMPT_SINGLE = (
    "This image is a crop from a small indoor scene. Identify the object in it.\n"
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
PROMPT_REPLACE = (
    "These two crops show the same spot in a small indoor scene at two times. "
    "The object visible in the FIRST image was replaced by a different object "
    "in the SECOND image.\n"
    'Answer with JSON only, no other text: {"label": "<the new object, 2-4 words>", '
    '"sentence": "<one sentence naming BOTH objects, e.g. the apple was replaced '
    'by a cup>"}'
)
PROMPT_REMOVED = (
    "These two crops show the same spot in a small indoor scene at two times. "
    "An object is visible in the FIRST image and gone in the SECOND; the space "
    "is empty now.\n"
    'Answer with JSON only, no other text: {"label": "<the removed object, 2-4 words>", '
    '"sentence": "<one sentence naming the object and its removal, e.g. the chair '
    'was removed from the room>"}'
)


# ---------------------------------------------------------------------------
# crop selection


def _pose_score(recon: dict, ref: int, other: int) -> float:
    """How different are two camera poses (degrees + 60 x metres)."""
    ei, ej = recon["extrinsic"][ref], recon["extrinsic"][other]
    r_rel = ei[:, :3].T @ ej[:, :3]
    cos = float(np.clip((np.trace(r_rel) - 1) / 2, -1, 1))
    angle = float(np.degrees(np.arccos(cos)))
    dist = float(np.linalg.norm(ei[:, 3] - ej[:, 3]))
    return angle + 60.0 * dist


def best_crop_for_cluster(cid: int, points: np.ndarray, tag: str, recon: dict, cfg: dict,
                          suffix: str = "", ref_frame: int | None = None) -> dict | None:
    """Pick the frame of the given video that sees the cluster best, crop it.

    When ``ref_frame`` is given (the own-video best frame), the frame is chosen
    to be *pose-similar* to it among the sufficiently-visible candidates, so
    the two verification crops show the same view - otherwise the VLM ends up
    comparing different scenes and invents changes.
    """
    scfg = cfg["semantics"]
    conf_q = float(np.percentile(recon["depth_conf"], cfg["reconstruction"]["conf_percentile"]))
    own_frames = [i for i, t in enumerate(recon["tags"]) if t == tag]
    if len(own_frames) > int(scfg["max_crop_frames"]):
        sel = np.linspace(0, len(own_frames) - 1, int(scfg["max_crop_frames"])).round().astype(int)
        own_frames = [own_frames[i] for i in sorted(set(sel.tolist()))]

    views = []
    occ_tol = float(scfg.get("crop_occlusion_tol", 0.15))
    for i in own_frames:
        uv, z = project_points(points, recon["extrinsic"][i], recon["intrinsic"][i])
        observed, in_bounds = sample_depth(recon["depth"][i], uv)
        conf_ok = sample_depth(recon["depth_conf"][i], uv)[0] >= conf_q
        # crop-framing visibility: the region is IN VIEW and NOT occluded by a
        # surface much closer to the camera.  NOTE: this is deliberately not
        # the "object is there" test (|z-depth| small) - in the other video a
        # removed object is *gone*, so its points sit in front of the
        # background; the old test made those frames look empty and the crop
        # landed on neighbouring objects (scene3's "chair replaced" misreads).
        visible = in_bounds & conf_ok & (z <= observed + occ_tol)
        views.append({"frame_index": i, "n_visible": int(visible.sum()),
                      "visible": visible, "uv": uv})
    if not views:
        return None
    n_max = max(v["n_visible"] for v in views)
    if n_max < 5:
        log.warning("cluster %d (%s%s): no good view found (%d visible points)",
                    cid, tag, suffix, n_max)
        return None
    if ref_frame is not None:
        # prefer frames where MOST of the cluster is in view; among those the
        # pose-closest to the own view. A frame where only a few points are in
        # view puts the crop on whatever happens to be there.
        enough = [v for v in views if v["n_visible"] >= max(5, 0.5 * n_max)]
        best = min(enough, key=lambda v: _pose_score(recon, ref_frame, v["frame_index"])) \
            if enough else max(views, key=lambda v: v["n_visible"])
    else:
        best = max(views, key=lambda v: v["n_visible"])

    uv = best["uv"][best["visible"]]
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    # verification crops need context around the cluster: a tight crop of a
    # fragment can miss the very object in the other video and fake a change
    margin = float(scfg.get("verify_crop_margin", scfg["crop_margin"]))
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    frame_idx = best["frame_index"]
    tr = recon["meta"]["frames"][frame_idx]
    bbox_orig = model_bbox_to_original((x1 - mx, y1 - my, x2 + mx, y2 + my), tr)

    # enforce a minimum crop size around its centre
    min_px = int(scfg.get("verify_min_crop_pixels", scfg["min_crop_pixels"]))
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
    crop_path = cache_dir(cfg, "semantics", "crops") / f"cluster_{cid:03d}{suffix}.png"
    cv2.imwrite(str(crop_path), crop)
    log.info("cluster %d (%s): best view %s (%d visible pts), crop %s",
             cid, tag, tr["file"], best["n_visible"], crop_path.name)
    return {"cluster_id": cid, "tag": tag, "frame": tr["file"],
            "frame_index": int(frame_idx), "bbox_orig": list(bbox_orig),
            "n_visible": best["n_visible"], "crop": str(crop_path)}


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
    kwargs = {"dtype": dtype, "device_map": "auto", "low_cpu_mem_usage": True}
    mm = cfg["semantics"].get("vlm_max_memory")
    if mm:
        # explicit per-GPU sharding for a shared node (accuracy sprint): keys are
        # the VISIBLE device indices, values e.g. "38GiB". Never quantizes.
        kwargs["max_memory"] = {int(k): v for k, v in mm.items()}
        log.info("VLM max_memory=%s", kwargs["max_memory"])
    model = ModelCls.from_pretrained(ref, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(ref)
    log.info("VLM ready on %s", getattr(model, "hf_device_map", "?"))
    return model, processor, ref


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


_STOPWORDS = {"the", "a", "an", "and", "of", "with", "on", "in", "is", "are", "no",
              "none", "empty", "table", "surface", "visible", "object", "objects",
              "there", "it", "its", "small", "large", "other",
              # descriptors: colours, shapes, qualities - two different objects
              # can share them ("dark surface", "white rectangular")
              "dark", "light", "bright", "blurred", "blurry", "sharp", "shiny",
              "edge", "edges", "corner", "top", "bottom", "left", "right",
              "front", "back", "side", "middle", "centre", "center",
              "rectangular", "round", "square", "circular", "cylindrical",
              "white", "black", "blue", "yellow", "green", "red", "brown",
              "orange", "pink", "purple", "grey", "gray", "silver", "golden",
              "thin", "thick", "long", "short", "flat", "smooth", "rough",
              "partially", "part", "piece", "portion", "area", "region",
              "possibly", "maybe", "appears", "seems", "like", "looks"}


def _objects_overlap(a, b) -> bool:
    """True when two free-text object descriptions name the same thing.

    Used for the verification self-consistency check: 'digital timer' vs
    'the digital timer on the table' -> True; 'wooden comb' vs 'empty table'
    -> False (all significant words are stopwords there).
    """
    def words(s):
        s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
        return {w for w in s.split() if w and w not in _STOPWORDS}

    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    return bool(wa & wb)


def _same_objects(a, b) -> bool:
    """True when the two descriptions contain exactly the same significant
    words (e.g. 'digital timer' vs 'digital timer' - no change named)."""
    def words(s):
        s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
        return {w for w in s.split() if w and w not in _STOPWORDS}

    wa, wb = words(a), words(b)
    return bool(wa) and wa == wb


def _background_words(s) -> bool:
    """True when a free-text description names no object (only background).

    'a black chair' -> False; 'empty table surface' -> True (every significant
    word is background vocabulary or a stopword).
    """
    words = {w for w in re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower()).split()
             if w and w not in _STOPWORDS}
    background = {"floor", "ground", "wall", "background", "room", "nothing",
                  "empty", "air", "space", "scene"}
    return not words or words <= background


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
                        help="crops only; no verification, labels or matching")
    parser.add_argument("--limit_clusters", type=int, default=None,
                        help="debug: only process the first N clusters")
    parser.add_argument("--min_change_score", type=float, default=None,
                        help="override semantics.min_change_score (per-backend "
                             "calibration; see docs/results.md backend comparison)")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    if args.min_change_score is not None:
        cfg["semantics"]["min_change_score"] = args.min_change_score

    sem_dir = cache_dir(cfg, "semantics")
    matches_path = sem_dir / "matches.json"
    if matches_path.exists() and not args.force:
        log.info("semantics already cached (%s); use --force to redo", matches_path)
        return 0

    recon = load_reconstruction(cfg)
    diff_dir = repo_path(cfg["paths"]["cache_dir"]) / "diff"
    clusters = read_json(diff_dir / "clusters.json")["clusters"]
    uncertain_path = diff_dir / "uncertain.json"
    uncertain = read_json(uncertain_path)["clusters"] if uncertain_path.exists() else []
    if args.limit_clusters:
        clusters = clusters[:args.limit_clusters]
        uncertain = uncertain[:args.limit_clusters]
    keep_ids = {c["id"] for c in clusters} | {c["id"] for c in uncertain}
    points_npz = np.load(diff_dir / "points.npz")
    cluster_points: dict[int, np.ndarray] = {}
    for c in clusters + uncertain:
        tag = c["source_tag"] if c["source_tag"] in ("a", "b") else "a"
        arr = points_npz[f"{tag}_points"]
        labels = points_npz[f"{tag}_labels"]
        cluster_points[c["id"]] = arr[labels == c["id"]].astype(np.float32)
    log.info("processing %d verified + %d unverified candidate(s)",
             len(clusters), len(uncertain))

    # ---- 1) crops -------------------------------------------------------
    crops_path = sem_dir / "crops.json"
    crops = read_json(crops_path) if crops_path.exists() and not args.force else {}
    crops = {k: v for k, v in crops.items() if int(k) in keep_ids}
    for c in clusters + uncertain:
        cid = c["id"]
        pts = cluster_points[cid]
        if not pts.size:
            continue
        entry = crops.get(str(cid), {})
        # own-video view
        own_tag = c["source_tag"] if c["source_tag"] in ("a", "b") else "a"
        if "crop" not in entry:
            info = best_crop_for_cluster(cid, pts, own_tag, recon, cfg)
            if info:
                entry.update(info)
        # other-video view (verified clusters only): pose-matched to the own view
        if c.get("verified", True) and "crop_other" not in entry and "frame_index" in entry:
            other_tag = "b" if own_tag == "a" else "a"
            info = best_crop_for_cluster(cid, pts, other_tag, recon, cfg, suffix="_other",
                                         ref_frame=int(entry["frame_index"]))
            if info:
                entry.update({f"{k}_other" if k != "cluster_id" else k: v
                              for k, v in info.items()})
                entry["tag_other"] = other_tag
        if entry:
            crops[str(cid)] = entry
    write_json(crops_path, crops)

    # ---- 2) VLM verification + labels -----------------------------------
    vlm_path = sem_dir / "vlm.json"
    # always reuse cached answers (they are keyed by model): --force re-runs
    # the stage but must not re-spend GPU time on identical VLM calls
    vlm_cache = read_json(vlm_path) if vlm_path.exists() else {}
    vlm_cache = {k: v for k, v in vlm_cache.items() if int(k) in keep_ids}
    rejected: dict[str, dict] = {}
    vlm_model = vlm_processor = None
    vlm_ref = None
    if not args.skip_vlm:
        from PIL import Image

        key = cfg["semantics"]["vlm"]
        vlm_ref = resolve_model(cfg["models"]["vlm_primary" if key == "primary"
                                             else "vlm_fallback"],
                               cfg["paths"]["model_dir"])

        # load the VLM lazily: skip the multi-minute model load entirely when
        # every answer is already cached (common when re-running post-filters)
        def _needs_vlm_call() -> bool:
            for c in clusters + uncertain:
                cached = vlm_cache.get(str(c["id"]))
                if not (cached and cached.get("model") == vlm_ref
                        and "changed" in cached):
                    return True
            return False

        if _needs_vlm_call():
            vlm_model, vlm_processor, _ = load_vlm(cfg)
        else:
            log.info("all VLM answers cached - skipping the model load")
        for c in clusters:
            cid = str(c["id"])
            entry = crops.get(cid, {})
            if "crop" not in entry:
                continue
            cached = vlm_cache.get(cid)
            if cached and cached.get("model") == vlm_ref and "changed" in cached:
                continue
            own_tag = entry.get("tag", "a")
            claim = CLAIMS.get(c["type"], c["type"])
            prompt = PROMPT_VERIFY.format(claim=claim)
            if "crop_other" in entry:
                # always present in A-then-B order
                first, second = ((entry["crop"], entry["crop_other"])
                                 if own_tag == "a" else (entry["crop_other"], entry["crop"]))
                reply = vlm_answer(vlm_model, vlm_processor,
                                   [Image.open(first).convert("RGB"),
                                    Image.open(second).convert("RGB")], prompt)
                two_view = True
            else:
                reply = vlm_answer(vlm_model, vlm_processor,
                                   [Image.open(entry["crop"]).convert("RGB")], PROMPT_SINGLE)
                two_view = False
            parsed = parse_json_reply(reply)
            log.debug("  raw reply for %s: %s", cid, reply[:300].replace("\n", " "))
            changed = parsed.get("changed") if two_view else None
            if isinstance(changed, str):
                changed = changed.strip().lower() in ("true", "yes", "1")
            # self-consistency override: if the model names the same object in
            # both images, a removed/added claim is false regardless of what it
            # answered (small blurry crops make it over-sensitive otherwise)
            if (two_view and c["type"] in ("removed", "added")
                    and _objects_overlap(parsed.get("first"), parsed.get("second"))):
                if changed is not False:
                    log.info("  override -> changed=False (%s vs %s)",
                             parsed.get("first"), parsed.get("second"))
                changed = False
            vlm_cache[cid] = {"label": parsed.get("label"),
                              "description": parsed.get("sentence") or parsed.get("description"),
                              "first": parsed.get("first"), "second": parsed.get("second"),
                              "changed": bool(changed) if changed is not None else True,
                              "raw": reply[:500], "model": vlm_ref, "created": stamp()}
            log.info("verify %s (%s) -> changed=%s label=%s",
                     cid, c["type"], vlm_cache[cid]["changed"], vlm_cache[cid]["label"])
            write_json(vlm_path, vlm_cache)
        for c in uncertain:
            cid = str(c["id"])
            entry = crops.get(cid, {})
            if "crop" not in entry:
                continue
            cached = vlm_cache.get(cid)
            if cached and cached.get("model") == vlm_ref:
                continue
            reply = vlm_answer(vlm_model, vlm_processor,
                               [Image.open(entry["crop"]).convert("RGB")], PROMPT_SINGLE)
            parsed = parse_json_reply(reply)
            vlm_cache[cid] = {"label": parsed.get("label"),
                              "description": parsed.get("description"),
                              "changed": None, "raw": reply[:500],
                              "model": vlm_ref, "created": stamp()}
            log.info("label %s (unverified %s) -> %s", cid, c["type"], vlm_cache[cid]["label"])
            write_json(vlm_path, vlm_cache)
    write_json(vlm_path, vlm_cache)

    # ---- 3) keep only VLM-confirmed verified clusters --------------------
    def _reject(c: dict, reason: str) -> None:
        """Record a rejected candidate (never drop silently)."""
        cid = str(c["id"])
        rejected[cid] = {"type": c["type"], "centroid": c["centroid"],
                         "reason": reason,
                         "vlm": {k: vlm_cache.get(cid, {}).get(k)
                                 for k in ("label", "first", "second")}}
        log.debug("  reject %s (%s): %s", cid, c["type"], reason)

    soft_veto = bool(cfg["semantics"].get("soft_veto", False))
    _min_pts = float(cfg["semantics"].get("soft_veto_min_points", 60))

    def _strong_geometry(c: dict) -> bool:
        n = float(c.get("n_points") or c.get("n_cells") or 0)
        return n >= _min_pts

    confirmed = []
    single_view: list[dict] = []
    soft_kept: list[dict] = []
    for c in clusters:
        cid = str(c["id"])
        v = vlm_cache.get(cid, {})
        if args.skip_vlm:
            confirmed.append(c)
        elif v.get("changed") is False:
            if soft_veto and _strong_geometry(c):
                c = dict(c); c["soft_veto"] = True
                soft_kept.append(c)
                log.info("soft veto -> keep denied %s as 'possibly' (n=%s)",
                         cid, c.get("n_points") or c.get("n_cells"))
            else:
                _reject(c, "VLM denied the change (same object visible in both videos)")
        elif v.get("changed") is None and v.get("first") is None and v.get("second") is None:
            # the two-view verification could not run (the other video's crop
            # was unavailable): a label-only answer must NOT count as a
            # confirmed change - downgrade to the unverified list
            single_view.append(c)
            log.info("cluster %s: no two-view verification available -> unverified", cid)
        else:
            confirmed.append(c)

    # mutual cancellation: a confirmed "removed X" and "added X" with the same
    # label at (nearly) the same spot mean X is present in BOTH videos - the
    # fragments came from lighting/blur differences, so both are dropped
    def _norm(s) -> str:
        return (s or "").lower().strip()

    def _dist(c1: dict, c2: dict) -> float:
        return float(np.linalg.norm(np.asarray(c1["centroid"]) - np.asarray(c2["centroid"])))

    cancel: set[int] = set()
    rem = [c for c in confirmed if c["type"] == "removed"]
    add = [c for c in confirmed if c["type"] == "added"]
    for r in rem:
        for a in add:
            vr = vlm_cache.get(str(r["id"]), {})
            va = vlm_cache.get(str(a["id"]), {})
            # compare the object each cluster sees in ITS OWN video: the
            # removed cluster's "first" (video A) against the added cluster's
            # "second" (video B). The stored label can name the other video's
            # object (scene4's added apple carried the label "scissors" from
            # its A-side view and cancelled a genuine apple arrival).
            lr = _norm(vr.get("first") or vr.get("label"))
            la = _norm(va.get("second") or va.get("label"))
            if lr and la and (lr in la or la in lr) and _dist(r, a) < 0.12:
                for c, why in ((r, "removed"), (a, "added")):
                    cid = str(c["id"])
                    if c["id"] not in cancel:
                        rejected[cid] = {"type": c["type"], "centroid": c["centroid"],
                                         "reason": "contradicted by a confirmed opposite change with "
                                                   "the same label nearby (object present in both videos)",
                                         "vlm": {"label": vlm_cache.get(cid, {}).get("label")}}
                    cancel.add(c["id"])
                log.info("cancel contradictory pair %s/%s (%s)",
                         r["id"], a["id"], vlm_cache.get(str(r["id"]), {}).get("label"))
    confirmed = [c for c in confirmed if c["id"] not in cancel]

    # drop unverified duplicates of confirmed changes (same physical change,
    # seen from the other video): bbox adjacency
    def _bbox_close(c1: dict, c2: dict, pad: float = 0.01) -> bool:
        return all(c1["bbox_min"][i] - pad <= c2["bbox_max"][i]
                   and c2["bbox_min"][i] - pad <= c1["bbox_max"][i] for i in range(3))

    uncertain_kept = []
    for c in list(uncertain) + single_view + soft_kept:
        cid = str(c["id"])
        dup = next((cc["id"] for cc in confirmed if _bbox_close(c, cc)), None)
        if dup is not None:
            rejected[cid] = {"type": c["type"], "centroid": c["centroid"],
                             "reason": f"duplicate of verified change #{dup} (overlapping region)",
                             "vlm": {"label": vlm_cache.get(cid, {}).get("label")}}
        else:
            uncertain_kept.append(c)
    log.info("VLM verification: %d confirmed, %d rejected, %d unverified kept",
             len(confirmed), len(rejected), len(uncertain_kept))

    # ---- 3a) move matching (runs BEFORE the type correction + filters) ----
    # A moved object appears as a confirmed "removed" cluster at its origin
    # and a confirmed "added" cluster at its destination; DINOv2 crop
    # similarity + a VLM move question turn the pair into one "moved" change.
    # This runs before the structural filters on purpose: those filters can
    # drop or rewrite the destination cluster (scene4's apple arrival was
    # rejected as a duplicate of a replaced scissor fragment, so the move was
    # never found).
    matches: list[dict] = []
    arrivals: list[dict] = []          # objects now occupying a move's origin
    moved_ids: set[int] = set()
    frag_ids: set[int] = set()         # duplicate arrivals dropped
    moved_origins: dict[int, list] = {}
    moved_centroids: dict[int, list] = {}
    if not args.skip_vlm and confirmed:
        import torch
        from PIL import Image

        matchable = [c for c in confirmed if c["type"] in ("removed", "added")
                     and str(c["id"]) in crops and "crop" in crops[str(c["id"])]]
        if matchable:
            matcher, matcher_processor = load_matcher(cfg)
            ids = [c["id"] for c in matchable]
            paths = [crops[str(i)]["crop"] for i in ids]
            emb = embed_crops(matcher, matcher_processor, paths)
            np.savez(sem_dir / "embeddings.npz",
                     ids=np.array(ids), vectors=emb,
                     files=np.array([crops[str(i)]["frame"] for i in ids]))
            del matcher, matcher_processor
            gc.collect()
            torch.cuda.empty_cache()

            scfg = cfg["semantics"]
            by_id = {c["id"]: c for c in matchable}
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
            # similarity first; within a 0.05 band prefer the cluster with
            # more evidence (scene4's apple had a clean 102-cell destination
            # and a cluttered 52-cell fragment, scored 0.704 vs 0.732)
            pairs.sort(key=lambda p: (-round(p["similarity"], 1),
                                      -by_id[p["added_id"]].get("n_cells",
                                          by_id[p["added_id"]].get("n_points", 0))))
            if pairs and vlm_model is None:
                vlm_model, vlm_processor, _ = load_vlm(cfg)
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
                moved_ids.update({p["removed_id"], p["added_id"]})
                moved_origins[p["removed_id"]] = by_id[p["removed_id"]]["centroid"]
                moved_centroids[p["added_id"]] = by_id[p["added_id"]]["centroid"]

                # other added fragments that show the same object are pieces
                # of the same arrival (scene4's apple had a second fragment)
                for c in confirmed:
                    if (c["type"] == "added" and c["id"] not in moved_ids
                            and _objects_overlap(
                                vlm_cache.get(str(c["id"]), {}).get("second"),
                                entry["label"])):
                        _reject(c, f"same object as the moved {entry['label']} "
                                   f"(fragment of its arrival)")
                        moved_ids.add(c["id"])
                        frag_ids.add(c["id"])

                # what occupies the origin now? the origin's own before/after
                # verification already saw it (scene4: a bottle appeared where
                # the apple used to be) - report it as its own added change
                v_rem = vlm_cache.get(str(p["removed_id"]), {})
                second = v_rem.get("second")
                if (second and not _objects_overlap(second, entry["label"])
                        and not _background_words(second)):
                    rem = by_id[p["removed_id"]]
                    arrivals.append({
                        "id": 900 + len(arrivals),
                        "type": "added",
                        "source_tag": "b",
                        "n_points": int(rem.get("n_points") or rem.get("n_cells") or 200),
                        "centroid": rem["centroid"],
                        "bbox_min": rem["bbox_min"], "bbox_max": rem["bbox_max"],
                        "extent": rem.get("extent"),
                        "sample_points": rem.get("sample_points", []),
                        "arrival_at": p["removed_id"], "moved_label": entry["label"],
                        "label": second,
                    })
                    log.info("arrival at #%s origin: %r", p["removed_id"], second)
            if frag_ids:
                confirmed = [c for c in confirmed if c["id"] not in frag_ids]
            log.info("move matches: %d (%d arrivals)", len(matches), len(arrivals))
            for a in arrivals:
                vlm_cache[str(a["id"])] = {
                    "label": a["label"],
                    "description": f"a {a['label']} appeared where the "
                                   f"{a['moved_label']} was before it moved",
                    "changed": True, "model": "derived", "created": stamp()}
            if arrivals:
                write_json(vlm_path, vlm_cache)
            write_json(matches_path, matches)

    # the matched clusters are reported through matches.json; they stay in
    # confirmed.json (build_change_list renders the move from the removed
    # cluster + the pair) but every later step skips them

    # ---- 3a) type correction with geometric evidence ---------------------
    # The photometric mask alone cannot separate "removed" from "replaced":
    # the other video's crop may show a NEIGHBOURING object (chair #2 behind
    # the removed chair) or the space may be occupied by a different object
    # (cup where the apple was). The pipeline's own design principle resolves
    # it: a *removed* object's space is seen EMPTY by the other video, while a
    # *replaced* object's space is OCCUPIED (by something at a similar depth).
    # We run the geometric empty-space test on the cluster's points and correct
    # the type; corrected clusters get a fresh sentence from the VLM.
    def _empty_fraction(c: dict) -> float:
        from src.diff import visibility_check
        pts = cluster_points.get(c["id"])
        if pts is None or not len(pts):
            return float("nan")
        own = c["source_tag"] if c["source_tag"] in ("a", "b") else "a"
        other = [i for i, t in enumerate(recon["tags"]) if t != own]
        frac, _ = visibility_check(pts, other, recon, cfg)
        return frac

    reclassified: list[dict] = []
    if not args.skip_vlm:
        keep_frac = float(cfg["semantics"].get("reclassify_empty_fraction", 0.30))
        occ_frac = float(cfg["semantics"].get("reclassify_occupied_fraction", 0.15))
        for c in confirmed:
            if c["type"] not in ("removed", "replaced") or c["id"] in moved_ids:
                continue
            frac = _empty_fraction(c)
            if not np.isfinite(frac):
                continue
            v = vlm_cache.get(str(c["id"]), {})
            if c["type"] == "replaced" and frac >= keep_frac:
                c["type"] = "removed"
                c["reclassified_from"] = "replaced"
                c["empty_fraction"] = round(frac, 3)
                reclassified.append(c)
                log.info("reclassify %s: replaced -> removed (space seen empty %.0f%%)",
                         c["id"], 100 * frac)
            elif (c["type"] == "removed" and frac <= occ_frac
                    and not _background_words(v.get("second"))):
                c["type"] = "replaced"
                c["reclassified_from"] = "removed"
                c["empty_fraction"] = round(frac, 3)
                reclassified.append(c)
                log.info("reclassify %s: removed -> replaced (space occupied %.0f%%, "
                         "other view names %r)", c["id"], 100 * frac, v.get("second"))

    # fresh sentences for corrected types (the old ones describe the wrong
    # event, e.g. "the black bowl was replaced by a black chair" for a removal)
    if reclassified and vlm_model is not None:
        from PIL import Image
        for c in reclassified:
            cid = str(c["id"])
            ci = crops.get(cid, {})
            if "crop" not in ci or "crop_other" not in ci:
                continue
            own_tag = ci.get("tag", "a")
            first, second = ((ci["crop"], ci["crop_other"]) if own_tag == "a"
                             else (ci["crop_other"], ci["crop"]))
            prompt = (PROMPT_REPLACE if c["type"] == "replaced" else PROMPT_REMOVED)
            reply = vlm_answer(vlm_model, vlm_processor,
                               [Image.open(first).convert("RGB"),
                                Image.open(second).convert("RGB")], prompt)
            parsed = parse_json_reply(reply)
            vlm_cache[cid]["label"] = parsed.get("label") or vlm_cache[cid].get("label")
            vlm_cache[cid]["description"] = (parsed.get("sentence")
                                             or parsed.get("description")
                                             or vlm_cache[cid].get("description"))
            vlm_cache[cid]["reclassified_from"] = c["reclassified_from"]
            log.info("reclassified %s sentence: %s", cid, vlm_cache[cid]["description"])
            write_json(vlm_path, vlm_cache)

    # ---- 3b) structural post-filters --------------------------------------
    # (_reject is defined above, before the confirmation filter)

    # weak fragments are noise, not changes (the pen scores 90, the comb 110).
    # NOTE: only photometric candidates carry a change score; geometric ones
    # are evidence-checked by the visibility test instead, so the floor must
    # not reject them just because the field is absent.
    min_score = float(cfg["semantics"].get("min_change_score", 80))
    kept = []
    for c in confirmed:
        if c["id"] in moved_ids:      # moves carry their own verification
            kept.append(c)
            continue
        score = c.get("mean_change_score")
        if (c["type"] in ("removed", "added") and score is not None
                and float(score) < min_score):
            _reject(c, f"weak evidence (change score {score} < {min_score})")
        else:
            kept.append(c)
    confirmed = kept

    # D: coverage-aware "added" - require the other video's depth to have
    # observed the space empty before reporting an addition (targets additions
    # on features present in both videos: a wall outlet, a floor reflection).
    if bool(cfg["semantics"].get("coverage_added", False)):
        cov_kept = []
        cov_min = float(cfg["diff"].get("visibility_min_fraction", 0.15))
        for c in confirmed:
            if c["type"] == "added" and c["id"] not in moved_ids:
                frac = _empty_fraction(c)
                if not (np.isfinite(frac) and frac >= cov_min):
                    _reject(c, f"coverage: the other video did not observe the space "
                               f"empty (empty fraction {frac:.2f} < {cov_min})")
                    continue
            cov_kept.append(c)
        confirmed = cov_kept
        log.info("coverage_added: %d kept", len(confirmed))

    # fragments of objects that exist in BOTH videos (timer, highlighter) are
    # edge/lighting differences, not changes
    try:
        from scipy.ndimage import binary_dilation

        from src.tabletop import static_object_mask
        smask, sbounds, scell = static_object_mask(recon, cfg)
        smask_d = binary_dilation(smask, iterations=2)
        kept = []
        for c in confirmed:
            if c["id"] in moved_ids:
                kept.append(c)
                continue
            pts = cluster_points.get(c["id"])
            # an explicit empty-space confirmation (from the type-correction
            # step) is stronger evidence than the both-present heuristic -
            # e.g. scene3's chair sits next to the OTHER chair, which made the
            # both-present mask fire; the empty-check proved it was removed.
            # BUT if the VLM named the SAME object in both views (the demo's
            # highlighter: 'highlighter' vs 'highlighter'), the object really
            # is present in both and the suppression stands.
            v = vlm_cache.get(str(c["id"]), {})
            if (c.get("empty_fraction") is not None and c.get("reclassified_from")
                    and not _same_objects(v.get("first"), v.get("second"))):
                kept.append(c)
                continue
            if c["type"] in ("removed", "added") and pts is not None and len(pts):
                ix = ((pts[:, 0] - sbounds[0]) / scell).astype(int)
                iy = ((pts[:, 1] - sbounds[2]) / scell).astype(int)
                ok = (ix >= 0) & (ix < smask_d.shape[0]) & (iy >= 0) & (iy < smask_d.shape[1])
                frac = float(smask_d[ix[ok], iy[ok]].mean()) if ok.any() else 0.0
                log.debug("  static-object overlap for %s: %.0f%% of its points",
                          c["id"], 100 * frac)
                if frac >= 0.5:
                    _reject(c, f"inside an object present in both videos "
                               f"({frac:.0%} of its points) - lighting/blur artefact")
                    continue
            kept.append(c)
        confirmed = kept
    except Exception as exc:  # noqa: BLE001 - never fail the stage over this
        log.warning("static-object suppression skipped: %s", exc)

    # composite replaced clusters: when BOTH sides of a "replaced" claim are
    # independently confirmed changes nearby, the replacement is a composite,
    # not a real event. scene4: the moved apple landed on the removed scissor's
    # pixels - A shows the scissor, B the apple - and the mask calls the mix
    # "replaced"; both sides are already reported (removed scissors + moved
    # apple), so the composite is dropped instead of double-counted.
    def _bbox_close(c1: dict, c2: dict, pad: float = 0.01) -> bool:
        return all(c1["bbox_min"][i] - pad <= c2["bbox_max"][i]
                   and c2["bbox_min"][i] - pad <= c1["bbox_max"][i] for i in range(3))

    comp_dist = float(cfg["semantics"].get("composite_match_distance", 0.15))

    def _label_near(centroid, label, pool: list[tuple]) -> str | None:
        for lab, cent, why in pool:
            if lab and _objects_overlap(label, lab) and float(np.linalg.norm(
                    np.asarray(centroid, dtype=float) - np.asarray(cent, dtype=float))) < comp_dist:
                return why
        return None

    origin_pool: list[tuple] = []
    dest_pool: list[tuple] = []
    for c in confirmed:
        lab = vlm_cache.get(str(c["id"]), {}).get("label")
        if not lab:
            continue
        if c["type"] == "removed":
            origin_pool.append((lab, c["centroid"], f"removed #{c['id']}"))
        elif c["type"] == "added":
            dest_pool.append((lab, c["centroid"], f"added #{c['id']}"))
    for m in matches:
        if m.get("label") and m["removed_id"] in moved_origins:
            origin_pool.append((m["label"], moved_origins[m["removed_id"]],
                                f"moved #{m['removed_id']}"))
        if m.get("label") and m["added_id"] in moved_centroids:
            dest_pool.append((m["label"], moved_centroids[m["added_id"]],
                              f"moved #{m['added_id']}"))

    replaced_boxes = []
    kept = []
    for c in confirmed:
        if c["type"] != "replaced":
            kept.append(c)
            continue
        v = vlm_cache.get(str(c["id"]), {})
        why_a = _label_near(c["centroid"], v.get("first"), origin_pool)
        why_b = _label_near(c["centroid"], v.get("second"), dest_pool)
        if why_a and why_b:
            _reject(c, f"composite of two confirmed changes ({why_a}; {why_b}) - "
                       f"not a replacement")
            continue
        replaced_boxes.append(c)
        kept.append(c)
    confirmed = kept

    # removed/added fragments overlapping a confirmed replaced change are the
    # same physical change seen twice
    kept = []
    for c in confirmed:
        if (c["type"] in ("removed", "added") and c["id"] not in moved_ids
                and any(_bbox_close(c, r) for r in replaced_boxes)):
            dup = next(r["id"] for r in replaced_boxes if _bbox_close(c, r))
            _reject(c, f"overlaps the verified replaced change #{dup} (same physical change)")
        else:
            kept.append(c)
    confirmed = kept

    # merge same-type confirmed fragments that name the same object nearby
    # (the comb's teeth cluster and its handle cluster are one change)
    def _dist(c1: dict, c2: dict) -> float:
        return float(np.linalg.norm(np.asarray(c1["centroid"]) - np.asarray(c2["centroid"])))

    merged_flags: set[int] = set()
    merged: list[dict] = []
    for i, c in enumerate(confirmed):
        if c["id"] in merged_flags:
            continue
        if c["id"] in moved_ids:      # a move is one change already
            merged.append(c)
            continue
        group = [c]
        for other in confirmed[i + 1:]:
            if (other["id"] in merged_flags or other["type"] != c["type"]
                    or other["id"] in moved_ids):
                continue
            if c["type"] not in ("removed", "added", "replaced"):
                continue
            li = vlm_cache.get(str(c["id"]), {}).get("label")
            lo = vlm_cache.get(str(other["id"]), {}).get("label")
            same_name = _objects_overlap(li, lo)
            adjacent = _bbox_close(c, other, pad=0.06)   # bbox within 6cm
            if (same_name or adjacent) and _dist(c, other) < 0.25:
                group.append(other)
                merged_flags.add(other["id"])
        if len(group) > 1:
            pts = np.concatenate([np.asarray(g["sample_points"], dtype=np.float32)
                                  for g in group if g.get("sample_points")])
            c = dict(c)
            # photometric clusters have n_cells; geometric ones n_points only
            c["n_cells"] = int(sum(g.get("n_cells", g.get("n_points", 0)) for g in group))
            c["merged_ids"] = [g["id"] for g in group]
            c["sample_points"] = [[round(float(v), 4) for v in p] for p in pts[:200]]
            c["centroid"] = [round(float(v), 4) for v in pts.mean(axis=0)]
            c["bbox_min"] = [round(float(v), 4) for v in pts.min(axis=0)]
            c["bbox_max"] = [round(float(v), 4) for v in pts.max(axis=0)]
            c["mean_change_score"] = max(g.get("mean_change_score") or 0 for g in group)
            log.info("merged %s into #%d (%s)", c["merged_ids"][1:], c["id"],
                     vlm_cache.get(str(c["id"]), {}).get("label"))
        merged.append(c)
    confirmed = merged

    log.info("post-filters: %d confirmed remain (%d rejected so far)",
             len(confirmed), len(rejected))
    if not args.skip_vlm and confirmed:
        import cv2
        import torch
        from PIL import Image

        pairs = [c for c in confirmed
                 if str(c["id"]) in crops and "crop" in crops[str(c["id"])]
                 and "crop_other" in crops[str(c["id"])]]
        if pairs:
            # replaced claims must show a strong pixel-level appearance change
            # (a real flip: black back vs white face); blur/lighting differences
            # score far lower
            min_pixel = float(cfg["semantics"].get("replaced_min_pixel_diff", 70))
            replaced_rejects: set[int] = set()
            for c in confirmed:
                if c["type"] != "replaced":
                    continue
                if c.get("reclassified_from"):
                    # the replacement was established from occupied-space
                    # evidence; the pixel gate exists for photometric
                    # candidates and would reject a visually odd (occluded)
                    # crop of a genuine replacement
                    continue
                ci = crops.get(str(c["id"]), {})
                if "crop" not in ci or "crop_other" not in ci:
                    continue
                a = cv2.resize(cv2.imread(ci["crop"]), (64, 64)).astype(float)
                b = cv2.resize(cv2.imread(ci["crop_other"]), (64, 64)).astype(float)
                pd = float(np.abs(a - b).mean())
                log.info("  replaced pixel-difference %s: %.1f (min %.0f)",
                         c["id"], pd, min_pixel)
                if pd < min_pixel:
                    _reject(c, f"replaced claim without a strong appearance change "
                               f"(pixel difference {pd:.1f} < {min_pixel})")
                    replaced_rejects.add(c["id"])
            confirmed = [c for c in confirmed if c["id"] not in replaced_rejects]

            matcher, matcher_processor = load_matcher(cfg)
            sims = {}
            for c in confirmed:
                if c["id"] in moved_ids:
                    continue
                if str(c["id"]) in crops and "crop" in crops[str(c["id"])] \
                        and "crop_other" in crops[str(c["id"])]:
                    ci = crops[str(c["id"])]
                    emb = embed_crops(matcher, matcher_processor, [ci["crop"], ci["crop_other"]])
                    sims[c["id"]] = float(emb[0] @ emb[1])
            threshold = float(cfg["semantics"].get("same_appearance_threshold", 0.80))
            still = []
            for c in confirmed:
                if c["id"] in moved_ids:
                    still.append(c)
                    continue
                sim = sims.get(c["id"])
                if sim is not None and sim > threshold:
                    cid = str(c["id"])
                    rejected[cid] = {"type": c["type"], "centroid": c["centroid"],
                                     "reason": f"before/after crops look like the same appearance "
                                               f"(DINOv2 similarity {sim:.2f} > {threshold})",
                                     "vlm": {k: vlm_cache.get(cid, {}).get(k)
                                             for k in ("label", "first", "second")}}
                    log.info("appearance-reject %s (sim %.2f): %s", cid, sim,
                             vlm_cache.get(cid, {}).get("label"))
                else:
                    still.append(c)
            for c in confirmed:
                if c["id"] in sims:
                    log.info("  crop-pair similarity %s (%s): %.2f", c["id"], c["type"], sims[c["id"]])
            confirmed = still
            log.info("appearance check: %d kept after crop-similarity filter", len(confirmed))
            del matcher, matcher_processor
            gc.collect()
            torch.cuda.empty_cache()

    write_json(sem_dir / "rejected.json",
               {"created": stamp(), "config_hash": cfg["_config_hash"],
                "note": "candidates the VLM denied or duplicates of confirmed changes",
                "rejected": rejected})
    # arrivals (objects that now occupy a moved object's origin) bypass the
    # structural filters: they are derived from the move's own verified
    # before/after evidence, not from an independent photometric candidate
    confirmed = confirmed + arrivals
    write_json(sem_dir / "confirmed.json",
               {"created": stamp(), "config_hash": cfg["_config_hash"],
                "clusters": confirmed, "uncertain": uncertain_kept})
    log.info("wrote %s, rejected.json, confirmed.json", matches_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
