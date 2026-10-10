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


# ---------------------------------------------------------------------------
# crop selection


def best_crop_for_cluster(cid: int, points: np.ndarray, tag: str, recon: dict, cfg: dict,
                          suffix: str = "") -> dict | None:
    """Pick the frame of the given video that sees the cluster best, crop it."""
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
        log.warning("cluster %d (%s%s): no good view found (%s visible points)",
                    cid, tag, suffix, None if best is None else best["n_visible"])
        return None

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
    model = ModelCls.from_pretrained(ref, dtype=dtype, device_map="auto", low_cpu_mem_usage=True)
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
              "there", "it", "its", "small", "large", "other"}


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
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

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
        # other-video view (verified clusters only)
        if c.get("verified", True) and "crop_other" not in entry:
            other_tag = "b" if own_tag == "a" else "a"
            info = best_crop_for_cluster(cid, pts, other_tag, recon, cfg, suffix="_other")
            if info:
                entry.update({f"{k}_other" if k != "cluster_id" else k: v
                              for k, v in info.items()})
                entry["tag_other"] = other_tag
        if entry:
            crops[str(cid)] = entry
    write_json(crops_path, crops)

    # ---- 2) VLM verification + labels -----------------------------------
    vlm_path = sem_dir / "vlm.json"
    vlm_cache = read_json(vlm_path) if vlm_path.exists() and not args.force else {}
    vlm_cache = {k: v for k, v in vlm_cache.items() if int(k) in keep_ids}
    rejected: dict[str, dict] = {}
    vlm_model = vlm_processor = None
    vlm_ref = None
    if not args.skip_vlm:
        from PIL import Image
        vlm_model, vlm_processor, vlm_ref = load_vlm(cfg)
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
            # for a "replaced" claim, identical object descriptions in both
            # images mean nothing actually changed (the timer blur case); a
            # real flip/replacement names different objects or attributes
            if (two_view and c["type"] == "replaced"
                    and _same_objects(parsed.get("first"), parsed.get("second"))):
                if changed is not False:
                    log.info("  override -> changed=False (identical names: %s)",
                             parsed.get("first"))
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
    confirmed = []
    for c in clusters:
        cid = str(c["id"])
        v = vlm_cache.get(cid, {})
        if args.skip_vlm or v.get("changed", True) is not False:
            confirmed.append(c)      # kept unless the VLM explicitly denied it
        else:
            rejected[cid] = {"type": c["type"], "centroid": c["centroid"],
                             "reason": "VLM denied the change (same object visible in both videos)",
                             "vlm": {k: v.get(k) for k in ("label", "raw")}}

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
            lr = _norm(vlm_cache.get(str(r["id"]), {}).get("label"))
            la = _norm(vlm_cache.get(str(a["id"]), {}).get("label"))
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
    for c in uncertain:
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

    # ---- 3b) appearance check: crop-pair similarity -----------------------
    # a candidate whose before/after crops embed as nearly the same image is a
    # lighting/blur artefact, not a change (the flipped remote scores low here,
    # the untouched timer scores high). Uses the same DINOv2 as matching.
    if not args.skip_vlm and confirmed:
        import torch
        from PIL import Image

        pairs = [c for c in confirmed
                 if str(c["id"]) in crops and "crop" in crops[str(c["id"])]
                 and "crop_other" in crops[str(c["id"])]]
        if pairs:
            matcher, matcher_processor = load_matcher(cfg)
            sims = {}
            for c in pairs:
                ci = crops[str(c["id"])]
                emb = embed_crops(matcher, matcher_processor, [ci["crop"], ci["crop_other"]])
                sims[c["id"]] = float(emb[0] @ emb[1])
            threshold = float(cfg["semantics"].get("same_appearance_threshold", 0.80))
            still = []
            for c in confirmed:
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

    # ---- 4) DINOv2 embeddings + move matching ---------------------------
    matches: list[dict] = []
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
    write_json(sem_dir / "rejected.json",
               {"created": stamp(), "config_hash": cfg["_config_hash"],
                "note": "candidates the VLM denied or duplicates of confirmed changes",
                "rejected": rejected})
    write_json(sem_dir / "confirmed.json",
               {"created": stamp(), "config_hash": cfg["_config_hash"],
                "clusters": confirmed, "uncertain": uncertain_kept})
    log.info("wrote %s, rejected.json, confirmed.json", matches_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
