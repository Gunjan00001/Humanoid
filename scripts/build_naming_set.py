#!/usr/bin/env python3
"""Build a labelled crop set for naming, from the SceneDiff VALIDATION pairs only.

Naming is measured independently of detection here: every crop is a ground-truth
object cut from a frame where its mask says it is visible, so the labels are
right by construction and no detector is involved.

  python scripts/build_naming_set.py                 # all validation pairs in the slice
  python scripts/build_naming_set.py --pairs gym_9_gym_10 --per-object 4

Output:
  data/naming_set/crops/<pair>__<objidx>__<frame>.png
  data/naming_set/manifest.json     one row per crop: path, pair, object index,
                                    ground-truth label, video side, frame index

The test split is never read. The pair list comes from
data/external/scenediff/scenediff_slice.json, validation section only.
"""
# EXPERIMENTAL, not used by detect.py. Part of the accuracy and naming
# experiments; kept as evidence. See docs/accuracy.md and docs/naming_v4.md.
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "external" / "scenediff" / "scenediff_benchmark" / "data"
SLICE = ROOT / "data" / "external" / "scenediff_slice.json"
OUT = ROOT / "data" / "naming_set"

log = logging.getLogger("namingset")


def val_pairs() -> list[str]:
    """Validation pairs only. The test split is deliberately not read.

    The slice file was pre-registered earlier and lists 20 validation plus 10
    test pairs; only the first list is used here.
    """
    data = json.loads(SLICE.read_text())
    pairs = data.get("val") or data.get("validation") or []
    log.info("validation pairs in the pre-registered slice: %d of %d "
             "(test split not read)", len(pairs), data.get("n_val", len(pairs)))
    return list(pairs)


def decode_mask(rle: dict) -> np.ndarray:
    """COCO RLE -> boolean mask. pycocotools wants byte counts."""
    rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    return mask_util.decode(rle).astype(bool)


def crop_indices(keys: list[str], limit: int) -> list[int]:
    """Evenly spaced frame indices out of the frames where the object is visible."""
    ordered = sorted((int(k) for k in keys))
    if len(ordered) <= limit:
        return ordered
    step = len(ordered) / limit
    return [ordered[int(i * step)] for i in range(limit)]


def crop_object(video: Path, masks: dict, limit: int, out_dir: Path, tag: str,
                margin: float = 0.08) -> list[dict]:
    """One pass over the video, writing a crop for each sampled frame."""
    wanted = set(crop_indices(list(masks), limit))
    if not wanted:
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log.warning("cannot open %s", video)
        return []
    rows: list[dict] = []
    index = 0
    last = max(wanted)
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            mask = decode_mask(masks[str(index)])
            # The mask is stored in the dataset's own frame size, which does not
            # always match what the decoder returns for the file on disk. Resize
            # with nearest neighbour so no mask pixels are invented.
            if mask.shape[:2] != frame.shape[:2]:
                log.debug("resizing mask %s -> %s for %s", mask.shape[:2],
                          frame.shape[:2], tag)
                mask = cv2.resize(mask.astype(np.uint8),
                                  (frame.shape[1], frame.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            ys, xs = np.where(mask)
            if len(xs) > 40:                     # ignore speck masks
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                mx = int((x1 - x0) * margin)
                my = int((y1 - y0) * margin)
                x0, y0 = max(0, x0 - mx), max(0, y0 - my)
                x1 = min(frame.shape[1], x1 + mx)
                y1 = min(frame.shape[0], y1 + my)
                patch = frame[y0:y1, x0:x1]
                if patch is None or patch.size == 0 or min(patch.shape[:2]) < 2:
                    index += 1
                    continue                  # degenerate box, skip this frame
                # Blank everything outside the mask so the crop is the object
                # alone: naming should not be helped or hurt by the background.
                patch = patch.copy()
                sub = mask[y0:y1, x0:x1]
                patch[~sub] = 0
                name = f"{tag}_{index:05d}.png"
                cv2.imwrite(str(out_dir / name), patch)
                rows.append({"path": str(out_dir / name), "frame": index,
                             "width": int(x1 - x0), "height": int(y1 - y0)})
        index += 1
    cap.release()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--per-object", type=int, default=6,
                    help="crops per object per video side")
    ap.add_argument("--min-size", type=int, default=24, help="skip crops smaller than this")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)-7s %(message)s")

    pairs = args.pairs or val_pairs()
    out_dir = OUT / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for pair in pairs:
        seg_file = BENCH / pair / "segments.pkl"
        if not seg_file.exists():
            log.warning("%s: no segments.pkl, skipping", pair)
            continue
        # Read as bytes: a text round trip rewrites newlines and corrupts the pickle.
        seg = pickle.loads(seg_file.read_bytes())
        videos = {1: sorted((BENCH / pair).glob("original_video1.*")),
                  2: sorted((BENCH / pair).glob("original_video2.*"))}
        for idx, obj in enumerate(seg["objects"]):
            label = obj["label"]
            for side, key in ((1, "video1_objects"), (2, "video2_objects")):
                masks = seg.get(key, {}).get(idx) or {}
                if not masks or not videos[side]:
                    continue
                tag = f"{pair}__o{idx}__v{side}"
                rows = crop_object(videos[side][0], masks, args.per_object,
                                   out_dir, tag)
                for row in rows:
                    if min(row["width"], row["height"]) < args.min_size:
                        Path(row["path"]).unlink(missing_ok=True)
                        continue
                    manifest.append({"pair": pair, "object_index": idx,
                                     "label": label, "side": side, **row})
        log.info("%s: %d crops so far", pair, len(manifest))

    (OUT / "manifest.json").write_text(json.dumps(
        {"source": "SceneDiff validation pairs only", "crops": manifest}, indent=2))
    labels = sorted({m["label"] for m in manifest})
    print(f"\nwrote {len(manifest)} crops from {len(pairs)} validation pairs")
    print(f"distinct ground-truth labels: {len(labels)}")
    print(f"manifest: {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
