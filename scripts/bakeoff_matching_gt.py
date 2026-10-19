#!/usr/bin/env python3
"""DINOv3 versus DINOv2 on GROUND-TRUTH pairs, with cluster-bootstrap intervals.

The first run used the pipeline's own matches (same cluster across two videos)
as positive pairs, which is circular: the pipeline had already decided those
crops belong to one object. This version takes positives from the ground truth:
the labelled crop manifest records, for every SceneDiff validation object, the
crop of that object in video 1 and in video 2, independently of any detector.
A positive pair is the same (pair, object index) on both sides; a negative pair
is two different object instances. Everything is CPU and local.

Reports per encoder and descriptor: mean positive and negative similarity, AUC,
a 95% cluster-bootstrap interval (object instances resampled, 500 draws), and
the PAIRED difference DINOv3 minus DINOv2 on the same pairs with its interval.
The adoption rule: DINOv3 stays only if the paired difference interval does not
include a loss.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = {"DINOv2-large": Path.home() / "models" / "dinov2-large",
          "DINOv3-ViT-L": Path.home() / "models" / "dinov3-vitl16"}


def load_pairs(max_groups: int = 120):
    """Ground-truth positive pairs and negative pairs from the manifest."""
    crops = json.loads((ROOT / "data/naming_set/manifest.json").read_text())["crops"]
    by_obj: dict[tuple, dict[int, str]] = defaultdict(dict)
    for c in crops:
        by_obj[(c["pair"], c["object_index"], c["label"])][c["side"]] = c["path"]
    groups = [(k, v[1], v[2]) for k, v in by_obj.items() if 1 in v and 2 in v]
    groups = groups[:max_groups]
    positives = [(g[1], g[2]) for g in groups]                # same instance, two videos
    rng = random.Random(0)
    negatives = []
    while len(negatives) < len(positives):
        a, b = rng.sample(groups, 2)
        negatives.append((a[1], b[2]))                       # different instances, cross side
    return groups, positives, negatives


def embed_all(paths, model, processor):
    import torch
    from PIL import Image

    out = {}
    for p in sorted(set(paths)):
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            feats = model(**processor(images=img, return_tensors="pt"))
        h = feats.last_hidden_state
        cls = torch.nn.functional.normalize(h[:, 0], dim=-1)[0].numpy()
        patch = torch.nn.functional.normalize(h[:, 1:], dim=-1).mean(dim=1)[0].numpy()
        out[p] = (cls, patch)
    return out


def auc(pos, neg) -> float:
    """Probability that a random positive scores above a random negative."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean() * 0.5
    return float(wins + ties)


def bootstrap_auc(pos, neg, groups_pos, groups_neg, draws=500, seed=0):
    """Cluster bootstrap: resample positive object instances (and negatives with them)."""
    rng = random.Random(seed)
    by_group_pos: dict = defaultdict(list)
    for p, g in zip(pos, groups_pos):
        by_group_pos[g].append(p)
    by_group_neg: dict = defaultdict(list)
    for n, g in zip(neg, groups_neg):
        by_group_neg[g].append(n)
    keys_p = list(by_group_pos)
    keys_n = list(by_group_neg)
    vals = []
    for _ in range(draws):
        rp = [v for k in (rng.choice(keys_p) for _ in keys_p) for v in by_group_pos[k]]
        rn = [v for k in (rng.choice(keys_n) for _ in keys_n) for v in by_group_neg[k]]
        vals.append(auc(rp, rn))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="logs/bakeoff_matching_gt.json")
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModel

    groups, positives, negatives = load_pairs()
    print(f"GT-based: {len(positives)} positive pairs (same instance, videos 1 and 2), "
          f"{len(negatives)} negative pairs (different instances)")
    all_paths = [p for pair in positives for p in pair] + [p for pair in negatives for p in pair]

    sims, vectors = {}, {}
    for name, path in MODELS.items():
        processor = AutoImageProcessor.from_pretrained(str(path))
        model = AutoModel.from_pretrained(str(path)).eval()
        vectors[name] = embed_all(all_paths, model, processor)
        pos_cls = [float(np.dot(vectors[name][a][0], vectors[name][b][0])) for a, b in positives]
        pos_patch = [float(np.dot(vectors[name][a][1], vectors[name][b][1])) for a, b in positives]
        neg_cls = [float(np.dot(vectors[name][a][0], vectors[name][b][0])) for a, b in negatives]
        neg_patch = [float(np.dot(vectors[name][a][1], vectors[name][b][1])) for a, b in negatives]
        sims[name] = {"pos_cls": pos_cls, "pos_patch": pos_patch,
                      "neg_cls": neg_cls, "neg_patch": neg_patch}
        print(f"  {name}: embedded {len(vectors[name])} crops")

    results = {}
    for name in MODELS:
        r = sims[name]
        entry = {}
        for kind in ("cls", "patch"):
            pos, neg = r[f"pos_{kind}"], r[f"neg_{kind}"]
            a = auc(pos, neg)
            lo, hi = bootstrap_auc(pos, neg, list(range(len(pos))),
                                   list(range(len(neg))), draws=args.draws)
            entry[kind] = {"positive_mean": round(float(np.mean(pos)), 3),
                           "negative_mean": round(float(np.mean(neg)), 3),
                           "auc": round(a, 3), "auc_ci": [round(lo, 3), round(hi, 3)]}
        results[name] = entry
        print(f"{name}: CLS AUC {entry['cls']['auc']} {entry['cls']['auc_ci']} | "
              f"patch AUC {entry['patch']['auc']} {entry['patch']['auc_ci']}")

    # Paired difference on the SAME pairs, with a bootstrap over pairs.
    names = list(MODELS)
    paired = {}
    rng = random.Random(0)
    for kind in ("cls", "patch"):
        d = [a - b for a, b in zip(sims[names[1]][f"pos_{kind}"], sims[names[0]][f"pos_{kind}"])]
        boots = [float(np.mean([d[rng.randrange(len(d))] for _ in d])) for _ in range(args.draws)]
        paired[kind] = {"mean": round(float(np.mean(d)), 3),
                        "ci": [round(float(np.percentile(boots, 2.5)), 3),
                               round(float(np.percentile(boots, 97.5)), 3)]}
        print(f"paired difference {names[1]} - {names[0]} ({kind}): "
              f"{paired[kind]['mean']} {paired[kind]['ci']}")

    decision = all(paired[k]["ci"][0] >= 0 for k in ("cls", "patch"))
    out = ROOT / args.out
    out.write_text(json.dumps({"positives": len(positives), "negatives": len(negatives),
                               "results": results, "paired": paired,
                               "decision_keep_dinov3": decision}, indent=2))
    print(f"decision: DINOv3 stays primary = {decision} "
          f"(both paired intervals must exclude a loss)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
