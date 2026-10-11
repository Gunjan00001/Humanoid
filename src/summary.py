"""Natural-language summary of a change report (v0.12, Step 2).

    python src/summary.py --config configs/default.yaml --cache cache --out outputs

Reads `outputs/<name>/report.json` and writes `summary.md`, `summary.txt` and
`summary.json`. Sentences come only from report.json fields:

- each change's sentence is phrased by Qwen3-VL in ONE call when a GPU is
  available; every returned sentence is validated against that change's label,
  the type's own action vocabulary and a non-object word list. A sentence is
  rejected when it uses any other word (objects, counts, or visual
  embellishments like "leaving only the cracked surface" - found in the scene5
  held-out review), describes a different change type (e.g. "replaced" for a
  removal - the scene4 scissors sentence), or states a direction without
  saying which video it refers to;
- on rejection (or without a GPU) a deterministic template is used.

This stage never changes detection: it only phrases what the report contains.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import (  # noqa: E402
    add_common_args,
    apply_overrides,
    load_config,
    out_dir,
    read_json,
    setup_logging,
    stamp,
    write_json,
)
from src.describe import _STOPWORDS  # noqa: E402

log = setup_logging("summary")

# --- vocabulary -------------------------------------------------------------

TYPE_ACTIONS = {
    "removed": ("removed", "gone", "disappeared", "vanished", "missing",
                "taken", "no longer", "absent", "not present"),
    "added": ("added", "appeared", "placed", "introduced", "new",
              "present", "appears"),
    "moved": ("moved", "relocated", "shifted", "repositioned",
              "different position", "compared", "in a different"),
    "replaced": ("replaced", "swapped", "changed", "instead"),
}
# actions that must NOT appear in a sentence of this type
FORBIDDEN_ACTIONS = {
    "removed": ("replac", "appeared", "was added"),
    "added": ("replac", "removed", "gone", "disappeared"),
    "moved": ("replac", "removed", "appeared"),
    "replaced": ("removed", "appeared"),
}
DIRECTION = {"left", "right", "up", "upward", "down", "downward", "leftward",
             "rightward", "forward", "backward", "corner"}
# scene words that may appear in a sentence without naming an object
GENERIC = {"video", "before", "after", "scene", "table", "room", "image",
           "picture", "crop", "frame", "first", "second", "slightly",
           "position", "spot", "place", "location", "area", "view", "viewer",
           "two", "different", "between", "still", "now", "then", "again",
           "tabletop", "surface", "top", "side", "background",
           "direction", "seen", "crops"}
POSSIBLY_BELOW = 0.40     # report confidence below this -> "possibly ..."
# common function words the object-description stopword list does not cover
FUNCTION = {
    "was", "were", "is", "are", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "may", "might",
    "from", "to", "by", "at", "for", "as", "than", "into", "upon", "over",
    "under", "not", "it", "its", "this", "that", "these", "those", "which",
    "who", "what", "when", "where", "while", "with", "without", "within",
    "during", "along", "toward", "towards", "against", "about", "also",
}
# words a rephrasing may add without naming a new object: motion/change verbs,
# positions, scene words, degree words. Object nouns are NOT here - they must
# come from the report's own label/description fields.
NON_OBJECT = {
    "moved", "removed", "added", "appeared", "replaced", "changed", "shifted",
    "relocated", "repositioned", "swapped", "introduced", "taken", "gone",
    "disappeared", "vanished", "missing", "placed", "left", "put",
    "position", "positions", "spot", "location", "place", "area", "region",
    "side", "corner", "edge", "centre", "center", "middle", "front", "back",
    "left", "right", "top", "bottom", "up", "upward", "down", "downward",
    "forward", "backward", "near", "nearby", "next", "behind", "beside",
    "between", "across", "around", "inside", "outside", "above", "below",
    "onto", "into", "out", "off", "away", "different", "same", "other",
    "another", "one", "two", "both", "new", "old", "scene", "video", "frame",
    "image", "picture", "crop", "view", "viewer", "table", "tabletop",
    "surface", "room", "background", "foreground", "shot", "recording",
    "slightly", "completely", "entirely", "fully", "partly", "partially",
    "still", "now", "then", "again", "before", "after", "during", "while",
    "when", "where", "longer", "only", "just", "also", "very", "much",
    "more", "less", "direction", "seen", "crops", "1", "2", "compared",
    "present", "a", "b", "floor", "ground", "wall",
}
VIDEO_REF = re.compile(r"video [ab]|before|after", re.I)


def _significant(text) -> set[str]:
    s = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower())
    return {w for w in s.split() if w and w not in _STOPWORDS}


def _singular(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def validate_sentence(sentence: str, change: dict) -> bool:
    """True when the sentence is safe for this change (report-only wording)."""
    if not sentence or not sentence.strip() or len(sentence) > 220:
        return False
    low = sentence.lower()
    actions = TYPE_ACTIONS.get(change["type"], ())
    if not any(a in low for a in actions):
        return False                                  # wrong or missing action
    if any(bad in low for bad in FORBIDDEN_ACTIONS.get(change["type"], ())):
        return False                                  # another type's action
    # Vocabulary = the change's LABEL + the type's own action words +
    # non-object words. The report's free-text description is deliberately NOT
    # a vocabulary source: it carries visual embellishments from the VLM
    # ("leaving only the cracked surface in video B") that are grounded in the
    # crop, not in the detection. (Found by the scene5 held-out review.)
    allowed: set[str] = set()
    allowed |= _significant(change.get("label"))
    allowed |= set(actions) | GENERIC | NON_OBJECT | FUNCTION
    allowed |= {_singular(w) for w in allowed}
    words = _significant(sentence)
    extra = {w for w in words if w not in allowed and _singular(w) not in allowed}
    if extra:
        return False                                  # names something not in the report
    if re.search(r"video\s*[12]\b", low):
        return False                                  # invalid video reference
    if any(w in DIRECTION for w in words) and not VIDEO_REF.search(low):
        return False                                  # direction without a video reference
    return True


def template_sentence(change: dict) -> str:
    t = change.get("type", "changed")
    label = (change.get("label") or "object").strip()
    if t == "removed":
        verb = "were" if label.endswith("s") else "was"
        return f"The {label} {verb} removed."
    if t == "added":
        return f"The {label} appeared."
    if t == "moved":
        return f"The {label} moved to a different position between the two videos."
    if t == "replaced":
        return f"The {label} changed between the two videos."
    return f"The {label} changed."


def _with_video_ref(sentence: str) -> str:
    return sentence.rstrip(". ") + " (direction as seen in the before/after crops)."


def sentence_for(change: dict, vlm_sentences: dict) -> tuple[str, str]:
    """(sentence, source) with source in {vlm, report, template}."""
    cand = vlm_sentences.get(str(change.get("id")))
    if cand and validate_sentence(cand, change):
        return cand.strip(), "vlm"
    desc = change.get("description")
    if desc and validate_sentence(desc, change):
        return desc.strip(), "report"
    if desc and any(w in DIRECTION for w in _significant(desc)) \
            and not VIDEO_REF.search(desc):
        cand2 = _with_video_ref(desc)
        if validate_sentence(cand2, change):
            return cand2, "report"
    return template_sentence(change), "template"


def vlm_phrase(changes: list[dict], cfg: dict) -> dict:
    """ONE Qwen3-VL call that rephrases the changes; {} on any failure."""
    try:
        import torch

        if not torch.cuda.is_available():
            log.info("no GPU available - using the deterministic templates")
            return {}
        from src.describe import load_vlm, parse_json_reply, vlm_answer

        model, processor, _ = load_vlm(cfg)
        facts = [{"id": c["id"], "type": c["type"], "label": c.get("label"),
                  "description": c.get("description")} for c in changes]
        prompt = (
            "Write one short factual sentence for each confirmed 3D change. "
            "Use ONLY the given fields; never add objects, counts or actions. "
            "Do not report distances. The two videos are called video A (the "
            "before video) and video B (the after video); if you name a "
            "direction, say which of those two videos it refers to. Never "
            "write 'video 1' or 'video 2'. Answer JSON only: "
            '{"changes": [{"id": 1, "sentence": "..."}]}\n'
            f"Changes: {json.dumps(facts, ensure_ascii=False)}")
        reply = vlm_answer(model, processor, [], prompt)
        log.info("VLM phrasing reply: %s", reply[:400].replace("\n", " "))
        parsed = parse_json_reply(reply)
        out = {}
        for item in parsed.get("changes", []) if isinstance(parsed, dict) else []:
            if isinstance(item, dict) and item.get("id") is not None:
                out[str(item["id"])] = str(item.get("sentence") or "")
        log.info("VLM phrased %d sentence(s) in one call", len(out))
        return out
    except Exception as exc:  # noqa: BLE001 - the template fallback is always valid
        log.warning("VLM phrasing failed (%s) - using the deterministic templates", exc)
        return {}


def build_summary(cfg: dict, use_vlm: bool = True) -> dict:
    out = out_dir(cfg)
    report_path = out / "report.json"
    if not report_path.exists():
        raise SystemExit(f"[summary] {report_path} not found - run report first")
    report = read_json(report_path)
    changes = report.get("changes", [])
    unverified = report.get("unverified_changes", [])
    name = report.get("meta", {}).get("name") or out.name

    vlm_sentences = vlm_phrase(changes, cfg) if (use_vlm and changes) else {}
    entries = []
    for c in changes:
        sentence, source = sentence_for(c, vlm_sentences)
        conf = float(c.get("confidence") or 0.0)
        possibly = conf < POSSIBLY_BELOW
        sentence = sentence[0].upper() + sentence[1:]
        if possibly:
            sentence = "Possibly " + sentence[0].lower() + sentence[1:]
        entries.append({"id": c.get("id"), "type": c.get("type"),
                        "label": c.get("label"), "sentence": sentence,
                        "confidence": round(conf, 2), "possibly": possibly,
                        "sentence_source": source})

    meta = report.get("meta", {})
    video_a = Path(meta.get("video_a", "video A")).name
    video_b = Path(meta.get("video_b", "video B")).name

    notes = []
    if unverified:
        notes.append(f"{len(unverified)} unverified candidate(s) were seen in one "
                     f"video only; they are listed in report.json but not counted "
                     f"as changes.")
    notes.append("Object names come from a vision-language model and can be wrong.")
    notes.append("Detection thresholds were tuned on the development scenes "
                 "(demo, scene2, scene3, scene4).")
    notes.append("No absolute distances are reported (the reconstruction scale is "
                 "not metric-validated); directions refer to the before/after "
                 "videos as stated.")

    files = [f for f in ("viewer.html", "change.gif", "report.json")
             if (out / f).exists()]

    if entries:
        head = (f"Compared {video_a} and {video_b}. I found "
                f"{len(entries)} change(s):")
        lines = [head]
        for i, e in enumerate(entries, 1):
            lines.append(f"{i}. {str(e['type']).capitalize()} - "
                         f"{e['label']}: {e['sentence']}")
    else:
        lines = [f"Compared {video_a} and {video_b}. No changes found."]
    lines.append("")
    lines.append("Notes: " + " ".join(notes))
    lines.append("Files: " + ", ".join(files))
    text = "\n".join(lines)

    return {
        "name": name,
        "video_a": meta.get("video_a"), "video_b": meta.get("video_b"),
        "n_changes": len(entries),
        "changes": entries,
        "n_unverified": len(unverified),
        "notes": notes,
        "files": files,
        "generated": stamp(),
        "text": text,
    }


def write_summaries(out: Path, summary: dict) -> list[Path]:
    write_json(out / "summary.json", summary)
    (out / "summary.txt").write_text(summary["text"] + "\n")
    (out / "summary.md").write_text(
        "# Change summary\n\n```\n" + summary["text"] + "\n```\n")
    return [out / "summary.md", out / "summary.txt", out / "summary.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--no-vlm", action="store_true",
                        help="skip the Qwen3-VL phrasing call (deterministic templates)")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    summary = build_summary(cfg, use_vlm=not args.no_vlm)
    paths = write_summaries(out_dir(cfg), summary)
    print(summary["text"])
    log.info("wrote %s", ", ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
