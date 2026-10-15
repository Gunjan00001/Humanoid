"""Inputs for the naming work, read from the caches the pipeline already wrote.

No model calls and no video decoding: this module only gathers what a previous
run produced, so the evaluation is cheap and repeatable.

Per scene it joins four files:

  outputs/<scene>/report.json          the entries the pipeline reported
  cache/<scene>/semantics/crops.json   the crop images per cluster
  cache/<scene>/semantics/vlm.json     the vision-language model's label per cluster
  data/<scene>/matches.json            the human-resolved scoring (names, types)
  data/<scene>/ground_truth.json       the true object per change

Every entry becomes one EntryInput, which carries everything the ladder and the
evaluation need. If a piece is missing the entry is kept with that field None
and a warning is logged, so a partial cache gives a partial table rather than a
crash.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("naming.cache")

DEFAULT_SCENES = ["demo", "scene2", "scene3", "scene4", "scene5", "scene6"]


def _read(path: Path) -> dict | None:
    """Read JSON, or None with a warning when it is missing or malformed."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        log.warning("missing %s", path)
        return None
    except json.JSONDecodeError as exc:
        log.warning("malformed JSON in %s: %s", path, exc)
        return None


@dataclass
class EntryInput:
    """One report entry, with its cached evidence and its ground truth."""

    scene: str
    entry_id: int
    change_type: str
    printed_label: str                  # what the pipeline printed at the time
    cluster_ids: list[int]
    views: list[str] = field(default_factory=list)       # crop image paths
    vlm_label: str = ""
    vlm_description: str = ""
    truth: str | None = None            # ground-truth object, None for an extra
    truth_type: str | None = None
    name_ok_at_the_time: bool | None = None              # human verdict, for reference
    kind: str = "change"                # "change" or "extra"

    @property
    def is_extra(self) -> bool:
        return self.truth is None


def scene_paths(scene: str, root: Path) -> dict[str, Path]:
    """All file locations for one scene.

    The demo is the odd one out: its committed output is outputs/report.json and
    its cache directory has changed name across the project's history, so the
    candidates are tried in order and the first one with crops wins.
    """
    if scene == "demo":
        base = root / "data"
        cache = None
        for candidate in ("demo", "run", "da3_run_tab", "da3_run"):
            if (root / "cache" / candidate / "semantics" / "crops.json").exists():
                cache = root / "cache" / candidate
                break
        cache = cache or root / "cache" / "demo"
        return {"report": root / "outputs" / "report.json",
                "crops": cache / "semantics" / "crops.json",
                "vlm": cache / "semantics" / "vlm.json",
                "matches": base / "matches.json",
                "truth": base / "ground_truth.json"}
    base = root / "data" / scene
    return {
        "report": root / "outputs" / scene / "report.json",
        "crops": root / "cache" / scene / "semantics" / "crops.json",
        "vlm": root / "cache" / scene / "semantics" / "vlm.json",
        "matches": base / "matches.json",
        "truth": base / "ground_truth.json",
    }


def load_scene(scene: str, root: Path, include_unverified: bool = False
               ) -> list[EntryInput]:
    """Build one EntryInput per reported entry for a scene.

    Only the entries the pipeline actually reported are scored by default.
    Unverified entries have no human resolution behind them, so including them
    would compare the ladder against a label nobody checked.
    """
    paths = scene_paths(scene, root)
    report = _read(paths["report"])
    if report is None:
        return []
    crops = _read(paths["crops"]) or {}
    vlm = _read(paths["vlm"]) or {}
    matches = (_read(paths["matches"]) or {}).get("entries", [])
    truth_changes = (_read(paths["truth"]) or {}).get("changes", [])
    truth_by_id = {c["id"]: c for c in truth_changes}
    match_by_entry = {m.get("entry_id"): m for m in matches}

    reported = list(report.get("changes", []))
    if include_unverified:
        reported += list(report.get("unverified_changes", []))
    out: list[EntryInput] = []
    for entry in reported:
        entry_id = int(entry.get("id", len(out) + 1))
        m = match_by_entry.get(entry_id, {})
        gt_id = m.get("gt_id")
        truth = truth_by_id.get(gt_id, {}) if gt_id else {}

        # Gather the crops for every cluster the entry covers, A side first.
        views: list[str] = []
        vlm_labels: list[str] = []
        vlm_descriptions: list[str] = []
        for cid in entry.get("cluster_ids", []):
            info = crops.get(str(cid), {})
            for key in ("crop", "crop_other"):
                if info.get(key) and Path(info[key]).exists():
                    views.append(info[key])
            answer = vlm.get(str(cid), {})
            if answer.get("label"):
                vlm_labels.append(answer["label"])
            if answer.get("description"):
                vlm_descriptions.append(answer["description"])

        item = EntryInput(
            scene=scene,
            entry_id=entry_id,
            change_type=entry.get("type", ""),
            printed_label=entry.get("label", ""),
            cluster_ids=list(entry.get("cluster_ids", [])),
            views=views,
            # The pipeline prints one label per entry; when the clusters behind
            # an entry disagree, join them so the ladder sees the whole story.
            vlm_label=" / ".join(vlm_labels),
            vlm_description=" ".join(vlm_descriptions),
            truth=truth.get("object"),
            truth_type=truth.get("type"),
            name_ok_at_the_time=m.get("name_ok"),
            kind="extra" if gt_id is None else "change",
        )
        if not views:
            log.warning("%s entry %d: no cached crops found (clusters %s)",
                        scene, entry_id, item.cluster_ids)
        out.append(item)

    log.info("%s: %d entries, %d with crops, %d with a ground-truth object",
             scene, len(out), sum(1 for e in out if e.views),
             sum(1 for e in out if e.truth))
    return out


def load_all(scenes: list[str] | None = None, root: Path | None = None
             ) -> list[EntryInput]:
    root = root or Path(__file__).resolve().parents[2]
    rows: list[EntryInput] = []
    for scene in scenes or DEFAULT_SCENES:
        rows.extend(load_scene(scene, root))
    return rows


if __name__ == "__main__":                       # quick inventory, no models
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for e in load_all():
        print(f"{e.scene:8} #{e.entry_id:<2} {e.change_type:9} printed={e.printed_label!r:28} "
              f"vlm={e.vlm_label!r:34} truth={e.truth!r:12} views={len(e.views)}")
