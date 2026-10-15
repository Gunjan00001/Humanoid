"""Metrics for naming with abstention.

Four numbers, defined precisely because they are the ones the gate is stated in
terms of:

  coverage        share of entries given any label at all (a name or a category)
  named precision share of *named* entries whose name is right
  wrong name rate  wrong names divided by all entries. This is the number the
                   gate wants to fall, because a wrong name is worse than an
                   abstention
  abstention rate  share of entries that fell through to a descriptor

Correctness is decided against the ground-truth object for the entry, using the
same canonicalisation as the vocabulary, so "remote" and "remote control" count
as the same object. A category label counts as "category correct, not a name",
which is reported separately rather than being folded into precision.

Leave one scene out: the caller tunes global knobs on the training scenes and
this module scores the held-out scene with those knobs, so no scene can tune
itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.naming.descriptor import is_descriptor

log = logging.getLogger("naming.evaluation")


@dataclass
class Outcomes:
    """Per-entry verdicts for one configuration, ready to score."""

    total: int = 0
    named_specific: int = 0          # printed an object name
    named_parent: int = 0            # printed a category
    abstained: int = 0               # printed a descriptor
    specific_correct: int = 0
    parent_correct: int = 0
    wrong: int = 0                   # named, but the name is not the right object
    rows: list[dict] = field(default_factory=list)   # one audit row per entry

    def add(self, *, kind: str, correct: bool, label: str, truth: str | None,
            scene: str, entry_id: int, detail: str = "") -> None:
        self.total += 1
        if is_descriptor(label):
            self.abstained += 1
        elif kind == "parent":
            self.named_parent += 1
        else:
            self.named_specific += 1
        if kind == "specific" and correct:
            self.specific_correct += 1
        elif kind == "parent" and correct:
            self.parent_correct += 1
        elif kind in ("specific", "parent") and not correct:
            self.wrong += 1
        self.rows.append({"scene": scene, "entry_id": entry_id, "label": label,
                          "truth": truth, "kind": kind, "correct": correct,
                          "detail": detail})

    # -- the four headline numbers ----------------------------------------
    @property
    def coverage(self) -> float:
        return (self.named_specific + self.named_parent) / self.total if self.total else 0.0

    @property
    def named_total(self) -> int:
        return self.named_specific + self.named_parent

    @property
    def named_precision(self) -> float:
        if not self.named_total:
            return 0.0
        return (self.specific_correct + self.parent_correct) / self.named_total

    @property
    def name_precision(self) -> float:
        """Precision counting only real names, ignoring category labels."""
        return self.specific_correct / self.named_specific if self.named_specific else 0.0

    @property
    def wrong_name_rate(self) -> float:
        return self.wrong / self.total if self.total else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.abstained / self.total if self.total else 0.0

    def table(self) -> str:
        return (f"entries {self.total:3d} | named {self.named_total:3d} "
                f"({100 * self.coverage:4.1f}%) | name precision "
                f"{100 * self.name_precision:4.1f}% | wrong names {self.wrong} "
                f"({100 * self.wrong_name_rate:4.1f}%) | abstained "
                f"{self.abstained} ({100 * self.abstention_rate:4.1f}%)")

    def summary(self) -> dict:
        return {"total": self.total, "named_specific": self.named_specific,
                "named_parent": self.named_parent, "abstained": self.abstained,
                "specific_correct": self.specific_correct,
                "parent_correct": self.parent_correct, "wrong": self.wrong,
                "coverage": round(self.coverage, 4),
                "named_precision": round(self.named_precision, 4),
                "name_precision": round(self.name_precision, 4),
                "wrong_name_rate": round(self.wrong_name_rate, 4),
                "abstention_rate": round(self.abstention_rate, 4)}


def label_is_correct(label: str, kind: str, truth: str | None, vocab) -> bool:
    """Is this label right for an entry whose ground-truth object is `truth`?

    A name is right when it canonicalises to the same object. A category is
    right when it is the ground-truth object's category. An entry with no
    ground truth (an extra) can never be right.
    """
    if truth is None:
        return False
    truth_canon = vocab.canonical_of(truth) or truth.lower()
    if kind == "specific":
        return vocab.canonical_of(label) == truth_canon
    if kind == "parent":
        return vocab.printed_parent(truth_canon) == label
    return False


def leave_one_scene_out(rows: list[dict], knobs: list[dict], score_fn) -> dict:
    """Tune on every scene but one, then score the held-out scene.

    rows   one dict per entry with a "scene" key
    knobs  the global settings to choose between, for example
           [{"min_sim": 0.10, "min_votes": 1}, ...]
    score_fn(rows, knob) -> Outcomes

    Returns {"choice": {scene: knob}, "per_scene": {...}, "overall": Outcomes}.
    """
    scenes = sorted({r["scene"] for r in rows})
    choice: dict[str, dict] = {}
    per_scene: dict[str, dict] = {}
    for held in scenes:
        train = [r for r in rows if r["scene"] != held]
        test = [r for r in rows if r["scene"] == held]
        if not train or not test:
            continue
        # Pick the knob with the best training score: fewest wrong names first,
        # then highest coverage, so the tie break favours answering.
        best, best_key = None, None
        for knob in knobs:
            o = score_fn(train, knob)
            key = (-o.wrong_name_rate, o.coverage)
            log.debug("LOSO train=%s knob=%s -> wrong=%.3f coverage=%.3f",
                      "all but " + held, knob, o.wrong_name_rate, o.coverage)
            if best_key is None or key > best_key:
                best, best_key = knob, key
        choice[held] = best
        out = score_fn(test, best)
        per_scene[held] = {"knob": best, **out.summary()}
        log.info("LOSO held out %s: %s", held, out.table())
    return {"choice": choice, "per_scene": per_scene}
