"""Scoring free-form answers without understating the models.

The problem this solves: a model that answers "remote" for a remote control, or
"paintbrush" for a paint brush, was counted wrong purely because the wording did
not match a vocabulary entry, which understates every model. The benchmark's
ground-truth labels are never added to the vocabulary here (that would leak the
targets); instead an answer is accepted in four documented steps:

  1. text normalisation: lowercase, punctuation removed, simple singular forms;
  2. the written synonym table in data/naming_synonyms_free.json, one entry per
     mapping, reviewable line by line;
  3. an embedding fallback: the answer is compared with the crop's own
     ground-truth label and its parent category using the SigLIP text encoder,
     accepted only when the similarity clears a threshold AND leads the runner-up
     by a margin (both written below);
  4. everything the fallback calls borderline is written out for manual review,
     marked "agent reviewed, pending user confirmation".

Every decision records which step made it, so the report can state how much of
the accuracy came from wording normalisation rather than recognition.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYNONYMS = ROOT / "data" / "naming_synonyms_free.json"

# Threshold and margin for step 3. Chosen before the rescore was run, from the
# observed SigLIP text-similarity range on the control pairs (same wording pairs
# such as cup/mug score well above 0.8, unrelated nouns below 0.6). Both are
# global; nothing here is per scene or per model.
EMBED_THRESHOLD = 0.72
EMBED_MARGIN = 0.03


def singular(word: str) -> str:
    """Simple singular forms; a full lemmatiser would be another dependency."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, singularise each word, collapse spaces."""
    words = re.sub(r"[^a-z0-9 ]", " ", str(text or "").lower()).split()
    return " ".join(singular(w) for w in words)


def load_synonyms(path: Path | None = None) -> dict[str, str]:
    raw = json.loads((path or SYNONYMS).read_text())
    return {normalize(k): v for k, v in raw["entries"].items()}


@dataclass
class Verdict:
    """One scoring decision, with the step that made it."""

    correct: bool
    step: str            # "exact" | "synonym" | "embedding" | "borderline" | "no"
    detail: str = ""


class FreeformScorer:
    """Score free-form answers against the crop's ground truth and its parent."""

    def __init__(self, vocabulary, synonyms: dict[str, str] | None = None,
                 embedder=None, threshold: float = EMBED_THRESHOLD,
                 margin: float = EMBED_MARGIN):
        self.vocab = vocabulary
        self.synonyms = synonyms or load_synonyms()
        self.embedder = embedder                # optional; sentence-transformers-like
        self.threshold = threshold
        self.margin = margin
        self.borderline: list[dict] = []

    # -- step 1 and 2 ------------------------------------------------------
    def canonical_guess(self, answer: str) -> tuple[str | None, str]:
        """(canonical name or None, the step that decided)."""
        n = normalize(answer)
        if not n:
            return None, "no"
        canon = self.vocab.canonical_of(n)
        if canon:
            return canon, "exact"
        if n in self.synonyms:
            target = self.synonyms[n]
            canon = self.vocab.canonical_of(target)
            if canon:
                return canon, "synonym"
        return None, "no"

    # -- step 3 ------------------------------------------------------------
    def _embed(self, text: str):
        if self.embedder is None:
            return None
        return self.embedder(text)

    def score(self, answer: str, truth_canonical: str, parent_label: str) -> Verdict:
        """Decide one crop. Accepted answers may be exact or parent-category."""
        canon, step = self.canonical_guess(answer)
        if canon is not None:
            if canon == truth_canonical:
                return Verdict(True, step)
            if self.vocab.parent_of(canon) and \
                    self.vocab.parent_of(canon) == self.vocab.parent_of(truth_canonical):
                return Verdict(True, step, "parent category agreed")
            return Verdict(False, "no", f"in vocabulary but {canon} != {truth_canonical}")

        # Step 3: embedding fallback. The answer is ranked against the WHOLE
        # vocabulary by text similarity; it counts when the nearest term is the
        # crop's truth (or its parent), clears the threshold and leads the
        # runner-up by the margin. Ranking the whole list is what makes the
        # margin meaningful; comparing two candidates makes it vacuous.
        vec_a = self._embed(normalize(answer))
        if vec_a is None:
            return Verdict(False, "no", "out of vocabulary, no embedder available")
        sims = []
        for name in self.vocab.canonical:
            vec_c = self._embed(name)
            if vec_c is None:
                continue
            sims.append((name, float(sum(x * y for x, y in zip(vec_a, vec_c)))))
        if not sims:
            return Verdict(False, "no", "out of vocabulary, embedder empty")
        sims.sort(key=lambda t: -t[1])
        best_name, best_sim = sims[0]
        second_sim = sims[1][1] if len(sims) > 1 else 0.0
        lead = best_sim - second_sim
        same_object = (best_name == truth_canonical or
                       (self.vocab.parent_of(best_name) is not None and
                        self.vocab.parent_of(best_name) == self.vocab.parent_of(truth_canonical)))
        if same_object and best_sim >= self.threshold and lead >= self.margin:
            return Verdict(True, "embedding", f"{best_name} at {best_sim:.3f}, lead {lead:.3f}")
        if abs(best_sim - self.threshold) < self.margin or (same_object and lead < self.margin):
            self.borderline.append({"answer": answer, "truth": truth_canonical,
                                    "nearest": best_name, "sim": round(best_sim, 3),
                                    "lead": round(lead, 3)})
            return Verdict(False, "borderline", f"nearest {best_name} at {best_sim:.3f}, lead {lead:.3f}")
        return Verdict(False, "no", f"nearest {best_name} at {best_sim:.3f}, lead {lead:.3f}")
        return Verdict(False, "no", f"best {best} at {best_sim:.3f} below the threshold")
