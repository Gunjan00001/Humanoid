"""The closed vocabulary: the only names the pipeline is allowed to print.

Loaded once from data/naming_vocabulary.json (a single global file, no scene
specific entries). This module answers three questions and nothing else:

  * is this phrase a known object?            -> Vocabulary.canonical_of
  * which category does it belong to?         -> Vocabulary.parent_of
  * is this phrase only a part of an object?  -> Vocabulary.is_part_only

Debugging tip: run `python -m src.naming.vocabulary` to print the size of the
loaded vocabulary and spot check a few labels.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.naming.text import clean, content_tokens, phrases

log = logging.getLogger("naming.vocabulary")

VOCAB_PATH = Path(__file__).resolve().parents[2] / "data" / "naming_vocabulary.json"


@dataclass(frozen=True)
class Vocabulary:
    """Immutable view of the vocabulary file.

    canonical: object name       -> parent category
    alias:     any accepted name -> canonical object name (canonical names map
                                    to themselves, so lookups are uniform)
    parts:     words that name a part of an object, never a whole object
    parents:   parent category   -> the words printed on screen for it
    """

    canonical: dict[str, str]
    alias: dict[str, str]
    parts: frozenset[str]
    parents: dict[str, str]

    @property
    def size(self) -> int:
        return len(self.canonical)

    def canonical_of(self, label: str) -> str | None:
        """Map free text to a canonical object name, or None if unknown.

        Matching is longest phrase first, so "remote control" resolves to
        "remote control" rather than matching the alias "remote" first.
        """
        if not clean(label):
            return None
        for phrase in phrases(label):
            if phrase in self.alias:
                canon = self.alias[phrase]
                log.debug("canonical_of(%r) -> %r via phrase %r", label, canon, phrase)
                return canon
        log.debug("canonical_of(%r) -> None (not in the vocabulary)", label)
        return None

    def parent_of(self, canonical: str) -> str | None:
        """Parent category of a canonical name, or None if it is not one."""
        return self.canonical.get(canonical)

    def is_part_only(self, label: str) -> bool:
        """True when every content word in the label names a part.

        "wooden leg" -> True, "leg of the chair" -> True, "chair leg" -> False
        (the word "chair" is not a part word, so the label still refers to a
        whole object and the ladder may keep it).
        """
        tokens = content_tokens(label)
        if not tokens:
            return False
        part_only = all(t in self.parts for t in tokens)
        log.debug("is_part_only(%r) -> %s (tokens=%s)", label, part_only, tokens)
        return part_only

    def printed_parent(self, canonical: str) -> str:
        """Human wording for a parent category, falling back to the category."""
        parent = self.parent_of(canonical)
        if parent is None:
            return "object"
        return self.parents.get(parent, parent)


def load_vocabulary(path: str | Path | None = None) -> Vocabulary:
    """Load and index the vocabulary file. Raises on a malformed file, because
    silently running with an empty vocabulary would produce all descriptors."""
    src = Path(path or VOCAB_PATH)
    raw = json.loads(src.read_text())
    canonical: dict[str, str] = {}
    alias: dict[str, str] = {}
    for parent, objects in raw["objects"].items():
        for name, extra_names in objects.items():
            canonical[name] = parent
            alias[name] = name                      # canonical names self-map
            for extra in extra_names:
                if extra in alias and alias[extra] != name:
                    log.warning("alias %r is claimed by both %r and %r; keeping the first",
                                extra, alias[extra], name)
                    continue
                alias[extra] = name
    vocab = Vocabulary(canonical=canonical, alias=alias,
                       parts=frozenset(raw["parts"]), parents=raw["parents"])
    log.info("loaded vocabulary: %d objects, %d accepted names, %d part words, "
             "from %s", vocab.size, len(alias), len(vocab.parts), src)
    return vocab


if __name__ == "__main__":                       # tiny self check, no models
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    v = load_vocabulary()
    for probe in ("remote control", "the small dark object", "wooden leg",
                  "chair leg", "coconut", "banana phone"):
        print(f"{probe!r:28} canonical={v.canonical_of(probe)!r:18} "
              f"part_only={v.is_part_only(probe)}")
