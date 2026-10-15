"""Text helpers for naming.

Everything that turns free text into tokens lives here, so the rest of the
naming code never has to care about punctuation, casing or stray whitespace.

Kept deliberately small and pure: every function takes a string and returns a
string or a list, which makes them trivial to test in a REPL or a unit test.
"""
from __future__ import annotations

import re

# Words the vision-language model sprinkles into labels that carry no object
# information. They are dropped before matching, otherwise "the cup" and "cup"
# would look like different objects.
STOPWORDS = {
    "the", "a", "an", "some", "this", "that", "these", "those",
    "with", "and", "or", "of", "on", "in", "at", "to", "from",
    "object", "item", "thing", "part", "piece",
    # hedges the model adds when it is unsure
    "possibly", "probably", "maybe", "appears", "seems", "like",
    "small", "large", "tiny", "medium",
}


def clean(label: str | None) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    "  Wooden Leg. " -> "wooden leg"
    """
    if not label:
        return ""
    text = label.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)   # punctuation and digits become spaces
    return " ".join(text.split())


def content_tokens(label: str | None) -> list[str]:
    """Tokens that carry object meaning, stopwords removed.

    "the small dark object" -> ["dark"]
    """
    return [t for t in clean(label).split() if t not in STOPWORDS]


def ngrams(tokens: list[str], n: int) -> list[str]:
    """All contiguous n-token phrases, in order.

    ["red", "apple", "box"] with n=2 -> ["red apple", "apple box"]
    """
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def phrases(label: str | None) -> list[str]:
    """Candidate phrases from longest to shortest, so "remote control" wins
    over the bare "remote" when both are in the vocabulary."""
    tokens = clean(label).split()
    out: list[str] = []
    for n in range(min(3, len(tokens)), 0, -1):
        out.extend(ngrams(tokens, n))
    return out
