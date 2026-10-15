"""Honest descriptors for objects the pipeline refuses to name.

When the ladder cannot agree on a specific name, it prints something like
"unidentified small dark object (#2) near the cup" instead of guessing.

Two rules are enforced here:

  * sizes are relative words only (tiny, small, large). No centimetres, because
    the reconstruction scale is not metric validated;
  * the index in "(#N)" is assigned once by the caller and then never changes,
    so report.json, the viewer, the GIF and the summary all agree.
"""
from __future__ import annotations

import logging

log = logging.getLogger("naming.descriptor")

# Size words, smallest first. "medium" is available but rarely needed; the
# comparison uses the object's extent relative to the scene's reference sizes.
SIZE_WORDS = ("tiny", "small", "medium", "large")

# Colour words the vision-language model may contribute. Anything outside this
# list is dropped rather than printed, so a hallucinated colour cannot leak into
# a label.
COLOR_WORDS = frozenset({
    "white", "black", "dark", "grey", "gray", "brown", "red", "orange",
    "yellow", "green", "blue", "purple", "pink", "beige", "cream", "silver",
    "gold", "metallic", "transparent", "clear",
})


def size_word(extent: float | None, tiny_below: float = 0.05,
              small_below: float = 0.15) -> str:
    """Pick a relative size word from an object extent in reconstruction units.

    The two cut points are global, not per scene. Callers that have no extent
    (for example an entry built from a cached crop alone) get "small", the
    neutral choice.
    """
    if extent is None:
        log.debug("size_word(None) -> small (no extent available)")
        return "small"
    if extent < tiny_below:
        word = "tiny"
    elif extent < small_below:
        word = "small"
    else:
        word = "large"
    log.debug("size_word(%.3f) -> %s", extent, word)
    return word


def color_word(text: str | None) -> str | None:
    """First colour word mentioned in a description, or None."""
    if not text:
        return None
    for token in text.lower().replace(",", " ").split():
        token = token.strip(".;:()")
        if token in COLOR_WORDS:
            log.debug("color_word(%r) -> %r", text[:40], token)
            return token
    return None


def describe(index: int, size: str = "small", color: str | None = None,
             anchor: str | None = None) -> str:
    """Build the descriptor string.

    index   stable number for this report entry, starting at 1
    size    relative size word; anything unknown becomes "small"
    color   already validated against COLOR_WORDS, or None
    anchor  a name the scene does contain, printed as "near the <anchor>"
    """
    size = size if size in SIZE_WORDS else "small"
    color = color if color in COLOR_WORDS else None
    words = [w for w in ("unidentified", size, color, "object") if w]
    label = f"{' '.join(words)} (#{index})"
    if anchor:
        label = f"{label} near the {anchor}"
    log.debug("describe -> %r", label)
    return label


def is_descriptor(label: str) -> bool:
    """True when a label is an abstention rather than a name. Used by the
    evaluation and by the summary wording."""
    return label.startswith("unidentified ") and "(#" in label


def index_of(label: str) -> int | None:
    """The stable number in a descriptor, or None if it is not one."""
    if not is_descriptor(label):
        return None
    try:
        return int(label.split("(#", 1)[1].split(")", 1)[0])
    except (IndexError, ValueError):             # pragma: no cover - defensive
        return None
