"""The naming ladder: specific name, else parent category, else descriptor.

One entry in, one label out. The function is pure, so the same code path serves
the live pipeline and the offline evaluation over cached crops.

The four rungs, in order, with the reason each exists:

  a. SPECIFIC. Both the vision-language model and the closed-vocabulary rerank
     have to agree, and enough views have to agree with them. This is the only
     rung that may print an object name, so a wrong name needs three separate
     things to go wrong at once.

  b. PARENT. The candidates disagree on the exact object but agree on the
     category ("mug" vs "cup" -> tableware). Printing the category is still
     informative and is far less likely to be wrong.

  c. DESCRIPTOR. Nothing agrees. Print what is actually known: relative size, a
     validated colour word, a stable index and, when available, the position
     relative to an object the scene does contain.

  d. PARTS ARE NEVER NAMES. "leg", "handle", "top" and friends are rejected at
     every rung. This is what produced "wooden leg" for a whole chair.

Debugging: set the log level to DEBUG and every decision prints the inputs, the
rung it stopped at and the votes behind it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.naming.descriptor import describe
from src.naming.vocabulary import Vocabulary

log = logging.getLogger("naming.ladder")


@dataclass
class Decision:
    """The outcome of the ladder for one entry.

    label       the string to print
    kind        "specific", "parent" or "descriptor"
    rung        short tag of the rung that produced it, for the evaluation
    votes       how many views agreed with the chosen name
    candidates  the per-view best names above the similarity floor
    detail      free text, safe to print in a debug log
    """

    label: str
    kind: str
    rung: str
    votes: int = 0
    candidates: list[str] = field(default_factory=list)
    detail: str = ""


def decide(vlm_label: str, views: list[list[tuple[str, float]]], vocab: Vocabulary, *,
           index: int = 1, size: str = "small", color: str | None = None,
           anchor: str | None = None, min_sim: float = 0.10,
           min_votes: int = 1) -> Decision:
    """Run the ladder for one entry.

    vlm_label  the free-text name the vision-language model produced
    views      one list of (canonical name, similarity) per available crop, best
               first. Similarities are cosine values in [-1, 1]; the caller
               decides how many views to pass in.
    min_sim    similarity floor: candidates below it are ignored, which is what
               keeps a weak match from becoming a name
    min_votes  how many views must agree before rung (a) prints a name
    """
    # Keep only candidates the rerank is at least somewhat confident about.
    floored = [[(n, float(s)) for n, s in v if s >= min_sim] for v in views]
    floored = [v for v in floored if v]
    heads = [v[0][0] for v in floored]           # best name per view

    vlm_canon = vocab.canonical_of(vlm_label)
    part_only = vocab.is_part_only(vlm_label)
    log.debug("ladder input: vlm=%r vlm_canon=%r part_only=%s heads=%s min_sim=%.2f "
              "min_votes=%d", vlm_label, vlm_canon, part_only, heads, min_sim, min_votes)

    # ---- rung (a): a specific name, only with agreement -------------------
    if vlm_canon is not None and not part_only:
        votes = sum(1 for h in heads if h == vlm_canon)
        if votes >= max(1, min_votes):
            log.debug("rung a: %r agreed by %d view(s)", vlm_canon, votes)
            return Decision(label=vlm_canon, kind="specific", rung="a",
                            votes=votes, candidates=heads,
                            detail=f"vlm and rerank agreed on {vlm_canon} ({votes} view(s))")
        log.debug("rung a rejected: %r in vocabulary but only %d view(s) agreed",
                  vlm_canon, votes)
    elif part_only:
        log.debug("rung a rejected: %r names only a part", vlm_label)
    else:
        log.debug("rung a rejected: %r is not in the vocabulary", vlm_label)

    # ---- rung (b): the parent category, when the candidates share one -----
    if heads:
        parents = {vocab.parent_of(h) for h in heads}
        if len(parents) == 1 and None not in parents:
            parent = parents.pop()
            # Two independent views agreeing is enough. With one view, the
            # model has to at least place the object in the same category.
            vlm_supports = vlm_canon is not None and vocab.parent_of(vlm_canon) == parent
            if len(heads) >= 2 or vlm_supports:
                label = vocab.printed_parent(heads[0])
                log.debug("rung b: candidates %s share parent %r -> %r",
                          heads, parent, label)
                return Decision(label=label, kind="parent", rung="b",
                                votes=len(heads), candidates=heads,
                                detail=f"candidates {heads} share the category {label}")
            log.debug("rung b rejected: one view, and the vlm category disagrees")
        else:
            log.debug("rung b rejected: candidates %s span categories %s", heads, parents)

    # ---- rung (c): an honest descriptor ----------------------------------
    label = describe(index, size=size, color=color, anchor=anchor)
    log.debug("rung c: %r (no agreement; heads=%s)", label, heads)
    return Decision(label=label, kind="descriptor", rung="c", votes=0,
                    candidates=heads,
                    detail="no agreement, descriptor with index and, if known, an anchor")
