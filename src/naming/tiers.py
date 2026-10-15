"""Three tier labels: plain name, "possibly", or an honest descriptor.

Flag `semantics.naming_v4`, default OFF.

The point of three tiers is that a single binary "name or abstain" hides the
middle ground. Often one signal is solid and another is not, and saying
"possibly cup (#2)" is more useful than either guessing "cup" or refusing to say
anything.

The tiers:

  plain          the VLM answer is in the SigLIP top 3 **and** the views agree.
                 Printed as the bare name: "coconut"
  possibly       exactly one of those two signals holds.
                 Printed as "possibly coconut (#2)"
  unidentified   neither signal holds.
                 Printed as "unidentified small dark object (#2)"

WHICH SCORE IS USED, and why it is not a cosine threshold: raw SigLIP cosine
values are not comparable across images or vocabularies, so a fixed cut like 0.2
is not defensible. This module uses **rank** (is the VLM answer inside the top
three for that crop?) and the **top-1 versus top-2 margin** (how far clear is the
winner?) only. The margin is a difference of scores, but it is used as a
tie-breaker between candidates for one crop, never as an absolute gate, and it
only ever separates "one view is confident" from "one view is nearly tied".
No absolute value of any similarity is ever compared against a constant.

Part words are rejected in every tier, as in v3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.naming.descriptor import describe
from src.naming.vocabulary import Vocabulary

log = logging.getLogger("naming.tiers")

PLAIN = "plain"
POSSIBLY = "possibly"
UNIDENTIFIED = "unidentified"

# The VLM answer has to sit inside this many top candidates of the reopen.
PLAIN_TOP_N = 3

# A margin below this between the best and second best candidate counts as a
# near tie for that crop, so the view does not count as confident agreement.
# It is used only to compare two candidates of the SAME crop, never as an
# absolute similarity gate.
TIE_MARGIN = 0.01


@dataclass
class TierDecision:
    """Outcome for one entry.

    label   the string to print, including "(#N)" for the hedged tiers
    tier    "plain", "possibly" or "unidentified"
    votes   views whose best candidate matched the chosen name
    rank    position of the chosen name in its best view, 1-based, 0 if absent
    signals which of the two signals held, for the audit trail
    margin  top-1 minus top-2 for the view the name came from
    detail  free text for logs
    """

    label: str
    tier: str
    votes: int = 0
    rank: int = 0
    signals: dict = field(default_factory=dict)
    margin: float = 0.0
    detail: str = ""


def _rank_of(name: str, view: list[tuple[str, float]]) -> int:
    """1-based position of `name` in one view's ranked candidates, 0 if absent."""
    for position, (candidate, _score) in enumerate(view, start=1):
        if candidate == name:
            return position
    return 0


def _margin(view: list[tuple[str, float]]) -> float:
    """Top-1 minus top-2 for one view. Used for same-crop tie breaking only."""
    if len(view) < 2:
        return float("inf")
    return float(view[0][1]) - float(view[1][1])


def decide_tier(vlm_label: str, views: list[list[tuple[str, float]]], vocab: Vocabulary,
                *, index: int = 1, size: str = "small", color: str | None = None,
                anchor: str | None = None, min_votes: int = 2,
                top_n: int = PLAIN_TOP_N) -> TierDecision:
    """Choose a tier for one entry.

    views  one ranked list of (canonical name, similarity) per crop, best first.
           Only the ORDER and the top-1 versus top-2 difference are used.
    min_votes
           how many views have to agree for the agreement signal to hold. It is
           deliberately NOT reduced when fewer views exist: with one crop there is
           no multi view agreement, so such an entry can reach "possibly" but
           never "plain". That is the honest reading of the tier definition.
    """
    usable = [v for v in views if v]
    required_votes = min_votes

    vlm_canon = vocab.canonical_of(vlm_label)
    part_only = vocab.is_part_only(vlm_label)

    # Signal 1: is the answer inside the top N of any view, by rank not by score?
    best_rank = 0
    best_margin = 0.0
    if vlm_canon:
        for v in usable:
            r = _rank_of(vlm_canon, v)
            if r and (best_rank == 0 or r < best_rank):
                best_rank, best_margin = r, _margin(v)
    # Rank is the gate; the margin is only recorded and used for tie breaking in
    # signal 2, never compared against an absolute constant.
    in_top_n = bool(best_rank) and best_rank <= top_n

    # Signal 2: how many views put the answer first, ignoring near ties
    votes = sum(1 for v in usable if v and v[0][0] == vlm_canon
                and _margin(v) >= TIE_MARGIN) if vlm_canon else 0
    agrees = votes >= required_votes

    log.debug("tier input: vlm=%r canon=%r part_only=%s rank=%d votes=%d/%d margin=%.3f",
              vlm_label, vlm_canon, part_only, best_rank, votes, required_votes, best_margin)

    signals = {"in_top_n": in_top_n, "votes": votes, "votes_needed": required_votes,
               "margin": round(best_margin, 4)}

    if part_only:
        # A part word can never be a label, whatever the rerank thinks.
        log.debug("tier: part only, refusing to name")
        return TierDecision(label=describe(index, size, color, anchor), tier=UNIDENTIFIED,
                            votes=votes, rank=best_rank, signals=signals,
                            margin=best_margin, detail="label named only a part")

    if vlm_canon and in_top_n and agrees:
        log.debug("tier plain: %r ranked %d and %d view(s) agreed", vlm_canon, best_rank, votes)
        return TierDecision(label=vlm_canon, tier=PLAIN, votes=votes, rank=best_rank,
                            signals=signals, margin=best_margin,
                            detail=f"{vlm_canon} in top {top_n} by rank and {votes} view(s) agreed")

    if vlm_canon and (in_top_n or agrees):
        which = "rank only" if in_top_n else "votes only"
        log.debug("tier possibly: %r, %s", vlm_canon, which)
        label = f"possibly {vlm_canon} (#{index})"
        if anchor:
            label += f" near the {anchor}"
        return TierDecision(label=label, tier=POSSIBLY, votes=votes, rank=best_rank,
                            signals=signals, margin=best_margin,
                            detail=f"one signal held ({which})")

    log.debug("tier unidentified: no signal held (rank=%d votes=%d)", best_rank, votes)
    return TierDecision(label=describe(index, size, color, anchor), tier=UNIDENTIFIED,
                        votes=votes, rank=best_rank, signals=signals, margin=best_margin,
                        detail="no signal held")


def tier_of(label: str) -> str:
    """Which tier a printed label belongs to. Used by the evaluation and by the
    test that checks tier ordering in the committed outputs."""
    if label.startswith("possibly "):
        return POSSIBLY
    if label.startswith("unidentified "):
        return UNIDENTIFIED
    return PLAIN
