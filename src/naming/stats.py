"""Small-sample statistics, in one place so it cannot drift again.

The Wilson score interval is the only interval this project uses for
proportions, and it has now been wrong twice in reporting code (see
docs/debugging.md): a local copy returned the centre as the lower bound. Every
caller imports from here instead of re-deriving the formula.

Definition used (z = 1.96 for 95%):

    centre = (p + z^2 / 2n) / (1 + z^2 / n)
    half   = z * sqrt(p(1-p)/n + z^2 / 4n^2) / (1 + z^2 / n)

with p = k / n. The interval always contains the point estimate p, and for
k = 0 or k = n it stays inside [0, 100] without special cases.
"""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k successes out of n, in percent.

    Returns (lower, upper). For n == 0 returns (0.0, 0.0) by convention, which
    callers must treat as "no data", never as a measurement of zero.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # The interval is provably inside [0, 100]; the clamp removes float noise
    # only (0/20 otherwise reports -1.4e-15 as its lower bound). It never
    # touches the proportion itself, which is what went wrong before.
    lower = max(0.0, min(100.0, 100 * (centre - half)))
    upper = max(0.0, min(100.0, 100 * (centre + half)))
    return (lower, upper)


def wilson_pair(k: int, n: int) -> tuple:
    """(point estimate, lower, upper) in percent, the shape the reports use."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    lo, hi = wilson(k, n)
    return (100 * k / n, lo, hi)
