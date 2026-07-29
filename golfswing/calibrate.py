"""Turn tagged clips into measured thresholds.

Every number in `thresholds.yaml` is invented. This is what replaces them with
measured ones, and it is why the capture brief insists on deliberately
exaggerated fault swings: **a threshold with no known-positive is
unfalsifiable.** If the early-extension rule never sees a swing where the golfer
deliberately thrust their hips at the ball, no amount of normal-swing data says
where the line goes.

The filename tag carries the expected answer (`..._11_posture.mov`), so parsing
it is load-bearing rather than cosmetic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from golfswing.faults import Comparison

# Filename shorthand -> rule name in faults.RULES
TAG_TO_RULE: dict[str, str] = {
    "posture": "loss_of_posture",
    "headlift": "head_lift",
    "earlyext": "early_extension",
    "kneestraight": "knee_straightening",
    "quicktempo": "quick_tempo",
}

# <date>_<angle>_<club>_<nn>[_<tag>]
_STEM = re.compile(r"^\d{4}-\d{2}-\d{2}_[a-z]+_[a-z0-9]+_\d+(?:_([a-z]+))?$")


def fault_tag(stem: str) -> str | None:
    """Rule name this clip is a known-positive for, or None if it is normal.

    Raises on an unrecognised tag rather than treating it as a normal swing —
    a typo would otherwise land a fault clip in the normal group and quietly
    push the threshold the wrong way.
    """
    match = _STEM.match(stem)
    if match is None:
        return None
    tag = match.group(1)
    if tag is None:
        return None
    if tag not in TAG_TO_RULE:
        raise ValueError(
            f"unknown fault tag {tag!r} in {stem!r}; "
            f"known tags: {', '.join(sorted(TAG_TO_RULE))}"
        )
    return TAG_TO_RULE[tag]


@dataclass(frozen=True)
class Score:
    """How a candidate threshold performs against tagged ground truth."""

    true_positive: int    # fault clips correctly flagged
    false_negative: int   # fault clips missed
    true_negative: int    # normal clips correctly left alone
    false_positive: int   # normal clips wrongly flagged

    @property
    def errors(self) -> int:
        return self.false_negative + self.false_positive


def _fires(value: float, threshold: float, comparison: Comparison) -> bool:
    if comparison == "above":
        return value > threshold
    if comparison == "below":
        return value < threshold
    return abs(value) > threshold


def _clean(values: Sequence[float], comparison: Comparison) -> np.ndarray:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return np.abs(array) if comparison == "magnitude" else array


def score(
    normal: Sequence[float],
    fault: Sequence[float],
    threshold: float,
    comparison: Comparison,
) -> Score:
    """Confusion counts for one threshold."""
    fault_fires = [_fires(v, threshold, comparison) for v in fault if np.isfinite(v)]
    normal_fires = [_fires(v, threshold, comparison) for v in normal if np.isfinite(v)]
    return Score(
        true_positive=sum(fault_fires),
        false_negative=len(fault_fires) - sum(fault_fires),
        true_negative=len(normal_fires) - sum(normal_fires),
        false_positive=sum(normal_fires),
    )


def suggest(
    normal: Sequence[float],
    fault: Sequence[float],
    comparison: Comparison,
) -> float:
    """A threshold separating normal swings from deliberate faults.

    Returns NaN unless both groups have examples: with no known-positive there
    is nothing to calibrate against, and with no normal swings there is no
    baseline to separate from.

    When the groups are cleanly separated, this is the midpoint of the gap —
    maximally far from both. When they overlap, it is the candidate with the
    fewest misclassifications, breaking ties toward the larger margin.
    """
    good = _clean(normal, comparison)
    bad = _clean(fault, comparison)
    if good.size == 0 or bad.size == 0:
        return float("nan")

    if comparison == "below":
        gap_low, gap_high = bad.max(), good.min()
    else:
        gap_low, gap_high = good.max(), bad.min()

    if gap_low < gap_high:
        return float((gap_low + gap_high) / 2.0)

    # Overlapping: search midpoints between adjacent observed values.
    observed = np.unique(np.concatenate([good, bad]))
    candidates = np.concatenate([
        observed - 1e-9,
        (observed[:-1] + observed[1:]) / 2.0 if observed.size > 1 else [],
        observed + 1e-9,
    ])
    best, best_key = float("nan"), None
    for candidate in candidates:
        result = score(good, bad, float(candidate), comparison)
        margin = min(abs(float(candidate) - v) for v in observed)
        key = (result.errors, -margin)
        if best_key is None or key < best_key:
            best, best_key = float(candidate), key
    return best
