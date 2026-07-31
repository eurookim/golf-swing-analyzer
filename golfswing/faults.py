"""Fault rules over metrics, ranked by severity.

Two design rules from PLAN.md drive the whole module:

**Prioritise, never enumerate.** Six faults on screen helps nobody — an amateur
can only work on one thing at a time, and picking the wrong one wastes a month.
So every result carries a severity and the list is sorted, with `primary()`
returning the single fault worth acting on.

**Honest uncertainty.** A metric that could not be computed is reported as
*unmeasurable*, never as "no fault." Silence and absence-of-evidence are
different answers, and conflating them is how a tool loses trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from golfswing import paths
from typing import Literal

import numpy as np
import yaml

from golfswing.metrics import SwingMetrics

DEFAULT_THRESHOLDS_PATH = paths.THRESHOLDS

Comparison = Literal["above", "below", "magnitude"]
Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class FaultRule:
    """One named fault: which metric, which direction, how much to trust it."""

    name: str
    metric: str
    comparison: Comparison
    confidence: Confidence
    description: str


# Confidence reflects what 60fps footage actually supports (see PLAN.md):
# address- and top-anchored metrics are solid, impact-anchored ones degrade,
# and tempo compounds errors at both ends.
RULES: tuple[FaultRule, ...] = (
    FaultRule("loss_of_posture", "posture_change", "magnitude", "high",
              "Spine angle at the top differs from address"),
    FaultRule("head_lift", "head_rise_p4", "above", "high",
              "Head rises during the backswing"),
    FaultRule("early_extension", "hip_depth_change", "above", "medium",
              "Hips move toward the ball through impact"),
    FaultRule("knee_straightening", "knee_extension_change", "above", "medium",
              "Trail leg straightens through impact"),
    FaultRule("quick_tempo", "tempo_ratio", "below", "low",
              "Backswing is fast relative to the downswing"),
)


@dataclass(frozen=True)
class Fault:
    """The outcome of one rule against one swing."""

    name: str
    fired: bool
    measurable: bool
    value: float
    threshold: float
    severity: float
    confidence: Confidence
    description: str


def _thresholds_for(thresholds: dict, club: str | None) -> dict:
    """Club section merged over the defaults.

    A driver's extra spine tilt is correct technique, so scoring it against iron
    thresholds would confidently report a fault for a good position.
    """
    merged = dict(thresholds.get("default", {}))
    if club:
        merged.update(thresholds.get(club, {}))
    return merged


def _fires(value: float, limit: float, comparison: Comparison) -> bool:
    if comparison == "above":
        return value > limit
    if comparison == "below":
        return value < limit
    return abs(value) > limit


def _severity(value: float, limit: float, comparison: Comparison) -> float:
    """Fraction over the limit, so unrelated units rank against each other.

    Degrees and torso-length ratios are not comparable as raw numbers, but
    "50% past its own threshold" means the same thing for both.
    """
    if limit == 0:
        return 0.0
    if comparison == "below":
        return max(0.0, (limit - value) / abs(limit))
    magnitude = abs(value) if comparison == "magnitude" else value
    return max(0.0, (magnitude - limit) / abs(limit))


def evaluate(
    metrics: SwingMetrics,
    thresholds: dict,
    club: str | None = None,
) -> list[Fault]:
    """Run every rule against one swing, worst first.

    Unmeasurable rules sort last regardless of anything else — they carry no
    severity, and letting one head the list would present a non-finding as the
    thing to work on.
    """
    limits = _thresholds_for(thresholds, club)
    results: list[Fault] = []

    for rule in RULES:
        value = float(getattr(metrics, rule.metric, float("nan")))
        limit = limits.get(rule.name, float("nan"))
        measurable = bool(np.isfinite(value) and np.isfinite(limit))

        fired = measurable and _fires(value, limit, rule.comparison)
        severity = _severity(value, limit, rule.comparison) if fired else 0.0

        results.append(Fault(
            name=rule.name,
            fired=fired,
            measurable=measurable,
            value=value,
            threshold=float(limit),
            severity=float(severity),
            confidence=rule.confidence,
            description=rule.description,
        ))

    results.sort(key=lambda f: (f.measurable, f.severity), reverse=True)
    return results


def find(results: list[Fault], name: str) -> Fault:
    """Look up one rule's outcome by name."""
    for fault in results:
        if fault.name == name:
            return fault
    raise KeyError(f"no such fault: {name}")


def primary(results: list[Fault]) -> Fault | None:
    """The single fault worth working on, or None if the swing is clean.

    Deliberately one value rather than a list — see the module docstring.
    """
    fired = [f for f in results if f.fired and f.measurable]
    return max(fired, key=lambda f: f.severity) if fired else None


def load_thresholds(path: Path | str = DEFAULT_THRESHOLDS_PATH) -> dict:
    """Read thresholds from YAML.

    Requires a `default` section: without it an unlisted club would silently
    score nothing at all, which reads identically to a clean swing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text()) or {}
    if "default" not in data:
        raise ValueError(f"{path} has no 'default' section")
    return data
