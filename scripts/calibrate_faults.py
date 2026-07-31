#!/usr/bin/env python3
"""Measure fault thresholds against deliberately-tagged clips.

Every number in thresholds.yaml is invented. This replaces them with measured
ones, using the fault tags in filenames as ground truth:

    2026-08-02_dtl_7iron_03.mov            -> a normal swing
    2026-08-02_dtl_7iron_11_posture.mov    -> a known loss-of-posture

For each rule it shows where your normal swings sit, where the deliberate fault
sits, whether the current threshold separates them, and what threshold would.

    .venv/bin/python calibrate_faults.py
    .venv/bin/python calibrate_faults.py --club 7iron
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from golfswing import calibrate, faults, labels, metrics, store

PROCESSED = Path("data/processed")


def _club_of(stem: str) -> str | None:
    match = re.search(r"^\d{4}-\d{2}-\d{2}_[a-z]+_([a-z0-9]+)_", stem)
    return match.group(1) if match else None


def _fmt(values: list[float]) -> str:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return "—"
    if len(finite) == 1:
        return f"{finite[0]:.3g}"
    return f"{min(finite):.3g} … {max(finite):.3g}  (n={len(finite)})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--club", help="only use clips of this club")
    ap.add_argument("--thresholds", type=Path, default=None)
    args = ap.parse_args()

    thresholds = faults.load_thresholds(args.thresholds) if args.thresholds \
        else faults.load_thresholds()

    # metric value per rule, split by whether the clip is a known-positive
    normal: dict[str, list[float]] = {r.name: [] for r in faults.RULES}
    tagged: dict[str, list[float]] = {r.name: [] for r in faults.RULES}
    used, skipped = 0, 0

    for path in sorted(PROCESSED.glob("*.npz")):
        events = labels.load_labels(path.stem, require_verified=True)
        if events is None:
            skipped += 1
            continue
        if args.club and _club_of(path.stem) != args.club:
            continue
        try:
            tag = calibrate.fault_tag(path.stem)
        except ValueError as exc:
            print(f"  ✗ {exc}", file=sys.stderr)
            return 1

        m = metrics.compute(store.load_sequence(path), events)
        used += 1
        for rule in faults.RULES:
            value = float(getattr(m, rule.metric))
            (tagged if tag == rule.name else normal)[rule.name].append(value)

    if used == 0:
        print("No clips with verified labels. Run label_swing.py first.")
        return 1

    print(f"Calibrating against {used} verified clip(s)"
          + (f", club={args.club}" if args.club else "")
          + (f"  ({skipped} unverified, skipped)" if skipped else ""))

    limits = faults._thresholds_for(thresholds, args.club)
    ready = 0

    for rule in faults.RULES:
        good, bad = normal[rule.name], tagged[rule.name]
        current = limits.get(rule.name, float("nan"))
        print(f"\n{rule.name}   [{rule.confidence} confidence]")
        print(f"   normal swings   {_fmt(good)}")
        print(f"   tagged faults   {_fmt(bad)}")
        print(f"   current limit   {current:.4g}")

        if not [v for v in bad if np.isfinite(v)]:
            print("   ⚠ no known-positive — threshold is unfalsifiable.")
            print(f"     Film a clip tagged with this fault, exaggerated.")
            continue

        now = calibrate.score(good, bad, current, rule.comparison)
        suggested = calibrate.suggest(good, bad, rule.comparison)
        then = calibrate.score(good, bad, suggested, rule.comparison)

        print(f"   at current      {now.true_positive} caught, "
              f"{now.false_negative} missed, {now.false_positive} false alarms")
        print(f"   suggested       {suggested:.4g}  -> "
              f"{then.true_positive} caught, {then.false_negative} missed, "
              f"{then.false_positive} false alarms")
        if then.errors == 0:
            print("   ✓ cleanly separates normal from deliberate")
            ready += 1
        else:
            print(f"   ⚠ best possible still misclassifies {then.errors} — the "
                  "fault may not have been exaggerated enough, or this metric "
                  "does not capture it")

    print(f"\n{ready}/{len(faults.RULES)} rules calibratable from this set.")
    print("Copy the suggested values into thresholds.yaml when you are happy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
