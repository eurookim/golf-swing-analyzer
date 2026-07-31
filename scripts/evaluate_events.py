#!/usr/bin/env python3
"""Score the event detector against hand-verified ground truth.

Only counts labels whose ``verified`` flag is true. Starter labels are seeded
from the detector's own output, so scoring against those would report zero error
and prove nothing.

    .venv/bin/python evaluate_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from golfswing import paths

from golfswing import events, labels, store

PROCESSED = paths.PROCESSED_DIR


def main() -> int:
    npz_files = sorted(PROCESSED.glob("*.npz"))
    if not npz_files:
        print("No cached swings. Run: .venv/bin/python -m golfswing")
        return 1

    per_clip: dict[str, dict[str, int]] = {}
    unverified: list[str] = []
    unlabeled: list[str] = []

    for path in npz_files:
        stem = path.stem
        truth = labels.load_labels(stem, require_verified=True)
        if truth is None:
            (unverified if labels.load_labels(stem) else unlabeled).append(stem)
            continue
        try:
            detected = events.detect_events(store.load_sequence(path))
        except events.NoSwingDetectedError as exc:
            print(f"  ✗ {stem}: {exc}")
            continue
        per_clip[stem] = labels.frame_errors(detected, truth)

    if unlabeled:
        print(f"Unlabeled ({len(unlabeled)}): run label_swing.py")
        for stem in unlabeled:
            print(f"   - {stem}")
    if unverified:
        print(f"\nNot yet verified ({len(unverified)}) — NOT scored:")
        for stem in unverified:
            print(f"   - {stem}")
        print('\n   Check outputs/<clip>_labelstrip.jpg, correct the frame numbers')
        print('   in data/labels/<clip>.json, then set "verified": true.')

    if not per_clip:
        print("\nNothing verified yet, so there is nothing to score.")
        return 1

    print(f"\nPer-clip frame error (negative = detected early):\n")
    header = f"  {'clip':<32}" + "".join(f"{k:>7}" for k in ("P1", "P4", "P7", "P10"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stem, errors in per_clip.items():
        row = "".join(f"{errors[k]:>7}" for k in ("P1", "P4", "P7", "P10"))
        print(f"  {stem:<32}{row}")

    summary = labels.summarise_errors(per_clip)
    print(f"\nSummary over {len(per_clip)} clip(s):\n")
    print(f"  {'event':<8}{'mean':>9}{'mean_abs':>11}{'max_abs':>10}")
    print("  " + "-" * 36)
    for name in ("P1", "P4", "P7", "P10"):
        s = summary[name]
        print(f"  {name:<8}{s['mean']:>9.1f}{s['mean_abs']:>11.1f}{s['max_abs']:>10.0f}")

    print("\n  mean_abs is the headline — signed means cancel out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
