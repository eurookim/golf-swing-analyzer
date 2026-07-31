"""Hand-labeled ground truth, and scoring the detector against it.

Without ground truth every change to the detector is a guess. With it, every
change is measurable. Labels live in `data/labels/` and are tracked in git on
purpose — they are the most expensive artifact in the project to recreate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from golfswing.events import SwingEvents

from golfswing.paths import LABELS_DIR

DEFAULT_LABELS_DIR = LABELS_DIR
EVENT_KEYS = ("p1", "p4", "p7", "p10")


def save_labels(
    clip_stem: str,
    events: SwingEvents,
    labels_dir: Path | str = DEFAULT_LABELS_DIR,
    verified: bool = False,
) -> Path:
    """Write ground-truth frame numbers as readable JSON.

    ``verified`` defaults to False because starter labels are seeded from the
    detector's own output. Scoring against those would report zero error and
    prove nothing — they only become ground truth once a human has looked at
    the frames and set the flag.

    Rejects out-of-order events: a swing cannot reach the top after impact, so
    that is a labeling mistake worth catching at write time rather than
    discovering as a bizarre error metric later.
    """
    frames = [getattr(events, key) for key in EVENT_KEYS]
    if not all(a < b for a, b in zip(frames, frames[1:])):
        raise ValueError(
            f"events must be strictly increasing, got "
            f"{dict(zip(EVENT_KEYS, frames))}"
        )

    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    path = labels_dir / f"{clip_stem}.json"
    payload = dict(zip(EVENT_KEYS, frames))
    payload["verified"] = verified
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _read(clip_stem: str, labels_dir: Path | str) -> dict | None:
    path = Path(labels_dir) / f"{clip_stem}.json"
    return json.loads(path.read_text()) if path.exists() else None


def is_verified(
    clip_stem: str,
    labels_dir: Path | str = DEFAULT_LABELS_DIR,
) -> bool:
    """True only when a human has confirmed these frames by eye."""
    data = _read(clip_stem, labels_dir)
    return bool(data.get("verified", False)) if data else False


def load_labels(
    clip_stem: str,
    labels_dir: Path | str = DEFAULT_LABELS_DIR,
    require_verified: bool = False,
) -> SwingEvents | None:
    """Read ground truth for a clip, or None if unavailable.

    With ``require_verified``, unconfirmed starter labels are treated as absent
    so they cannot silently inflate a score.
    """
    data = _read(clip_stem, labels_dir)
    if data is None:
        return None
    if require_verified and not data.get("verified", False):
        return None
    return SwingEvents(**{key: int(data[key]) for key in EVENT_KEYS})


def frame_errors(detected: SwingEvents, labeled: SwingEvents) -> dict[str, int]:
    """Signed per-event error in frames: negative is early, positive is late.

    Signed rather than absolute because a detector consistently 5 frames early
    is a fixable bias, while errors scattered both ways are noise. Collapsing to
    absolute values at this stage would hide the difference.
    """
    return {
        key.upper(): getattr(detected, key) - getattr(labeled, key)
        for key in EVENT_KEYS
    }


def summarise_errors(
    per_clip: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-clip errors into per-event statistics.

    ``mean_abs`` is the honest headline: signed means cancel out, so a detector
    10 frames early on one clip and 10 late on another reports a mean of zero
    while being wrong on both.
    """
    events: list[str] = []
    for errors in per_clip.values():
        for name in errors:
            if name not in events:
                events.append(name)

    summary: dict[str, dict[str, float]] = {}
    for name in events:
        values = np.array(
            [errors[name] for errors in per_clip.values() if name in errors],
            dtype=float,
        )
        summary[name] = {
            "mean": float(values.mean()),
            "mean_abs": float(np.abs(values).mean()),
            "max_abs": float(np.abs(values).max()),
            "n": int(values.size),
        }
    return summary
