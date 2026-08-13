"""Turn a hand-picked frame into corrected metrics and verified ground truth.

The detector is right most of the time and wrong quietly: a misplaced impact
frame yields a plausible number rather than an obvious failure. Correcting one
used to mean leaving the app — run `scripts/label_swing.py`, hand-edit the JSON,
re-run the pipeline. This is the path that does it in place.

Because the expensive part is deciding which frame is right, and that decision
is being made by hand anyway, it is recorded as ground truth at the same time.
Labelling stops being a chore and becomes a side effect of noticing.

Nothing here re-runs pose estimation. The landmarks are already cached, so a
correction is a file read, an arithmetic pass, and two writes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from golfswing import db, labels, metrics, store
from golfswing.events import SwingEvents
from golfswing.metrics import SwingMetrics
from golfswing.paths import LABELS_DIR, PROCESSED_DIR

EVENT_KEYS = ("p1", "p4", "p7", "p10")


class OutOfOrderError(ValueError):
    """The corrected frames do not run address → top → impact → finish."""


def corrected(events: SwingEvents, key: str, frame: int) -> SwingEvents:
    """A copy of ``events`` with one event moved. SwingEvents is frozen."""
    if key not in EVENT_KEYS:
        raise KeyError(f"unknown event {key!r} — expected one of {EVENT_KEYS}")
    return replace(events, **{key: int(frame)})


def check_order(events: SwingEvents) -> None:
    """Raise unless the frames run address → top → impact → finish.

    Public because the UI previews validity to decide whether to offer the
    confirm button, rather than letting the user press it and read a traceback.
    """
    frames = [getattr(events, key) for key in EVENT_KEYS]
    if not all(a < b for a, b in zip(frames, frames[1:])):
        raise OutOfOrderError(
            "events must run address → top → impact → finish, got "
            f"{dict(zip(EVENT_KEYS, frames))}"
        )


def apply_correction(
    conn: sqlite3.Connection,
    clip_stem: str,
    events: SwingEvents,
    *,
    processed_dir: Path | str = PROCESSED_DIR,
    labels_dir: Path | str = LABELS_DIR,
) -> SwingMetrics:
    """Recompute and persist one swing from hand-corrected event frames.

    Order matters. Validation runs before anything is written, and the label is
    written last: a rejected correction must leave the database and the label
    file exactly as they were, rather than moving one and not the other.
    """
    check_order(events)

    sequence = store.load_sequence(Path(processed_dir) / f"{clip_stem}.npz")
    computed = metrics.compute(sequence, events)

    db.update_events(conn, clip_stem, events, computed)
    labels.save_labels(clip_stem, events, labels_dir=labels_dir, verified=True)
    return computed
