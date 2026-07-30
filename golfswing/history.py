"""Turn cached clips into history rows.

Filename is the source of date, angle, club, and fault tag — but a clip that
does not follow the convention is still worth analysing, just unattributed. An
unrecognised fault *tag* is different: it raises, because silently treating a
mislabeled fault clip as a normal swing corrupts calibration.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path

from golfswing import calibrate, db, events, labels, metrics, store

PROCESSED_DIR = Path("data/processed")

_STEM = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<angle>[a-z]+)_(?P<club>[a-z0-9]+)_\d+"
    r"(?:_(?P<tag>[a-z]+))?$"
)


@dataclass(frozen=True)
class ClipMeta:
    clip: str
    date: str
    angle: str | None
    club: str | None
    fault_tag: str | None


def parse_clip(stem: str) -> ClipMeta:
    """Pull date / angle / club / fault tag out of a clip name."""
    match = _STEM.match(stem)
    if match is None:
        # Unattributed rather than rejected — an oddly-named clip can still be
        # measured, it just cannot join a per-club trend.
        return ClipMeta(clip=stem, date=_date.today().isoformat(),
                        angle=None, club=None, fault_tag=None)
    return ClipMeta(
        clip=stem,
        date=match.group("date"),
        angle=match.group("angle"),
        club=match.group("club"),
        fault_tag=calibrate.fault_tag(stem),   # raises on an unknown tag
    )


def sync(
    conn: sqlite3.Connection,
    processed_dir: Path | str = PROCESSED_DIR,
    prefer_labels: bool = True,
) -> tuple[int, list[str]]:
    """Compute and store metrics for every cached clip.

    Uses hand-verified event labels where they exist, falling back to detection.
    Returns (rows written, list of clips that could not be analysed).
    """
    written, skipped = 0, []
    for path in sorted(Path(processed_dir).glob("*.npz")):
        meta = parse_clip(path.stem)
        sequence = store.load_sequence(path)

        found = labels.load_labels(path.stem, require_verified=True) if prefer_labels else None
        if found is None:
            try:
                found = events.detect_events(sequence)
            except events.NoSwingDetectedError as exc:
                skipped.append(f"{path.stem}: {exc}")
                continue

        db.save_swing(
            conn,
            clip=meta.clip,
            date=meta.date,
            metrics=metrics.compute(sequence, found),
            club=meta.club,
            angle=meta.angle,
            fps=sequence.fps,
            fault_tag=meta.fault_tag,
        )
        written += 1
    return written, skipped
