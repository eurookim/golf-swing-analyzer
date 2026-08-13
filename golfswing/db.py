"""Swing history.

One row per clip. Re-processing a clip updates its row rather than appending, so
history reflects swings taken rather than times the pipeline was run.

**NaN is stored as NULL.** SQLite has no NaN, and collapsing "not measured" into
a number would destroy the distinction the whole metrics layer maintains.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from golfswing.paths import DB_PATH
from typing import Any

from golfswing.events import SwingEvents
from golfswing.metrics import SwingMetrics

DEFAULT_DB_PATH = DB_PATH

METRIC_COLUMNS = (
    "spine_tilt_p1", "spine_tilt_p4", "spine_tilt_p7", "posture_change",
    "hip_depth_change", "head_rise_p4", "head_rise_p7", "head_depth_p7",
    "knee_extension_change", "tempo_ratio",
)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS swings (
    clip        TEXT PRIMARY KEY,
    date        TEXT NOT NULL,
    club        TEXT,
    angle       TEXT,
    fps         REAL,
    fault_tag   TEXT,
    outcome     TEXT,
    p1 INTEGER, p4 INTEGER, p7 INTEGER, p10 INTEGER,
    events_source TEXT NOT NULL DEFAULT 'detected',
    {', '.join(f'{name} REAL' for name in METRIC_COLUMNS)}
);
CREATE INDEX IF NOT EXISTS swings_by_club_date ON swings (club, date);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing database up to the current schema.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so
    a column added to _SCHEMA never reaches a database that predates it. Every
    real database here predates events_source.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(swings)")}
    if "events_source" not in columns:
        conn.execute(
            "ALTER TABLE swings "
            "ADD COLUMN events_source TEXT NOT NULL DEFAULT 'detected'"
        )


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the history database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because Streamlit caches this connection across
    # script runs, and each run happens on a script-runner thread that may not
    # be the one that opened it. Safe here: Streamlit serialises script runs, so
    # only one thread touches the connection at a time, and this is a
    # single-user local app with no concurrent writers.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


OUTCOMES = {"flushed", "mishit", "unsure"}


def set_outcome(conn: sqlite3.Connection, clip: str, outcome: str | None) -> None:
    """Record how the shot turned out, or clear it."""
    if outcome is not None and outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r} — expected {sorted(OUTCOMES)}")
    conn.execute("UPDATE swings SET outcome = ? WHERE clip = ?", (outcome, clip))
    conn.commit()


def _nullable(value: float) -> float | None:
    """NaN -> NULL, so 'unmeasured' survives the round trip."""
    return None if value is None or math.isnan(value) else float(value)


def save_swing(
    conn: sqlite3.Connection,
    clip: str,
    date: str,
    metrics: SwingMetrics,
    club: str | None = None,
    angle: str | None = None,
    fps: float | None = None,
    fault_tag: str | None = None,
) -> None:
    """Insert or replace one swing. Keyed on clip name."""
    values = metrics.as_dict()
    columns = ["clip", "date", "club", "angle", "fps", "fault_tag",
               "p1", "p4", "p7", "p10", *METRIC_COLUMNS]
    row = [
        clip, date, club, angle, fps, fault_tag,
        metrics.events.p1, metrics.events.p4, metrics.events.p7, metrics.events.p10,
        *[_nullable(values[name]) for name in METRIC_COLUMNS],
    ]
    # INSERT OR REPLACE rewrites the whole row, blanking any column not listed.
    # The outcome is typed by hand and cannot be recomputed from the video, so
    # carry it across a re-sync rather than silently discarding it.
    existing = conn.execute(
        "SELECT outcome FROM swings WHERE clip = ?", (clip,)).fetchone()
    conn.execute(
        f"INSERT OR REPLACE INTO swings ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' * len(columns))})",
        row,
    )
    if existing and existing["outcome"]:
        conn.execute("UPDATE swings SET outcome = ? WHERE clip = ?",
                     (existing["outcome"], clip))
    conn.commit()


def update_events(
    conn: sqlite3.Connection,
    clip: str,
    events: SwingEvents,
    metrics: SwingMetrics,
) -> None:
    """Replace one swing's event frames and the metrics derived from them.

    Deliberately not `save_swing`: that uses INSERT OR REPLACE, which rewrites
    the whole row and blanks anything not listed. A correction must leave
    `outcome`, `fault_tag`, `club` and `date` untouched — losing a fault tag
    would quietly pull a deliberately-botched swing into the baseline. An UPDATE
    cannot blank a column it does not name.
    """
    values = metrics.as_dict()
    assignments = ", ".join(f"{name} = ?" for name in METRIC_COLUMNS)
    cursor = conn.execute(
        f"UPDATE swings SET p1 = ?, p4 = ?, p7 = ?, p10 = ?, {assignments}, "
        "events_source = 'corrected' WHERE clip = ?",
        (
            events.p1, events.p4, events.p7, events.p10,
            *(_nullable(values[name]) for name in METRIC_COLUMNS),
            clip,
        ),
    )
    if cursor.rowcount == 0:
        raise KeyError(f"no swing named {clip!r}")
    conn.commit()


def load_swings(
    conn: sqlite3.Connection,
    club: str | None = None,
    exclude_tagged: bool = False,
) -> list[dict[str, Any]]:
    """Swings oldest first, so trend plots read left to right.

    ``exclude_tagged`` drops deliberate-fault clips. They are calibration data,
    not swings you took — leaving them in a progress trend would show phantom
    regressions on the days you filmed them.
    """
    where, params = [], []
    if club:
        where.append("club = ?")
        params.append(club)
    if exclude_tagged:
        where.append("fault_tag IS NULL")

    sql = "SELECT * FROM swings"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY date, clip"

    return [dict(row) for row in conn.execute(sql, params)]
