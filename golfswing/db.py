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
    p1 INTEGER, p4 INTEGER, p7 INTEGER, p10 INTEGER,
    {', '.join(f'{name} REAL' for name in METRIC_COLUMNS)}
);
CREATE INDEX IF NOT EXISTS swings_by_club_date ON swings (club, date);
"""


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the history database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


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
    conn.execute(
        f"INSERT OR REPLACE INTO swings ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' * len(columns))})",
        row,
    )
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
