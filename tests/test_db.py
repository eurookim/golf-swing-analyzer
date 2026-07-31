"""Tests for golfswing.db — swing history."""

import numpy as np
import pytest

from golfswing import db
from golfswing.events import SwingEvents
from golfswing.metrics import SwingMetrics

EVENTS = SwingEvents(p1=10, p4=100, p7=140, p10=180)


def _metrics(**overrides) -> SwingMetrics:
    base = dict(
        spine_tilt_p1=39.0, spine_tilt_p4=39.5, spine_tilt_p7=35.0,
        posture_change=0.5, hip_depth_change=0.20,
        head_rise_p4=0.01, head_rise_p7=-0.03, head_depth_p7=0.05,
        knee_extension_change=20.0, tempo_ratio=3.1,
    )
    base.update(overrides)
    return SwingMetrics(events=EVENTS, **base)


def _record(clip="2026-07-29_dtl_7iron_01", **overrides):
    fields = dict(
        clip=clip, date="2026-07-29", club="7iron", angle="dtl",
        fps=120.0, metrics=_metrics(), fault_tag=None,
    )
    fields.update(overrides)
    return fields


class TestSchema:
    def test_creates_the_database_file(self, tmp_path):
        path = tmp_path / "swings.db"
        db.connect(path).close()
        assert path.exists()

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "swings.db"
        db.connect(path).close()
        assert path.exists()

    def test_is_safe_to_open_twice(self, tmp_path):
        path = tmp_path / "swings.db"
        db.connect(path).close()
        db.connect(path).close()   # must not fail on existing tables


class TestSaveAndLoad:
    def test_round_trips_a_swing(self, tmp_path):
        conn = db.connect(tmp_path / "s.db")
        db.save_swing(conn, **_record())

        rows = db.load_swings(conn)

        assert len(rows) == 1
        assert rows[0]["clip"] == "2026-07-29_dtl_7iron_01"

    def test_preserves_metric_values(self, tmp_path):
        conn = db.connect(tmp_path / "s.db")
        db.save_swing(conn, **_record(metrics=_metrics(tempo_ratio=4.25)))

        assert db.load_swings(conn)[0]["tempo_ratio"] == pytest.approx(4.25)

    def test_preserves_event_frames(self, tmp_path):
        conn = db.connect(tmp_path / "s.db")
        db.save_swing(conn, **_record())

        row = db.load_swings(conn)[0]
        assert (row["p1"], row["p4"], row["p7"], row["p10"]) == (10, 100, 140, 180)

    def test_nan_metrics_round_trip_as_none(self, tmp_path):
        """SQLite has no NaN. Storing it as NULL keeps 'unmeasured' distinct
        from a real value, which is the whole point of the NaN convention."""
        conn = db.connect(tmp_path / "s.db")
        db.save_swing(conn, **_record(metrics=_metrics(tempo_ratio=float("nan"))))

        assert db.load_swings(conn)[0]["tempo_ratio"] is None

    def test_saving_the_same_clip_twice_updates_rather_than_duplicates(self, tmp_path):
        """Re-processing a clip must not create a second history entry."""
        conn = db.connect(tmp_path / "s.db")
        db.save_swing(conn, **_record(metrics=_metrics(tempo_ratio=3.0)))
        db.save_swing(conn, **_record(metrics=_metrics(tempo_ratio=9.9)))

        rows = db.load_swings(conn)
        assert len(rows) == 1
        assert rows[0]["tempo_ratio"] == pytest.approx(9.9)

    def test_stores_the_fault_tag(self, tmp_path):
        conn = db.connect(tmp_path / "s.db")
        db.save_swing(conn, **_record(clip="x_10_posture", fault_tag="loss_of_posture"))

        assert db.load_swings(conn)[0]["fault_tag"] == "loss_of_posture"


class TestQuerying:
    def _seeded(self, tmp_path):
        conn = db.connect(tmp_path / "s.db")
        db.save_swing(conn, **_record(clip="a", club="7iron", date="2026-07-01"))
        db.save_swing(conn, **_record(clip="b", club="driver", date="2026-07-15"))
        db.save_swing(conn, **_record(clip="c", club="7iron", date="2026-07-29"))
        return conn

    def test_filters_by_club(self, tmp_path):
        rows = db.load_swings(self._seeded(tmp_path), club="7iron")
        assert {r["clip"] for r in rows} == {"a", "c"}

    def test_orders_by_date_for_trends(self, tmp_path):
        rows = db.load_swings(self._seeded(tmp_path))
        assert [r["date"] for r in rows] == sorted(r["date"] for r in rows)

    def test_excludes_tagged_faults_when_asked(self, tmp_path):
        """Deliberate fault swings must not pollute a progress trend."""
        conn = self._seeded(tmp_path)
        db.save_swing(conn, **_record(clip="d", club="7iron",
                                      fault_tag="early_extension"))

        rows = db.load_swings(conn, club="7iron", exclude_tagged=True)

        assert {r["clip"] for r in rows} == {"a", "c"}

    def test_empty_database_returns_no_rows(self, tmp_path):
        assert db.load_swings(db.connect(tmp_path / "s.db")) == []


class TestThreadSafety:
    def test_a_connection_works_from_another_thread(self, tmp_path):
        """Streamlit runs each script run on a script-runner thread, and the
        cached connection outlives any one of them. SQLite rejects cross-thread
        use by default, which surfaced as a ProgrammingError traceback filling
        the whole page the moment a rerun landed on a different thread."""
        import threading

        conn = db.connect(tmp_path / "s.db")
        db.save_swing(conn, **_record())

        result, error = [], []

        def read():
            try:
                result.extend(db.load_swings(conn))
            except Exception as exc:            # noqa: BLE001 - recording it
                error.append(exc)

        thread = threading.Thread(target=read)
        thread.start()
        thread.join()

        assert not error, f"cross-thread use raised {error}"
        assert len(result) == 1
