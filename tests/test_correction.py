"""Tests for golfswing.correction — a hand-picked frame becomes metrics + truth."""

import json

import numpy as np
import pytest

from golfswing import correction, db, store
from golfswing.events import SwingEvents
from golfswing.sequence import PoseSequence


def _sequence(n_frames=200):
    # (n, 33, 4): landmarks carry visibility as their fourth channel, and
    # PoseSequence rejects anything narrower.
    rng = np.random.default_rng(0)
    return PoseSequence(
        landmarks=rng.random((n_frames, 33, 4)),
        times=np.arange(n_frames) / 120.0,
        fps=120.0,
        source="test.mov",
    )


@pytest.fixture
def workspace(tmp_path):
    processed = tmp_path / "processed"
    labels = tmp_path / "labels"
    store.save_sequence(processed / "c1.npz", _sequence())
    conn = db.connect(tmp_path / "s.sqlite")
    conn.execute(
        "INSERT INTO swings (clip, date, club, fault_tag) "
        "VALUES ('c1', '2026-07-29', '7iron', 'early_extension')"
    )
    conn.execute("UPDATE swings SET outcome = 'flushed' WHERE clip = 'c1'")
    conn.commit()
    return conn, processed, labels


def test_recomputes_metrics_from_the_corrected_events(workspace):
    conn, processed, labels = workspace
    events = SwingEvents(p1=10, p4=100, p7=140, p10=180)

    result = correction.apply_correction(
        conn, "c1", events, processed_dir=processed, labels_dir=labels
    )

    assert result.events == events
    row = conn.execute(
        "SELECT p7, events_source FROM swings WHERE clip='c1'").fetchone()
    assert row["p7"] == 140
    assert row["events_source"] == "corrected"


def test_is_deterministic(workspace):
    conn, processed, labels = workspace
    events = SwingEvents(p1=10, p4=100, p7=140, p10=180)

    first = correction.apply_correction(
        conn, "c1", events, processed_dir=processed, labels_dir=labels
    )
    second = correction.apply_correction(
        conn, "c1", events, processed_dir=processed, labels_dir=labels
    )

    assert first.as_dict() == second.as_dict()


def test_writes_a_verified_label(workspace):
    conn, processed, labels = workspace
    events = SwingEvents(p1=10, p4=100, p7=140, p10=180)

    correction.apply_correction(
        conn, "c1", events, processed_dir=processed, labels_dir=labels
    )

    payload = json.loads((labels / "c1.json").read_text())
    assert payload["p7"] == 140
    assert payload["verified"] is True, "a human looked at the frame — that is the point"


def test_preserves_the_columns_it_does_not_own(workspace):
    conn, processed, labels = workspace

    correction.apply_correction(
        conn, "c1", SwingEvents(10, 100, 140, 180),
        processed_dir=processed, labels_dir=labels,
    )

    row = conn.execute("SELECT * FROM swings WHERE clip='c1'").fetchone()
    assert row["outcome"] == "flushed"
    assert row["fault_tag"] == "early_extension"
    assert row["club"] == "7iron"


def test_rejects_out_of_order_events(workspace):
    conn, processed, labels = workspace

    with pytest.raises(correction.OutOfOrderError):
        correction.apply_correction(
            conn, "c1", SwingEvents(p1=10, p4=150, p7=140, p10=180),
            processed_dir=processed, labels_dir=labels,
        )


def test_writes_nothing_when_rejected(workspace):
    """No partial state: the label must not move while the metrics stay put."""
    conn, processed, labels = workspace
    before = conn.execute("SELECT p7 FROM swings WHERE clip='c1'").fetchone()["p7"]

    with pytest.raises(correction.OutOfOrderError):
        correction.apply_correction(
            conn, "c1", SwingEvents(p1=10, p4=150, p7=140, p10=180),
            processed_dir=processed, labels_dir=labels,
        )

    assert conn.execute(
        "SELECT p7 FROM swings WHERE clip='c1'").fetchone()["p7"] == before
    assert not (labels / "c1.json").exists()


def test_confirming_the_detectors_own_frame_still_counts_as_corrected(workspace):
    """A human looked and agreed. That is worth more than the detector's guess,
    and evaluate_events.py only scores labels a human verified."""
    conn, processed, labels = workspace
    conn.execute("UPDATE swings SET p7 = 140 WHERE clip = 'c1'")
    conn.commit()

    correction.apply_correction(
        conn, "c1", SwingEvents(10, 100, 140, 180),
        processed_dir=processed, labels_dir=labels,
    )

    row = conn.execute(
        "SELECT events_source FROM swings WHERE clip='c1'").fetchone()
    assert row["events_source"] == "corrected"
    assert json.loads((labels / "c1.json").read_text())["verified"] is True


def test_missing_landmarks_file_raises(workspace):
    conn, processed, labels = workspace

    with pytest.raises(FileNotFoundError):
        correction.apply_correction(
            conn, "nope", SwingEvents(10, 100, 140, 180),
            processed_dir=processed, labels_dir=labels,
        )


class TestCorrected:
    def test_moves_one_event_and_leaves_the_rest(self):
        moved = correction.corrected(SwingEvents(1, 2, 3, 4), "p7", 30)

        assert (moved.p1, moved.p4, moved.p7, moved.p10) == (1, 2, 30, 4)

    def test_does_not_mutate_the_original(self):
        """SwingEvents is frozen; a correction copies rather than writes."""
        events = SwingEvents(1, 2, 3, 4)

        correction.corrected(events, "p7", 30)

        assert events.p7 == 3

    def test_rejects_an_unknown_event(self):
        with pytest.raises(KeyError):
            correction.corrected(SwingEvents(1, 2, 3, 4), "p9", 30)
