"""Tests for golfswing.labels — hand-labeled ground truth and detector scoring."""

import json

import pytest

from golfswing import labels
from golfswing.events import SwingEvents


def _events(p1=30, p4=60, p7=80, p10=110):
    return SwingEvents(p1=p1, p4=p4, p7=p7, p10=p10)


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        labels.save_labels("clip_01", _events(), labels_dir=tmp_path)
        assert labels.load_labels("clip_01", labels_dir=tmp_path) == _events()

    def test_writes_readable_json(self, tmp_path):
        """Labels are hand-edited, so the file has to be human-readable."""
        labels.save_labels("clip_01", _events(p4=61), labels_dir=tmp_path)

        data = json.loads((tmp_path / "clip_01.json").read_text())

        assert data["p4"] == 61

    def test_missing_labels_return_none(self, tmp_path):
        assert labels.load_labels("never_labeled", labels_dir=tmp_path) is None

    def test_creates_directory(self, tmp_path):
        target = tmp_path / "nested" / "labels"
        labels.save_labels("clip_01", _events(), labels_dir=target)
        assert (target / "clip_01.json").exists()

    def test_rejects_out_of_order_events(self, tmp_path):
        """P4 after P7 is a labeling mistake, not a valid swing."""
        with pytest.raises(ValueError):
            labels.save_labels(
                "bad", SwingEvents(p1=10, p4=90, p7=80, p10=110), labels_dir=tmp_path
            )


class TestVerification:
    """Starter labels are seeded FROM the detector, so scoring against an
    unverified file reports zero error and proves nothing. Labels only count
    once a human has actually looked at the frames."""

    def test_labels_are_unverified_by_default(self, tmp_path):
        labels.save_labels("clip_01", _events(), labels_dir=tmp_path)
        assert labels.is_verified("clip_01", labels_dir=tmp_path) is False

    def test_can_be_marked_verified(self, tmp_path):
        labels.save_labels("clip_01", _events(), labels_dir=tmp_path, verified=True)
        assert labels.is_verified("clip_01", labels_dir=tmp_path) is True

    def test_verified_flag_survives_round_trip_in_json(self, tmp_path):
        labels.save_labels("clip_01", _events(), labels_dir=tmp_path, verified=True)
        data = json.loads((tmp_path / "clip_01.json").read_text())
        assert data["verified"] is True

    def test_require_verified_hides_unverified_labels(self, tmp_path):
        labels.save_labels("clip_01", _events(), labels_dir=tmp_path)
        assert labels.load_labels(
            "clip_01", labels_dir=tmp_path, require_verified=True
        ) is None

    def test_require_verified_returns_verified_labels(self, tmp_path):
        labels.save_labels("clip_01", _events(), labels_dir=tmp_path, verified=True)
        assert labels.load_labels(
            "clip_01", labels_dir=tmp_path, require_verified=True
        ) == _events()

    def test_missing_file_is_not_verified(self, tmp_path):
        assert labels.is_verified("nope", labels_dir=tmp_path) is False


class TestFrameErrors:
    def test_zero_when_detection_matches_labels(self):
        assert labels.frame_errors(_events(), _events()) == {
            "P1": 0, "P4": 0, "P7": 0, "P10": 0
        }

    def test_signed_so_early_and_late_are_distinguishable(self):
        """Sign matters — a detector consistently 5 frames early is a fixable
        bias, whereas errors scattered either way are noise."""
        detected = _events(p1=25, p4=65)
        errors = labels.frame_errors(detected, _events(p1=30, p4=60))

        assert errors["P1"] == -5
        assert errors["P4"] == 5

    def test_summarises_across_clips(self):
        per_clip = {
            "a": {"P1": -5, "P4": 2, "P7": 0, "P10": 3},
            "b": {"P1": -3, "P4": -2, "P7": 1, "P10": -1},
        }

        summary = labels.summarise_errors(per_clip)

        assert summary["P1"]["mean"] == pytest.approx(-4.0)
        assert summary["P1"]["mean_abs"] == pytest.approx(4.0)
        assert summary["P7"]["max_abs"] == 1

    def test_mean_abs_is_the_honest_headline(self):
        """Mean signed error cancels out; mean absolute error does not.

        A detector 10 frames early on one clip and 10 late on another has a mean
        of 0 and is not accurate.
        """
        per_clip = {"a": {"P4": -10}, "b": {"P4": 10}}

        summary = labels.summarise_errors(per_clip)

        assert summary["P4"]["mean"] == pytest.approx(0.0)
        assert summary["P4"]["mean_abs"] == pytest.approx(10.0)
