"""Tests for golfswing.skeleton — joint overlay coloured by deviation."""

import numpy as np
import pytest

from golfswing import skeleton
from golfswing.coach import Standing


def _standing(metric, verdict_value, median):
    return Standing(metric=metric, label=metric, short=metric, unit="",
                    meaning="", value=verdict_value, rank=1, n_peers=11,
                    median=median, baseline="flushed")


def _frame(h=400, w=300):
    return np.full((h, w, 3), 40, dtype=np.uint8)


def _landmarks():
    """A crude standing figure — enough for the drawing code to have geometry."""
    lm = np.zeros((33, 4))
    lm[:, 3] = 1.0
    for index, (x, y) in {
        0: (0.50, 0.15),                      # nose
        11: (0.42, 0.30), 12: (0.58, 0.30),   # shoulders
        23: (0.45, 0.55), 24: (0.55, 0.55),   # hips
        25: (0.44, 0.75), 26: (0.56, 0.75),   # knees
        27: (0.43, 0.95), 28: (0.57, 0.95),   # ankles
    }.items():
        lm[index, 0], lm[index, 1] = x, y
    return lm


class TestPartsForEvent:
    def test_impact_shows_the_metrics_measured_at_impact(self):
        parts = skeleton.parts_for_event("p7")
        assert set(parts.values()) >= {"hip_depth_change", "knee_extension_change"}

    def test_the_top_shows_posture_and_head_rise_to_the_top(self):
        parts = skeleton.parts_for_event("p4")
        assert "posture_change" in parts.values()
        assert "head_rise_p4" in parts.values()

    def test_address_is_the_reference_so_nothing_is_flagged(self):
        """P1 is where the others are measured FROM — it cannot deviate from
        itself, so colouring it would imply a judgement that does not exist."""
        assert skeleton.parts_for_event("p1") == {}

    def test_tempo_is_never_mapped_to_a_body_part(self):
        """Tempo is timing. There is nothing on the body to colour for it."""
        for event in ("p1", "p4", "p7", "p10"):
            assert "tempo_ratio" not in skeleton.parts_for_event(event).values()


class TestColours:
    def test_a_metric_worse_than_usual_is_flagged(self):
        found = [_standing("hip_depth_change", 0.90, 0.20)]
        colours = skeleton.colours_for(found, "p7")
        assert colours["hips"] == skeleton.WORSE

    def test_a_metric_better_than_usual_reads_as_better(self):
        found = [_standing("hip_depth_change", 0.05, 0.20)]
        assert skeleton.colours_for(found, "p7")["hips"] == skeleton.BETTER

    def test_a_typical_metric_is_left_neutral(self):
        found = [_standing("hip_depth_change", 0.201, 0.20)]
        assert skeleton.colours_for(found, "p7")["hips"] == skeleton.NEUTRAL

    def test_an_unranked_metric_is_left_neutral(self):
        found = [Standing(metric="hip_depth_change", label="", short="", unit="",
                          meaning="", value=0.9, rank=None, n_peers=3,
                          median=None)]
        assert skeleton.colours_for(found, "p7")["hips"] == skeleton.NEUTRAL


class TestDraw:
    def test_returns_an_image_of_the_same_shape(self):
        frame = _frame()
        out = skeleton.draw(frame, _landmarks(), {})
        assert out.shape == frame.shape

    def test_it_does_not_modify_the_frame_it_was_given(self):
        frame = _frame()
        before = frame.copy()
        skeleton.draw(frame, _landmarks(), {})
        assert np.array_equal(frame, before)

    def test_drawing_actually_marks_the_image(self):
        out = skeleton.draw(_frame(), _landmarks(), {})
        assert not np.array_equal(out, _frame()), "nothing was drawn"

    def test_a_highlighted_part_looks_different_from_a_neutral_one(self):
        plain = skeleton.draw(_frame(), _landmarks(), {})
        flagged = skeleton.draw(_frame(), _landmarks(), {"hips": skeleton.WORSE})
        assert not np.array_equal(plain, flagged)

    def test_low_confidence_joints_are_skipped(self):
        """MediaPipe reports a visibility score; drawing an invented limb
        across the frame would look like a real measurement."""
        lm = _landmarks()
        lm[25:29, 3] = 0.0            # both legs untrustworthy
        sparse = skeleton.draw(_frame(), lm, {})
        full = skeleton.draw(_frame(), _landmarks(), {})
        assert (sparse != full).any()
