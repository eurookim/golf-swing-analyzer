"""Tests for golfswing.events — swing signals and P1/P4/P7/P10 detection."""

import numpy as np
import pytest

from golfswing import events
from golfswing.sequence import PoseSequence

L_WR, R_WR = 15, 16
TORSO = (11, 12, 23, 24)


def _sequence(landmarks, fps=60.0, times=None):
    n = landmarks.shape[0]
    if times is None:
        times = np.arange(n) / fps
    return PoseSequence(landmarks=landmarks, times=times, fps=fps, source="synthetic")


def _still(n=60):
    lm = np.zeros((n, 33, 4))
    lm[:, :, 3] = 1.0
    return lm


def _synthetic_swing(n=140, p1=30, p4=60, p7=80):
    """A swing with known event frames.

    Deliberately gives the FOLLOW-THROUGH a higher wrist peak (0.66) than the
    backswing (0.62) — that is what the real clips do, and it is the trap that
    breaks a naive global-maximum search for the top of the backswing.
    """
    lm = np.zeros((n, 33, 4))
    lm[:, :, 3] = 1.0

    height = np.full(n, 0.40)
    height[p1:p4 + 1] = np.linspace(0.40, 0.62, p4 - p1 + 1)      # backswing
    height[p4:p7 + 1] = np.linspace(0.62, 0.34, p7 - p4 + 1)      # downswing
    follow_end = min(p7 + 22, n)
    height[p7:follow_end] = np.linspace(0.34, 0.66, follow_end - p7)
    height[follow_end:] = np.linspace(0.66, 0.45, n - follow_end)

    lm[:, L_WR, 1] = 1.0 - height
    lm[:, R_WR, 1] = 1.0 - height

    # Torso: a velocity burst peaking exactly at p7, decaying fast afterwards —
    # the torso stops well before the swing does.
    burst = np.exp(-((np.arange(n) - p7) ** 2) / (2 * 4.0 ** 2))
    travel = np.cumsum(burst) * 0.01
    for j in TORSO:
        lm[:, j, 0] = travel
        lm[:, j, 1] = 0.5

    # Arms keep travelling through the follow-through and only settle at
    # p7 + 30. This is the real behaviour the torso-only finish test missed.
    arm_motion = np.exp(-((np.arange(n) - p7) ** 2) / (2 * 12.0 ** 2))
    lm[:, L_WR, 0] = np.cumsum(arm_motion) * 0.01
    lm[:, R_WR, 0] = np.cumsum(arm_motion) * 0.01

    return lm


class TestWristHeight:
    def test_one_value_per_frame(self):
        seq = _sequence(_still(40))
        assert events.wrist_height(seq).shape == (40,)

    def test_higher_when_hands_are_higher_in_frame(self):
        """Image y grows downward, so a smaller y must read as a greater height."""
        lm = _still(2)
        lm[0, [L_WR, R_WR], 1] = 0.8   # low in frame
        lm[1, [L_WR, R_WR], 1] = 0.2   # high in frame

        height = events.wrist_height(_sequence(lm))

        assert height[1] > height[0]

    def test_peaks_at_the_top_of_the_backswing(self):
        seq = _sequence(_synthetic_swing())
        height = events.wrist_height(seq)
        assert np.argmax(height[:75]) == pytest.approx(60, abs=2)


class TestTorsoSpeed:
    def test_one_value_per_frame(self):
        assert events.torso_speed(_sequence(_still(40))).shape == (40,)

    def test_zero_for_a_stationary_body(self):
        assert np.allclose(events.torso_speed(_sequence(_still(40))), 0.0)

    def test_peaks_at_impact(self):
        seq = _sequence(_synthetic_swing())
        assert np.argmax(events.torso_speed(seq)) == pytest.approx(80, abs=2)

    def test_uses_real_timestamps_not_frame_index(self):
        """Same motion over twice the elapsed time must read as half the speed."""
        lm = _still(20)
        lm[:, TORSO, 0] = np.linspace(0, 1, 20)[:, None]

        fast = events.torso_speed(_sequence(lm, times=np.arange(20) * 0.01))
        slow = events.torso_speed(_sequence(lm, times=np.arange(20) * 0.02))

        assert np.nanmean(fast) == pytest.approx(2 * np.nanmean(slow), rel=1e-6)


class TestDetectEvents:
    def test_finds_impact_at_the_speed_spike(self):
        got = events.detect_events(_sequence(_synthetic_swing(p7=80)))
        assert got.p7 == pytest.approx(80, abs=2)

    def test_finds_top_of_backswing(self):
        got = events.detect_events(_sequence(_synthetic_swing(p4=60)))
        assert got.p4 == pytest.approx(60, abs=3)

    def test_top_is_the_peak_before_impact_not_the_global_maximum(self):
        """The follow-through peak is higher; a global argmax would return it."""
        seq = _sequence(_synthetic_swing(p4=60, p7=80))
        height = events.wrist_height(seq)

        assert np.argmax(height) > 80, "fixture must have a higher follow-through"
        assert events.detect_events(seq).p4 < 80

    def test_address_precedes_the_top(self):
        got = events.detect_events(_sequence(_synthetic_swing(p1=30, p4=60)))
        assert got.p1 == pytest.approx(30, abs=6)
        assert got.p1 < got.p4

    def test_events_are_ordered(self):
        got = events.detect_events(_sequence(_synthetic_swing()))
        assert got.p1 < got.p4 < got.p7 < got.p10

    def test_finish_comes_after_impact(self):
        got = events.detect_events(_sequence(_synthetic_swing()))
        assert got.p10 > got.p7

    def test_finish_waits_for_the_whole_body_to_settle(self):
        """The torso stops almost immediately at impact; the arms do not.

        Detecting the finish from torso speed alone lands ~0.15s after impact,
        mid-follow-through — verified wrong on real footage via the contact
        sheet. The finish must wait for full-body motion to settle.
        """
        seq = _sequence(_synthetic_swing(p7=80))
        got = events.detect_events(seq)

        settle_seconds = seq.times[got.p10] - seq.times[got.p7]
        assert settle_seconds > 0.30, (
            f"finish only {settle_seconds:.2f}s after impact — still mid-swing"
        )

    def test_downswing_duration_is_physically_plausible(self):
        seq = _sequence(_synthetic_swing(p4=60, p7=80))
        got = events.detect_events(seq)
        downswing = seq.times[got.p7] - seq.times[got.p4]
        assert 0.15 < downswing < 0.60

    def test_raises_when_no_swing_is_present(self):
        """A clip of someone standing still has no impact to find."""
        with pytest.raises(events.NoSwingDetectedError):
            events.detect_events(_sequence(_still(120)))
