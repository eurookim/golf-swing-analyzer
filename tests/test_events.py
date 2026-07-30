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


FLAT_TAKEAWAY = 10  # frames of horizontal-only club movement at the start


def _synthetic_swing(n=140, p1=30, p4=60, p7=80):
    """A swing with known event frames.

    Two traps from real footage are built in deliberately:

    1. The FOLLOW-THROUGH peaks higher (0.66) than the backswing (0.62), which
       breaks a naive global-maximum search for the top.
    2. The takeaway is HORIZONTAL for its first `FLAT_TAKEAWAY` frames — the club
       travels back before it travels up, so wrist HEIGHT stays flat while the
       hands are already moving. Detecting address from height alone lands late.
    """
    lm = np.zeros((n, 33, 4))
    lm[:, :, 3] = 1.0

    lift = p1 + FLAT_TAKEAWAY
    height = np.full(n, 0.40)
    height[lift:p4 + 1] = np.linspace(0.40, 0.62, p4 - lift + 1)   # backswing lift
    height[p4:p7 + 1] = np.linspace(0.62, 0.34, p7 - p4 + 1)       # downswing
    follow_end = min(p7 + 22, n)
    height[p7:follow_end] = np.linspace(0.34, 0.66, follow_end - p7)
    height[follow_end:] = np.linspace(0.66, 0.45, n - follow_end)

    lm[:, L_WR, 1] = 1.0 - height
    lm[:, R_WR, 1] = 1.0 - height

    # Hands travel backwards horizontally from p1, including through the flat
    # part where height reveals nothing.
    hand_x = np.zeros(n)
    hand_x[p1:p4 + 1] = np.linspace(0.0, 0.25, p4 - p1 + 1)
    hand_x[p4:] = 0.25
    lm[:, L_WR, 0] += hand_x
    lm[:, R_WR, 0] += hand_x

    # Shoulder line foreshortens to its narrowest exactly AT impact: from
    # down-the-line the torso squares to the target line and the shoulder line
    # points at the camera. Measured on all three real clips as the most
    # accurate impact cue (mean_abs 1.0 frames).
    # Squareness bottoms ~3 frames AFTER contact — rotation continues through
    # release. Measured on 15 labeled 120fps clips: shoulder-width minimum runs
    # mean_abs 4.47 frames with a max of 14, while the hands bottom AT contact
    # (mean_abs 0.40, max 1).
    squareness = np.exp(-((np.arange(n) - (p7 + 3)) ** 2) / (2 * 5.0 ** 2))
    half_width = 0.16 * (1.0 - 0.85 * squareness)
    lm[:, 11, 0] = 0.5 - half_width
    lm[:, 12, 0] = 0.5 + half_width
    lm[:, 11, 1] = 0.35
    lm[:, 12, 1] = 0.35

    # Torso velocity peaks 3 frames AFTER contact — the body keeps accelerating
    # through release, which is why torso speed lags true impact on real
    # footage.
    # Amplitude has to dominate the shoulder-squareness motion above, or the
    # torso-speed peak is pulled earlier by it and the fixture stops
    # demonstrating the lag it exists to demonstrate.
    burst = np.exp(-((np.arange(n) - (p7 + 3)) ** 2) / (2 * 4.0 ** 2))
    travel = np.cumsum(burst) * 0.05
    for j in TORSO:
        lm[:, j, 0] += travel
        if j in (23, 24):
            lm[:, j, 1] = 0.55

    # Arms keep travelling through the follow-through and only settle at
    # p7 + 30. This is the real behaviour the torso-only finish test missed.
    # NOTE: must accumulate, not assign — assigning here would wipe out the
    # backswing hand travel set above and silently change what the tests mean.
    arm_motion = np.exp(-((np.arange(n) - p7) ** 2) / (2 * 12.0 ** 2))
    lm[:, L_WR, 0] += np.cumsum(arm_motion) * 0.01
    lm[:, R_WR, 0] += np.cumsum(arm_motion) * 0.01

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

    def test_lags_impact_rather_than_marking_it(self):
        """Documents why torso speed cannot be the impact signal on its own.

        Measured on 15 labeled 120fps clips: mean +3.40 frames, always late,
        because the body keeps accelerating through release. Useful for
        locating the swing; too coarse for pinning the moment.
        """
        seq = _sequence(_synthetic_swing(p7=80))
        assert np.argmax(events.torso_speed(seq)) >= 80

    def test_a_degenerate_first_interval_does_not_fake_a_spike(self):
        """Decoders emit a near-zero first interval; dividing by it explodes.

        Measured on real 120fps clips: the first dt came back as 0.417ms against
        a median of 8.333ms — 20x too small — inflating frame 0's speed past the
        real impact, so two clips failed with 'impact detected on the first
        frame'.
        """
        # Displacements taken from a real clip: near-static at address, ~11x
        # larger at impact. Uniform motion would be a useless fixture here —
        # every frame would share the same speed and argmax would be 0 anyway.
        displacement = np.full(60, 0.00087)
        displacement[28:33] = 0.0095                          # impact burst

        lm = _still(60)
        lm[:, TORSO, 0] = np.cumsum(displacement)[:, None]
        times = np.arange(60) / 120.0
        times[1] = 0.000417                                   # the artifact

        speed = events.torso_speed(_sequence(lm, times=times))

        assert np.nanargmax(speed) != 0, "frame 0 spike from a degenerate dt"
        assert 27 <= np.nanargmax(speed) <= 33, "should peak at the real burst"

    def test_uses_real_timestamps_not_frame_index(self):
        """Same motion over twice the elapsed time must read as half the speed."""
        lm = _still(20)
        lm[:, TORSO, 0] = np.linspace(0, 1, 20)[:, None]

        fast = events.torso_speed(_sequence(lm, times=np.arange(20) * 0.01))
        slow = events.torso_speed(_sequence(lm, times=np.arange(20) * 0.02))

        assert np.nanmean(fast) == pytest.approx(2 * np.nanmean(slow), rel=1e-6)


class TestDetectEvents:
    def test_finds_impact_at_the_hand_low_point(self):
        got = events.detect_events(_sequence(_synthetic_swing(p7=80)))
        assert got.p7 == pytest.approx(80, abs=2)

    def test_ignores_a_larger_speed_peak_late_in_the_follow_through(self):
        """Some swings spike harder rotating into the finish than at impact.

        Measured on real 120fps clips: 5 of 20 had a follow-through peak
        exceeding the impact peak, and the global-maximum search picked it —
        producing downswings of 0.67-0.90s against a real range of 0.28-0.36s.
        Impact must be bounded to a physically plausible window after the top.
        """
        n, p4, p7 = 400, 100, 140
        lm = _synthetic_swing(n=n, p1=40, p4=p4, p7=p7)
        # A bigger burst 0.7s after the top, well past any real downswing.
        late = int(p4 + 0.70 * 120)
        extra = np.exp(-((np.arange(n) - late) ** 2) / (2 * 4.0 ** 2)) * 1.4
        for j in TORSO:
            lm[:, j, 0] += np.cumsum(extra) * 0.01

        got = events.detect_events(_sequence(lm, fps=120.0))

        assert abs(got.p7 - p7) < 20, (
            f"impact at f{got.p7} chased the follow-through spike at f{late}"
        )

    def test_impact_is_not_dragged_late_by_torso_acceleration(self):
        """Peak torso speed occurs AFTER contact — the body is still
        accelerating through release.

        Measured on three verified clips: shoulder-width minimum scores
        mean_abs 1.0 frames against torso-speed max at 1.67 (max error 3).
        Impact must key on the torso squaring to the target line, not on how
        fast it is rotating.
        """
        got = events.detect_events(_sequence(_synthetic_swing(p7=80)))
        assert got.p7 == pytest.approx(80, abs=1)

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

    def test_address_is_when_the_hands_start_moving_not_when_they_rise(self):
        """The club goes BACK before it goes UP.

        Verified on real footage: detecting address from wrist height alone put
        P1 well into the takeaway on all three clips, with the club visibly
        lifted. A late P1 shortens the measured backswing and drags the tempo
        ratio down, which is what made three swings read 1.5-2.0:1 against a
        ~3:1 norm.
        """
        p1 = 30
        got = events.detect_events(_sequence(_synthetic_swing(p1=p1, p4=60)))

        assert got.p1 < p1 + FLAT_TAKEAWAY, (
            f"P1 at frame {got.p1} is inside the flat takeaway "
            f"(starts {p1}, lift begins {p1 + FLAT_TAKEAWAY}) — detected from "
            f"height rather than motion onset"
        )

    def test_leading_dead_time_does_not_change_the_measured_swing(self):
        """Standing still before the swing must not inflate the backswing.

        Detection anchors on impact and scans BACKWARD, so extra stillness at
        the head of a clip is skipped rather than counted. Durations between
        events must be identical however long the golfer stands there.
        """
        base = _synthetic_swing(n=140, p1=30, p4=60, p7=80)
        dead = np.repeat(base[:1], 120, axis=0)     # 2 extra seconds of address
        padded = np.concatenate([dead, base], axis=0)

        short = events.detect_events(_sequence(base))
        long = events.detect_events(_sequence(padded))

        assert long.p4 - long.p1 == pytest.approx(short.p4 - short.p1, abs=2)
        assert long.p7 - long.p4 == pytest.approx(short.p7 - short.p4, abs=2)

    def test_address_ignores_a_waggle(self):
        """A pre-shot waggle is motion, but it is not the takeaway."""
        lm = _synthetic_swing(p1=40, p4=70, p7=90)
        lm[10:16, [L_WR, R_WR], 0] += 0.02   # brief jiggle, then still again

        got = events.detect_events(_sequence(lm))

        assert got.p1 > 20, f"P1 at {got.p1} latched onto the waggle"

    def test_events_are_ordered(self):
        got = events.detect_events(_sequence(_synthetic_swing()))
        assert got.p1 < got.p4 < got.p7 < got.p10

    def test_finish_comes_after_impact(self):
        got = events.detect_events(_sequence(_synthetic_swing()))
        assert got.p10 > got.p7

    def test_finish_ignores_a_momentary_dip_in_the_follow_through(self):
        """Motion must STAY settled, not merely touch the threshold once.

        Regression guard, not a fix: this passed on first write, so it did not
        drive any change. The real early-finish bias on four verified clips
        (-7, -1, -6, -3) turned out to be definitional rather than a lull — the
        detector marks when motion stops, while a human marks the fully-wrapped
        finish pose, which is still being moved into. See PLAN.md.
        """
        p7 = 80
        lm = _synthetic_swing(n=200, p1=30, p4=60, p7=p7)
        # Brief lull mid-follow-through, then motion resumes before the real stop.
        for j in (L_WR, R_WR):
            lm[p7 + 8:p7 + 12, j, 0] = lm[p7 + 8, j, 0]

        got = events.detect_events(_sequence(lm))

        assert got.p10 > p7 + 14, (
            f"finish at f{got.p10} latched onto the lull at f{p7 + 8}-{p7 + 12}"
        )

    def test_finish_waits_for_the_whole_body_to_settle(self):
        """The torso stops almost immediately at impact; the arms do not.

        Detecting the finish from torso speed alone lands ~0.15s after impact,
        mid-follow-through — verified wrong on real footage via the contact
        sheet. The finish must wait for full-body motion to settle.
        """
        seq = _sequence(_synthetic_swing(p7=80))
        got = events.detect_events(seq)

        settle_seconds = seq.times[got.p10] - seq.times[got.p7]
        # The bug this guards against put the finish 0.15s after impact. The
        # bar is set clear of that rather than at a precise value, because the
        # exact figure depends on fixture amplitudes rather than on behaviour.
        assert settle_seconds > 0.20, (
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
