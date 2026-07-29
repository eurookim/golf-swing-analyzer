"""Tests for golfswing.metrics — DTL body measurements at the key frames."""

import numpy as np
import pytest

from golfswing import metrics
from golfswing.events import SwingEvents
from golfswing.sequence import PoseSequence

L_SH, R_SH = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
NOSE = 0


def _frame(positions: dict[int, tuple[float, float]] | None = None) -> np.ndarray:
    """One frame of landmarks; positions given as index -> (x, y)."""
    lm = np.full((1, 33, 4), np.nan)
    lm[:, :, 3] = 1.0
    for index, (x, y) in (positions or {}).items():
        lm[0, index, 0] = x
        lm[0, index, 1] = y
        lm[0, index, 2] = 0.0
    return lm


def _upright(shoulder_y=0.30, hip_y=0.60, lean=0.0, half_width=0.08):
    """A body whose shoulders sit `lean` to the right of the hips."""
    return _frame({
        L_SH: (0.5 + lean - half_width, shoulder_y),
        R_SH: (0.5 + lean + half_width, shoulder_y),
        L_HIP: (0.5 - half_width, hip_y),
        R_HIP: (0.5 + half_width, hip_y),
    })


def _sequence(landmarks, fps=60.0):
    n = landmarks.shape[0]
    return PoseSequence(landmarks=landmarks, times=np.arange(n) / fps,
                        fps=fps, source="synthetic")


class TestSpineTilt:
    """Angle between the hip->shoulder vector and image-vertical, in degrees.

    Positive means the shoulders sit at greater x than the hips. Image y grows
    downward, so the vertical reference has to be flipped or an upright spine
    reads as 180 degrees.
    """

    def test_upright_spine_is_zero(self):
        seq = _sequence(_upright(lean=0.0))
        assert metrics.spine_tilt(seq, 0) == pytest.approx(0.0, abs=0.01)

    def test_shoulders_forward_of_hips_is_positive(self):
        seq = _sequence(_upright(lean=0.10))
        assert metrics.spine_tilt(seq, 0) > 0

    def test_shoulders_behind_hips_is_negative(self):
        seq = _sequence(_upright(lean=-0.10))
        assert metrics.spine_tilt(seq, 0) < 0

    def test_forty_five_degrees(self):
        # equal horizontal and vertical separation -> 45 degrees
        seq = _sequence(_upright(shoulder_y=0.30, hip_y=0.60, lean=0.30))
        assert metrics.spine_tilt(seq, 0) == pytest.approx(45.0, abs=0.01)

    def test_is_scale_invariant(self):
        """Standing closer to the camera must not change the angle."""
        near = _sequence(_upright(shoulder_y=0.20, hip_y=0.80, lean=0.20))
        far = _sequence(_upright(shoulder_y=0.40, hip_y=0.70, lean=0.10))
        assert metrics.spine_tilt(near, 0) == pytest.approx(
            metrics.spine_tilt(far, 0), abs=0.01
        )

    def test_missing_landmarks_give_nan_not_a_number(self):
        """A guessed value here would silently become a plausible angle."""
        lm = _frame()  # nothing populated
        assert np.isnan(metrics.spine_tilt(_sequence(lm), 0))


class TestBodyScaleNormaliser:
    """The normaliser must be club-invariant AND viewpoint-invariant.

    Shoulder width satisfies the first but not the second: from down-the-line
    the shoulder line points nearly at the camera at address and foreshortens
    to almost nothing. Measured across four real clips it ranged 0.0087-0.0329
    for the same body — a 3.8x spread driven by camera placement, not anatomy.
    Dividing by it inflated every length metric by an unpredictable factor.

    Torso length is a vertical measure, so rotation about the body's vertical
    axis does not foreshorten it.
    """

    def test_returns_hip_to_shoulder_distance(self):
        seq = _sequence(_upright(shoulder_y=0.30, hip_y=0.60))
        assert metrics.body_scale(seq, 0) == pytest.approx(0.30, abs=1e-6)

    def test_survives_the_shoulders_foreshortening(self):
        """Rotation collapses shoulder width but must not change body scale."""
        square = _upright(half_width=0.10)
        turned = _upright(half_width=0.01)      # shoulder line pointing at camera
        assert metrics.body_scale(_sequence(square), 0) == pytest.approx(
            metrics.body_scale(_sequence(turned), 0), abs=1e-6
        )

    def test_scales_with_distance_from_camera(self):
        """Which is the point — it absorbs how close the golfer stands."""
        near = _sequence(_upright(shoulder_y=0.20, hip_y=0.80))
        far = _sequence(_upright(shoulder_y=0.40, hip_y=0.70))
        assert metrics.body_scale(near, 0) > metrics.body_scale(far, 0)

    def test_nan_when_the_body_is_missing(self):
        assert np.isnan(metrics.body_scale(_sequence(_frame()), 0))


class TestPostureChange:
    """Loss of posture: spine tilt at the top differs from address."""

    def test_zero_when_posture_is_maintained(self):
        lm = np.concatenate([_upright(lean=0.20), _upright(lean=0.20)])
        assert metrics.posture_change(_sequence(lm), p1=0, p4=1) == pytest.approx(0.0, abs=0.01)

    def test_positive_when_standing_up_out_of_posture(self):
        """Standing up reduces forward lean, so the change is signed negative
        in tilt but reported as a magnitude of deviation."""
        lm = np.concatenate([_upright(lean=0.30), _upright(lean=0.10)])
        change = metrics.posture_change(_sequence(lm), p1=0, p4=1)
        assert change != pytest.approx(0.0, abs=1.0)

    def test_is_the_signed_difference_top_minus_address(self):
        lm = np.concatenate([_upright(lean=0.0), _upright(lean=0.30)])
        seq = _sequence(lm)
        expected = metrics.spine_tilt(seq, 1) - metrics.spine_tilt(seq, 0)
        assert metrics.posture_change(seq, p1=0, p4=1) == pytest.approx(expected)


class TestBallDirection:
    """Which way is the ball, in image x?

    Not hardcoded: at address the golfer leans toward the ball, so the
    horizontal component of the hip->shoulder vector points at it. That makes
    the sign work for either handedness and either side the camera sits on.
    """

    def test_leaning_right_means_ball_is_to_the_right(self):
        assert metrics.ball_direction(_sequence(_upright(lean=0.20)), 0) == 1.0

    def test_leaning_left_means_ball_is_to_the_left(self):
        assert metrics.ball_direction(_sequence(_upright(lean=-0.20)), 0) == -1.0

    def test_nan_when_the_body_is_missing(self):
        assert np.isnan(metrics.ball_direction(_sequence(_frame()), 0))


class TestHipDepth:
    """Early extension: hips thrusting toward the ball between top and impact."""

    def test_zero_when_the_hips_hold_their_position(self):
        lm = np.concatenate([_upright(lean=0.20), _upright(lean=0.20)])
        assert metrics.hip_depth_change(_sequence(lm), p1=0, p7=1) == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_hips_move_toward_the_ball(self):
        address = _upright(lean=0.20)
        impact = _upright(lean=0.20)
        impact[0, [L_HIP, R_HIP], 0] += 0.04    # hips shift toward the ball (+x)
        lm = np.concatenate([address, impact])
        assert metrics.hip_depth_change(_sequence(lm), p1=0, p7=1) > 0

    def test_negative_when_hips_move_away_from_the_ball(self):
        address = _upright(lean=0.20)
        impact = _upright(lean=0.20)
        impact[0, [L_HIP, R_HIP], 0] -= 0.04
        lm = np.concatenate([address, impact])
        assert metrics.hip_depth_change(_sequence(lm), p1=0, p7=1) < 0

    def test_sign_is_independent_of_which_way_the_golfer_faces(self):
        """A mirrored setup must report the same fault, not the opposite one."""
        right = np.concatenate([_upright(lean=0.20), _upright(lean=0.20)])
        right[1, [L_HIP, R_HIP], 0] += 0.04
        left = np.concatenate([_upright(lean=-0.20), _upright(lean=-0.20)])
        left[1, [L_HIP, R_HIP], 0] -= 0.04     # same movement, mirrored

        assert metrics.hip_depth_change(_sequence(right), p1=0, p7=1) == pytest.approx(
            metrics.hip_depth_change(_sequence(left), p1=0, p7=1)
        )

    def test_is_normalised_by_body_scale_not_raw_pixels(self):
        """Standing closer to the camera must not inflate the reading."""
        # Every dimension must scale by 2x, lean included, or the two bodies are
        # not similar shapes and the comparison means nothing.
        near = np.concatenate([_upright(lean=0.20, shoulder_y=0.20, hip_y=0.80)] * 2)
        near[1, [L_HIP, R_HIP], 0] += 0.08
        far = np.concatenate([_upright(lean=0.10, shoulder_y=0.40, hip_y=0.70)] * 2)
        far[1, [L_HIP, R_HIP], 0] += 0.04

        assert metrics.hip_depth_change(_sequence(near), p1=0, p7=1) == pytest.approx(
            metrics.hip_depth_change(_sequence(far), p1=0, p7=1), abs=1e-9
        )


class TestHeadMovement:
    def test_head_rise_is_positive_when_the_head_goes_up(self):
        address = _upright()
        address[0, NOSE] = (0.5, 0.20, 0.0, 1.0)
        top = _upright()
        top[0, NOSE] = (0.5, 0.16, 0.0, 1.0)     # smaller y = higher in frame
        lm = np.concatenate([address, top])
        assert metrics.head_rise(_sequence(lm), p1=0, other=1) > 0

    def test_head_rise_is_negative_when_the_head_drops(self):
        address = _upright()
        address[0, NOSE] = (0.5, 0.20, 0.0, 1.0)
        top = _upright()
        top[0, NOSE] = (0.5, 0.26, 0.0, 1.0)
        lm = np.concatenate([address, top])
        assert metrics.head_rise(_sequence(lm), p1=0, other=1) < 0

    def test_nan_when_the_nose_is_missing(self):
        lm = np.concatenate([_upright(), _upright()])
        assert np.isnan(metrics.head_rise(_sequence(lm), p1=0, other=1))


class TestKneeAngle:
    """Interior angle at the knee: 180 degrees is a straight leg."""

    def _leg(self, hip_xy, knee_xy, ankle_xy):
        return _frame({
            L_HIP: hip_xy, R_HIP: hip_xy,
            L_KNEE: knee_xy, R_KNEE: knee_xy,
            L_ANKLE: ankle_xy, R_ANKLE: ankle_xy,
            L_SH: (0.5, 0.2), R_SH: (0.5, 0.2),
        })

    def test_straight_leg_is_180_degrees(self):
        lm = self._leg((0.5, 0.50), (0.5, 0.70), (0.5, 0.90))
        assert metrics.knee_angle(_sequence(lm), 0) == pytest.approx(180.0, abs=0.01)

    def test_right_angle_is_90_degrees(self):
        lm = self._leg((0.5, 0.50), (0.5, 0.70), (0.7, 0.70))
        assert metrics.knee_angle(_sequence(lm), 0) == pytest.approx(90.0, abs=0.01)

    def test_is_scale_invariant(self):
        small = self._leg((0.5, 0.50), (0.5, 0.60), (0.6, 0.60))
        large = self._leg((0.5, 0.30), (0.5, 0.70), (0.9, 0.70))
        assert metrics.knee_angle(_sequence(small), 0) == pytest.approx(
            metrics.knee_angle(_sequence(large), 0), abs=0.01
        )

    def test_nan_when_the_leg_is_missing(self):
        assert np.isnan(metrics.knee_angle(_sequence(_frame()), 0))


class TestKneeExtension:
    def test_positive_when_the_leg_straightens_into_impact(self):
        bent = _frame({L_HIP: (0.5, 0.5), R_HIP: (0.5, 0.5),
                       L_KNEE: (0.5, 0.7), R_KNEE: (0.5, 0.7),
                       L_ANKLE: (0.62, 0.88), R_ANKLE: (0.62, 0.88),
                       L_SH: (0.5, 0.2), R_SH: (0.5, 0.2)})
        straight = _frame({L_HIP: (0.5, 0.5), R_HIP: (0.5, 0.5),
                           L_KNEE: (0.5, 0.7), R_KNEE: (0.5, 0.7),
                           L_ANKLE: (0.5, 0.9), R_ANKLE: (0.5, 0.9),
                           L_SH: (0.5, 0.2), R_SH: (0.5, 0.2)})
        seq = _sequence(np.concatenate([bent, straight]))
        assert metrics.knee_extension_change(seq, p1=0, p7=1) > 0


class TestHeadDepth:
    """Head drifting toward or away from the ball, along image x."""

    def _with_nose(self, lean, nose_x):
        lm = _upright(lean=lean)
        lm[0, NOSE] = (nose_x, 0.20, 0.0, 1.0)
        return lm

    def test_positive_when_the_head_moves_toward_the_ball(self):
        lm = np.concatenate([self._with_nose(0.2, 0.50), self._with_nose(0.2, 0.56)])
        assert metrics.head_depth_change(_sequence(lm), p1=0, other=1) > 0

    def test_sign_is_independent_of_which_way_the_golfer_faces(self):
        right = np.concatenate([self._with_nose(0.2, 0.50), self._with_nose(0.2, 0.56)])
        left = np.concatenate([self._with_nose(-0.2, 0.50), self._with_nose(-0.2, 0.44)])
        assert metrics.head_depth_change(_sequence(right), p1=0, other=1) == pytest.approx(
            metrics.head_depth_change(_sequence(left), p1=0, other=1)
        )


class TestSwingMetricsRecord:
    """Everything a fault rule needs, computed in one pass."""

    def _swing(self):
        frames = [_upright(lean=0.30), _upright(lean=0.24), _upright(lean=0.28)]
        for f in frames:
            f[0, NOSE] = (0.5, 0.20, 0.0, 1.0)
            f[0, L_KNEE] = (0.5, 0.72, 0.0, 1.0)
            f[0, R_KNEE] = (0.5, 0.72, 0.0, 1.0)
            f[0, L_ANKLE] = (0.55, 0.90, 0.0, 1.0)
            f[0, R_ANKLE] = (0.55, 0.90, 0.0, 1.0)
        return _sequence(np.concatenate(frames), fps=10.0)

    def test_computes_every_v1_metric(self):
        m = metrics.compute(self._swing(), SwingEvents(p1=0, p4=1, p7=2, p10=2))
        for field in ("spine_tilt_p1", "spine_tilt_p4", "spine_tilt_p7",
                      "posture_change", "hip_depth_change", "head_rise_p4",
                      "head_rise_p7", "head_depth_p7", "knee_extension_change",
                      "tempo_ratio"):
            assert hasattr(m, field), f"missing {field}"

    def test_carries_the_events_it_was_computed_at(self):
        events = SwingEvents(p1=0, p4=1, p7=2, p10=2)
        assert metrics.compute(self._swing(), events).events == events

    def test_survives_a_sequence_with_missing_landmarks(self):
        """Must return NaNs, not raise — a partly-tracked swing is still worth
        reporting on for the metrics that did resolve."""
        blank = _sequence(np.concatenate([_frame(), _frame(), _frame()]), fps=10.0)
        m = metrics.compute(blank, SwingEvents(p1=0, p4=1, p7=2, p10=2))
        assert np.isnan(m.spine_tilt_p1)

    def test_as_dict_is_flat_and_serialisable(self):
        m = metrics.compute(self._swing(), SwingEvents(p1=0, p4=1, p7=2, p10=2))
        d = m.as_dict()
        assert all(isinstance(v, float) for v in d.values())


class TestTempo:
    def test_is_the_ratio_of_backswing_to_downswing_in_seconds(self):
        seq = _sequence(np.repeat(_upright(), 100, axis=0), fps=50.0)
        # P1=0s, P4=1.2s (frame 60), P7=1.6s (frame 80) -> 3:1
        assert metrics.tempo_ratio(seq, p1=0, p4=60, p7=80) == pytest.approx(3.0)

    def test_uses_real_timestamps_not_frame_counts(self):
        lm = np.repeat(_upright(), 100, axis=0)
        times = np.arange(100) / 50.0
        times[60:] += 0.5                       # a gap the frame count cannot see
        seq = PoseSequence(landmarks=lm, times=times, fps=50.0, source="x")
        assert metrics.tempo_ratio(seq, p1=0, p4=60, p7=80) != pytest.approx(3.0)

    def test_nan_when_the_downswing_has_no_duration(self):
        seq = _sequence(np.repeat(_upright(), 100, axis=0))
        assert np.isnan(metrics.tempo_ratio(seq, p1=0, p4=60, p7=60))
