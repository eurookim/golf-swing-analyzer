"""Tests for golfswing.smooth."""

import numpy as np
import pytest

from golfswing import smooth

N_LANDMARKS = 33
X, Y, Z, VIS = 0, 1, 2, 3


def _seq(n_frames, fill=0.0):
    return np.full((n_frames, N_LANDMARKS, 4), fill, dtype=np.float64)


class TestWindowLength:
    """Window is given in SECONDS so smoothing is identical at any frame rate."""

    def test_converts_seconds_to_odd_frame_count(self):
        # 0.1s at 60fps = 6 frames -> must be odd for Savitzky-Golay
        assert smooth.window_frames(0.1, fps=60.0) == 7

    def test_scales_with_frame_rate(self):
        # same real-world duration, twice the frames
        assert smooth.window_frames(0.1, fps=120.0) == 13

    def test_never_returns_below_polyorder_plus_two(self):
        assert smooth.window_frames(0.001, fps=60.0, polyorder=2) >= 5

    def test_result_is_always_odd(self):
        for fps in (24.0, 30.0, 59.94, 60.0, 120.0, 240.0):
            assert smooth.window_frames(0.1, fps=fps) % 2 == 1


class TestSmoothLandmarks:
    def test_preserves_shape(self):
        seq = _seq(100)
        assert smooth.smooth_landmarks(seq, fps=60.0).shape == seq.shape

    def test_moves_a_noisy_signal_closer_to_the_truth(self):
        """Property test: filtering must reduce error against the known signal.

        Robust to window tuning — it asserts the filter helps, not by how much.
        """
        rng = np.random.default_rng(0)
        truth = 0.5
        seq = _seq(120, truth)
        seq[:, :, X] += rng.normal(0, 0.02, size=(120, N_LANDMARKS))

        out = smooth.smooth_landmarks(seq, fps=60.0)

        noisy_err = np.abs(seq[:, :, X] - truth).mean()
        smoothed_err = np.abs(out[:, :, X] - truth).mean()
        assert smoothed_err < noisy_err

    def test_noise_reduction_matches_savitzky_golay_theory(self):
        """Pins the filter to its known characteristics at a fixed window.

        A Savitzky-Golay filter of window 7, polyorder 2 has coefficients
        [-2,3,6,7,6,3,-2]/21, giving a white-noise gain of
        sqrt(sum(c^2)) = 0.577. Not an arbitrary threshold: this catches a filter
        that smooths too little (broken) or too much (destroying real motion).

        0.1s at 60fps = 6 frames, rounded up to 7 for the odd-window requirement.
        """
        rng = np.random.default_rng(0)
        seq = _seq(120, 0.5)
        seq[:, :, X] += rng.normal(0, 0.02, size=(120, N_LANDMARKS))

        out = smooth.smooth_landmarks(seq, fps=60.0, window_seconds=0.1)

        ratio = out[:, :, X].std() / seq[:, :, X].std()
        assert ratio == pytest.approx(0.577, abs=0.05)

    def test_preserves_a_linear_ramp(self):
        """Savitzky-Golay with polyorder>=1 is exact on polynomial signals.

        This is the counterpart to the noise test: together they show the filter
        removes jitter WITHOUT distorting real motion.
        """
        n = 100
        seq = _seq(n)
        ramp = np.linspace(0.0, 1.0, n)
        seq[:, :, X] = ramp[:, None]

        out = smooth.smooth_landmarks(seq, fps=60.0)

        np.testing.assert_allclose(out[:, :, X], seq[:, :, X], atol=1e-9)

    def test_leaves_visibility_channel_untouched(self):
        """Visibility is a confidence score, not a trajectory — never filter it."""
        rng = np.random.default_rng(1)
        seq = _seq(80)
        seq[:, :, VIS] = rng.random((80, N_LANDMARKS))

        out = smooth.smooth_landmarks(seq, fps=60.0)

        np.testing.assert_array_equal(out[:, :, VIS], seq[:, :, VIS])

    def test_smooths_x_y_and_z(self):
        rng = np.random.default_rng(2)
        seq = _seq(120, 0.5)
        for ch in (X, Y, Z):
            seq[:, :, ch] += rng.normal(0, 0.02, size=(120, N_LANDMARKS))

        out = smooth.smooth_landmarks(seq, fps=60.0)

        for ch in (X, Y, Z):
            assert out[:, :, ch].std() < seq[:, :, ch].std()

    def test_short_clip_shorter_than_window_still_works(self):
        """A 5-frame clip must not crash — window clamps to the data length."""
        seq = _seq(5, 0.3)
        out = smooth.smooth_landmarks(seq, fps=60.0)
        assert out.shape == seq.shape
        assert np.all(np.isfinite(out))

    def test_rejects_sequence_too_short_to_filter(self):
        with pytest.raises(ValueError):
            smooth.smooth_landmarks(_seq(2), fps=60.0)
