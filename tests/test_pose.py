"""Tests for golfswing.pose.

Pose extraction needs the MediaPipe model on disk; tests skip if absent rather
than downloading 30 MB inside a test run.
"""

from pathlib import Path

import numpy as np
import pytest

from golfswing import pose
from golfswing.sequence import N_LANDMARKS, VISIBILITY

MODEL = Path("models/pose_landmarker_heavy.task")
REAL_CLIPS = sorted(Path("data/raw").glob("*.mov")) if Path("data/raw").is_dir() else []

requires_model = pytest.mark.skipif(
    not MODEL.exists(), reason="pose model not downloaded"
)


@requires_model
class TestExtractContract:
    """Shape and alignment guarantees, checked on a clip with no person in it."""

    def test_returns_a_pose_sequence(self, cfr_clip):
        seq = pose.extract_sequence(cfr_clip)
        assert seq.n_frames > 0

    def test_landmark_array_shape(self, cfr_clip):
        seq = pose.extract_sequence(cfr_clip)
        assert seq.landmarks.shape == (seq.n_frames, N_LANDMARKS, 4)

    def test_times_align_with_frames(self, cfr_clip):
        seq = pose.extract_sequence(cfr_clip)
        assert len(seq.times) == seq.n_frames
        assert np.all(np.diff(seq.times) > 0)

    def test_records_true_fps_from_container(self, cfr_clip):
        assert pose.extract_sequence(cfr_clip).fps == pytest.approx(50.0, abs=0.01)

    def test_records_source_path(self, cfr_clip):
        assert pose.extract_sequence(cfr_clip).source == str(cfr_clip)

    def test_undetected_frames_are_nan_not_zero(self, cfr_clip):
        """A colour test pattern has no person in it.

        Undetected frames must be NaN, never plausible-looking zeros — zeros are
        a valid coordinate and would silently produce real-looking angles from
        frames where nothing was seen.
        """
        seq = pose.extract_sequence(cfr_clip)
        assert np.all(np.isnan(seq.landmarks[:, :, :3]))
        assert np.all(seq.landmarks[:, :, VISIBILITY] == 0.0)


@requires_model
@pytest.mark.skipif(not REAL_CLIPS, reason="no swing footage in data/raw")
class TestAgainstRealFootage:
    """Integration check against actual swing video when it is present."""

    def test_detects_a_person_in_most_frames(self):
        seq = pose.extract_sequence(REAL_CLIPS[0])
        detected = ~np.isnan(seq.landmarks[:, 0, 0])
        assert detected.mean() > 0.8

    def test_torso_landmarks_are_high_confidence(self):
        """Phase 0 measured shoulders and hips at 1.00 visibility on DTL."""
        seq = pose.extract_sequence(REAL_CLIPS[0])
        torso = seq.landmarks[:, [11, 12, 23, 24], VISIBILITY]
        assert np.nanmean(torso) > 0.9


class TestTrackingTimestamps:
    """MediaPipe VIDEO mode requires strictly increasing integer milliseconds.

    Real presentation times cannot supply that: at 120fps frames are 8.33ms
    apart, and the decoder emitted the first two frames of a real clip only
    0.42ms apart, so both rounded to 0 and extraction raised. The tracking
    clock is therefore derived from the frame index — it only has to be
    monotonic, while the timestamps stored for metrics stay real.
    """

    def test_strictly_increasing_at_every_frame_rate(self):
        for fps in (24.0, 30.0, 59.94, 60.0, 120.0, 240.0):
            stamps = pose.tracking_timestamps_ms(200, fps)
            assert all(b > a for a, b in zip(stamps, stamps[1:])), f"collided at {fps}fps"

    def test_spacing_reflects_the_real_frame_rate(self):
        stamps = pose.tracking_timestamps_ms(100, 120.0)
        assert (stamps[-1] - stamps[0]) / 99 == pytest.approx(1000 / 120, abs=0.5)

    def test_survives_a_degenerate_frame_rate(self):
        stamps = pose.tracking_timestamps_ms(10, 0.0)
        assert all(b > a for a, b in zip(stamps, stamps[1:]))


@requires_model
class TestHighFrameRateExtraction:
    def test_120fps_clip_extracts_without_raising(self, high_fps_clip):
        seq = pose.extract_sequence(high_fps_clip)
        assert seq.n_frames > 0

    def test_real_timestamps_are_preserved_not_replaced(self, high_fps_clip):
        """The tracking clock is internal; stored times must stay real."""
        seq = pose.extract_sequence(high_fps_clip)
        assert np.diff(seq.times).mean() == pytest.approx(1 / 120, abs=1e-3)
