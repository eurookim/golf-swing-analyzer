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
