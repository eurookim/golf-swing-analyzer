"""Tests for golfswing.ingest."""

import numpy as np
import pytest

from golfswing import ingest


class TestRotationCorrection:
    """ffprobe reports the display-matrix angle; the correction is its negative.

    Verified empirically against ffmpeg's own output on a real iPhone clip:
    a file tagged rotation=-90 needs a 90 degree CLOCKWISE rotation to display
    upright (2.2 mean abs diff vs 71.6 for counter-clockwise).
    """

    def test_iphone_minus_90_needs_90_clockwise(self):
        assert ingest.rotation_correction(-90) == 90

    def test_zero_rotation_is_noop(self):
        assert ingest.rotation_correction(0) == 0

    def test_180_is_symmetric(self):
        assert ingest.rotation_correction(180) == 180

    def test_positive_90_needs_270_clockwise(self):
        assert ingest.rotation_correction(90) == 270

    def test_normalises_angles_outside_one_turn(self):
        assert ingest.rotation_correction(-450) == 90


class TestApplyRotation:
    """Rotating actual pixels, not just computing an angle."""

    def test_iphone_minus_90_rotates_pixels_clockwise(self):
        frame = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype=np.uint8)
        # 90 clockwise: the original bottom-left value lands top-left.
        expected = np.array([[4, 1],
                             [5, 2],
                             [6, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(ingest.apply_rotation(frame, -90), expected)

    def test_zero_rotation_returns_frame_unchanged(self):
        frame = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(ingest.apply_rotation(frame, 0), frame)

    def test_180_flips_both_axes(self):
        frame = np.array([[1, 2, 3],
                          [4, 5, 6]], dtype=np.uint8)
        expected = np.array([[6, 5, 4],
                             [3, 2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(ingest.apply_rotation(frame, 180), expected)

    def test_portrait_iphone_frame_becomes_portrait(self):
        """1920x1080 stored landscape + rotation=-90 must display 1080x1920."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert ingest.apply_rotation(frame, -90).shape == (1920, 1080, 3)


class TestProbe:
    def test_reads_true_capture_fps(self, cfr_clip):
        assert ingest.probe(cfr_clip).fps == pytest.approx(50.0, abs=0.01)

    def test_reads_dimensions(self, cfr_clip):
        info = ingest.probe(cfr_clip)
        assert (info.width, info.height) == (320, 240)

    def test_reports_zero_rotation_when_absent(self, cfr_clip):
        assert ingest.probe(cfr_clip).rotation == 0

    def test_reads_negative_rotation_metadata(self, rotated_clip):
        assert ingest.probe(rotated_clip).rotation == -90

    def test_display_size_accounts_for_rotation(self, rotated_clip):
        info = ingest.probe(rotated_clip)
        assert (info.width, info.height) == (320, 240)
        assert (info.display_width, info.display_height) == (240, 320)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest.probe(tmp_path / "nope.mov")


class TestFrameTimes:
    """The VFR finding: iPhone clips advertise 60fps but average ~55.

    Computing time as ``index / fps`` therefore drifts, which corrupts tempo.
    Timestamps must come from the decoder, one per frame ACTUALLY decoded.
    """

    def test_returns_one_timestamp_per_decoded_frame(self, cfr_clip):
        times, frames = ingest.read_frames_with_times(cfr_clip)
        assert len(times) == len(frames)
        assert len(frames) > 0

    def test_timestamps_are_monotonically_increasing(self, cfr_clip):
        times, _ = ingest.read_frames_with_times(cfr_clip)
        assert np.all(np.diff(times) > 0)

    def test_timestamps_are_in_seconds_starting_near_zero(self, cfr_clip):
        times, _ = ingest.read_frames_with_times(cfr_clip)
        assert times[0] == pytest.approx(0.0, abs=0.05)
        assert times[-1] == pytest.approx(1.0, abs=0.10)

    def test_frames_are_rotation_corrected(self, rotated_clip):
        _, frames = ingest.read_frames_with_times(rotated_clip)
        # stored 320x240 landscape, rotation -90 -> displays 240x320 portrait
        assert frames[0].shape[:2] == (320, 240)
