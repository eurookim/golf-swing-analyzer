"""Tests for golfswing.store and golfswing.sequence."""

import numpy as np
import pytest

from golfswing import store
from golfswing.sequence import PoseSequence

N_LANDMARKS = 33


def _sequence(n_frames=40, fps=60.0):
    rng = np.random.default_rng(7)
    return PoseSequence(
        landmarks=rng.random((n_frames, N_LANDMARKS, 4)),
        times=np.linspace(0.0, (n_frames - 1) / fps, n_frames),
        fps=fps,
        source="data/raw/2026-07-28_dtl_7iron_01.mov",
    )


class TestPoseSequence:
    def test_n_frames_reflects_landmark_count(self):
        assert _sequence(n_frames=40).n_frames == 40

    def test_duration_comes_from_timestamps_not_frame_count(self):
        """Duration must be measured, never computed as n_frames / fps."""
        seq = PoseSequence(
            landmarks=np.zeros((3, N_LANDMARKS, 4)),
            times=np.array([0.0, 0.5, 2.0]),
            fps=60.0,
            source="x.mov",
        )
        assert seq.duration == pytest.approx(2.0)

    def test_rejects_times_length_mismatch(self):
        with pytest.raises(ValueError):
            PoseSequence(
                landmarks=np.zeros((10, N_LANDMARKS, 4)),
                times=np.zeros(9),
                fps=60.0,
                source="x.mov",
            )

    def test_rejects_wrong_landmark_shape(self):
        with pytest.raises(ValueError):
            PoseSequence(
                landmarks=np.zeros((10, N_LANDMARKS)),
                times=np.zeros(10),
                fps=60.0,
                source="x.mov",
            )


class TestRoundTrip:
    def test_landmarks_survive_exactly(self, tmp_path):
        seq = _sequence()
        path = tmp_path / "swing.npz"

        store.save_sequence(path, seq)
        loaded = store.load_sequence(path)

        np.testing.assert_array_equal(loaded.landmarks, seq.landmarks)

    def test_timestamps_survive_exactly(self, tmp_path):
        seq = _sequence()
        path = tmp_path / "swing.npz"

        store.save_sequence(path, seq)

        np.testing.assert_array_equal(store.load_sequence(path).times, seq.times)

    def test_fps_survives(self, tmp_path):
        seq = _sequence(fps=59.94)
        path = tmp_path / "swing.npz"

        store.save_sequence(path, seq)

        assert store.load_sequence(path).fps == pytest.approx(59.94)

    def test_source_path_survives(self, tmp_path):
        seq = _sequence()
        path = tmp_path / "swing.npz"

        store.save_sequence(path, seq)

        assert store.load_sequence(path).source == seq.source

    def test_float_precision_is_not_degraded(self, tmp_path):
        """float32 would quietly cost precision on normalised 0-1 coordinates."""
        seq = _sequence()
        path = tmp_path / "swing.npz"

        store.save_sequence(path, seq)

        assert store.load_sequence(path).landmarks.dtype == seq.landmarks.dtype

    def test_creates_parent_directories(self, tmp_path):
        seq = _sequence()
        path = tmp_path / "nested" / "deeper" / "swing.npz"

        store.save_sequence(path, seq)

        assert path.exists()


class TestLoadErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            store.load_sequence(tmp_path / "nope.npz")
