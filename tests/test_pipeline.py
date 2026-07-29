"""Tests for golfswing.pipeline — the ingest -> pose -> smooth -> store chain."""

from pathlib import Path

import numpy as np
import pytest

from golfswing import pipeline, store

MODEL = Path("models/pose_landmarker_heavy.task")
REAL_CLIPS = sorted(Path("data/raw").glob("*.mov")) if Path("data/raw").is_dir() else []

requires_model = pytest.mark.skipif(
    not MODEL.exists(), reason="pose model not downloaded"
)
requires_footage = pytest.mark.skipif(
    not REAL_CLIPS, reason="no swing footage in data/raw"
)


@requires_model
class TestNoPersonDetected:
    def test_raises_rather_than_writing_an_all_nan_file(self, cfr_clip, tmp_path):
        """A clip with no golfer in it is an error, not an empty result.

        Writing an all-NaN cache would look like a successful run and fail
        confusingly three stages later.
        """
        with pytest.raises(pipeline.NoPoseDetectedError):
            pipeline.process_clip(cfr_clip, out_dir=tmp_path)

    def test_writes_nothing_when_it_fails(self, cfr_clip, tmp_path):
        with pytest.raises(pipeline.NoPoseDetectedError):
            pipeline.process_clip(cfr_clip, out_dir=tmp_path)
        assert list(tmp_path.glob("*.npz")) == []


@requires_model
@requires_footage
class TestRealClip:
    def test_writes_npz_named_after_the_clip(self, tmp_path):
        clip = REAL_CLIPS[0]
        out = pipeline.process_clip(clip, out_dir=tmp_path)
        assert out.name == f"{clip.stem}.npz"
        assert out.exists()

    def test_result_loads_back_with_matching_frame_count(self, tmp_path):
        out = pipeline.process_clip(REAL_CLIPS[0], out_dir=tmp_path)
        seq = store.load_sequence(out)
        assert seq.n_frames == len(seq.times)
        assert seq.n_frames > 0

    def test_stored_landmarks_are_smoothed(self, tmp_path):
        """The cache must hold filtered trajectories, not raw jittery ones."""
        from golfswing import pose

        clip = REAL_CLIPS[0]
        raw = pose.extract_sequence(clip)
        smoothed = store.load_sequence(pipeline.process_clip(clip, out_dir=tmp_path))

        assert not np.allclose(smoothed.landmarks[:, :, :3],
                               raw.landmarks[:, :, :3], equal_nan=True)

    def test_visibility_is_preserved_through_the_chain(self, tmp_path):
        from golfswing import pose

        clip = REAL_CLIPS[0]
        raw = pose.extract_sequence(clip)
        smoothed = store.load_sequence(pipeline.process_clip(clip, out_dir=tmp_path))

        np.testing.assert_array_equal(smoothed.landmarks[:, :, 3],
                                      raw.landmarks[:, :, 3])

    def test_reuses_cache_on_second_run(self, tmp_path):
        clip = REAL_CLIPS[0]
        first = pipeline.process_clip(clip, out_dir=tmp_path)
        mtime = first.stat().st_mtime_ns

        second = pipeline.process_clip(clip, out_dir=tmp_path)

        assert second == first
        assert second.stat().st_mtime_ns == mtime, "should not have re-processed"

    def test_force_reprocesses(self, tmp_path):
        clip = REAL_CLIPS[0]
        first = pipeline.process_clip(clip, out_dir=tmp_path)
        mtime = first.stat().st_mtime_ns

        again = pipeline.process_clip(clip, out_dir=tmp_path, force=True)

        assert again.stat().st_mtime_ns != mtime
