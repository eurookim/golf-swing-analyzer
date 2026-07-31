"""Tests for golfswing.history — cached clips into history rows."""

import pytest

from golfswing import history


class TestParseClipName:
    def test_reads_date_angle_and_club(self):
        meta = history.parse_clip("2026-07-29_dtl_7iron_04")
        assert (meta.date, meta.angle, meta.club) == ("2026-07-29", "dtl", "7iron")

    def test_reads_the_fault_tag(self):
        assert history.parse_clip("2026-07-29_dtl_7iron_10_posture").fault_tag \
            == "loss_of_posture"

    def test_untagged_clip_has_no_fault(self):
        assert history.parse_clip("2026-07-29_dtl_7iron_04").fault_tag is None

    def test_unparseable_name_still_yields_a_record(self):
        """A clip named IMG_4471 should still be analysable, just unattributed."""
        meta = history.parse_clip("IMG_4471")
        assert meta.club is None and meta.angle is None
        assert meta.date is not None      # falls back rather than failing

    def test_unknown_fault_tag_raises(self):
        with pytest.raises(ValueError):
            history.parse_clip("2026-07-29_dtl_7iron_10_slice")


class TestPendingClips:
    """Which raw clips still need pose extraction. This only existed inside the
    CLI, so the app could not tell whether there was anything new to process."""

    def _raw(self, tmp_path, *names):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        for n in names:
            (raw / n).write_bytes(b"x")
        return raw

    def test_finds_a_clip_with_no_cached_keypoints(self, tmp_path):
        raw = self._raw(tmp_path, "a.mov")
        assert [p.name for p in history.pending_clips(raw, tmp_path / "p")] == ["a.mov"]

    def test_skips_a_clip_that_is_already_processed(self, tmp_path):
        raw = self._raw(tmp_path, "a.mov", "b.mov")
        processed = tmp_path / "processed"
        processed.mkdir()
        (processed / "a.npz").write_bytes(b"x")
        assert [p.name for p in history.pending_clips(raw, processed)] == ["b.mov"]

    def test_ignores_files_that_are_not_video(self, tmp_path):
        raw = self._raw(tmp_path, "a.mov", "notes.txt", ".DS_Store")
        assert [p.name for p in history.pending_clips(raw, tmp_path / "p")] == ["a.mov"]

    def test_is_case_insensitive_about_the_suffix(self, tmp_path):
        raw = self._raw(tmp_path, "a.MOV", "b.MP4")
        assert len(history.pending_clips(raw, tmp_path / "p")) == 2

    def test_a_missing_raw_directory_is_not_an_error(self, tmp_path):
        assert history.pending_clips(tmp_path / "nope", tmp_path / "p") == []

    def test_results_are_sorted_so_progress_is_predictable(self, tmp_path):
        raw = self._raw(tmp_path, "c.mov", "a.mov", "b.mov")
        assert [p.name for p in history.pending_clips(raw, tmp_path / "p")] \
            == ["a.mov", "b.mov", "c.mov"]
