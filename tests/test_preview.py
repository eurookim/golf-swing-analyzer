"""Tests for golfswing.preview — browser-playable copies of raw clips."""

import subprocess

import pytest

from golfswing import preview


def _make_clip(path, brand="qt", seconds=0.2):
    """Render a real tiny clip. ffmpeg picks the container brand from -f."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=32x32:rate=10:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-f", "mov" if brand == "qt" else "mp4", str(path)],
        check=True,
    )
    return path


def _brand(path) -> str:
    return path.read_bytes()[8:12].decode("ascii", "replace").strip()


class TestBrandDetection:
    def test_quicktime_brand_is_not_browser_playable(self, tmp_path):
        clip = _make_clip(tmp_path / "a.mov", brand="qt")
        assert preview.is_browser_playable(clip) is False

    def test_isom_brand_is_browser_playable(self, tmp_path):
        clip = _make_clip(tmp_path / "a.mp4", brand="isom")
        assert preview.is_browser_playable(clip) is True

    def test_a_missing_file_is_not_playable(self, tmp_path):
        assert preview.is_browser_playable(tmp_path / "nope.mp4") is False


class TestPlayableCopy:
    def test_converts_quicktime_to_an_isom_mp4(self, tmp_path):
        clip = _make_clip(tmp_path / "raw" / "swing.mov", brand="qt")

        out = preview.playable(clip, cache_dir=tmp_path / "cache")

        assert out.exists() and _brand(out) == "isom"

    def test_an_already_playable_file_is_returned_untouched(self, tmp_path):
        clip = _make_clip(tmp_path / "raw" / "swing.mp4", brand="isom")

        assert preview.playable(clip, cache_dir=tmp_path / "cache") == clip

    def test_reuses_the_cached_copy(self, tmp_path):
        """Remuxing on every page render would make the app feel broken."""
        clip = _make_clip(tmp_path / "raw" / "swing.mov", brand="qt")
        cache = tmp_path / "cache"

        first = preview.playable(clip, cache_dir=cache)
        stamp = first.stat().st_mtime_ns
        second = preview.playable(clip, cache_dir=cache)

        assert second == first
        assert second.stat().st_mtime_ns == stamp, "should not have rewritten"

    def test_regenerates_when_the_source_changes(self, tmp_path):
        clip = _make_clip(tmp_path / "raw" / "swing.mov", brand="qt")
        cache = tmp_path / "cache"
        first = preview.playable(clip, cache_dir=cache)
        stamp = first.stat().st_mtime_ns

        # Re-record the same path with different content.
        _make_clip(tmp_path / "raw" / "swing.mov", brand="qt", seconds=0.4)
        second = preview.playable(clip, cache_dir=cache)

        assert second.stat().st_mtime_ns != stamp

    def test_a_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preview.playable(tmp_path / "nope.mov", cache_dir=tmp_path / "c")
