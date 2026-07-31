"""Tests for golfswing.naming — building conventional clip filenames."""

from datetime import date

import pytest

from golfswing import naming


class TestConventionalName:
    def test_builds_the_documented_shape(self):
        assert naming.conventional_name(
            date(2026, 8, 5), "dtl", "7iron", 1, suffix=".mov"
        ) == "2026-08-05_dtl_7iron_01.mov"

    def test_zero_pads_the_index(self):
        """Unpadded numbers sort 1, 10, 11, 2 — which scrambles a session."""
        name = naming.conventional_name(date(2026, 8, 5), "dtl", "7iron", 9)
        assert "_09" in name

    def test_keeps_three_digits_past_ninety_nine(self):
        name = naming.conventional_name(date(2026, 8, 5), "dtl", "7iron", 100)
        assert "_100" in name

    def test_appends_a_fault_tag(self):
        name = naming.conventional_name(
            date(2026, 8, 5), "dtl", "7iron", 3, fault="earlyext")
        assert name.startswith("2026-08-05_dtl_7iron_03_earlyext")

    def test_preserves_the_original_suffix(self):
        name = naming.conventional_name(
            date(2026, 8, 5), "dtl", "7iron", 1, suffix=".MP4")
        assert name.endswith(".mp4"), "suffix should be normalised to lowercase"

    def test_rejects_an_unknown_angle(self):
        with pytest.raises(ValueError, match="angle"):
            naming.conventional_name(date(2026, 8, 5), "sideways", "7iron", 1)

    def test_rejects_an_unknown_fault_tag(self):
        """A typo'd tag would be filed as a normal swing and quietly pollute
        the baseline it is supposed to be measured against."""
        with pytest.raises(ValueError, match="fault"):
            naming.conventional_name(
                date(2026, 8, 5), "dtl", "7iron", 1, fault="slice")

    def test_rejects_a_club_with_spaces_or_punctuation(self):
        with pytest.raises(ValueError, match="club"):
            naming.conventional_name(date(2026, 8, 5), "dtl", "7 iron", 1)


class TestFollowsConvention:
    def test_recognises_a_conventional_name(self):
        assert naming.follows_convention("2026-08-05_dtl_7iron_01") is True

    def test_recognises_one_with_a_fault_tag(self):
        assert naming.follows_convention("2026-08-05_dtl_7iron_10_posture") is True

    def test_rejects_a_camera_export(self):
        assert naming.follows_convention("IMG_4471") is False

    def test_rejects_an_unpadded_index(self):
        assert naming.follows_convention("2026-08-05_dtl_7iron_1") is False


class TestNextIndex:
    def test_starts_at_one_in_an_empty_directory(self, tmp_path):
        assert naming.next_index(tmp_path, date(2026, 8, 5), "dtl", "7iron") == 1

    def test_continues_after_existing_clips_of_the_same_session(self, tmp_path):
        for n in (1, 2, 3):
            (tmp_path / f"2026-08-05_dtl_7iron_{n:02d}.mov").write_bytes(b"")
        assert naming.next_index(tmp_path, date(2026, 8, 5), "dtl", "7iron") == 4

    def test_ignores_a_different_day_or_club(self, tmp_path):
        (tmp_path / "2026-08-05_dtl_driver_01.mov").write_bytes(b"")
        (tmp_path / "2026-07-01_dtl_7iron_09.mov").write_bytes(b"")
        assert naming.next_index(tmp_path, date(2026, 8, 5), "dtl", "7iron") == 1

    def test_counts_tagged_clips_in_the_same_session(self, tmp_path):
        (tmp_path / "2026-08-05_dtl_7iron_01_posture.mov").write_bytes(b"")
        assert naming.next_index(tmp_path, date(2026, 8, 5), "dtl", "7iron") == 2

    def test_fills_past_the_highest_rather_than_the_count(self, tmp_path):
        """Deleting a middle clip must not cause a collision."""
        (tmp_path / "2026-08-05_dtl_7iron_01.mov").write_bytes(b"")
        (tmp_path / "2026-08-05_dtl_7iron_07.mov").write_bytes(b"")
        assert naming.next_index(tmp_path, date(2026, 8, 5), "dtl", "7iron") == 8


class TestRenameToConvention:
    def _clip(self, tmp_path, name="IMG_4471.mov"):
        p = tmp_path / name
        p.write_bytes(b"video")
        return p

    def test_renames_the_file_on_disk(self, tmp_path):
        clip = self._clip(tmp_path)
        out = naming.rename_to_convention(
            clip, date(2026, 8, 5), "dtl", "7iron", 1)
        assert out.name == "2026-08-05_dtl_7iron_01.mov"
        assert out.exists() and not clip.exists()

    def test_keeps_the_original_bytes(self, tmp_path):
        clip = self._clip(tmp_path)
        out = naming.rename_to_convention(
            clip, date(2026, 8, 5), "dtl", "7iron", 1)
        assert out.read_bytes() == b"video"

    def test_refuses_to_overwrite_an_existing_clip(self, tmp_path):
        """Two clips landing on one name would silently destroy a swing."""
        self._clip(tmp_path, "2026-08-05_dtl_7iron_01.mov")
        clip = self._clip(tmp_path, "IMG_4471.mov")
        with pytest.raises(FileExistsError):
            naming.rename_to_convention(clip, date(2026, 8, 5), "dtl", "7iron", 1)
        assert clip.exists(), "source must survive a refused rename"

    def test_carries_the_suffix_across(self, tmp_path):
        clip = self._clip(tmp_path, "IMG_4471.MP4")
        out = naming.rename_to_convention(
            clip, date(2026, 8, 5), "dtl", "7iron", 2)
        assert out.suffix == ".mp4"

    def test_an_invalid_field_renames_nothing(self, tmp_path):
        clip = self._clip(tmp_path)
        with pytest.raises(ValueError):
            naming.rename_to_convention(clip, date(2026, 8, 5), "dtl", "7iron",
                                        1, fault="slice")
        assert clip.exists()
