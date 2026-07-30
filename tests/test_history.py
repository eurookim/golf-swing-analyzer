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
