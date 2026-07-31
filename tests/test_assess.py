"""Tests for calibrate.assess — per-rule calibration status.

This logic lived inside scripts/calibrate_faults.py, so the app could not show
it. Extracted here so the script and the UI share one implementation.
"""

import pytest

from golfswing import calibrate, faults


def _row(clip, *, club="7iron", tag=None, **metrics):
    base = dict(
        clip=clip, club=club, date="2026-07-29", fault_tag=tag,
        p1=10, p4=100, p7=140, p10=180, fps=120.0,
        posture_change=0.0, hip_depth_change=0.20, head_rise_p4=0.0,
        head_rise_p7=0.0, head_depth_p7=0.0, knee_extension_change=20.0,
        tempo_ratio=3.0, spine_tilt_p1=39.0, spine_tilt_p4=39.0,
        spine_tilt_p7=35.0,
    )
    base.update(metrics)
    return base


def _find(assessments, name):
    return next(a for a in assessments if a.rule == name)


class TestAssess:
    def test_reports_one_entry_per_rule(self):
        rows = [_row(f"c{i}") for i in range(6)]
        assert {a.rule for a in calibrate.assess(rows)} == {r.name for r in faults.RULES}

    def test_a_rule_with_no_example_cannot_be_calibrated(self):
        rows = [_row(f"c{i}") for i in range(6)]
        found = _find(calibrate.assess(rows), "knee_straightening")
        assert found.status == "no_example"
        assert found.suggested is None

    def test_a_cleanly_separated_fault_is_ready(self):
        """Normals tightly clustered, the deliberate fault far outside."""
        rows = [_row(f"c{i}", head_rise_p4=0.01) for i in range(6)]
        rows.append(_row("bad", tag="head_lift", head_rise_p4=0.90))

        found = _find(calibrate.assess(rows), "head_lift")

        assert found.status == "ready"
        assert found.suggested is not None
        assert 0.01 < found.suggested <= 0.90

    def test_a_fault_inside_the_normal_range_is_not_separable(self):
        """What actually happened on the 2026-07-29 session."""
        rows = [_row(f"c{i}", hip_depth_change=0.10 + 0.02 * i) for i in range(6)]
        rows.append(_row("bad", tag="early_extension", hip_depth_change=0.14))

        found = _find(calibrate.assess(rows), "early_extension")

        assert found.status == "not_separable"

    def test_it_carries_the_ranges_for_display(self):
        rows = [_row(f"c{i}", head_rise_p4=0.01 * i) for i in range(6)]
        rows.append(_row("bad", tag="head_lift", head_rise_p4=0.9))
        found = _find(calibrate.assess(rows), "head_lift")
        assert found.normal_low is not None and found.normal_high is not None
        assert found.fault_values == [0.9]

    def test_only_the_requested_club_is_used(self):
        rows = [_row(f"i{i}", club="7iron", head_rise_p4=0.01) for i in range(6)]
        rows += [_row(f"d{i}", club="driver", head_rise_p4=9.0) for i in range(6)]

        found = _find(calibrate.assess(rows, club="7iron"), "head_lift")

        assert found.n_normal == 6

    def test_a_tagged_clip_never_counts_as_normal_for_its_own_rule(self):
        rows = [_row(f"c{i}", head_rise_p4=0.01) for i in range(6)]
        rows.append(_row("bad", tag="head_lift", head_rise_p4=0.9))
        found = _find(calibrate.assess(rows), "head_lift")
        assert found.n_normal == 6

    def test_no_swings_at_all_yields_no_example_everywhere(self):
        for found in calibrate.assess([]):
            assert found.status == "no_example"
