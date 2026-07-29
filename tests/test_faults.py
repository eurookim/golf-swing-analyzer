"""Tests for golfswing.faults — rules over metrics, ranked by severity."""

import numpy as np
import pytest

from golfswing import faults
from golfswing.events import SwingEvents
from golfswing.metrics import SwingMetrics

EVENTS = SwingEvents(p1=0, p4=30, p7=45, p10=70)


def _metrics(**overrides) -> SwingMetrics:
    """A clean swing; override individual metrics to trip specific rules."""
    base = dict(
        spine_tilt_p1=34.0, spine_tilt_p4=34.0, spine_tilt_p7=34.0,
        posture_change=0.0, hip_depth_change=0.0,
        head_rise_p4=0.0, head_rise_p7=0.0, head_depth_p7=0.0,
        knee_extension_change=0.0, tempo_ratio=3.0,
    )
    base.update(overrides)
    return SwingMetrics(events=EVENTS, **base)


THRESHOLDS = {
    "default": {
        "loss_of_posture": 8.0,
        "head_lift": 0.03,
        "early_extension": 0.04,
        "knee_straightening": 15.0,
        "quick_tempo": 2.2,
    }
}


class TestFiring:
    def test_clean_swing_fires_nothing(self):
        results = faults.evaluate(_metrics(), thresholds=THRESHOLDS)
        assert [f for f in results if f.fired] == []

    def test_magnitude_rule_fires_in_either_direction(self):
        """Losing posture and diving into it are both faults."""
        up = faults.find(faults.evaluate(_metrics(posture_change=12.0), thresholds=THRESHOLDS),
                         "loss_of_posture")
        down = faults.find(faults.evaluate(_metrics(posture_change=-12.0), thresholds=THRESHOLDS),
                           "loss_of_posture")
        assert up.fired and down.fired

    def test_above_rule_does_not_fire_on_the_opposite_sign(self):
        """Hips moving AWAY from the ball is not early extension."""
        result = faults.find(
            faults.evaluate(_metrics(hip_depth_change=-0.20), thresholds=THRESHOLDS),
            "early_extension",
        )
        assert not result.fired

    def test_below_rule_fires_when_the_value_is_too_small(self):
        """Quick tempo is a LOW ratio, unlike every other rule."""
        quick = faults.find(faults.evaluate(_metrics(tempo_ratio=1.8), thresholds=THRESHOLDS),
                            "quick_tempo")
        normal = faults.find(faults.evaluate(_metrics(tempo_ratio=3.0), thresholds=THRESHOLDS),
                             "quick_tempo")
        assert quick.fired and not normal.fired

    def test_exactly_at_the_threshold_does_not_fire(self):
        result = faults.find(
            faults.evaluate(_metrics(posture_change=8.0), thresholds=THRESHOLDS),
            "loss_of_posture",
        )
        assert not result.fired


class TestUnmeasurable:
    """A metric that could not be computed must not read as 'no fault'."""

    def test_nan_metric_does_not_fire(self):
        result = faults.find(
            faults.evaluate(_metrics(posture_change=float("nan")), thresholds=THRESHOLDS),
            "loss_of_posture",
        )
        assert not result.fired

    def test_nan_metric_is_marked_unmeasurable(self):
        result = faults.find(
            faults.evaluate(_metrics(posture_change=float("nan")), thresholds=THRESHOLDS),
            "loss_of_posture",
        )
        assert result.measurable is False

    def test_measured_metric_is_marked_measurable(self):
        result = faults.find(
            faults.evaluate(_metrics(posture_change=2.0), thresholds=THRESHOLDS),
            "loss_of_posture",
        )
        assert result.measurable is True


class TestSeverity:
    """Severity is 'fraction over the limit', so faults in different units
    (degrees vs torso-length ratios) can be ranked against each other."""

    def test_zero_when_not_fired(self):
        result = faults.find(faults.evaluate(_metrics(), thresholds=THRESHOLDS),
                             "loss_of_posture")
        assert result.severity == 0.0

    def test_doubles_the_threshold_gives_severity_one(self):
        result = faults.find(
            faults.evaluate(_metrics(posture_change=16.0), thresholds=THRESHOLDS),
            "loss_of_posture",
        )
        assert result.severity == pytest.approx(1.0)

    def test_comparable_across_different_units(self):
        """12 degrees on an 8 degree limit and 0.06 on a 0.04 limit are both
        50% over, so they must rank equal despite unrelated units."""
        results = faults.evaluate(
            _metrics(posture_change=12.0, hip_depth_change=0.06), thresholds=THRESHOLDS
        )
        posture = faults.find(results, "loss_of_posture")
        extension = faults.find(results, "early_extension")
        assert posture.severity == pytest.approx(extension.severity, abs=1e-9)


class TestRanking:
    """Design principle: show ONE fault, collapse the rest.

    Six faults displayed helps nobody — an amateur can only work on one thing,
    and the wrong one wastes a month.
    """

    def test_results_are_sorted_by_severity_descending(self):
        results = faults.evaluate(
            _metrics(posture_change=10.0, hip_depth_change=0.20, tempo_ratio=1.5),
            thresholds=THRESHOLDS,
        )
        fired = [f.severity for f in results if f.fired]
        assert fired == sorted(fired, reverse=True)

    def test_primary_is_the_worst_fault(self):
        results = faults.evaluate(
            _metrics(posture_change=10.0, hip_depth_change=0.20), thresholds=THRESHOLDS
        )
        assert faults.primary(results).name == "early_extension"

    def test_primary_is_none_on_a_clean_swing(self):
        assert faults.primary(faults.evaluate(_metrics(), thresholds=THRESHOLDS)) is None

    def test_unmeasurable_faults_never_become_primary(self):
        results = faults.evaluate(
            _metrics(posture_change=float("nan"), hip_depth_change=0.06),
            thresholds=THRESHOLDS,
        )
        assert faults.primary(results).name == "early_extension"


class TestPerClubThresholds:
    """A driver's extra spine tilt is correct technique, not a posture fault."""

    def test_club_section_overrides_the_default(self):
        thresholds = {"default": {"loss_of_posture": 8.0},
                      "driver": {"loss_of_posture": 14.0}}
        m = _metrics(posture_change=10.0)

        with_iron = faults.find(faults.evaluate(m, thresholds, club="7iron"),
                                "loss_of_posture")
        with_driver = faults.find(faults.evaluate(m, thresholds, club="driver"),
                                  "loss_of_posture")

        assert with_iron.fired
        assert not with_driver.fired

    def test_falls_back_to_default_for_an_unlisted_club(self):
        thresholds = {"default": {"loss_of_posture": 8.0},
                      "driver": {"loss_of_posture": 14.0}}
        result = faults.find(
            faults.evaluate(_metrics(posture_change=10.0), thresholds, club="5wood"),
            "loss_of_posture",
        )
        assert result.fired

    def test_missing_threshold_makes_the_rule_unmeasurable(self):
        result = faults.find(
            faults.evaluate(_metrics(posture_change=99.0), thresholds={"default": {}}),
            "loss_of_posture",
        )
        assert not result.fired and result.measurable is False


class TestThresholdLoading:
    def test_loads_a_yaml_file(self, tmp_path):
        path = tmp_path / "thresholds.yaml"
        path.write_text("default:\n  loss_of_posture: 8.0\n")
        assert faults.load_thresholds(path)["default"]["loss_of_posture"] == 8.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            faults.load_thresholds(tmp_path / "nope.yaml")

    def test_rejects_a_file_with_no_default_section(self, tmp_path):
        """Without defaults, an unlisted club would silently score nothing."""
        path = tmp_path / "thresholds.yaml"
        path.write_text("driver:\n  loss_of_posture: 14.0\n")
        with pytest.raises(ValueError):
            faults.load_thresholds(path)

    def test_the_shipped_file_covers_every_rule(self):
        """A rule with no default threshold is permanently unmeasurable."""
        shipped = faults.load_thresholds()
        for rule in faults.RULES:
            assert rule.name in shipped["default"], f"no default for {rule.name}"
