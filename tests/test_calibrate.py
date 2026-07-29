"""Tests for golfswing.calibrate — turning tagged clips into real thresholds."""

import pytest

from golfswing import calibrate


class TestFaultTagFromFilename:
    """The filename tag is the expected answer, so parsing it is load-bearing."""

    def test_reads_a_tagged_clip(self):
        assert calibrate.fault_tag("2026-08-02_dtl_7iron_11_posture") == "loss_of_posture"

    def test_untagged_clip_is_a_normal_swing(self):
        assert calibrate.fault_tag("2026-08-02_dtl_7iron_03") is None

    def test_maps_every_shorthand_tag_to_a_real_rule(self):
        from golfswing.faults import RULES
        rule_names = {r.name for r in RULES}
        for shorthand, rule in calibrate.TAG_TO_RULE.items():
            assert rule in rule_names, f"{shorthand} maps to unknown rule {rule}"

    def test_unknown_tag_raises_rather_than_silently_ignoring(self):
        """A typo'd tag would otherwise become a normal swing and quietly
        corrupt the calibration."""
        with pytest.raises(ValueError):
            calibrate.fault_tag("2026-08-02_dtl_7iron_11_slice")


class TestScoreThreshold:
    """How a candidate threshold performs against tagged ground truth."""

    def test_perfect_separation(self):
        score = calibrate.score(normal=[1.0, 2.0], fault=[9.0, 10.0],
                                threshold=5.0, comparison="above")
        assert (score.true_positive, score.false_negative) == (2, 0)
        assert (score.true_negative, score.false_positive) == (2, 0)

    def test_threshold_too_low_produces_false_positives(self):
        score = calibrate.score(normal=[1.0, 2.0], fault=[9.0],
                                threshold=0.5, comparison="above")
        assert score.false_positive == 2

    def test_threshold_too_high_misses_real_faults(self):
        score = calibrate.score(normal=[1.0], fault=[9.0, 10.0],
                                threshold=20.0, comparison="above")
        assert score.false_negative == 2

    def test_below_comparison_inverts_the_logic(self):
        """Quick tempo fires on a LOW value."""
        score = calibrate.score(normal=[3.0, 4.0], fault=[1.5],
                                threshold=2.2, comparison="below")
        assert score.true_positive == 1 and score.false_positive == 0

    def test_magnitude_comparison_uses_absolute_value(self):
        score = calibrate.score(normal=[1.0], fault=[-12.0, 12.0],
                                threshold=8.0, comparison="magnitude")
        assert score.true_positive == 2


class TestSuggestThreshold:
    def test_lands_between_cleanly_separated_distributions(self):
        got = calibrate.suggest(normal=[1.0, 2.0, 3.0], fault=[9.0, 10.0],
                                comparison="above")
        assert 3.0 < got < 9.0

    def test_prefers_the_midpoint_of_the_gap(self):
        got = calibrate.suggest(normal=[2.0], fault=[8.0], comparison="above")
        assert got == pytest.approx(5.0)

    def test_below_comparison_lands_between_too(self):
        got = calibrate.suggest(normal=[3.0, 4.0], fault=[1.0, 1.5],
                                comparison="below")
        assert 1.5 < got < 3.0

    def test_overlapping_distributions_minimise_total_error(self):
        """No perfect split exists, so pick the fewest misclassifications."""
        got = calibrate.suggest(normal=[1.0, 2.0, 8.0], fault=[3.0, 9.0, 10.0],
                                comparison="above")
        score = calibrate.score(normal=[1.0, 2.0, 8.0], fault=[3.0, 9.0, 10.0],
                                threshold=got, comparison="above")
        assert score.false_positive + score.false_negative <= 2

    def test_nan_when_there_are_no_fault_examples(self):
        """Without a known-positive a threshold is unfalsifiable — the whole
        reason the capture brief insists on deliberate faults."""
        import numpy as np
        assert np.isnan(calibrate.suggest(normal=[1.0, 2.0], fault=[],
                                          comparison="above"))

    def test_nan_when_there_are_no_normal_examples(self):
        import numpy as np
        assert np.isnan(calibrate.suggest(normal=[], fault=[9.0],
                                          comparison="above"))

    def test_ignores_nan_values_in_either_group(self):
        got = calibrate.suggest(normal=[1.0, float("nan")], fault=[9.0],
                                comparison="above")
        assert got == pytest.approx(5.0)
