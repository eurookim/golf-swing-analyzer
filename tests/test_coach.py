"""Tests for golfswing.coach — swing metrics into coaching context."""

import pytest

from golfswing import coach


def _row(clip, *, tag=None, club="7iron", **metrics):
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


class TestStandings:
    def test_reports_the_swings_own_value(self):
        rows = [_row(f"c{i}", hip_depth_change=0.1 * i) for i in range(8)]

        found = coach.standings(rows, "c3")

        hip = next(s for s in found if s.metric == "hip_depth_change")
        assert hip.value == pytest.approx(0.3)

    def test_ranks_the_swing_against_the_others(self):
        """The whole point: position in YOUR distribution, not a fixed threshold."""
        rows = [_row(f"c{i}", hip_depth_change=float(i)) for i in range(10)]

        hip = next(s for s in coach.standings(rows, "c9")
                   if s.metric == "hip_depth_change")

        assert hip.rank == 1, "largest value should rank first"
        assert hip.n_peers == 10

    def test_median_comes_from_the_other_swings(self):
        rows = [_row(f"c{i}", tempo_ratio=float(i)) for i in range(11)]

        tempo = next(s for s in coach.standings(rows, "c0")
                     if s.metric == "tempo_ratio")

        assert tempo.median == pytest.approx(5.0)

    def test_deliberate_faults_are_excluded_from_the_baseline(self):
        """A swing you botched on purpose is not part of your normal range."""
        rows = [_row(f"c{i}", hip_depth_change=0.20) for i in range(6)]
        rows.append(_row("bad", tag="early_extension", hip_depth_change=9.9))

        hip = next(s for s in coach.standings(rows, "c0")
                   if s.metric == "hip_depth_change")

        assert hip.n_peers == 6, "tagged clip must not inflate the baseline"

    def test_a_tagged_swing_can_still_be_examined(self):
        rows = [_row(f"c{i}", hip_depth_change=0.20) for i in range(6)]
        rows.append(_row("bad", tag="early_extension", hip_depth_change=9.9))

        hip = next(s for s in coach.standings(rows, "bad")
                   if s.metric == "hip_depth_change")

        assert hip.value == pytest.approx(9.9)

    def test_rank_never_exceeds_the_number_of_swings_compared(self):
        """A tagged subject is not in its own baseline, so a naive rank of
        'peers above me, plus one' can exceed the peer count — producing the
        nonsense '17 of 16'. The subject has to be counted in the denominator.
        """
        rows = [_row(f"c{i}", hip_depth_change=0.20) for i in range(6)]
        rows.append(_row("bad", tag="early_extension", hip_depth_change=0.01))

        hip = next(s for s in coach.standings(rows, "bad")
                   if s.metric == "hip_depth_change")

        assert hip.rank == 7 and hip.n_peers == 7, (
            f"got rank {hip.rank} of {hip.n_peers}"
        )

    def test_an_untagged_subject_is_not_double_counted(self):
        rows = [_row(f"c{i}", hip_depth_change=float(i)) for i in range(6)]

        hip = next(s for s in coach.standings(rows, "c5")
                   if s.metric == "hip_depth_change")

        assert hip.rank == 1 and hip.n_peers == 6

    def test_too_few_peers_means_no_ranking(self):
        """Three swings cannot establish a personal range — say so, don't imply it."""
        rows = [_row(f"c{i}", hip_depth_change=0.1 * i) for i in range(3)]

        hip = next(s for s in coach.standings(rows, "c1")
                   if s.metric == "hip_depth_change")

        assert hip.rank is None and hip.median is None
        assert hip.value == pytest.approx(0.1)

    def test_unmeasured_metrics_are_skipped(self):
        rows = [_row(f"c{i}", tempo_ratio=None) for i in range(8)]

        metrics = {s.metric for s in coach.standings(rows, "c0")}

        assert "tempo_ratio" not in metrics

    def test_unknown_clip_raises(self):
        with pytest.raises(KeyError):
            coach.standings([_row("a")], "nope")


class TestPrompt:
    def _prompt(self, rows, clip):
        return coach.build_prompt(coach.standings(rows, clip), clip)

    def test_names_the_clip(self):
        rows = [_row(f"c{i}") for i in range(8)]
        assert "c2" in self._prompt(rows, "c2")

    def test_states_that_thresholds_are_uncalibrated(self):
        """The model must not invent a verdict the data cannot support."""
        rows = [_row(f"c{i}") for i in range(8)]
        assert "not calibrated" in self._prompt(rows, "c0").lower()

    def test_includes_the_ranking_when_available(self):
        rows = [_row(f"c{i}", hip_depth_change=float(i)) for i in range(10)]
        assert "1 of 10" in self._prompt(rows, "c9")

    def test_says_so_when_the_sample_is_too_small(self):
        rows = [_row(f"c{i}") for i in range(3)]
        assert "too few" in self._prompt(rows, "c0").lower()

    def test_explains_what_each_metric_means(self):
        """Raw metric names are meaningless without the DTL geometry."""
        rows = [_row(f"c{i}") for i in range(8)]
        assert "toward the ball" in self._prompt(rows, "c0").lower()


class TestAvailability:
    def test_reports_unavailable_without_credentials(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(coach, "_has_cli_profile", lambda: False)

        assert coach.available() is False

    def test_reports_available_with_an_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        assert coach.available() is True
