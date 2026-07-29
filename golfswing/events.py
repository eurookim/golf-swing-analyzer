"""Swing event detection: P1 address, P4 top, P7 impact, P10 finish.

Every metric in the project is measured at, or between, these frames, so this is
the highest-risk component.

Signals were chosen by plotting candidates across real footage rather than by
reasoning about landmark confidence — see docs/phase2_candidate_signals.png.
Smoothed wrist height gives a plateau -> peak -> trough -> peak shape in every
swing; torso speed gives one unambiguous spike at impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from golfswing.sequence import PoseSequence

LEFT_WRIST, RIGHT_WRIST = 15, 16
TORSO = (11, 12, 23, 24)  # shoulders + hips, the highest-confidence landmarks

# Upper body including the arms. The torso decelerates almost immediately after
# impact while the arms and club travel on for roughly another half second, so
# the finish must be judged on this set, not on TORSO alone.
UPPER_BODY = (11, 12, 13, 14, 15, 16, 23, 24)

# Impact must stand this far above the median to count as a swing at all.
MIN_PEAK_OVER_BASELINE = 3.0

# Address is the last frame still within this fraction of the backswing's rise.
ADDRESS_TOLERANCE = 0.05

# Finish is where speed first falls back to this fraction of the impact spike.
FINISH_TOLERANCE = 0.10


class NoSwingDetectedError(RuntimeError):
    """Raised when a clip contains no identifiable swing."""


@dataclass(frozen=True)
class SwingEvents:
    """Frame indices of the four key positions."""

    p1: int   # address
    p4: int   # top of backswing
    p7: int   # impact
    p10: int  # finish

    def as_dict(self) -> dict[str, int]:
        return {"P1": self.p1, "P4": self.p4, "P7": self.p7, "P10": self.p10}


def wrist_height(sequence: PoseSequence) -> np.ndarray:
    """Mean hand height per frame, with up as positive.

    Image y grows downward, so this is flipped — otherwise the top of the
    backswing would appear as a minimum.
    """
    y = sequence.landmarks[:, [LEFT_WRIST, RIGHT_WRIST], 1]
    return 1.0 - np.nanmean(y, axis=1)


def _mean_speed(sequence: PoseSequence, landmark_indices) -> np.ndarray:
    """Mean speed of the given landmarks, in normalised units per second.

    Divided by real elapsed time rather than frame count, so the values mean the
    same thing at 60fps and 240fps.
    """
    xy = sequence.landmarks[:, list(landmark_indices), :2]
    dt = np.gradient(sequence.times)
    per_frame = np.linalg.norm(np.gradient(xy, axis=0), axis=-1)
    return np.nanmean(per_frame, axis=1) / dt


def torso_speed(sequence: PoseSequence) -> np.ndarray:
    """Speed of shoulders and hips — the impact signal."""
    return _mean_speed(sequence, TORSO)


def upper_body_speed(sequence: PoseSequence) -> np.ndarray:
    """Speed of torso plus arms — the finish signal."""
    return _mean_speed(sequence, UPPER_BODY)


def detect_events(sequence: PoseSequence) -> SwingEvents:
    """Locate P1 / P4 / P7 / P10.

    Impact is found first because it is the single most unambiguous feature in
    the signal; everything else is located relative to it.
    """
    speed = torso_speed(sequence)
    height = wrist_height(sequence)

    if len(speed) < 5 or not np.any(np.isfinite(speed)):
        raise NoSwingDetectedError("too few usable frames")

    p7 = int(np.nanargmax(speed))
    peak = float(speed[p7])
    baseline = float(np.nanmedian(speed))

    if not np.isfinite(peak) or peak <= 0 or peak < MIN_PEAK_OVER_BASELINE * baseline:
        raise NoSwingDetectedError(
            f"no impact spike found (peak {peak:.4f} vs baseline {baseline:.4f}) — "
            "the clip may not contain a swing"
        )
    if p7 == 0:
        raise NoSwingDetectedError("impact detected on the first frame; clip starts mid-swing")

    # P4: highest hands BEFORE impact. Deliberately not the global maximum —
    # on real footage the follow-through often peaks higher than the backswing.
    p4 = int(np.nanargmax(height[:p7]))

    # P1: last frame still at address height, walking back from the top.
    pre = height[: p4 + 1]
    base = float(np.nanmin(pre))
    rise = float(height[p4]) - base
    if rise <= 0:
        p1 = 0
    else:
        at_address = np.where(pre <= base + ADDRESS_TOLERANCE * rise)[0]
        p1 = int(at_address[-1]) if len(at_address) else 0

    # P10: first frame after impact where the WHOLE upper body has settled.
    # Judging this on torso speed alone lands mid-follow-through, because the
    # torso stops at impact while the arms and club carry on.
    whole = upper_body_speed(sequence)
    whole_peak = float(np.nanmax(whole))
    whole_base = float(np.nanmedian(whole))
    settled = whole_base + FINISH_TOLERANCE * (whole_peak - whole_base)
    after = np.where(whole[p7 + 1:] <= settled)[0]
    p10 = int(p7 + 1 + after[0]) if len(after) else len(whole) - 1

    return SwingEvents(p1=p1, p4=p4, p7=p7, p10=p10)
