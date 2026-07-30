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
from scipy.signal import find_peaks

from golfswing.sequence import PoseSequence

LEFT_WRIST, RIGHT_WRIST = 15, 16
TORSO = (11, 12, 23, 24)  # shoulders + hips, the highest-confidence landmarks

# Upper body including the arms. The torso decelerates almost immediately after
# impact while the arms and club travel on for roughly another half second, so
# the finish must be judged on this set, not on TORSO alone.
UPPER_BODY = (11, 12, 13, 14, 15, 16, 23, 24)

# Impact must stand this far above the median to count as a swing at all.
MIN_PEAK_OVER_BASELINE = 3.0

# How far either side of the coarse speed peak to search for true contact.
IMPACT_REFINE_SECONDS = 0.12

# A downswing is physically bounded. Measured across 15 hand-labeled swings:
# 0.283-0.358s. This window is generous around that, and its job is to stop the
# search reaching a follow-through spike — 5 of 20 real clips had one LARGER
# than the impact spike, which a global maximum picked, yielding 0.67-0.90s
# "downswings".
DOWNSWING_MIN_SECONDS = 0.15
DOWNSWING_MAX_SECONDS = 0.50

# A wrist-height peak counts as the top if it rises this fraction of the
# swing's full height range above its surroundings.
TOP_PROMINENCE_FRACTION = 0.25

# Hands are "still" below this fraction of the backswing's typical hand speed.
ADDRESS_MOTION_FRACTION = 0.15

# The backswing lasts about a second, so this window captures real hand motion
# without being diluted by however long the golfer stood still beforehand.
BACKSWING_WINDOW_SECONDS = 1.5

# Stillness must persist this long to count as address rather than a waggle
# pause. Specified in seconds so it behaves the same at 60fps and 240fps.
ADDRESS_STILL_SECONDS = 0.05

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


def _safe_intervals(times: np.ndarray) -> np.ndarray:
    """Per-frame time deltas, floored so a decoder artifact cannot explode them.

    Decoders emit a near-zero interval at the head of a file — measured at
    0.417ms against a median of 8.333ms on real 120fps clips, 20x too small.
    Dividing a normal displacement by it fabricates a speed spike larger than
    impact, which made two clips fail outright with "impact detected on the
    first frame."

    Floored at half the median: wide enough to pass genuine variation, tight
    enough to kill an order-of-magnitude artifact.
    """
    dt = np.gradient(times)
    median = float(np.median(dt))
    if not np.isfinite(median) or median <= 0:
        return np.where(dt > 0, dt, 1.0)
    return np.maximum(dt, median * 0.5)


def _mean_speed(sequence: PoseSequence, landmark_indices) -> np.ndarray:
    """Mean speed of the given landmarks, in normalised units per second.

    Divided by real elapsed time rather than frame count, so the values mean the
    same thing at 60fps and 240fps.
    """
    xy = sequence.landmarks[:, list(landmark_indices), :2]
    per_frame = np.linalg.norm(np.gradient(xy, axis=0), axis=-1)
    return np.nanmean(per_frame, axis=1) / _safe_intervals(sequence.times)


def torso_speed(sequence: PoseSequence) -> np.ndarray:
    """Speed of shoulders and hips — the impact signal."""
    return _mean_speed(sequence, TORSO)


def upper_body_speed(sequence: PoseSequence) -> np.ndarray:
    """Speed of torso plus arms — the finish signal."""
    return _mean_speed(sequence, UPPER_BODY)


def hand_speed(sequence: PoseSequence) -> np.ndarray:
    """Speed of the hands — the address signal."""
    return _mean_speed(sequence, (LEFT_WRIST, RIGHT_WRIST))


def shoulder_width(sequence: PoseSequence) -> np.ndarray:
    """Shoulder separation in frame, normalised by torso length.

    The impact signal. From down-the-line the torso squares to the target line
    at contact, so the shoulder line points at the camera and foreshortens to
    its narrowest. Normalised by torso length so it survives the golfer
    standing nearer or further from the camera.
    """
    lm = sequence.landmarks
    shoulders = np.linalg.norm(lm[:, 11, :2] - lm[:, 12, :2], axis=1)
    mid_shoulder = np.nanmean(lm[:, [11, 12], :2], axis=1)
    mid_hip = np.nanmean(lm[:, [23, 24], :2], axis=1)
    torso = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(torso > 0, shoulders / torso, np.nan)


def _find_top(height: np.ndarray) -> int | None:
    """First prominent peak in hand height — the top of the backswing.

    Not the global maximum: on real footage the follow-through peaks higher
    than the backswing on most swings, so a global search finds the wrong one.
    Prominence is measured against the swing's own height range, so it scales
    with how big the swing is rather than needing an absolute threshold.
    """
    finite = height[np.isfinite(height)]
    if finite.size == 0:
        return None
    span = float(finite.max() - finite.min())
    if span <= 0:
        return None
    peaks, _ = find_peaks(height, prominence=span * TOP_PROMINENCE_FRACTION)
    return int(peaks[0]) if len(peaks) else None


def _refine_impact(sequence: PoseSequence, coarse: int) -> int:
    """Sharpen a coarse impact estimate using the hand-height minimum.

    The hands reach the bottom of the swing arc AT contact — a geometric
    coincidence rather than a rotational one, which is why it does not lag.

    Measured against 15 hand-labeled 120fps clips:

        wrist height MIN     mean -0.27  mean_abs 0.40  max 1
        torso speed MAX      mean +3.40  mean_abs 3.40  max 6
        shoulder width MIN   mean +0.87  mean_abs 4.47  max 14

    This reverses an earlier choice. At 60fps shoulder-width minimum scored
    better (1.0 vs 1.67), so it was picked — but 60fps was too coarse to show
    that it is simply noisy. Doubling the frame rate exposed it, and also
    revealed the torso-speed lag as a consistent ~26ms rather than a rounding
    error. A signal with no bias beats a biased signal plus a correction.
    """
    height = wrist_height(sequence)
    fps = sequence.fps if sequence.fps > 0 else 60.0
    half = max(2, int(round(IMPACT_REFINE_SECONDS * fps)))

    lo = max(0, coarse - half)
    hi = min(len(height), coarse + half + 1)
    window = height[lo:hi]
    if not np.any(np.isfinite(window)):
        return coarse
    return int(lo + np.nanargmin(window))


def _find_address(sequence: PoseSequence, p4: int) -> int:
    """Last frame before the top at which the hands are still.

    Scans backward from the top so an earlier waggle is never reached.
    """
    speed = hand_speed(sequence)[: p4 + 1]
    if len(speed) < 2:
        return 0

    fps_for_window = sequence.fps if sequence.fps > 0 else 60.0
    # Derive the "moving" scale from a fixed window before the top, NOT the
    # whole clip. Taken over everything, a long static setup floods the
    # distribution with near-zero frames, drags the percentile down, and drops
    # the threshold until faint motion counts as moving — so how long the
    # golfer stood there would change where address is detected.
    window_start = max(0, p4 - int(round(BACKSWING_WINDOW_SECONDS * fps_for_window)))
    scale = float(np.nanpercentile(speed[window_start:], 75))
    if not np.isfinite(scale) or scale <= 0:
        return 0
    threshold = ADDRESS_MOTION_FRACTION * scale

    fps = sequence.fps if sequence.fps > 0 else 60.0
    min_still = max(3, int(round(ADDRESS_STILL_SECONDS * fps)))

    still = speed <= threshold
    run = 0
    for i in range(len(speed) - 1, -1, -1):
        if still[i]:
            run += 1
            if run >= min_still:
                # Last frame of this still stretch — the one nearest the top.
                return int(i + run - 1)
        else:
            run = 0
    return 0


def detect_events(sequence: PoseSequence) -> SwingEvents:
    """Locate P1 / P4 / P7 / P10.

    Impact is found first because it is the single most unambiguous feature in
    the signal; everything else is located relative to it.
    """
    speed = torso_speed(sequence)
    height = wrist_height(sequence)

    if len(speed) < 5 or not np.any(np.isfinite(speed)):
        raise NoSwingDetectedError("too few usable frames")

    peak = float(np.nanmax(speed))
    baseline = float(np.nanmedian(speed))
    if not np.isfinite(peak) or peak <= 0 or peak < MIN_PEAK_OVER_BASELINE * baseline:
        raise NoSwingDetectedError(
            f"no impact spike found (peak {peak:.4f} vs baseline {baseline:.4f}) — "
            "the clip may not contain a swing"
        )

    # P4 first, independently of impact. The top is the FIRST prominent peak in
    # wrist height — the follow-through frequently peaks higher, so a global
    # maximum finds the wrong one. Verified against 20 real clips.
    p4 = _find_top(height)
    if p4 is None:
        raise NoSwingDetectedError("no backswing peak found in hand height")

    # P7 inside a physically plausible downswing window after the top. Without
    # the bound, a follow-through spike larger than impact wins the global
    # maximum — which happened on 5 of 20 real clips.
    fps = sequence.fps if sequence.fps > 0 else 60.0
    lo = p4 + max(1, int(round(DOWNSWING_MIN_SECONDS * fps)))
    hi = min(len(speed), p4 + int(round(DOWNSWING_MAX_SECONDS * fps)) + 1)
    if lo >= hi:
        raise NoSwingDetectedError("clip ends before a downswing could complete")
    coarse = int(lo + np.nanargmax(speed[lo:hi]))

    # The speed peak lags contact — the body keeps accelerating through release.
    # The shoulder line bottoms AT contact, so refine against that.
    p7 = _refine_impact(sequence, coarse)

    # P1: the last frame the hands are still, walking BACK from the top.
    #
    # Keyed on hand MOTION, not hand height. The club travels back before it
    # travels up, so height stays flat through the early takeaway and a
    # height-based rule lands well inside the backswing with the club already
    # lifted — verified on all three clips.
    #
    # Walking backward (rather than forward from frame 0) is what makes a
    # pre-shot waggle harmless: the search stops at the first sustained still
    # period before the top and never reaches the waggle.
    p1 = _find_address(sequence, p4)

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
