"""Body measurements at the key swing frames — down-the-line only.

Everything is measured at, or between, P1/P4/P7. Two rules govern the whole
module:

1. **Normalise by torso length.** The normaliser must be club-invariant AND
   viewpoint-invariant. Stance width fails the first (a driver stance is wider
   than a 7-iron). Shoulder width fails the second: from down-the-line the
   shoulder line points nearly at the camera at address and foreshortens to
   almost nothing — measured 0.0087-0.0329 across four clips of the same body,
   a 3.8x spread driven by camera placement rather than anatomy. Torso length is
   a vertical measure, so rotation about the body's vertical axis leaves it
   alone.

2. **Return NaN rather than a plausible number.** A missing landmark must not
   silently become a real-looking angle.
"""

from __future__ import annotations

import numpy as np

from golfswing.sequence import PoseSequence

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
NOSE = 0


def _midpoint(sequence: PoseSequence, frame: int, a: int, b: int) -> np.ndarray:
    """Midpoint of two landmarks in normalised image coordinates."""
    pair = sequence.landmarks[frame, [a, b], :2]
    if not np.all(np.isfinite(pair)):
        return np.array([np.nan, np.nan])
    return pair.mean(axis=0)


def body_scale(sequence: PoseSequence, frame: int) -> float:
    """Hip-to-shoulder distance — the normaliser for every length here.

    Deliberately not shoulder width: see the module docstring. This absorbs how
    far the golfer stands from the camera, which is the whole job, while staying
    stable under the body rotation that collapses the shoulder line.
    """
    mid_shoulder = _midpoint(sequence, frame, LEFT_SHOULDER, RIGHT_SHOULDER)
    mid_hip = _midpoint(sequence, frame, LEFT_HIP, RIGHT_HIP)
    if not (np.all(np.isfinite(mid_shoulder)) and np.all(np.isfinite(mid_hip))):
        return float("nan")
    return float(np.linalg.norm(mid_shoulder - mid_hip))


def spine_tilt(sequence: PoseSequence, frame: int) -> float:
    """Angle of the hip->shoulder vector from vertical, in degrees.

    Positive when the shoulders sit at greater x than the hips. Image y grows
    downward, so the vertical component is negated — without that an upright
    spine would read as 180 degrees.

    Being an angle, this is inherently scale-invariant: no normalisation needed.
    """
    mid_shoulder = _midpoint(sequence, frame, LEFT_SHOULDER, RIGHT_SHOULDER)
    mid_hip = _midpoint(sequence, frame, LEFT_HIP, RIGHT_HIP)
    if not (np.all(np.isfinite(mid_shoulder)) and np.all(np.isfinite(mid_hip))):
        return float("nan")

    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    if dx == 0.0 and dy == 0.0:
        return float("nan")
    return float(np.degrees(np.arctan2(dx, -dy)))


def ball_direction(sequence: PoseSequence, p1: int) -> float:
    """+1 or -1: which way the ball lies along image x.

    Derived rather than hardcoded. At address the golfer leans toward the ball,
    so the horizontal component of the hip->shoulder vector points at it. This
    keeps every "toward the ball" metric correct for either handedness and for
    a camera on either side, instead of silently reporting the opposite fault
    when the setup mirrors.
    """
    mid_shoulder = _midpoint(sequence, p1, LEFT_SHOULDER, RIGHT_SHOULDER)
    mid_hip = _midpoint(sequence, p1, LEFT_HIP, RIGHT_HIP)
    if not (np.all(np.isfinite(mid_shoulder)) and np.all(np.isfinite(mid_hip))):
        return float("nan")
    dx = mid_shoulder[0] - mid_hip[0]
    if dx == 0.0:
        return float("nan")
    return 1.0 if dx > 0 else -1.0


def hip_depth_change(sequence: PoseSequence, p1: int, p7: int) -> float:
    """Hip movement toward the ball between address and impact.

    Early extension. Expressed as a fraction of torso length, positive toward
    the ball.
    """
    direction = ball_direction(sequence, p1)
    width = body_scale(sequence, p1)
    hip_address = _midpoint(sequence, p1, LEFT_HIP, RIGHT_HIP)
    hip_impact = _midpoint(sequence, p7, LEFT_HIP, RIGHT_HIP)
    if not np.isfinite(direction) or not np.isfinite(width) or width == 0:
        return float("nan")
    if not (np.all(np.isfinite(hip_address)) and np.all(np.isfinite(hip_impact))):
        return float("nan")
    return float((hip_impact[0] - hip_address[0]) * direction / width)


def head_rise(sequence: PoseSequence, p1: int, other: int) -> float:
    """Vertical head movement from address, as a fraction of torso length.

    Positive is upward. Image y grows downward, so the difference is negated.
    """
    width = body_scale(sequence, p1)
    y_address = sequence.landmarks[p1, NOSE, 1]
    y_other = sequence.landmarks[other, NOSE, 1]
    if not np.isfinite(width) or width == 0:
        return float("nan")
    if not (np.isfinite(y_address) and np.isfinite(y_other)):
        return float("nan")
    return float((y_address - y_other) / width)


def tempo_ratio(sequence: PoseSequence, p1: int, p4: int, p7: int) -> float:
    """Backswing duration divided by downswing duration.

    Computed from real timestamps, never frame counts.

    **Report this relative to the golfer's own history, never against a tour
    benchmark.** P1 carries ~8 frames of error and the ratio compounds errors at
    both ends, so the absolute value is not trustworthy — but the error is a
    consistent offset, so change over time is.
    """
    backswing = float(sequence.times[p4] - sequence.times[p1])
    downswing = float(sequence.times[p7] - sequence.times[p4])
    if downswing <= 0 or backswing <= 0:
        return float("nan")
    return backswing / downswing


def posture_change(sequence: PoseSequence, p1: int, p4: int) -> float:
    """Spine tilt at the top minus spine tilt at address, in degrees.

    Signed on purpose: standing up out of posture and diving into it are
    different faults, and collapsing to a magnitude here would hide which.
    """
    return spine_tilt(sequence, p4) - spine_tilt(sequence, p1)
