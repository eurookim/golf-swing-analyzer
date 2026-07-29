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

from dataclasses import dataclass, fields

import numpy as np

from golfswing.events import SwingEvents
from golfswing.sequence import PoseSequence

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
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


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at b, in degrees. 180 means a-b-c are collinear."""
    if not all(np.all(np.isfinite(p)) for p in (a, b, c)):
        return float("nan")
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na == 0 or nc == 0:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def knee_angle(sequence: PoseSequence, frame: int) -> float:
    """Mean interior knee angle across both legs, in degrees (180 = straight).

    Averaged rather than picking a side: from down-the-line one leg occludes
    the other, and which one is visible depends on handedness and camera side.
    Straightening shows up in the mean either way.
    """
    lm = sequence.landmarks
    angles = [
        _joint_angle(lm[frame, hip, :2], lm[frame, knee, :2], lm[frame, ankle, :2])
        for hip, knee, ankle in (
            (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
            (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        )
    ]
    finite = [a for a in angles if np.isfinite(a)]
    return float(np.mean(finite)) if finite else float("nan")


def knee_extension_change(sequence: PoseSequence, p1: int, p7: int) -> float:
    """Knee angle at impact minus at address. Positive means straightening."""
    return knee_angle(sequence, p7) - knee_angle(sequence, p1)


def head_depth_change(sequence: PoseSequence, p1: int, other: int) -> float:
    """Head movement toward the ball, as a fraction of torso length."""
    direction = ball_direction(sequence, p1)
    scale = body_scale(sequence, p1)
    x_address = sequence.landmarks[p1, NOSE, 0]
    x_other = sequence.landmarks[other, NOSE, 0]
    if not np.isfinite(direction) or not np.isfinite(scale) or scale == 0:
        return float("nan")
    if not (np.isfinite(x_address) and np.isfinite(x_other)):
        return float("nan")
    return float((x_other - x_address) * direction / scale)


@dataclass(frozen=True)
class SwingMetrics:
    """Every v1 DTL measurement for one swing.

    Lengths are fractions of torso length; angles are degrees. Any value may be
    NaN when the landmarks it needs were not tracked — reported rather than
    guessed, so a partly-tracked swing still yields the metrics that resolved.
    """

    events: SwingEvents

    spine_tilt_p1: float
    spine_tilt_p4: float
    spine_tilt_p7: float
    posture_change: float
    hip_depth_change: float
    head_rise_p4: float
    head_rise_p7: float
    head_depth_p7: float
    knee_extension_change: float
    tempo_ratio: float

    def as_dict(self) -> dict[str, float]:
        """Flat float mapping, for storage and display. Excludes the events."""
        return {
            f.name: float(getattr(self, f.name))
            for f in fields(self)
            if f.name != "events"
        }


def compute(sequence: PoseSequence, events: SwingEvents) -> SwingMetrics:
    """Compute every v1 metric for one swing in a single pass."""
    p1, p4, p7 = events.p1, events.p4, events.p7
    return SwingMetrics(
        events=events,
        spine_tilt_p1=spine_tilt(sequence, p1),
        spine_tilt_p4=spine_tilt(sequence, p4),
        spine_tilt_p7=spine_tilt(sequence, p7),
        posture_change=posture_change(sequence, p1, p4),
        hip_depth_change=hip_depth_change(sequence, p1, p7),
        head_rise_p4=head_rise(sequence, p1, p4),
        head_rise_p7=head_rise(sequence, p1, p7),
        head_depth_p7=head_depth_change(sequence, p1, p7),
        knee_extension_change=knee_extension_change(sequence, p1, p7),
        tempo_ratio=tempo_ratio(sequence, p1, p4, p7),
    )
