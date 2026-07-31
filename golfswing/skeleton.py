"""Joint overlay for key frames, coloured by deviation from the golfer's own
good swings.

Each metric is measured on a specific part of the body, so "what's off" can be
drawn rather than only listed. The mapping is the point of this module:

    posture_change        -> spine      measured at the top
    head_rise_p4          -> head       measured at the top
    hip_depth_change      -> hips       measured at impact
    head_rise_p7          -> head       measured at impact
    knee_extension_change -> knees      measured at impact
    tempo_ratio           -> nothing    it is timing; there is no body part

**A coloured segment means "unlike your flushed swings", not "wrong".** No
threshold exists to support a verdict, so the colour carries the same claim
the tiles do — a comparison against the golfer's own good swings, nothing more.

Address (P1) is deliberately never coloured: it is the reference the other
measurements are taken FROM, so it cannot deviate from itself.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from golfswing.coach import Standing, comparison

# Validated palette: teal-green against red passes the colourblind checks that
# a naive red/green pair fails (deltaE 4.5 under deuteranopia). BGR for OpenCV.
# Lines are drawn twice: a dark casing, then the colour on top. Against bright
# sky and a light shirt a single thin stroke is close to invisible, and the
# overlay has to read on every background in the frame.
CASING = (20, 20, 20)
NEUTRAL = (245, 245, 245)
BETTER = (106, 143, 26)      # #1a8f6a
WORSE = (59, 69, 209)        # #d1453b

# MediaPipe Pose indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

# Which segments belong to which named part, so a part can be coloured as one.
PARTS: dict[str, list[tuple[int, int]]] = {
    "spine": [(L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP)],
    "hips": [(L_HIP, R_HIP)],
    "knees": [(L_HIP, L_KNEE), (R_HIP, R_KNEE),
              (L_KNEE, L_ANKLE), (R_KNEE, R_ANKLE)],
    "arms": [(L_SHOULDER, L_ELBOW), (R_SHOULDER, R_ELBOW),
             (L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)],
    "shoulders": [(L_SHOULDER, R_SHOULDER)],
}

# Metric -> body part, per event. Only where the metric is actually measured:
# hip depth is a change from address to impact, so it belongs on the impact
# frame and nowhere else.
MEASURED_AT: dict[str, dict[str, str]] = {
    "p1": {},
    "p4": {"spine": "posture_change", "head": "head_rise_p4"},
    "p7": {"hips": "hip_depth_change", "head": "head_rise_p7",
           "knees": "knee_extension_change"},
    "p10": {},
}

# Below this the landmark is a guess. Drawing an invented limb across the frame
# would look exactly as authoritative as a measured one.
MIN_VISIBILITY = 0.5


def parts_for_event(event: str) -> dict[str, str]:
    """Which body parts carry a measurement at this event."""
    return dict(MEASURED_AT.get(event, {}))


def colours_for(found: list[Standing], event: str) -> dict[str, tuple]:
    """Colour per body part, from each metric's verdict against the baseline."""
    by_metric = {s.metric: s for s in found}
    colours = {}
    for part, metric in parts_for_event(event).items():
        standing = by_metric.get(metric)
        verdict = comparison(standing) if standing else None
        colours[part] = {"worse": WORSE, "better": BETTER}.get(verdict, NEUTRAL)
    return colours


def _point(landmarks: np.ndarray, index: int, shape) -> tuple[int, int] | None:
    if landmarks[index, 3] < MIN_VISIBILITY:
        return None
    height, width = shape[:2]
    return int(landmarks[index, 0] * width), int(landmarks[index, 1] * height)


def draw(frame: np.ndarray, landmarks: np.ndarray,
         colours: dict[str, tuple]) -> np.ndarray:
    """Return a copy of the frame with the skeleton drawn over it."""
    out = frame.copy()
    shape = out.shape

    thin = max(2, shape[0] // 240)
    for part, segments in PARTS.items():
        colour = colours.get(part, NEUTRAL)
        flagged = colour is not NEUTRAL
        width = thin * 2 if flagged else thin
        for start, end in segments:
            a, b = _point(landmarks, start, shape), _point(landmarks, end, shape)
            if a is None or b is None:
                continue
            cv2.line(out, a, b, CASING, width + 4, cv2.LINE_AA)
            cv2.line(out, a, b, colour, width, cv2.LINE_AA)

    # Head is a ring, not a disc — a filled marker hides the face, and where
    # the head is pointing is worth seeing.
    head = _point(landmarks, NOSE, shape)
    if head is not None:
        colour = colours.get("head", NEUTRAL)
        radius = max(7, shape[0] // 70)
        cv2.circle(out, head, radius, CASING, thin + 4, cv2.LINE_AA)
        cv2.circle(out, head, radius, colour,
                   thin + 1 if colour is not NEUTRAL else thin, cv2.LINE_AA)

    for index in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE):
        joint = _point(landmarks, index, shape)
        if joint is not None:
            cv2.circle(out, joint, thin + 2, CASING, -1, cv2.LINE_AA)
            cv2.circle(out, joint, thin, NEUTRAL, -1, cv2.LINE_AA)
    return out


EVENT_LABELS = {"p1": "P1 address", "p4": "P4 top",
                "p7": "P7 impact", "p10": "P10 finish"}


def key_frames(video: "Path", sequence, events, found: list[Standing],
               height: int = 460) -> "np.ndarray | None":
    """A strip of the four key frames with the skeleton drawn on each.

    Rebuilt on demand rather than cached to disk: the colours depend on the
    golfer's current baseline, which shifts every time a swing is labelled.
    A cached image would quietly show yesterday's comparison.
    """
    from golfswing import ingest        # imported here to keep the module light

    _, frames = ingest.read_frames_with_times(video)
    wanted = [("p1", events.p1), ("p4", events.p4),
              ("p7", events.p7), ("p10", events.p10)]

    panels = []
    for event, index in wanted:
        if not 0 <= index < len(frames):
            continue
        drawn = draw(frames[index], sequence.landmarks[index],
                     colours_for(found, event))
        scale = height / drawn.shape[0]
        drawn = cv2.resize(drawn, (int(drawn.shape[1] * scale), height))

        bar = np.full((34, drawn.shape[1], 3), (251, 252, 252), np.uint8)
        cv2.putText(bar, EVENT_LABELS[event], (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (11, 11, 11), 1, cv2.LINE_AA)
        panels.append(np.vstack([bar, drawn]))

    if not panels:
        return None
    width = min(p.shape[1] for p in panels)
    return np.hstack([p[:, :width] for p in panels])
