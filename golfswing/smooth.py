"""Temporal smoothing of landmark trajectories.

Raw pose keypoints jitter frame to frame. Smoothing MUST happen before any
differentiation — differentiating raw keypoints amplifies that jitter and wrecks
the velocity signals event detection depends on.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

# Landmark channel layout: (x, y, z, visibility)
_COORD_CHANNELS = (0, 1, 2)
_VISIBILITY_CHANNEL = 3

DEFAULT_WINDOW_SECONDS = 0.1
DEFAULT_POLYORDER = 2


def window_frames(
    window_seconds: float,
    fps: float,
    polyorder: int = DEFAULT_POLYORDER,
) -> int:
    """Frames spanning ``window_seconds``, valid as a Savitzky-Golay window.

    The window is specified in SECONDS rather than frames so that the same
    physical amount of smoothing is applied regardless of capture rate — 60fps
    footage today and 240fps later must not be filtered differently.

    Savitzky-Golay requires an odd window strictly greater than ``polyorder``.
    """
    n = int(round(window_seconds * fps))
    n = max(n, polyorder + 2)
    if n % 2 == 0:
        n += 1
    return n


def smooth_landmarks(
    landmarks: np.ndarray,
    fps: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    polyorder: int = DEFAULT_POLYORDER,
) -> np.ndarray:
    """Savitzky-Golay filter the x/y/z trajectories along the time axis.

    ``landmarks`` is ``(n_frames, n_landmarks, 4)``. The visibility channel is
    passed through untouched: it is a per-frame confidence score, not a
    trajectory, and filtering it would blend confident and guessed frames into a
    value that means neither.
    """
    if landmarks.ndim != 3 or landmarks.shape[2] < 4:
        raise ValueError(
            f"expected (n_frames, n_landmarks, 4), got {landmarks.shape}"
        )

    n_frames = landmarks.shape[0]
    min_frames = polyorder + 2
    if n_frames < min_frames:
        raise ValueError(
            f"need at least {min_frames} frames to filter with polyorder="
            f"{polyorder}, got {n_frames}"
        )

    window = window_frames(window_seconds, fps, polyorder)
    if window > n_frames:
        # Clamp to the clip and keep it odd.
        window = n_frames if n_frames % 2 == 1 else n_frames - 1

    out = landmarks.copy()
    for channel in _COORD_CHANNELS:
        out[:, :, channel] = savgol_filter(
            landmarks[:, :, channel], window, polyorder, axis=0
        )
    return out
