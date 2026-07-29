"""The core data structure: one swing's pose trajectory over time.

Lives in its own module so persistence (`store`) does not have to import the
pose extractor, which pulls in MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_LANDMARKS = 33

# Landmark channel layout
X, Y, Z, VISIBILITY = 0, 1, 2, 3


@dataclass(frozen=True)
class PoseSequence:
    """Landmarks over time for a single swing.

    ``landmarks`` is ``(n_frames, N_LANDMARKS, 4)`` holding ``(x, y, z,
    visibility)``. x/y are normalised to [0, 1] against frame dimensions.

    ``times`` holds the real presentation timestamp of each frame in seconds —
    not a frame index, and not reconstructable from ``fps``.
    """

    landmarks: np.ndarray
    times: np.ndarray
    fps: float
    source: str

    def __post_init__(self) -> None:
        if self.landmarks.ndim != 3 or self.landmarks.shape[2] < 4:
            raise ValueError(
                f"landmarks must be (n_frames, n_landmarks, 4), got "
                f"{self.landmarks.shape}"
            )
        if len(self.times) != self.landmarks.shape[0]:
            raise ValueError(
                f"times has {len(self.times)} entries but landmarks has "
                f"{self.landmarks.shape[0]} frames"
            )

    @property
    def n_frames(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def duration(self) -> float:
        """Measured span in seconds.

        Derived from timestamps rather than ``n_frames / fps`` — the decoded
        frame count and the container's advertised count disagree, so anything
        computed from counts is suspect.
        """
        if self.n_frames == 0:
            return 0.0
        return float(self.times[-1] - self.times[0])
