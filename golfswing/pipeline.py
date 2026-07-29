"""The Phase 1 chain: video -> pose -> smoothed -> cached keypoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from golfswing import pose, smooth, store
from golfswing.sequence import PoseSequence

DEFAULT_OUT_DIR = Path("data/processed")

# Below this fraction of frames containing a detected pose, the clip is not
# usable. Matches the Phase 0 GO/NO-GO threshold.
MIN_DETECTION_RATE = 0.8


class NoPoseDetectedError(RuntimeError):
    """Raised when too few frames contain a detectable person."""


def detection_rate(sequence: PoseSequence) -> float:
    """Fraction of frames in which a pose was found."""
    if sequence.n_frames == 0:
        return 0.0
    return float(np.mean(~np.isnan(sequence.landmarks[:, 0, 0])))


def process_clip(
    video_path: Path | str,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    force: bool = False,
) -> Path:
    """Extract, smooth, and cache keypoints for one clip.

    Returns the path to the .npz. Re-running is a no-op unless ``force`` — pose
    extraction is the slow stage and its output is deterministic for a given
    clip, so there is nothing to gain by repeating it.

    Raises ``NoPoseDetectedError`` if the clip contains no golfer. Caching an
    all-NaN file would look like success and fail confusingly several stages
    later, so nothing is written in that case.
    """
    video_path = Path(video_path)
    out_path = Path(out_dir) / f"{video_path.stem}.npz"

    if out_path.exists() and not force:
        return out_path

    raw = pose.extract_sequence(video_path)

    rate = detection_rate(raw)
    if rate < MIN_DETECTION_RATE:
        raise NoPoseDetectedError(
            f"pose found in only {rate:.0%} of frames in {video_path.name} "
            f"(need {MIN_DETECTION_RATE:.0%}). Check lighting, framing, and that "
            f"a golfer is actually in shot."
        )

    smoothed = replace(
        raw, landmarks=smooth.smooth_landmarks(raw.landmarks, fps=raw.fps)
    )
    return store.save_sequence(out_path, smoothed)
