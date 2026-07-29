"""Persistence for extracted pose sequences.

Keypoints are cached as .npz so pose extraction — the slow part — runs once per
clip. These files are derived and disposable: `data/raw/` is the only thing that
cannot be regenerated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from golfswing.sequence import PoseSequence


def save_sequence(path: Path | str, sequence: PoseSequence) -> Path:
    """Write a sequence to a compressed .npz, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        landmarks=sequence.landmarks,
        times=sequence.times,
        fps=np.float64(sequence.fps),
        source=np.array(sequence.source),
    )
    return path


def load_sequence(path: Path | str) -> PoseSequence:
    """Read a sequence back. Raises FileNotFoundError if it is not there."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # allow_pickle=False: these files are data, never executable payloads.
    with np.load(path, allow_pickle=False) as data:
        return PoseSequence(
            landmarks=data["landmarks"],
            times=data["times"],
            fps=float(data["fps"]),
            source=str(data["source"]),
        )
