"""Pose extraction: video in, landmark trajectories out.

The slow stage. Results are cached via `store` so a clip is posed once.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from golfswing import ingest
from golfswing.sequence import N_LANDMARKS, PoseSequence

DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    """Download the pose model if it is not already on disk (~30 MB, once)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def extract_sequence(
    video_path: Path | str,
    model_path: Path | str = DEFAULT_MODEL,
) -> PoseSequence:
    """Run pose over every frame of a clip.

    Frames where no pose is found are filled with NaN coordinates and zero
    visibility. NaN rather than zero because zero is a *valid* normalised
    coordinate — filling with it would silently yield real-looking angles from
    frames in which nothing was detected. NaN forces callers to handle the gap.

    Runs in VIDEO mode, which tracks across frames and is markedly smoother than
    treating each frame independently.
    """
    info = ingest.probe(video_path)
    times, frames = ingest.read_frames_with_times(video_path)

    landmarks = np.full((len(frames), N_LANDMARKS, 4), np.nan, dtype=np.float64)
    landmarks[:, :, 3] = 0.0  # visibility: 0 until proven otherwise

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, (frame, t) in enumerate(zip(frames, times)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(t * 1000))
            if not result.pose_landmarks:
                continue
            for j, lm in enumerate(result.pose_landmarks[0]):
                landmarks[i, j] = (
                    lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0),
                )

    return PoseSequence(
        landmarks=landmarks,
        times=times,
        fps=info.fps,
        source=str(video_path),
    )
