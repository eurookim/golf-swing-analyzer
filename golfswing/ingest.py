"""Video ingest: metadata, true per-frame timestamps, rotation correction."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROTATION_OPS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotation_correction(metadata_rotation: int) -> int:
    """Degrees CLOCKWISE to rotate a decoded frame so it displays upright.

    ffprobe reports the display-matrix angle, so the correction is its negative.
    An iPhone clip tagged ``rotation=-90`` needs a 90 degree clockwise rotation.
    """
    return (-metadata_rotation) % 360


def apply_rotation(frame: np.ndarray, metadata_rotation: int) -> np.ndarray:
    """Orient a decoded frame upright using the container's rotation metadata.

    OpenCV does not reliably honour the rotation matrix in .mov files, so this is
    applied explicitly rather than trusted to the decoder.
    """
    op = _ROTATION_OPS.get(rotation_correction(metadata_rotation))
    return frame if op is None else cv2.rotate(frame, op)


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata as reported by ffprobe."""

    path: Path
    fps: float           # r_frame_rate — the true capture rate
    avg_fps: float       # avg_frame_rate — lower than fps on variable-rate files
    width: int           # as stored, before rotation
    height: int
    rotation: int        # display-matrix angle; see rotation_correction()
    duration: float
    codec: str
    nb_frames: int       # advertised; decoders often return fewer

    @property
    def display_width(self) -> int:
        return self.height if rotation_correction(self.rotation) in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        return self.width if rotation_correction(self.rotation) in (90, 270) else self.height

    # Note: avg_fps is often well below fps on iPhone clips (60.00 vs 55.61),
    # but that reflects an unreliable nb_frames estimate rather than uneven
    # frame spacing. Measured intervals on real footage are a constant 16.67ms.
    # Judge variability from decoded timestamps, not from this ratio.


def _parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    num, _, den = value.partition("/")
    return float(num) / float(den or 1)


def probe(path: Path | str) -> VideoInfo:
    """Read container metadata. Never infers anything from the filename."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    if not data.get("streams"):
        raise ValueError(f"no video stream in {path}")
    stream = data["streams"][0]

    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
            break
    else:
        rotation = int(stream.get("tags", {}).get("rotate", 0) or 0)

    return VideoInfo(
        path=path,
        fps=_parse_rate(stream.get("r_frame_rate", "")),
        avg_fps=_parse_rate(stream.get("avg_frame_rate", "")),
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        rotation=rotation,
        duration=float(data.get("format", {}).get("duration", 0.0)),
        codec=stream.get("codec_name", "?"),
        nb_frames=int(stream.get("nb_frames", 0) or 0),
    )


def read_frames_with_times(
    path: Path | str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Decode a clip into upright frames plus their REAL timestamps in seconds.

    Timestamps come from the decoder's presentation time rather than
    ``index / fps``. On the iPhone footage measured so far the two agree to
    within 3ms across a whole clip, so this is not currently load-bearing — but
    it costs nothing, removes an assumption, and is the only thing that stays
    correct if a clip ever does arrive with uneven spacing or dropped frames.

    Returns fewer frames than ``VideoInfo.nb_frames`` advertises (322 vs 351 on
    one real clip). The decoded count is the trustworthy one: it matches
    duration x fps. Never size an array from ``nb_frames``.
    """
    info = probe(path)

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {info.path}")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except cv2.error:
        pass

    times: list[float] = []
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Query AFTER the read: OpenCV updates POS_MSEC to the timestamp of
            # the frame just decoded. Reading it beforehand yields the previous
            # frame's time, which duplicates frame 0 and breaks monotonicity.
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            frames.append(apply_rotation(frame, info.rotation))
    finally:
        cap.release()

    return np.asarray(times, dtype=np.float64), frames
