#!/usr/bin/env python3
"""
Phase 0 — GO / NO-GO check.

Answers one question: does MediaPipe Pose actually track a golf swing in YOUR
footage? Everything else in this project is built on that assumption, so it gets
tested before any real code exists.

For each video in data/raw/ this:
  1. Reports TRUE capture fps + rotation from ffprobe (never trusts the filename)
  2. Runs MediaPipe Pose Landmarker over every frame
  3. Writes an annotated copy with the skeleton drawn on you
  4. Reports per-landmark-group visibility so you can see NUMERICALLY where
     tracking degrades — on down-the-line footage expect the trail arm and trail
     leg to be worst, since they're occluded by your body.

Usage:
    .venv/bin/python phase0_check.py                 # everything in data/raw/
    .venv/bin/python phase0_check.py path/to/clip.mov
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

# MediaPipe Pose landmark indices, grouped so the report is readable.
# For a RIGHT-handed golfer filmed down-the-line, the "trail" side is the right.
LANDMARK_GROUPS = {
    "head":        [0],
    "shoulders":   [11, 12],
    "lead arm":    [11, 13, 15],   # left arm (RH golfer)
    "trail arm":   [12, 14, 16],   # right arm — expect occlusion on DTL
    "hips":        [23, 24],
    "lead leg":    [23, 25, 27],
    "trail leg":   [24, 26, 28],   # expect occlusion on DTL
}

# Below this, a landmark is being guessed rather than seen.
VIS_THRESHOLD = 0.5

# A real golf swing is ~1.2s of actual motion. Much longer means the clip is
# either very padded or the slo-mo ramp was baked in on export.
TYPICAL_SWING_SECONDS = 1.2


def probe(path: Path) -> dict:
    """Read true capture fps, rotation, and dimensions via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]

    # r_frame_rate is the real capture rate; avg_frame_rate can differ on
    # variable-frame-rate files.
    def parse_rate(value: str) -> float:
        if not value or value == "0/0":
            return 0.0
        num, _, den = value.partition("/")
        return float(num) / float(den or 1)

    rotation = 0
    for side_data in stream.get("side_data_list", []) or []:
        if "rotation" in side_data:
            rotation = int(side_data["rotation"])
    if rotation == 0:
        rotation = int(stream.get("tags", {}).get("rotate", 0) or 0)

    return {
        "fps": parse_rate(stream.get("r_frame_rate", "")),
        "avg_fps": parse_rate(stream.get("avg_frame_rate", "")),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "rotation": rotation % 360,
        "codec": stream.get("codec_name", "?"),
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "nb_frames": int(stream.get("nb_frames", 0) or 0),
    }


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply the container's rotation metadata ourselves.

    OpenCV does not reliably honour the rotation matrix in .mov files, which is
    how you end up with a sideways skeleton and every angle silently wrong.

    Sign convention matters and is easy to invert. ffprobe reports the display
    matrix angle, so the correction is the NEGATIVE of it: an iPhone clip tagged
    `rotation=-90` must be rotated 90° CLOCKWISE to display upright. Verified
    against ffmpeg's own output — it applies rotation correctly, so a frame
    extracted with ffmpeg is ground truth to diff against.
    """
    angle = (-rotation) % 360
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model -> {MODEL_PATH.name} (~30 MB, one time)")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("  done\n")


def report_source(info: dict) -> None:
    print(f"  codec        {info['codec']}  {info['width']}x{info['height']}")
    print(f"  TRUE fps     {info['fps']:.2f}   (avg {info['avg_fps']:.2f})")
    print(f"  rotation     {info['rotation']}°")
    print(f"  duration     {info['duration']:.2f}s")

    fps = info["fps"]
    downswing_frames = 0.25 * fps
    print(f"  downswing    ~{downswing_frames:.0f} frames at this rate")

    if fps < 50:
        print("  ⚠️  UNDER 50 FPS — the downswing is too few frames to analyse.")
        print("      If you shot this in Slo-Mo, the slow-motion was likely baked")
        print("      in on export. Re-export with Photos > File > Export >")
        print("      'Export Unmodified Original'.")
    elif fps < 100:
        print("  ✓  Workable. Address/top-of-backswing metrics are solid;")
        print("     impact-anchored ones (tempo, early extension) degrade.")
    else:
        print("  ✓  Good frame rate.")

    if info["duration"] > TYPICAL_SWING_SECONDS * 6 and fps < 50:
        print("  ⚠️  Long clip at low fps — classic signature of baked-in slo-mo.")

    if info["rotation"]:
        print("  ℹ️  Rotation metadata present; applying it manually.")


def analyse(path: Path) -> bool:
    """Run pose over one clip. Returns True if tracking looks usable."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp

    print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")

    try:
        info = probe(path)
    except subprocess.CalledProcessError as exc:
        print(f"  ✗ ffprobe failed: {exc.stderr.strip()[:200]}")
        return False

    report_source(info)

    fps = info["fps"] or 30.0

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,  # temporal tracking, smoother
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("  ✗ OpenCV could not open this file.")
        return False

    # Handle rotation ourselves rather than relying on OpenCV.
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except Exception:
        pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{path.stem}_annotated.mp4"
    writer = None

    vis_sums = {name: 0.0 for name in LANDMARK_GROUPS}
    vis_lowframes = {name: 0 for name in LANDMARK_GROUPS}
    frames = 0
    detected = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = rotate_frame(frame, info["rotation"])
            h, w = frame.shape[:2]

            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frames / fps * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frames += 1

            if result.pose_landmarks:
                detected += 1
                lms = result.pose_landmarks[0]

                for name, idxs in LANDMARK_GROUPS.items():
                    vals = [getattr(lms[i], "visibility", 1.0) for i in idxs]
                    mean_vis = sum(vals) / len(vals)
                    vis_sums[name] += mean_vis
                    if mean_vis < VIS_THRESHOLD:
                        vis_lowframes[name] += 1

                # Draw skeleton: green = confident, red = guessed.
                for i, lm in enumerate(lms):
                    vis = getattr(lm, "visibility", 1.0)
                    x, y = int(lm.x * w), int(lm.y * h)
                    color = (0, 220, 0) if vis >= VIS_THRESHOLD else (0, 0, 255)
                    cv2.circle(frame, (x, y), 4, color, -1)

                for conn in mp_vision.PoseLandmarksConnections.POSE_LANDMARKS:
                    pa, pb = lms[conn.start], lms[conn.end]
                    cv2.line(
                        frame,
                        (int(pa.x * w), int(pa.y * h)),
                        (int(pb.x * w), int(pb.y * h)),
                        (255, 200, 0), 2,
                    )

            cv2.putText(
                frame, f"frame {frames}  {fps:.0f}fps", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()

    if frames == 0:
        print("  ✗ No frames read.")
        return False

    detect_rate = detected / frames
    print(f"\n  Frames processed        {frames}")
    print(f"  Pose detected           {detected}/{frames}  ({detect_rate:.0%})")
    print(f"  Annotated video         {out_path.relative_to(ROOT)}")

    print("\n  Landmark visibility (mean, and % of frames below "
          f"{VIS_THRESHOLD}):")
    weakest = []
    for name in LANDMARK_GROUPS:
        mean_vis = vis_sums[name] / detected if detected else 0.0
        low_pct = vis_lowframes[name] / detected if detected else 1.0
        flag = "  ⚠️" if low_pct > 0.25 else ""
        print(f"    {name:<12} {mean_vis:.2f}   {low_pct:5.0%} low{flag}")
        if low_pct > 0.25:
            weakest.append(name)

    print()
    if detect_rate < 0.8:
        print("  ✗ NO-GO for this clip — pose found in under 80% of frames.")
        print("    Usual causes: poor lighting, baggy clothing, golfer too small")
        print("    in frame, or heavy motion blur.")
        return False

    print("  ✓ GO — tracking is usable.")
    if weakest:
        print(f"    Weak groups: {', '.join(weakest)}.")
        print("    On down-the-line footage this is expected for the trail arm")
        print("    and trail leg (occluded by your body). It matters only if a")
        print("    metric you need depends on those joints.")
    print("\n  → Now WATCH the annotated video. Numbers can look fine while the")
    print("    skeleton visibly detaches during the swing. Trust your eyes.")
    return True


def main() -> int:
    ensure_model()

    if len(sys.argv) > 1:
        videos = [Path(a) for a in sys.argv[1:]]
    else:
        videos = sorted(
            p for p in RAW_DIR.iterdir()
            if p.suffix.lower() in VIDEO_SUFFIXES
        ) if RAW_DIR.exists() else []

    if not videos:
        print(f"No videos found in {RAW_DIR}/")
        print("\nExport them from Photos with:")
        print("  File > Export > 'Export Unmodified Original for N Items...'")
        print("Plain 'Export' or drag-and-drop re-encodes and can bake in the")
        print("slo-mo ramp, leaving you with a 30fps file.")
        return 1

    results = [analyse(v) for v in videos]

    print(f"\n{'=' * 70}")
    ok = sum(results)
    print(f"VERDICT: {ok}/{len(results)} clips usable")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
