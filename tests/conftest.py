"""Shared test fixtures.

Clips are generated with ffmpeg rather than committed, so the suite is
self-contained and does not depend on the user's private swing footage
(which is gitignored and machine-local).
"""

import shutil
import subprocess

import pytest

FFMPEG = shutil.which("ffmpeg")

requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def _synth(path, *, seconds=1.0, fps=50, size="320x240"):
    subprocess.run(
        [FFMPEG, "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size={size}:rate={fps}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="session")
def cfr_clip(tmp_path_factory):
    """Constant-frame-rate clip: 50 fps, 1 second, no rotation."""
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    return _synth(tmp_path_factory.mktemp("clips") / "cfr.mp4")


@pytest.fixture(scope="session")
def rotated_clip(tmp_path_factory):
    """Clip carrying rotation=-90, mimicking an iPhone portrait capture."""
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("clips_rot")
    base = _synth(d / "base.mp4")
    out = d / "rot.mov"
    subprocess.run(
        [FFMPEG, "-v", "error", "-display_rotation", "-90",
         "-i", str(base), "-c", "copy", "-y", str(out)],
        check=True,
    )
    return out
