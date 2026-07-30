"""Browser-playable copies of raw swing clips.

iPhone footage is a QuickTime-brand container (`ftypqt  `) even though the
video stream inside is ordinary H.264. Chrome's MP4 demuxer rejects that brand
outright — no error, no console message, just a `<video>` element that loads the
bytes and then sits at `readyState: 0` forever. Renaming the file or overriding
the MIME type does not help; the brand is in the container header.

Safari and WKWebView play QuickTime happily, so this only bites in Chrome — but
the fix is cheap and makes the app work in both. Remuxing copies the streams
without re-encoding: ~65ms for an 8.5MB clip, and the output is bit-identical
video.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CACHE_DIR = Path("data/previews")

# A .app launched from Finder inherits a minimal PATH — notably WITHOUT
# /opt/homebrew/bin — so shutil.which("ffmpeg") finds nothing even when ffmpeg
# is installed and works fine in a terminal. Symptom: previews generate when
# you run streamlit by hand, and fail with "No such file or directory:
# 'ffmpeg'" for every clip when you double-click the app.
KNOWN_LOCATIONS = (
    "/opt/homebrew/bin/ffmpeg",   # Apple silicon Homebrew
    "/usr/local/bin/ffmpeg",      # Intel Homebrew
    "/opt/local/bin/ffmpeg",      # MacPorts
)

# Container brands browsers will demux. Anything else needs a remux.
PLAYABLE_BRANDS = {"isom", "mp42", "avc1", "iso2", "M4V", "dash"}


def ffmpeg_path() -> str:
    """Absolute path to ffmpeg, searched beyond the inherited PATH."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in KNOWN_LOCATIONS:
        if os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "ffmpeg was not found. It is needed to convert iPhone clips into a "
        "format browsers can play.\n\nInstall it with:  brew install ffmpeg"
    )


def dimensions(path: Path) -> tuple[int, int] | None:
    """(width, height) of the video stream, or None if unreadable.

    The browser does not know a video's shape until it loads, so a layout that
    depends on aspect ratio cannot be right until then. Reading it up front
    lets the page reserve the correct box immediately.
    """
    probe = ffmpeg_path().replace("ffmpeg", "ffprobe")
    try:
        result = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    parts = result.stdout.strip().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def brand(path: Path) -> str | None:
    """The container's major brand, from bytes 8-12 of the ftyp box."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
    except OSError:
        return None
    if len(header) < 12 or header[4:8] != b"ftyp":
        return None
    return header[8:12].decode("ascii", "replace").strip()


def is_browser_playable(path: Path) -> bool:
    return brand(path) in PLAYABLE_BRANDS


def playable(source: Path, cache_dir: Path | str = CACHE_DIR) -> Path:
    """Return a path a browser can actually play, remuxing and caching if needed.

    Cached on source mtime and size, so re-processed footage regenerates but an
    unchanged clip is remuxed only once — doing it on every page render would
    make the app feel broken.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    if is_browser_playable(source):
        return source

    info = source.stat()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{source.stem}_{int(info.st_mtime)}_{info.st_size}.mp4"

    if target.exists():
        return target

    # Drop stale copies of this clip so the cache does not grow without bound.
    for old in cache_dir.glob(f"{source.stem}_*.mp4"):
        old.unlink(missing_ok=True)

    subprocess.run(
        [ffmpeg_path(), "-y", "-v", "error", "-i", str(source),
         "-c", "copy",                  # no re-encode: same streams, new box layout
         "-movflags", "+faststart",     # moov atom first, so playback starts early
         str(target)],
        check=True,
    )
    return target
