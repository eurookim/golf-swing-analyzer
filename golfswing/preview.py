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

import subprocess
from pathlib import Path

CACHE_DIR = Path("data/previews")

# Container brands browsers will demux. Anything else needs a remux.
PLAYABLE_BRANDS = {"isom", "mp42", "avc1", "iso2", "M4V", "dash"}


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
        ["ffmpeg", "-y", "-v", "error", "-i", str(source),
         "-c", "copy",                  # no re-encode: same streams, new box layout
         "-movflags", "+faststart",     # moov atom first, so playback starts early
         str(target)],
        check=True,
    )
    return target
