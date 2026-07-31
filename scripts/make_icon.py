#!/usr/bin/env python3
"""Turn any image into a macOS app icon.

    .venv/bin/python make_icon.py docs/icon-source.png

Writes docs/appicon.icns, then re-run ./install_app.sh to apply it.

macOS icons are not bare squares: since Big Sur the artwork sits inside a
rounded rectangle occupying about 80% of the canvas, with transparent padding
around it. A raw square image dropped into a .icns looks visibly wrong beside
every other icon in the Dock, so this does the shaping rather than just
resizing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Proportions from Apple's icon grid: the rounded rect covers ~80% of the
# canvas, with a corner radius of ~22.37% of the rect's own width.
CONTENT_FRACTION = 0.80
CORNER_FRACTION = 0.2237

# The sizes a .iconset must contain; iconutil rejects an incomplete set.
SIZES = [16, 32, 64, 128, 256, 512, 1024]


def square(image: Image.Image) -> Image.Image:
    """Centre-crop to a square without distorting the subject."""
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def shape(image: Image.Image, canvas: int = 1024) -> Image.Image:
    """Round the corners and inset onto a transparent canvas."""
    content = int(canvas * CONTENT_FRACTION)
    art = square(image).convert("RGBA").resize((content, content), Image.LANCZOS)

    # Supersample the mask so the rounded corners are not visibly stair-stepped.
    scale = 4
    mask = Image.new("L", (content * scale, content * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, content * scale - 1, content * scale - 1),
        radius=int(content * scale * CORNER_FRACTION),
        fill=255,
    )
    art.putalpha(mask.resize((content, content), Image.LANCZOS))

    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    offset = (canvas - content) // 2
    out.paste(art, (offset, offset), art)
    return out


def build(source: Path, target: Path) -> None:
    icon = shape(Image.open(source))

    iconset = target.with_suffix(".iconset")
    iconset.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        resized = icon.resize((size, size), Image.LANCZOS)
        resized.save(iconset / f"icon_{size}x{size}.png")
        # Retina variant: a 32px @2x file is the 64px rendering.
        if size >= 32:
            resized.save(iconset / f"icon_{size // 2}x{size // 2}@2x.png")

    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(target)], check=True
    )
    for leftover in iconset.iterdir():
        leftover.unlink()
    iconset.rmdir()


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/icon-source.png")
    if not source.exists():
        print(f"No such image: {source}", file=sys.stderr)
        return 1

    target = Path("docs/appicon.icns")
    target.parent.mkdir(parents=True, exist_ok=True)
    build(source, target)

    print(f"Wrote {target} ({target.stat().st_size // 1024} KB)")
    print("Now run:  ./install_app.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
