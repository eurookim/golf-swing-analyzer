#!/usr/bin/env python3
"""Render the four detected key frames side by side, for eyeball verification.

The detector reports frame numbers. Numbers can be confidently wrong — this puts
the actual frames on screen so you can see whether "top of backswing" really is
the top.

Usage:
    .venv/bin/python render_contact_sheet.py                 # all cached swings
    .venv/bin/python render_contact_sheet.py clip.mov
"""

from __future__ import annotations

import sys
from pathlib import Path

from golfswing import paths

import cv2
import numpy as np

from golfswing import events, ingest, store

PROCESSED = paths.PROCESSED_DIR
RAW = paths.RAW_DIR
OUT_DIR = paths.OUTPUTS_DIR

LABEL_H = 54
PAD = 8
TARGET_H = 720

# Chart palette, reused so the sheets match the diagnostic plots.
SURFACE = (251, 252, 252)   # BGR of #fcfcfb
INK = (11, 11, 11)
ACCENT = (214, 120, 42)     # BGR of #2a78d6


def _find_video(stem: str) -> Path | None:
    for suffix in (".mov", ".mp4", ".m4v", ".MOV", ".MP4"):
        candidate = RAW / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def contact_sheet(npz_path: Path) -> Path | None:
    sequence = store.load_sequence(npz_path)
    video = _find_video(npz_path.stem)
    if video is None:
        print(f"  ✗ {npz_path.stem}: source video not found in data/raw/")
        return None

    try:
        found = events.detect_events(sequence)
    except events.NoSwingDetectedError as exc:
        print(f"  ✗ {npz_path.stem}: {exc}")
        return None

    wanted = found.as_dict()
    info = ingest.probe(video)
    _, frames = ingest.read_frames_with_times(video)

    panels = []
    for label, index in wanted.items():
        index = min(index, len(frames) - 1)
        frame = frames[index].copy()

        scale = TARGET_H / frame.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), TARGET_H))

        strip = np.full((LABEL_H, frame.shape[1], 3), SURFACE, dtype=np.uint8)
        cv2.putText(strip, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT, 2)
        seconds = sequence.times[index]
        cv2.putText(strip, f"f{index}  {seconds:.2f}s", (10 + 70, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1)
        panels.append(np.vstack([strip, frame]))

    gap = np.full((panels[0].shape[0], PAD, 3), SURFACE, dtype=np.uint8)
    sheet = panels[0]
    for panel in panels[1:]:
        sheet = np.hstack([sheet, gap, panel])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{npz_path.stem}_keyframes.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    t = sequence.times
    backswing = t[found.p4] - t[found.p1]
    downswing = t[found.p7] - t[found.p4]
    print(f"  ✓ {npz_path.stem}  backswing {backswing:.2f}s  "
          f"downswing {downswing:.2f}s  tempo {backswing / downswing:.2f}:1  -> {out.name}")
    return out


def main() -> int:
    if len(sys.argv) > 1:
        targets = [PROCESSED / f"{Path(a).stem}.npz" for a in sys.argv[1:]]
    else:
        targets = sorted(PROCESSED.glob("*.npz"))

    if not targets:
        print("No cached swings. Run: .venv/bin/python -m golfswing")
        return 1

    made = [contact_sheet(t) for t in targets]
    ok = [m for m in made if m]
    print(f"\n{len(ok)}/{len(targets)} sheets written to {OUT_DIR}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
