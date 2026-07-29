#!/usr/bin/env python3
"""Zoom into the frames around one event, auto-centred on the hands.

The full-frame label strip shrinks the golfer to a few hundred pixels, which is
too small to judge address or impact. This crops around the wrist landmarks —
so it follows the swing regardless of how the shot is framed — and blows the
crop up.

    .venv/bin/python zoom_event.py <clip> --event P7 --at 76 --span 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from golfswing import events, ingest, store

PROCESSED = Path("data/processed")
RAW = Path("data/raw")
OUT_DIR = Path("outputs")

SURFACE = (251, 252, 252)
INK = (11, 11, 11)
ACCENT = (214, 120, 42)

TARGET_SHEET_WIDTH = 2300


def _find_video(stem: str) -> Path | None:
    for suffix in (".mov", ".mp4", ".m4v", ".MOV", ".MP4"):
        candidate = RAW / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("--event", default="P7", choices=["P1", "P4", "P7", "P10"])
    ap.add_argument("--at", type=int, help="centre frame (default: the detection)")
    ap.add_argument("--span", type=int, default=5)
    ap.add_argument("--box", type=float, default=0.55,
                    help="crop width as a fraction of frame width")
    ap.add_argument("--zoom", type=float, default=2.2)
    args = ap.parse_args()

    stem = Path(args.clip).stem
    npz = PROCESSED / f"{stem}.npz"
    if not npz.exists():
        print(f"no cached keypoints for {stem}")
        return 1
    video = _find_video(stem)
    if video is None:
        print(f"no source video for {stem}")
        return 1

    sequence = store.load_sequence(npz)
    detected = events.detect_events(sequence)
    centre = args.at if args.at is not None else detected.as_dict()[args.event]

    _, frames = ingest.read_frames_with_times(video)
    h, w = frames[0].shape[:2]

    # Centre the crop on the hands at the centre frame, from the pose data, so
    # the window follows the swing instead of assuming where the golfer stands.
    wrists = sequence.landmarks[centre, [15, 16], :2]
    cx, cy = np.nanmean(wrists, axis=0)
    if not np.isfinite(cx):
        cx, cy = 0.5, 0.6
    box_w = int(w * args.box)
    box_h = int(box_w * 0.75)
    x0 = int(np.clip(cx * w - box_w / 2, 0, w - box_w))
    y0 = int(np.clip(cy * h - box_h / 2, 0, h - box_h))

    panels = []
    for index in range(centre - args.span, centre + args.span + 1):
        clamped = int(np.clip(index, 0, len(frames) - 1))
        crop = frames[clamped][y0:y0 + box_h, x0:x0 + box_w]
        crop = cv2.resize(crop, None, fx=args.zoom, fy=args.zoom,
                          interpolation=cv2.INTER_CUBIC)
        picked = index == centre
        # Scale label chrome with the panel so text stays legible at any zoom.
        bar_h = max(28, crop.shape[1] // 12)
        font = bar_h / 42.0
        bar = np.full((bar_h, crop.shape[1], 3), SURFACE, dtype=np.uint8)
        cv2.putText(bar, f"f{clamped}", (8, int(bar_h * 0.72)),
                    cv2.FONT_HERSHEY_SIMPLEX, font,
                    ACCENT if picked else INK, max(1, int(font * 2)))
        panel = np.vstack([bar, crop])
        if picked:
            cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1),
                          ACCENT, max(2, int(font * 4)))
        panels.append(panel)

    # Keep the sheet a readable width rather than one enormous row.
    per_row = max(1, min(len(panels), TARGET_SHEET_WIDTH // panels[0].shape[1]))
    rows = []
    for start in range(0, len(panels), per_row):
        chunk = panels[start:start + per_row]
        while len(chunk) < per_row:
            chunk.append(np.full_like(panels[0], SURFACE))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stem}_{args.event}_zoom.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  ✓ {stem} {args.event} centred f{centre} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
