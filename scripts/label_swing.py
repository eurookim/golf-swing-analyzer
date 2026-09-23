#!/usr/bin/env python3
"""Render candidate frames around each detected event, for hand-labeling.

Cheaper than an interactive scrubber and enough for a handful of clips: it puts
the neighbouring frames on screen with their numbers, so you can see whether the
detector picked the right one.

Workflow:
    1. .venv/bin/python label_swing.py <clip>     # writes a strip + starter JSON
    2. look at outputs/<clip>_labelstrip.jpg (±12 frames; --span to widen)
       and zoom_event.py <clip> --event P1 --at <frame> for a close-up
    3. edit data/labels/<clip>.json if the detector was off, set "verified": true
    4. .venv/bin/python evaluate_events.py        # score detector vs labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from golfswing import paths

import cv2
import numpy as np

from golfswing import events, ingest, labels, store

PROCESSED = paths.PROCESSED_DIR
RAW = paths.RAW_DIR
OUT_DIR = paths.OUTPUTS_DIR

# Frames either side of the detection. Wide on purpose: a narrow window only
# shows frames the detector already picked, so a label can't land far from it.
SPAN = 12
PANEL_H = 260
TARGET_SHEET_WIDTH = 2400  # wrap each event's frames onto rows no wider than this
LABEL_H = 30
PAD = 4

SURFACE = (251, 252, 252)
INK = (11, 11, 11)
ACCENT = (214, 120, 42)   # BGR of #2a78d6 — marks the detector's pick


def _find_video(stem: str) -> Path | None:
    for suffix in (".mov", ".mp4", ".m4v", ".MOV", ".MP4"):
        candidate = RAW / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def label_strip(
    npz_path: Path,
    at: dict[str, int] | None = None,
    span: int = SPAN,
) -> Path | None:
    sequence = store.load_sequence(npz_path)
    video = _find_video(npz_path.stem)
    if video is None:
        print(f"  ✗ {npz_path.stem}: source video not found")
        return None

    try:
        detected = events.detect_events(sequence)
    except events.NoSwingDetectedError as exc:
        print(f"  ✗ {npz_path.stem}: {exc}")
        return None

    _, frames = ingest.read_frames_with_times(video)

    centres = detected.as_dict()
    if at:
        centres = {**centres, **at}

    rows = []
    for name, centre in centres.items():
        panels = []
        for index in range(centre - span, centre + span + 1):
            clamped = int(np.clip(index, 0, len(frames) - 1))
            frame = frames[clamped]
            scale = PANEL_H / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), PANEL_H))

            picked = index == centre
            strip = np.full((LABEL_H, frame.shape[1], 3), SURFACE, dtype=np.uint8)
            # Every panel names its event, since an event can wrap onto several rows.
            cv2.putText(strip, f"{name} f{clamped}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, ACCENT if picked else INK, 2 if picked else 1)

            panel = np.vstack([strip, frame])
            if picked:
                cv2.rectangle(panel, (0, 0),
                              (panel.shape[1] - 1, panel.shape[0] - 1), ACCENT, 3)
            panels.append(panel)

        gap = np.full((panels[0].shape[0], PAD, 3), SURFACE, dtype=np.uint8)
        per_row = max(1, TARGET_SHEET_WIDTH // (panels[0].shape[1] + PAD))
        for start in range(0, len(panels), per_row):
            chunk = panels[start:start + per_row]
            row = chunk[0]
            for panel in chunk[1:]:
                row = np.hstack([row, gap, panel])
            rows.append((name, row))

    width = max(r.shape[1] for _, r in rows)
    padded = []
    for i, (name, row) in enumerate(rows):
        if row.shape[1] < width:
            fill = np.full((row.shape[0], width - row.shape[1], 3), SURFACE, np.uint8)
            row = np.hstack([row, fill])
        padded.append(row)
        if i + 1 < len(rows):
            # A wider gap where one event ends and the next begins.
            gap_h = PAD * 2 if rows[i + 1][0] == name else PAD * 8
            padded.append(np.full((gap_h, width, 3), SURFACE, dtype=np.uint8))
    sheet = np.vstack(padded)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{npz_path.stem}_labelstrip.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Starter labels, pre-filled with the detection. Edit by hand where wrong.
    existing = labels.load_labels(npz_path.stem)
    if existing is None:
        labels.save_labels(npz_path.stem, detected)
        note = "starter labels written"
    else:
        note = "labels already exist, left alone"

    print(f"  ✓ {npz_path.stem}  -> {out.name}  ({note})")
    return out


def _parse_at(values: list[str]) -> dict[str, int]:
    """Turn ['P1=17', 'P4=61'] into {'P1': 17, 'P4': 61}."""
    out: dict[str, int] = {}
    for item in values:
        name, _, frame = item.partition("=")
        name = name.strip().upper()
        if name not in {"P1", "P4", "P7", "P10"} or not frame.strip().lstrip("-").isdigit():
            sys.exit(f"error: --at expects P1=17 style, got {item!r}")
        out[name] = int(frame)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--at", action="append", default=[], metavar="P1=17",
                    help="centre a row on a specific frame instead of the detection")
    ap.add_argument("--span", type=int, default=SPAN,
                    help=f"frames either side of centre (default {SPAN})")
    args = ap.parse_args()

    targets = (
        [PROCESSED / f"{Path(a).stem}.npz" for a in args.clips]
        if args.clips
        else sorted(PROCESSED.glob("*.npz"))
    )
    if not targets:
        print("No cached swings. Run: .venv/bin/python -m golfswing")
        return 1

    at = _parse_at(args.at)
    made = [label_strip(t, at=at, span=args.span) for t in targets]
    print(f"\n{sum(m is not None for m in made)}/{len(targets)} strips written")
    print(f"Edit ground truth in {labels.DEFAULT_LABELS_DIR}/ where the pick is wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
