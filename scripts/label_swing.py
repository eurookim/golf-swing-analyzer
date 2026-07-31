#!/usr/bin/env python3
"""Render candidate frames around each detected event, for hand-labeling.

Cheaper than an interactive scrubber and enough for a handful of clips: it puts
the neighbouring frames on screen with their numbers, so you can see whether the
detector picked the right one.

Workflow:
    1. .venv/bin/python label_swing.py <clip>     # writes a strip + starter JSON
    2. look at outputs/<clip>_labelstrip.jpg
    3. edit data/labels/<clip>.json if the detector was off
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

SPAN = 3          # frames either side of the detection
PANEL_H = 260
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
            text = f"{name} f{clamped}" if picked else f"f{clamped}"
            cv2.putText(strip, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, ACCENT if picked else INK, 2 if picked else 1)

            panel = np.vstack([strip, frame])
            if picked:
                cv2.rectangle(panel, (0, 0),
                              (panel.shape[1] - 1, panel.shape[0] - 1), ACCENT, 3)
            panels.append(panel)

        gap = np.full((panels[0].shape[0], PAD, 3), SURFACE, dtype=np.uint8)
        row = panels[0]
        for panel in panels[1:]:
            row = np.hstack([row, gap, panel])
        rows.append(row)

    width = max(r.shape[1] for r in rows)
    spacer = np.full((PAD * 2, width, 3), SURFACE, dtype=np.uint8)
    padded = []
    for row in rows:
        if row.shape[1] < width:
            fill = np.full((row.shape[0], width - row.shape[1], 3), SURFACE, np.uint8)
            row = np.hstack([row, fill])
        padded.extend([row, spacer])
    sheet = np.vstack(padded[:-1])

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
