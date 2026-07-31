"""CLI: extract and cache keypoints for swing clips.

    .venv/bin/python -m golfswing                    # everything in data/raw/
    .venv/bin/python -m golfswing clip.mov           # one clip
    .venv/bin/python -m golfswing --force            # re-process cached clips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from golfswing import pipeline, pose, store
from golfswing.paths import RAW_DIR, VIDEO_SUFFIXES



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="golfswing", description=__doc__)
    ap.add_argument("clips", nargs="*", type=Path,
                    help="clips to process (default: everything in data/raw/)")
    ap.add_argument("--out-dir", type=Path, default=pipeline.DEFAULT_OUT_DIR)
    ap.add_argument("--force", action="store_true",
                    help="re-process clips that are already cached")
    args = ap.parse_args(argv)

    clips = args.clips or sorted(
        p for p in RAW_DIR.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES
    ) if (args.clips or RAW_DIR.is_dir()) else []

    if not clips:
        print(f"No clips found in {RAW_DIR}/", file=sys.stderr)
        return 1

    pose.ensure_model()

    failures = 0
    for clip in clips:
        cached = (args.out_dir / f"{clip.stem}.npz").exists() and not args.force
        try:
            out = pipeline.process_clip(clip, out_dir=args.out_dir, force=args.force)
        except (pipeline.NoPoseDetectedError, FileNotFoundError, RuntimeError) as exc:
            print(f"  ✗ {clip.name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        seq = store.load_sequence(out)
        tag = "cached" if cached else "processed"
        print(f"  ✓ {clip.name}  [{tag}]  {seq.n_frames} frames  "
              f"{seq.duration:.2f}s  {seq.fps:.2f}fps  -> {out}")

    print(f"\n{len(clips) - failures}/{len(clips)} clips ready in {args.out_dir}/")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
