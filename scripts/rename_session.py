#!/usr/bin/env python3
"""
Bulk-rename one range session's clips into the project convention:

    YYYY-MM-DD_<angle>_<club>_<nn>[_<fault>].mov

Dry-run by default — it prints what it WOULD do and changes nothing until you
pass --apply. Re-running is safe: files already matching the convention are
skipped, so numbering never shifts underneath you.

Examples
--------
    # preview (changes nothing)
    .venv/bin/python rename_session.py --angle dtl --club 7iron

    # actually rename
    .venv/bin/python rename_session.py --angle dtl --club 7iron --apply

    # mark clips 9, 10, 11 as deliberate faults
    .venv/bin/python rename_session.py --angle dtl --club 7iron \
        --fault 9=posture --fault 10=earlyext --fault 11=headlift --apply
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path

from golfswing import paths

RAW_DIR = paths.RAW_DIR

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
ANGLES = {"dtl", "fo"}
FAULT_TAGS = {"posture", "earlyext", "headlift", "quicktempo", "sway", "reversepivot"}

# Matches a filename that is ALREADY in the convention, so we can skip it.
ALREADY_NAMED = re.compile(
    r"^\d{4}-\d{2}-\d{2}_(dtl|fo)_[a-z0-9]+_\d{2,}(_[a-z]+)?$"
)


def capture_time(path: Path) -> datetime:
    """Best-effort shooting time, used to order clips correctly.

    Filename order is usually right for Photos exports (IMG_4471 < IMG_4472),
    but it breaks the moment clips come from more than one source. The
    container's creation_time is the real answer when it's present; fall back to
    the file's mtime if it isn't.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        tags = json.loads(out).get("format", {}).get("tags", {})
        raw = tags.get("creation_time")
        if raw:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError, OSError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def parse_fault(values: list[str]) -> dict[int, str]:
    """Turn ['9=posture', '10=earlyext'] into {9: 'posture', 10: 'earlyext'}."""
    mapping: dict[int, str] = {}
    for item in values:
        if "=" not in item:
            sys.exit(f"error: --fault expects N=tag, got {item!r}")
        num_s, tag = item.split("=", 1)
        tag = tag.strip().lower()
        if not num_s.strip().isdigit():
            sys.exit(f"error: --fault index must be a number, got {num_s!r}")
        if tag not in FAULT_TAGS:
            sys.exit(
                f"error: unknown fault tag {tag!r}\n"
                f"       known tags: {', '.join(sorted(FAULT_TAGS))}"
            )
        mapping[int(num_s)] = tag
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rename one session's clips into the project convention.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--angle", required=True, choices=sorted(ANGLES),
                    help="camera angle: dtl (down-the-line) or fo (face-on)")
    ap.add_argument("--club", required=True,
                    help="club used, lowercase no spaces, e.g. 7iron / driver / pw")
    ap.add_argument("--date", help="session date YYYY-MM-DD (default: from video metadata)")
    ap.add_argument("--fault", action="append", default=[], metavar="N=tag",
                    help="mark clip N as a deliberate fault; repeatable")
    ap.add_argument("--start", type=int, default=1, help="first sequence number (default 1)")
    ap.add_argument("--dir", type=Path, default=RAW_DIR, help="directory to rename in")
    ap.add_argument("--apply", action="store_true",
                    help="actually rename (without this, previews only)")
    args = ap.parse_args()

    club = args.club.strip().lower()
    if not club.isalnum():
        sys.exit(f"error: --club must be alphanumeric, no spaces or dashes: {club!r}")

    faults = parse_fault(args.fault)

    if not args.dir.is_dir():
        sys.exit(f"error: no such directory: {args.dir}")

    all_videos = [p for p in args.dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if not all_videos:
        sys.exit(f"error: no video files in {args.dir}")

    skipped = [p for p in all_videos if ALREADY_NAMED.match(p.stem)]
    todo = [p for p in all_videos if not ALREADY_NAMED.match(p.stem)]

    for p in skipped:
        print(f"  skip (already named)  {p.name}")
    if not todo:
        print("\nNothing to rename — every clip already follows the convention.")
        return 0

    # Order by shooting time so sequence numbers reflect the order you actually hit.
    todo.sort(key=lambda p: (capture_time(p), p.name))

    if args.date:
        try:
            session_date = date.fromisoformat(args.date)
        except ValueError:
            sys.exit(f"error: --date must be YYYY-MM-DD, got {args.date!r}")
    else:
        session_date = capture_time(todo[0]).date()
        print(f"\nInferred session date from video metadata: {session_date}")

    width = max(2, len(str(args.start + len(todo) - 1)))
    plan: list[tuple[Path, Path]] = []
    for offset, src in enumerate(todo):
        n = args.start + offset
        stem = f"{session_date}_{args.angle}_{club}_{n:0{width}d}"
        if n in faults:
            stem += f"_{faults[n]}"
        plan.append((src, src.with_name(stem + src.suffix.lower())))

    # Never clobber. Check against files on disk AND against our own targets.
    existing = {p.name for p in args.dir.iterdir()} - {s.name for s, _ in plan}
    collisions = [dst for _, dst in plan if dst.name in existing]
    seen: set[str] = set()
    for _, dst in plan:
        if dst.name in seen:
            collisions.append(dst)
        seen.add(dst.name)
    if collisions:
        print("\nerror: these target names already exist or collide:")
        for c in collisions:
            print(f"  {c.name}")
        print("Refusing to overwrite anything. Resolve, then re-run.")
        return 1

    print(f"\n{'CURRENT':<34}  {'NEW':<44}")
    print(f"{'-' * 34}  {'-' * 44}")
    for src, dst in plan:
        mark = "  ← fault" if len(dst.stem.split("_")) > 4 else ""
        print(f"{src.name:<34}  {dst.name:<44}{mark}")

    unused = sorted(set(faults) - {args.start + i for i in range(len(todo))})
    if unused:
        print(f"\n⚠️  --fault given for clip(s) {unused}, but there "
              f"{'is' if len(unused) == 1 else 'are'} only {len(todo)} clip(s) to rename.")

    if not args.apply:
        print(f"\nDRY RUN — nothing changed. {len(plan)} file(s) would be renamed.")
        print("Re-run with --apply to do it.")
        return 0

    for src, dst in plan:
        src.rename(dst)
    print(f"\n✓ Renamed {len(plan)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
