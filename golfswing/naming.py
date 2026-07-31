"""Building and recognising conventional clip filenames.

    YYYY-MM-DD_<angle>_<club>_<nn>[_<fault>].<ext>

The filename is not cosmetic — it is the only place a clip's club, angle, date
and deliberate-fault tag are recorded. history.parse_clip() reads them straight
back out of it, and a clip whose club cannot be read can never be ranked,
because ranking compares against swings of the same club only.

That is why every field is validated here rather than accepted and filed away:
a typo'd fault tag would be read as a normal swing and quietly pollute the very
baseline it is meant to be measured against.
"""

from __future__ import annotations

import re
from datetime import date as _date
from pathlib import Path

ANGLES = {"dtl", "fo"}

# Must stay in step with calibrate.FAULT_TAGS — those map a tag to a rule name.
FAULT_TAGS = {"posture", "earlyext", "headlift", "quicktempo",
              "kneestraight", "sway", "reversepivot"}

_CLUB = re.compile(r"^[a-z0-9]+$")
_CONVENTIONAL = re.compile(
    r"^\d{4}-\d{2}-\d{2}_(?:dtl|fo)_[a-z0-9]+_\d{2,}(?:_[a-z]+)?$"
)


def conventional_name(
    when: _date,
    angle: str,
    club: str,
    index: int,
    fault: str | None = None,
    suffix: str = ".mov",
) -> str:
    """Assemble a filename, validating every field."""
    if angle not in ANGLES:
        raise ValueError(f"unknown angle {angle!r} — expected one of {sorted(ANGLES)}")
    if not _CLUB.match(club or ""):
        raise ValueError(
            f"club {club!r} must be lowercase letters and digits only, no spaces"
        )
    if fault is not None and fault not in FAULT_TAGS:
        raise ValueError(
            f"unknown fault tag {fault!r} — expected one of {sorted(FAULT_TAGS)}"
        )

    stem = f"{when.isoformat()}_{angle}_{club}_{index:02d}"
    if fault:
        stem += f"_{fault}"
    return stem + suffix.lower()


def follows_convention(stem: str) -> bool:
    """Whether a filename stem already carries date, angle, club and index."""
    return bool(_CONVENTIONAL.match(stem))


def rename_to_convention(
    path: Path,
    when: _date,
    angle: str,
    club: str,
    index: int,
    fault: str | None = None,
) -> Path:
    """Rename a clip in place, refusing to overwrite anything.

    Two clips landing on one name would destroy a swing with no warning, and
    the raw footage is the one artifact here that cannot be regenerated.
    """
    target = path.with_name(
        conventional_name(when, angle, club, index, fault, suffix=path.suffix)
    )
    if target != path and target.exists():
        raise FileExistsError(
            f"{target.name} already exists — refusing to overwrite it"
        )
    path.rename(target)
    return target


def next_index(directory: Path, when: _date, angle: str, club: str) -> int:
    """The next free clip number for one session.

    Counts past the HIGHEST existing number rather than counting files, so
    deleting a clip from the middle of a session cannot cause a collision.
    """
    prefix = f"{when.isoformat()}_{angle}_{club}_"
    highest = 0
    if directory.is_dir():
        for path in directory.iterdir():
            if not path.stem.startswith(prefix):
                continue
            number = path.stem[len(prefix):].split("_")[0]
            if number.isdigit():
                highest = max(highest, int(number))
    return highest + 1
