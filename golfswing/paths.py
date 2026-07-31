"""Where the project's data lives.

Every data location resolves from the package's own position on disk, not from
the working directory. Two bugs made this necessary:

1. Scripts moved into `scripts/` computed `Path(__file__).parent` as their root,
   which silently became `scripts/` — so they looked for `scripts/data/raw`.
   Importing them still worked, which is why an import check missed it.
2. Everything else used bare relative paths like `Path("data/processed")`, so
   the project only worked when run from the repo root. That was never stated
   anywhere, and breaks the moment the app is launched from Finder with a
   different working directory.

Set GOLFSWING_ROOT to point at a different data tree without touching code.
"""

from __future__ import annotations

import os
from pathlib import Path

# paths.py lives in golfswing/, so its parent's parent is the project root.
# With an editable install this is the real source tree, which is what we want.
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT = Path(os.environ.get("GOLFSWING_ROOT", _DEFAULT_ROOT)).resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                  # swing videos, as filmed
PROCESSED_DIR = DATA_DIR / "processed"      # cached keypoints (.npz)
LABELS_DIR = DATA_DIR / "labels"            # hand-labeled ground truth
PREVIEWS_DIR = DATA_DIR / "previews"        # browser-playable copies
DB_PATH = DATA_DIR / "swings.db"            # swing history

OUTPUTS_DIR = PROJECT_ROOT / "outputs"      # contact sheets, annotated video
MODELS_DIR = PROJECT_ROOT / "models"        # MediaPipe weights
THRESHOLDS = PROJECT_ROOT / "thresholds.yaml"

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
