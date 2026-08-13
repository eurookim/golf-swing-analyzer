# Event Frame Correction Implementation Plan

> **Executed and merged 2026-08-13.** Five defects found against the real code
> during execution have been corrected in place, so this document matches what
> shipped: `read_frames_with_times` returns `(times, frames)` and the tests
> unpacked it backwards; `PoseSequence` requires four landmark channels, not
> three; `window_bounds`'s sample implementation contradicted two of its own four
> tests (the tests were right); the database is `data/swings.db`, not
> `data/history.sqlite`; and `tests/test_db.py` already had a `_metrics` helper
> with a different signature. Baseline test count was 316, not 318.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user scrub to the correct frame for any swing event in the app, recompute that swing's metrics from the cached landmarks, and record the choice as verified ground truth.

**Architecture:** Four layers, each testable alone. `ingest` gains a windowed frame reader. `db` gains a migration and a narrow `UPDATE` that cannot blank neighbouring columns. A new `correction` module orchestrates load → validate → recompute → persist. `app.py` gets the Streamlit scrubber on top.

**Tech Stack:** Python, SQLite, OpenCV, NumPy, Streamlit, pytest.

## Global Constraints

- **Never re-run pose extraction.** Corrections read the cached `.npz` via `store.load_sequence()`. Pose is the slow part and it has already happened.
- **`SwingEvents` is a frozen dataclass.** Build corrected copies with `dataclasses.replace(events, p7=n)`, never by mutation.
- **`p1 < p4 < p7 < p10` is validated before any write.** `labels.save_labels()` already raises on out-of-order events, but it runs last — validating only there would leave the database updated and the label unwritten.
- **The correction path must not use `INSERT OR REPLACE`.** `db.save_swing()`'s own comment warns it "rewrites the whole row, blanking any column not listed." Corrections use an explicit `UPDATE`.
- **Metrics are always recomputed by `metrics.compute()`, never hand-set.** A hand-edited metric cannot be reproduced or checked.
- **`events_source` values are exactly `'detected'` and `'corrected'`** (TEXT, default `'detected'`).
- **`CORRECTION_WINDOW_FRAMES = 15`.**
- Match the surrounding code's style: module docstrings explaining *why*, comments only where the reason is not visible in the code.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `golfswing/ingest.py` | Video decoding | **Modify** — add `frames_in_range()` |
| `golfswing/db.py` | Swing persistence | **Modify** — schema migration + `update_events()` |
| `golfswing/correction.py` | Orchestrate a correction end to end | **Create** |
| `app.py` | Streamlit UI | **Modify** — scrubber block |
| `tests/test_ingest.py` | | **Modify** |
| `tests/test_db.py` | | **Modify** |
| `tests/test_correction.py` | | **Create** |

---

### Task 1: Windowed frame reader

**Files:**
- Modify: `golfswing/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: existing `probe()`, `apply_rotation()` in the same module
- Produces: `frames_in_range(path, lo, hi) -> list[np.ndarray]` — upright frames for indices `lo..hi` inclusive, clamped to the clip

**Why sequential rather than seeking:** `cv2.CAP_PROP_POS_FRAMES` can land on the wrong frame with inter-frame-compressed video, which would show the user one frame while recording a different index — a silent correctness bug. Decoding forward is always accurate, and a scrub window is small enough that the caller can decode it once and cache it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingest.py`:

```python
def test_frames_in_range_returns_the_requested_window(tmp_path):
    video = _make_video(tmp_path, n_frames=30)

    frames = ingest.frames_in_range(video, 10, 14)

    assert len(frames) == 5


def test_frames_in_range_clamps_to_the_clip(tmp_path):
    video = _make_video(tmp_path, n_frames=30)

    frames = ingest.frames_in_range(video, -5, 200)

    assert len(frames) == 30, "clamped at both ends, not an error"


def test_frames_in_range_is_upright(tmp_path):
    """Same rotation handling as the full read — a sideways scrubber is useless."""
    video = _make_video(tmp_path, n_frames=10, width=40, height=20)

    _, whole = ingest.read_frames_with_times(video)
    windowed = ingest.frames_in_range(video, 3, 3)

    assert windowed[0].shape == whole[3].shape


def test_frames_in_range_matches_the_full_read(tmp_path):
    """The window must be the same pixels the full decode produces."""
    video = _make_video(tmp_path, n_frames=20)

    _, whole = ingest.read_frames_with_times(video)
    windowed = ingest.frames_in_range(video, 7, 9)

    for offset, frame in enumerate(windowed):
        assert np.array_equal(frame, whole[7 + offset]), f"frame {7 + offset} differs"


def test_frames_in_range_empty_when_window_is_past_the_end(tmp_path):
    video = _make_video(tmp_path, n_frames=10)

    assert ingest.frames_in_range(video, 50, 60) == []
```

If `tests/test_ingest.py` has no `_make_video` helper, add one that writes a short clip with `cv2.VideoWriter` using distinguishable per-frame content (e.g. a filled rectangle whose position depends on the frame index) so `test_frames_in_range_matches_the_full_read` can actually tell frames apart:

```python
def _make_video(tmp_path, n_frames=30, width=64, height=48, fps=30.0):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, : max(1, (i + 1) % width)] = 255   # a bar that grows per frame
        writer.write(frame)
    writer.release()
    return path
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_ingest.py -k frames_in_range -v`
Expected: FAIL — `AttributeError: module 'golfswing.ingest' has no attribute 'frames_in_range'`

- [ ] **Step 3: Implement**

Add to `golfswing/ingest.py`, beside `read_frames_with_times`:

```python
def frames_in_range(path: Path | str, lo: int, hi: int) -> list[np.ndarray]:
    """Upright frames for indices ``lo..hi`` inclusive, clamped to the clip.

    Decodes forward rather than seeking. ``CAP_PROP_POS_FRAMES`` can land on the
    wrong frame with inter-frame compression, which would show one frame while
    the caller records a different index — the scrubber's whole job is that the
    number and the picture agree.

    Cheap enough because the caller decodes one window and reuses it: the cost
    is one pass to ``hi``, not one pass per frame examined.
    """
    info = probe(path)
    lo = max(0, lo)
    if hi < lo:
        return []

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {info.path}")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except cv2.error:
        pass

    frames: list[np.ndarray] = []
    try:
        for index in range(hi + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if index >= lo:
                frames.append(apply_rotation(frame, info.rotation))
    finally:
        cap.release()
    return frames
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_ingest.py -k frames_in_range -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Run the suite**

Run: `python3 -m pytest -q --ignore=tests/test_pipeline.py --ignore=tests/test_pose.py`
Expected: PASS (316 before this task, 322 after)

- [ ] **Step 6: Commit**

```bash
git add golfswing/ingest.py tests/test_ingest.py
git commit -m "Read a window of frames without seeking"
```

---

### Task 2: Schema migration and a narrow update

**Files:**
- Modify: `golfswing/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: existing `METRIC_COLUMNS`, `_nullable()`, `connect()`
- Produces:
  - `events_source` column on `swings`, TEXT NOT NULL DEFAULT `'detected'`
  - `update_events(conn, clip, events, metrics) -> None` — raises `KeyError` if the clip is not present

**Why a migration is required:** `connect()` runs `CREATE TABLE IF NOT EXISTS`. On an existing database that statement is a no-op, so adding the column to `_SCHEMA` alone would leave every real database without it and every correction failing with `no such column`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_db.py`:

```python
def test_connect_adds_events_source_to_a_legacy_database(tmp_path):
    """A database created before this feature must gain the column, not break."""
    path = tmp_path / "legacy.sqlite"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE swings (clip TEXT PRIMARY KEY, date TEXT NOT NULL, "
        "club TEXT, angle TEXT, fps REAL, fault_tag TEXT, outcome TEXT, "
        "p1 INTEGER, p4 INTEGER, p7 INTEGER, p10 INTEGER, "
        + ", ".join(f"{name} REAL" for name in db.METRIC_COLUMNS) + ")"
    )
    legacy.execute("INSERT INTO swings (clip, date) VALUES ('old', '2026-01-01')")
    legacy.commit()
    legacy.close()

    conn = db.connect(path)

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(swings)")}
    assert "events_source" in columns
    row = conn.execute("SELECT events_source FROM swings WHERE clip = 'old'").fetchone()
    assert row["events_source"] == "detected", "existing rows are detected, not corrected"


def test_connect_is_idempotent(tmp_path):
    path = tmp_path / "twice.sqlite"
    db.connect(path).close()

    conn = db.connect(path)

    columns = [row["name"] for row in conn.execute("PRAGMA table_info(swings)")]
    assert columns.count("events_source") == 1


def test_update_events_writes_frames_and_metrics(tmp_path):
    conn = db.connect(tmp_path / "s.sqlite")
    _insert_swing(conn, "c1")
    events = SwingEvents(p1=5, p4=50, p7=90, p10=120)
    metrics = _metrics(events, tempo_ratio=2.5)

    db.update_events(conn, "c1", events, metrics)

    row = conn.execute("SELECT * FROM swings WHERE clip = 'c1'").fetchone()
    assert (row["p1"], row["p4"], row["p7"], row["p10"]) == (5, 50, 90, 120)
    assert row["tempo_ratio"] == pytest.approx(2.5)
    assert row["events_source"] == "corrected"


def test_update_events_preserves_everything_it_does_not_own(tmp_path):
    """The reason this is not save_swing: INSERT OR REPLACE blanks unlisted
    columns, and losing a fault_tag would silently corrupt the baseline."""
    conn = db.connect(tmp_path / "s.sqlite")
    _insert_swing(conn, "c1", club="driver", fault_tag="early_extension",
                  date="2026-07-01")
    conn.execute("UPDATE swings SET outcome = 'flushed' WHERE clip = 'c1'")

    db.update_events(conn, "c1", SwingEvents(1, 2, 3, 4),
                     _metrics(SwingEvents(1, 2, 3, 4)))

    row = conn.execute("SELECT * FROM swings WHERE clip = 'c1'").fetchone()
    assert row["outcome"] == "flushed"
    assert row["fault_tag"] == "early_extension"
    assert row["club"] == "driver"
    assert row["date"] == "2026-07-01"


def test_update_events_stores_nan_as_null(tmp_path):
    conn = db.connect(tmp_path / "s.sqlite")
    _insert_swing(conn, "c1")
    events = SwingEvents(1, 2, 3, 4)

    db.update_events(conn, "c1", events, _metrics(events, tempo_ratio=float("nan")))

    row = conn.execute("SELECT tempo_ratio FROM swings WHERE clip = 'c1'").fetchone()
    assert row["tempo_ratio"] is None


def test_update_events_rejects_an_unknown_clip(tmp_path):
    conn = db.connect(tmp_path / "s.sqlite")
    events = SwingEvents(1, 2, 3, 4)

    with pytest.raises(KeyError):
        db.update_events(conn, "nope", events, _metrics(events))
```

Add these helpers to the same test module if it lacks equivalents:

```python
def _insert_swing(conn, clip, *, club="7iron", fault_tag=None, date="2026-07-29"):
    conn.execute(
        "INSERT INTO swings (clip, date, club, fault_tag) VALUES (?, ?, ?, ?)",
        (clip, date, club, fault_tag),
    )
    conn.commit()
```

`tests/test_db.py` already has a `_metrics(**overrides)` helper that binds a
module-level `EVENTS`. Reuse it rather than redefining it with a different
signature — `update_events` takes the events separately, so the metrics object's
own `.events` is not what gets written. Add `import sqlite3` at the top;
`pytest`, `SwingEvents` and `SwingMetrics` are already imported there.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_db.py -k "events_source or update_events" -v`
Expected: FAIL — `AttributeError: module 'golfswing.db' has no attribute 'update_events'`, and the migration tests fail on the missing column

- [ ] **Step 3: Add the column to the schema and migrate existing databases**

In `golfswing/db.py`, add the column to `_SCHEMA` so fresh databases have it:

```python
    p1 INTEGER, p4 INTEGER, p7 INTEGER, p10 INTEGER,
    events_source TEXT NOT NULL DEFAULT 'detected',
    {', '.join(f'{name} REAL' for name in METRIC_COLUMNS)}
```

Then add the migration and call it from `connect()`:

```python
def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing database up to the current schema.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so
    a column added to _SCHEMA never reaches a database that predates it. Every
    real database here predates events_source.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(swings)")}
    if "events_source" not in columns:
        conn.execute(
            "ALTER TABLE swings "
            "ADD COLUMN events_source TEXT NOT NULL DEFAULT 'detected'"
        )
```

In `connect()`, after `conn.executescript(_SCHEMA)` and before `conn.commit()`:

```python
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()
```

- [ ] **Step 4: Implement `update_events`**

Add to `golfswing/db.py`, after `save_swing`:

```python
def update_events(
    conn: sqlite3.Connection,
    clip: str,
    events: SwingEvents,
    metrics: SwingMetrics,
) -> None:
    """Replace one swing's event frames and the metrics derived from them.

    Deliberately not `save_swing`: that uses INSERT OR REPLACE, which rewrites
    the whole row and blanks anything not listed. A correction must leave
    `outcome`, `fault_tag`, `club` and `date` untouched — losing a fault tag
    would quietly pull a deliberately-botched swing into the baseline. An UPDATE
    cannot blank a column it does not name.
    """
    values = metrics.as_dict()
    assignments = ", ".join(f"{name} = ?" for name in METRIC_COLUMNS)
    cursor = conn.execute(
        f"UPDATE swings SET p1 = ?, p4 = ?, p7 = ?, p10 = ?, {assignments}, "
        "events_source = 'corrected' WHERE clip = ?",
        (
            events.p1, events.p4, events.p7, events.p10,
            *(_nullable(values[name]) for name in METRIC_COLUMNS),
            clip,
        ),
    )
    if cursor.rowcount == 0:
        raise KeyError(f"no swing named {clip!r}")
    conn.commit()
```

Add the imports `from golfswing.events import SwingEvents` and `from golfswing.metrics import SwingMetrics` if `db.py` does not already have them. If that introduces a circular import, annotate the parameters as strings and guard the imports under `if TYPE_CHECKING:` — the function body only reads attributes.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_db.py -v`
Expected: PASS, including the six new tests

- [ ] **Step 6: Run the suite**

Run: `python3 -m pytest -q --ignore=tests/test_pipeline.py --ignore=tests/test_pose.py`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add golfswing/db.py tests/test_db.py
git commit -m "Record whether a swing's events were detected or corrected"
```

---

### Task 3: The correction itself

**Files:**
- Create: `golfswing/correction.py`
- Test: `tests/test_correction.py`

**Interfaces:**
- Consumes: `store.load_sequence()`, `metrics.compute()`, `db.update_events()`, `labels.save_labels()`, `events.SwingEvents`
- Produces:
  - `EVENT_KEYS = ("p1", "p4", "p7", "p10")`
  - `OutOfOrderError(ValueError)`
  - `apply_correction(conn, clip_stem, events, *, processed_dir, labels_dir) -> SwingMetrics`

- [ ] **Step 1: Write the failing test**

Create `tests/test_correction.py`:

```python
"""Tests for golfswing.correction — a hand-picked frame becomes metrics + truth."""

import json

import numpy as np
import pytest

from golfswing import correction, db, store
from golfswing.events import SwingEvents
from golfswing.sequence import PoseSequence


def _sequence(n_frames=200):
    rng = np.random.default_rng(0)
    return PoseSequence(
        landmarks=rng.random((n_frames, 33, 4)),
        times=np.arange(n_frames) / 120.0,
        fps=120.0,
        source="test.mov",
    )


@pytest.fixture
def workspace(tmp_path):
    processed = tmp_path / "processed"
    labels = tmp_path / "labels"
    store.save_sequence(processed / "c1.npz", _sequence())
    conn = db.connect(tmp_path / "s.sqlite")
    conn.execute(
        "INSERT INTO swings (clip, date, club, fault_tag) "
        "VALUES ('c1', '2026-07-29', '7iron', 'early_extension')"
    )
    conn.execute("UPDATE swings SET outcome = 'flushed' WHERE clip = 'c1'")
    conn.commit()
    return conn, processed, labels


def test_recomputes_metrics_from_the_corrected_events(workspace):
    conn, processed, labels = workspace
    events = SwingEvents(p1=10, p4=100, p7=140, p10=180)

    result = correction.apply_correction(
        conn, "c1", events, processed_dir=processed, labels_dir=labels
    )

    assert result.events == events
    row = conn.execute("SELECT p7, events_source FROM swings WHERE clip='c1'").fetchone()
    assert row["p7"] == 140
    assert row["events_source"] == "corrected"


def test_is_deterministic(workspace):
    conn, processed, labels = workspace
    events = SwingEvents(p1=10, p4=100, p7=140, p10=180)

    first = correction.apply_correction(
        conn, "c1", events, processed_dir=processed, labels_dir=labels
    )
    second = correction.apply_correction(
        conn, "c1", events, processed_dir=processed, labels_dir=labels
    )

    assert first.as_dict() == second.as_dict()


def test_writes_a_verified_label(workspace):
    conn, processed, labels = workspace
    events = SwingEvents(p1=10, p4=100, p7=140, p10=180)

    correction.apply_correction(
        conn, "c1", events, processed_dir=processed, labels_dir=labels
    )

    payload = json.loads((labels / "c1.json").read_text())
    assert payload["p7"] == 140
    assert payload["verified"] is True, "a human looked at the frame — that is the point"


def test_preserves_the_columns_it_does_not_own(workspace):
    conn, processed, labels = workspace

    correction.apply_correction(
        conn, "c1", SwingEvents(10, 100, 140, 180),
        processed_dir=processed, labels_dir=labels,
    )

    row = conn.execute("SELECT * FROM swings WHERE clip='c1'").fetchone()
    assert row["outcome"] == "flushed"
    assert row["fault_tag"] == "early_extension"
    assert row["club"] == "7iron"


def test_rejects_out_of_order_events(workspace):
    conn, processed, labels = workspace

    with pytest.raises(correction.OutOfOrderError):
        correction.apply_correction(
            conn, "c1", SwingEvents(p1=10, p4=150, p7=140, p10=180),
            processed_dir=processed, labels_dir=labels,
        )


def test_writes_nothing_when_rejected(workspace):
    """No partial state: the label must not move while the metrics stay put."""
    conn, processed, labels = workspace
    before = conn.execute("SELECT p7 FROM swings WHERE clip='c1'").fetchone()["p7"]

    with pytest.raises(correction.OutOfOrderError):
        correction.apply_correction(
            conn, "c1", SwingEvents(p1=10, p4=150, p7=140, p10=180),
            processed_dir=processed, labels_dir=labels,
        )

    assert conn.execute("SELECT p7 FROM swings WHERE clip='c1'").fetchone()["p7"] == before
    assert not (labels / "c1.json").exists()


def test_confirming_the_detectors_own_frame_still_counts_as_corrected(workspace):
    """A human looked and agreed. That is worth more than the detector's guess,
    and evaluate_events.py only scores labels a human verified."""
    conn, processed, labels = workspace
    conn.execute("UPDATE swings SET p7 = 140 WHERE clip = 'c1'")
    conn.commit()

    correction.apply_correction(
        conn, "c1", SwingEvents(10, 100, 140, 180),
        processed_dir=processed, labels_dir=labels,
    )

    row = conn.execute("SELECT events_source FROM swings WHERE clip='c1'").fetchone()
    assert row["events_source"] == "corrected"
    assert json.loads((labels / "c1.json").read_text())["verified"] is True


def test_missing_landmarks_file_raises(workspace):
    conn, processed, labels = workspace

    with pytest.raises(FileNotFoundError):
        correction.apply_correction(
            conn, "nope", SwingEvents(10, 100, 140, 180),
            processed_dir=processed, labels_dir=labels,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_correction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'golfswing.correction'`

- [ ] **Step 3: Implement**

Create `golfswing/correction.py`:

```python
"""Turn a hand-picked frame into corrected metrics and verified ground truth.

The detector is right most of the time and wrong quietly: a misplaced impact
frame yields a plausible number rather than an obvious failure. Correcting one
used to mean leaving the app. This is the path that does it in place — and,
because the expensive part is deciding which frame is right, it records that
decision as ground truth at the same time. Labelling stops being a chore and
becomes a side effect of noticing.

Nothing here re-runs pose estimation. The landmarks are already cached; a
correction is a file read, an arithmetic pass, and two writes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from golfswing import db, labels, metrics, store
from golfswing.events import SwingEvents
from golfswing.metrics import SwingMetrics
from golfswing.paths import LABELS_DIR, PROCESSED_DIR

EVENT_KEYS = ("p1", "p4", "p7", "p10")


class OutOfOrderError(ValueError):
    """The corrected frames do not run address → top → impact → finish."""


def corrected(events: SwingEvents, key: str, frame: int) -> SwingEvents:
    """A copy of ``events`` with one event moved. SwingEvents is frozen."""
    if key not in EVENT_KEYS:
        raise KeyError(f"unknown event {key!r} — expected one of {EVENT_KEYS}")
    return replace(events, **{key: int(frame)})


def _check_order(events: SwingEvents) -> None:
    frames = [getattr(events, key) for key in EVENT_KEYS]
    if not all(a < b for a, b in zip(frames, frames[1:])):
        raise OutOfOrderError(
            "events must run address → top → impact → finish, got "
            f"{dict(zip(EVENT_KEYS, frames))}"
        )


def apply_correction(
    conn: sqlite3.Connection,
    clip_stem: str,
    events: SwingEvents,
    *,
    processed_dir: Path | str = PROCESSED_DIR,
    labels_dir: Path | str = LABELS_DIR,
) -> SwingMetrics:
    """Recompute and persist one swing from hand-corrected event frames.

    Order matters. Validation runs before anything is written, and the label is
    written last: a rejected correction must leave the database and the label
    file exactly as they were, rather than moving one and not the other.
    """
    _check_order(events)

    sequence = store.load_sequence(Path(processed_dir) / f"{clip_stem}.npz")
    computed = metrics.compute(sequence, events)

    db.update_events(conn, clip_stem, events, computed)
    labels.save_labels(clip_stem, events, labels_dir=labels_dir, verified=True)
    return computed
```

Note `test_missing_landmarks_file_raises` passes `"nope"`, for which no `.npz` exists — `store.load_sequence` raises `FileNotFoundError` before any write, which is the behaviour under test.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_correction.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Run the suite**

Run: `python3 -m pytest -q --ignore=tests/test_pipeline.py --ignore=tests/test_pose.py`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add golfswing/correction.py tests/test_correction.py
git commit -m "Apply a hand-picked event frame, and keep it as ground truth"
```

---

### Task 4: The scrubber

**Files:**
- Modify: `app.py`
- Modify: `golfswing/correction.py` (add `window_bounds`)
- Test: `tests/test_correction.py`

**Interfaces:**
- Consumes: `correction.apply_correction()`, `correction.corrected()`, `ingest.frames_in_range()`, `skeleton.draw()`
- Produces: `correction.window_bounds(current, n_frames, window=CORRECTION_WINDOW_FRAMES) -> tuple[int, int]`

Streamlit widgets are not worth unit-testing; the arithmetic behind them is. `window_bounds` is extracted so the clamping has real coverage and the UI block stays thin.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_correction.py`:

```python
class TestWindowBounds:
    def test_centres_on_the_current_frame(self):
        assert correction.window_bounds(100, 300, window=15) == (85, 115)

    def test_clamps_at_the_start(self):
        assert correction.window_bounds(3, 300, window=15) == (0, 18)

    def test_clamps_at_the_end(self):
        assert correction.window_bounds(295, 300, window=15) == (280, 299)

    def test_a_clip_shorter_than_the_window_is_fully_covered(self):
        assert correction.window_bounds(2, 5, window=15) == (0, 4)
```

Note the clamped cases keep the full window width on the side that has room — a scrubber that shrinks near the ends would hide exactly the frames a mis-detected event tends to land on.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_correction.py -k window_bounds -v`
Expected: FAIL — `AttributeError: module 'golfswing.correction' has no attribute 'window_bounds'`

- [ ] **Step 3: Implement `window_bounds`**

Add to `golfswing/correction.py`:

```python
CORRECTION_WINDOW_FRAMES = 15


def window_bounds(
    current: int, n_frames: int, window: int = CORRECTION_WINDOW_FRAMES
) -> tuple[int, int]:
    """Inclusive frame range to scrub, centred on ``current`` and clamped.

    Stays centred on the detector's own pick rather than sliding inward to keep
    a constant width near the ends. The error being corrected is centred on that
    pick, so a shifted window would put the frame you are looking for off to one
    side — and a clip trimmed close to the finish is exactly where P10 lands.
    """
    if n_frames <= 0:
        return (0, 0)
    return (max(0, current - window), min(n_frames - 1, current + window))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_correction.py -k window_bounds -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Make the order check public**

The UI previews validity before enabling the button, so it needs the check by a
public name rather than reaching into a private one. In `golfswing/correction.py`
rename `_check_order` to `check_order` and update its call site inside
`apply_correction`.

- [ ] **Step 6: Add the UI block to `app.py`**

Place this inside the swing page, after the existing key-frame images. Match the surrounding code's conventions for how it reaches the connection, the clip stem, the video path and the current `SwingEvents` — this snippet names them `conn`, `clip_stem`, `video_path` and `events`; rename to whatever that page already uses rather than introducing new lookups.

```python
with st.expander("Event frames look wrong?"):
    st.caption(
        "Pick the right frame and every metric derived from it is recomputed. "
        "Your choice is also saved as ground truth, so the detector can be "
        "scored against it."
    )

    key = st.radio(
        "Event", correction.EVENT_KEYS, index=2, horizontal=True,
        format_func=str.upper,
        help="P7 (impact) is the one most often off.",
    )
    current = getattr(events, key)
    lo, hi = correction.window_bounds(current, sequence.n_frames)

    chosen = st.slider("Frame", min_value=lo, max_value=hi, value=current)

    window = _scrub_window(video_path, lo, hi)      # cached, see below
    # draw() returns a copy and takes a per-part colour map; {} means every part
    # renders NEUTRAL, which is what this view wants — the scrubber shows the
    # pose so you can judge the frame, not deviations from a baseline.
    frame = skeleton.draw(window[chosen - lo], sequence.landmarks[chosen], {})
    st.image(frame, channels="BGR",
             caption=f"frame {chosen}"
                     + ("  (detector's pick)" if chosen == current else ""))

    candidate = correction.corrected(events, key, chosen)
    try:
        correction.check_order(candidate)
        blocked = None
    except correction.OutOfOrderError as exc:
        blocked = str(exc)

    if blocked:
        st.warning(blocked)
    elif st.button(f"Use frame {chosen} as {key.upper()}", type="primary"):
        correction.apply_correction(conn, clip_stem, candidate)
        st.success(f"{key.upper()} set to frame {chosen}; metrics recomputed.")
        st.rerun()
```

Add the cached window reader near the app's other helpers:

```python
@st.cache_data(show_spinner="Decoding frames…")
def _scrub_window(video_path: str, lo: int, hi: int) -> list[np.ndarray]:
    """Decode the scrub window once; the slider then costs nothing to move."""
    return ingest.frames_in_range(video_path, lo, hi)
```

- [ ] **Step 7: Show that a swing was corrected**

Wherever the swing page displays the event frames, mark a corrected swing so a hand-set frame is never mistaken for a detected one. Read `events_source` from the swing row already loaded on that page:

```python
if row["events_source"] == "corrected":
    st.caption("Event frames were set by hand.")
```

- [ ] **Step 8: Run the suite**

Run: `python3 -m pytest -q --ignore=tests/test_pipeline.py --ignore=tests/test_pose.py`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add app.py golfswing/correction.py tests/test_correction.py
git commit -m "Scrub to the right event frame from the swing page"
```

---

### Task 5: Verify against a real clip

**Files:** none — this task is manual verification and a docs note.

This is the first task that exercises the feature end to end on real video and a real database. Everything before it ran on synthetic fixtures.

- [ ] **Step 1: Back up the database**

```bash
cp data/swings.db data/swings.db.bak
```

Do this before the first run against real data. The migration is additive and tested, but a one-command undo costs nothing.

- [ ] **Step 2: Confirm the migration ran**

```bash
python3 -c "
from golfswing import db
conn = db.connect()
cols = [r['name'] for r in conn.execute('PRAGMA table_info(swings)')]
print('events_source present:', 'events_source' in cols)
print(conn.execute(\"SELECT COUNT(*) c FROM swings WHERE events_source='detected'\").fetchone()['c'], 'existing swings marked detected')
"
```

Expected: the column exists and every pre-existing swing reads `detected`.

- [ ] **Step 3: Record the detector's current score**

```bash
python3 scripts/evaluate_events.py
```

Save the output. This is the before-measurement — the number the corrections you are about to make will move.

- [ ] **Step 4: Correct a clip with a visibly wrong P7**

Run the app, open a swing whose impact frame is wrong, scrub to the right frame, confirm.

Check three things:
- the tempo ratio on the page changed
- the swing now shows the corrected marker
- `data/labels/<clip>.json` has the new `p7` and `"verified": true`

- [ ] **Step 5: Confirm nothing else moved**

```bash
python3 -c "
from golfswing import db
conn = db.connect()
row = conn.execute('SELECT clip, club, fault_tag, outcome, events_source FROM swings WHERE events_source = \"corrected\"').fetchone()
print(dict(row))
"
```

Expected: `club`, `fault_tag` and `outcome` are unchanged from before the correction. This is the `INSERT OR REPLACE` hazard the design named — confirm on real data that it did not fire.

- [ ] **Step 6: Confirm the label reached the scorer**

```bash
python3 scripts/evaluate_events.py
```

Expected: the verified-label count is one higher than Step 3.

- [ ] **Step 7: Note the workflow in the README**

Add a short paragraph to the README's usage section: event frames can be corrected from the swing page, corrections recompute the swing's metrics, and each one becomes verified ground truth that `evaluate_events.py` scores against. Mention that `scripts/label_swing.py` remains for bulk labelling outside the app.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "Document correcting event frames from the app"
```

---

## Follow-up — done, with a different result than expected

**The `_refine_impact` p4-floor bug.** Fixed 2026-08-13 (`lo = max(p4 + 1, coarse - half)`), but the expectation behind it was wrong. The window can only reach `p4` at exactly 30fps and below ~8fps, where the `max()` floors on `DOWNSWING_MIN_SECONDS` and `IMPACT_REFINE_SECONDS` invert their normal relationship; at 50/60/120/240fps it provably cannot. These clips are 60 and 120fps, and `evaluate_events.py` scored identically before and after the fix. It is a latent-bug guard for 30fps imports, not a P7 accuracy improvement.

**Still open: what actually causes the bad P7s.** That remains the natural first use of the labels this feature produces.
