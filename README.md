# Golf Swing Analyzer

Personal tool that analyzes golf swing videos recorded at the range and reports
**named swing faults** plus **numeric metrics**, with history tracked over time.

Runs locally on a laptop. No cloud, no accounts, no mobile app.

See [`PLAN.md`](PLAN.md) for the full architecture, metric definitions, fault
rules, and design rationale.

---

## Status

**Phases 0-5 complete.** 15 clips hand-labeled at 120fps; event detection
measured against them at **P4 0.4 / P7 0.4 frames** mean absolute error.

**Phase 4 is blocked on footage, not code.** The fault engine works, but every
threshold in `thresholds.yaml` is still invented: a session of deliberately
exaggerated faults measured *inside* the normal range, so no rule can yet
separate a fault from a normal swing. Until that is re-shot, the app compares
each swing against the golfer's own distribution instead — which needs no
thresholds and is true today. See `CAPTURE.md`.

```bash
# One-time setup
.venv/bin/pip install -e ".[dev]"

# Everything below is run from the repo root
.venv/bin/python -m golfswing                     # extract + cache keypoints
.venv/bin/streamlit run app.py                    # the app (or open the macOS app)
./install_app.sh                                  # build a double-clickable .app
.venv/bin/pytest -q                               # 256 tests
```

**v1 angle: down-the-line only.** Face-on deferred to v2.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **0** | Run MediaPipe on existing 60fps DTL clips, dump annotated video. **GO / NO-GO.** | ✅ |
| **1** | `ingest` + `pose` + smoothing + persisted keypoints (CLI only) | ✅ |
| **2** | Event detection (P1/P4/P7/P10) + hand-labeled ground-truth set | ✅ |
| **3** | DTL metrics with torso-length normalization | ✅ |
| **4** | Fault rules + per-club `thresholds.yaml` + tuning | 🔄 engine done, thresholds uncalibrated |
| **5** | Streamlit UI, SQLite history, trend charts | ✅ |
| **6** | Swing segmentation — split a continuous range video into individual swings | ☐ |
| **7** | *Optional.* Shaft-line detection → swing plane. Decide after Phase 3. | ☐ |

---

## The three constraints that shape everything

1. **Frame rate decides which metrics are trustworthy.** A downswing is ~0.25s —
   7 frames at 30fps (unusable), ~15 at 60fps (workable), ~60 at 240fps (ideal).
   60fps holds up here because most faults are measured at address and top of
   backswing, the slow moments. Only impact-anchored metrics degrade. **fps is
   always read from `ffprobe`, never assumed.**
2. **The club is invisible.** MediaPipe gives 33 *body* joints and nothing about
   the club. Body-driven metrics are cheap; swing plane and club path need a
   separate detector — optional Phase 7.
3. **Camera angle changes the meaning of every metric.** Face-on and
   down-the-line support different metrics. Every video is tagged with its angle
   at ingest, and only angle-valid metrics are computed.

**Why DTL first:** footage already exists so Phase 0 starts now; it's the harder
pose problem (arms cross the torso, trail leg hides behind lead leg), making it a
stronger GO/NO-GO gate; and early extension — a very common amateur fault — is
invisible face-on.

---

## Capture setup

- **Down-the-line**: camera *on the target line* behind you (extend the
  ball-to-target line backward through yourself), hand height, ~10–12 ft.
  Placing it behind your *body* rather than the *line* skews every angle.
- **Slo-Mo (120/240fps) when possible**; 60fps works for v1. Bright light matters
  more than frame rate — motion blur wrecks pose accuracy.
- **Tripod at a fixed height** — consistency beats correctness. Mark the spot.
- **Fitted clothing** contrasting with the background
- **One club (7-iron)** for the first session
- **Stop-start recording, one clip per swing** until Phase 6 lands — the pipeline
  currently assumes one clip = one swing, and separate files make the
  deliberate-fault labels trivial
- Filenames: `2026-07-28_dtl_7iron_01.mov`

---

## Repo layout

```
golf-swing-analyzer/
├── PLAN.md            architecture, metrics, fault rules, phases
├── CAPTURE.md         what to film, and how — read before a range session
├── thresholds.yaml    fault thresholds (INVENTED until calibrated)
├── pyproject.toml
│
├── app.py             Streamlit UI — the page layout
├── desktop.py         runs the app in a native macOS window
├── install_app.sh     builds the double-clickable .app
│
├── golfswing/         the package — everything importable
│   ├── ingest.py      ffprobe true fps + rotation, downscale, tag angle
│   ├── pose.py        MediaPipe Pose Landmarker → (n_frames, 33, 4)
│   ├── smooth.py      Savitzky-Golay along time axis
│   ├── sequence.py    the (landmarks, times, fps) value type
│   ├── events.py      detect P1 / P4 / P7 / P10
│   ├── metrics.py     angles + normalised distances at key frames
│   ├── faults.py      rule engine over metrics
│   ├── coach.py       rank a swing against the golfer's own history
│   ├── calibrate.py   measure thresholds from tagged fault clips
│   ├── labels.py      hand-labeled ground truth, read + write
│   ├── store.py       keypoint cache (.npz)
│   ├── db.py          SQLite swing history
│   ├── history.py     cached clips → history rows
│   ├── preview.py     browser-playable copies of raw clips
│   ├── pipeline.py    ingest → pose → smooth → cache
│   └── ui.py          CSS and stat-tile markup for app.py
│
├── scripts/           one-off tools, all run from the repo root
│   ├── phase0_check.py         GO/NO-GO pose check on raw footage
│   ├── rename_session.py       bulk-rename exports to the convention
│   ├── label_swing.py          frame strips for hand-labeling events
│   ├── zoom_event.py           close-up of a single event frame
│   ├── render_contact_sheet.py 4 key frames per swing
│   ├── evaluate_events.py      score the detector against ground truth
│   ├── calibrate_faults.py     suggest thresholds from tagged clips
│   └── make_icon.py            build the macOS app icon
│
├── tests/             256 tests, one module per package module
└── data/
    ├── raw/           swing videos            (gitignored — too large)
    ├── processed/     keypoint dumps          (gitignored — regenerable)
    ├── previews/      browser-playable clips  (gitignored — regenerable)
    └── labels/        hand-labeled P-frames   (TRACKED — the valuable part)
```

## What's tracked vs. ignored

Videos and derived artifacts stay out of git — a 240fps clip is hundreds of MB,
GitHub rejects anything over 100MB, and committed files live in history forever.

**`data/labels/` is deliberately tracked.** The hand-labeled P-frames are the
highest-value artifact here: without ground truth, every change to the event
detector is a guess. Losing them means re-labeling every clip by hand.
