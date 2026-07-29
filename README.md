# Golf Swing Analyzer

Personal tool that analyzes golf swing videos recorded at the range and reports
**named swing faults** plus **numeric metrics**, with history tracked over time.

Runs locally on a laptop. No cloud, no accounts, no mobile app.

See [`PLAN.md`](PLAN.md) for the full architecture, metric definitions, fault
rules, and design rationale.

---

## Status

**Phase 1 complete.** Phase 0 returned **GO** — 100% pose detection on all clips,
with shoulders and hips at 1.00 visibility.

```bash
.venv/bin/python -m golfswing      # extract + cache keypoints from data/raw/
.venv/bin/pytest -q                # 58 tests
```

**v1 angle: down-the-line only.** Face-on deferred to v2.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **0** | Run MediaPipe on existing 60fps DTL clips, dump annotated video. **GO / NO-GO.** | ✅ |
| **1** | `ingest` + `pose` + smoothing + persisted keypoints (CLI only) | ✅ |
| **2** | Event detection (P1/P4/P7/P10) + hand-labeled ground-truth set | ☐ |
| **3** | DTL metrics with torso-length normalization | ✅ |
| **4** | Fault rules + per-club `thresholds.yaml` + tuning | ☐ |
| **5** | Streamlit UI, SQLite history, trend charts | ☐ |
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
   separate detector — optional Phase 6.
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
├── README.md
├── thresholds.yaml    fault thresholds (tuned constantly — never hardcode)
├── golfswing/         the package
│   ├── ingest.py      ffprobe true fps + rotation, downscale, tag angle
│   ├── pose.py        MediaPipe Pose Landmarker → (n_frames, 33, 4)
│   ├── smooth.py      Savitzky-Golay along time axis
│   ├── events.py      detect P1 / P4 / P7 / P10
│   ├── metrics.py     angles + normalized distances at key frames
│   ├── faults.py      rule engine over metrics
│   ├── render.py      annotated video + key-frame contact sheet
│   └── store.py       SQLite swing history
├── app.py             Streamlit UI
└── data/
    ├── raw/           swing videos            (gitignored — too large)
    ├── processed/     keypoint dumps          (gitignored — regenerable)
    └── labels/        hand-labeled P-frames   (TRACKED — the valuable part)
```

## What's tracked vs. ignored

Videos and derived artifacts stay out of git — a 240fps clip is hundreds of MB,
GitHub rejects anything over 100MB, and committed files live in history forever.

**`data/labels/` is deliberately tracked.** The hand-labeled P-frames are the
highest-value artifact here: without ground truth, every change to the event
detector is a guess. Losing them means re-labeling every clip by hand.
