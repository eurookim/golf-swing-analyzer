# Golf Swing Analyzer

Personal tool that analyzes golf swing videos recorded at the range and reports
**named swing faults** plus **numeric metrics**, with history tracked over time.

Runs locally on a laptop. No cloud, no accounts, no mobile app.

See [`PLAN.md`](PLAN.md) for the full architecture, metric definitions, fault
rules, and design rationale.

---

## Status

**Phase 0 — not started.** Nothing built yet.

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **0** | Record 5–10 swings, run MediaPipe, dump annotated video. **GO / NO-GO.** | ☐ |
| **1** | `ingest` + `pose` + smoothing + persisted keypoints (CLI only) | ☐ |
| **2** | Event detection (P1/P4/P7/P10) + hand-labeled ground-truth set | ☐ |
| **3** | Metrics with body-scale normalization | ☐ |
| **4** | Fault rules + `thresholds.yaml` + tuning | ☐ |
| **5** | Streamlit UI, SQLite history, trend charts | ☐ |

---

## The three constraints that shape everything

1. **Record in Slo-Mo (120 or 240fps).** A downswing is ~0.25s — at 30fps that's
   7 frames, which is unusable.
2. **The club is invisible.** MediaPipe gives 33 *body* joints and nothing about
   the club. Body-driven metrics are cheap; club path / face angle are out of
   scope for v1.
3. **Camera angle changes the meaning of every metric.** Face-on and
   down-the-line support different metrics. Every video is tagged with its angle
   at ingest.

---

## Capture setup

- **Slo-Mo mode, 120 or 240fps**, bright light (motion blur wrecks pose accuracy)
- **Tripod at a fixed height** — consistency beats correctness
- **Face-on**: perpendicular to target line, belt height, ~10–12 ft away
- **Down-the-line**: on the target line behind you, hand height, ~10–12 ft
- **Fitted clothing** contrasting with the background

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
