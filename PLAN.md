# Golf Swing Analyzer — Plan

**Scope decided:** personal tool, runs locally on the laptop, analyzes swings recorded earlier at the range.
Output = named swing faults + numeric metrics. No club tracking, no cloud, no accounts, no mobile app.

---

## The three constraints that shape everything

1. **Frame rate.** Downswing (top → impact) is ~0.25s. At 30fps that's 7 frames — unusable.
   Record in **iPhone Slo-Mo (120 or 240fps)**. This is non-negotiable and affects file sizes downstream.

2. **The club is invisible.** MediaPipe gives 33 *body* joints and nothing about the club. Metrics driven by
   body joints (turn, tilt, sway, tempo, extension) are cheap. Club path / face angle / swing plane require a
   custom-trained detector — explicitly out of scope for v1.

3. **Camera angle changes the meaning of every metric.** Face-on and down-the-line (DTL) support *different*
   metrics. Every video must be tagged with its angle at ingest, and only angle-valid metrics computed.
   Baking this in from day one; retrofitting it later is painful.

---

## Core primitive: swing event detection

Golf has a standard position framework. We need four frames:

| Event | Name | Detection signal |
|-------|------|------------------|
| P1 | Address | Last sustained low-motion plateau before motion energy crosses threshold |
| P4 | Top of backswing | Wrist vertical velocity crosses zero + wrist reaches highest point, direction reverses |
| P7 | Impact | Peak hand speed / hands return through address position on the way down |
| P10 | Finish | Motion energy returns to near-zero after P7 |

**Every metric is measured at, or between, these frames.** Get this right and the rest is arithmetic.
This is also the highest-risk component — budget the most time here.

Fallback if heuristics prove fragile: the **GolfDB** dataset (McNally et al., 2019) — ~1400 labeled swing
videos with 8 event frames each, plus a baseline model (SwingNet). Worth knowing it exists before hand-rolling
something complicated.

---

## Pipeline

```
video.mp4 (120/240fps)
  │
  ├─ ingest.py   ffprobe for TRUE capture fps + rotation metadata; downscale to 720p; tag camera angle
  ├─ pose.py     MediaPipe Pose Landmarker, VIDEO mode, heavy model → (n_frames, 33, 4) array
  ├─ smooth.py   Savitzky-Golay along time axis, per keypoint per coord  ← MUST happen before angles
  ├─ events.py   detect P1 / P4 / P7 / P10
  ├─ metrics.py  angles + normalized distances at each key frame
  ├─ faults.py   rule engine over metrics, thresholds from YAML
  ├─ render.py   annotated video + 4-up key-frame contact sheet
  └─ store.py    SQLite: swings, metrics, faults → enables trend charts over time
```

### Gotchas per stage

- **ingest**: iOS slo-mo may be a true 240fps file *or* a 30fps file with playback already baked in. Always
  `ffprobe` — don't trust the filename. Also: phone videos carry a rotation tag; ignoring it gives you a
  sideways skeleton (classic bug).
- **pose**: use VIDEO mode, not IMAGE mode — VIDEO mode does temporal tracking and is much smoother.
  The `z` coordinate is relative-depth and noisy; treat with suspicion, prefer 2D reasoning.
- **smooth**: smoothing must precede any differentiation. Differentiating raw noisy keypoints is catastrophic
  for velocity-based event detection. SavGol window ~7–11 frames at 120fps, polyorder 2–3.

---

## Metrics

Key landmarks: 11/12 shoulders, 23/24 hips, 25/26 knees, 27/28 ankles, 15/16 wrists, 0 nose.

| Metric | Angle | How |
|--------|-------|-----|
| Spine tilt | DTL | mid-hip→mid-shoulder vector vs vertical, at P1 / P4 / P7 |
| Shoulder turn | Face-on | `arccos(current_shoulder_width / address_shoulder_width)` — foreshortening trick |
| Hip turn | Face-on | same trick on the hip line |
| X-factor | Face-on | shoulder turn − hip turn at P4 (good players ~40–50°) |
| Head drift | Both | nose displacement from P1, normalized by shoulder width |
| Weight shift | Face-on | mid-hip horizontal position over time, normalized by stance width |
| Tempo ratio | Both | `(P4−P1) / (P7−P4)` in frames. Tour average ≈ 3:1 |
| Knee flex | Both | knee angle at P1 vs P7 |
| Hip depth | DTL | hip horizontal distance from the original address butt-line |

**Normalization is mandatory.** Divide every pixel distance by a body-scale reference (shoulder width, or
hip→shoulder distance). Otherwise standing 2 feet closer to the camera silently changes all your numbers and
session-over-session comparison is meaningless.

---

## Fault rules

Each fault = a pure function over metrics → `(fired, measured_value, threshold, severity)`.

| Fault | Angle | Rough rule |
|-------|-------|------------|
| Loss of posture | DTL | spine tilt at P4 differs from P1 by > 8° |
| Early extension | DTL | hips move ballward > 4% of body-scale between P4 and P7 |
| Sway | Face-on | mid-hip drifts away from target > 6% of stance width at P4 |
| Slide | Face-on | excessive lateral hip drift toward target at P7 |
| Reverse pivot | Face-on | head/weight moves *toward* target during backswing |
| Head lift | Both | nose rises beyond threshold P1→P7 |
| Quick tempo | Both | tempo ratio < 2.2 |
| Restricted turn | Face-on | shoulder turn at P4 < 80° |

Thresholds live in `thresholds.yaml`, never in code — tuning these is the main activity for weeks 3–4.

**These numbers are invented until calibrated.** They are starting points, not truth.

---

## Stack

Single user, local, "make it real" effort — so the weeks go into *analysis quality*, not deployment plumbing.

- Python 3.11 + `uv`
- `mediapipe`, `opencv-python`, `numpy`, `scipy`, `pandas`, `ffmpeg-python`
- SQLite for swing history
- **UI: Streamlit or Gradio**, deliberately *not* React + FastAPI. Zero other users means every hour on a
  custom frontend is an hour stolen from the actual project. File upload, video playback, tables and charts
  come free.
- Clean package boundaries + a CLI entry point, so bolting on a real API later is trivial if it ever matters.

---

## Phases

| Phase | Time | Deliverable |
|-------|------|-------------|
| **0** | 2–3 hrs | Record 5–10 swings. ffprobe them. Run MediaPipe. Dump annotated video. **GO / NO-GO on the whole idea.** |
| **1** | Week 1 | ingest + pose + smoothing + persisted keypoints. CLI only. |
| **2** | Week 1–2 | Event detection + a frame-scrubber to hand-label ground truth. Highest risk. |
| **3** | Week 2 | Metrics with normalization. |
| **4** | Week 3 | Fault rules + YAML thresholds + tuning. |
| **5** | Week 3–4 | Streamlit UI, SQLite history, trend charts. |

---

## The single highest-leverage thing

**Build a hand-labeled test set of your own swings in Phase 2.** 10–20 clips where you've manually marked the
P1/P4/P7/P10 frames.

Without it, every change to the detector is a guess and you'll thrash. With it, every change is measurable
(mean frame error vs. your labels). This is what separates the version that works from the version that's
abandoned in week 3.

---

## Capture setup (do this before writing any code)

- **Slo-Mo mode, 120 or 240fps.** Bright light — motion blur destroys pose accuracy.
- **Tripod, fixed height.** Camera height changes measured angles, so *consistency beats correctness*.
  Same spot, same height, every session.
- **Face-on**: perpendicular to target line, hand/belt height, ~10–12 ft away.
- **Down-the-line**: on the target line behind you, hand height, ~10–12 ft.
- **Fitted clothing**, contrasting with the background. Baggy clothes badly degrade pose estimation.
- Record both angles of the same swing where possible; label filenames with angle + club.
