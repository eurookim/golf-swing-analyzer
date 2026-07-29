# Golf Swing Analyzer — Plan

**Scope decided:** personal tool, runs locally on the laptop, analyzes swings recorded earlier at the range.
Output = named swing faults + numeric metrics. No club tracking, no cloud, no accounts, no mobile app.

**v1 angle: down-the-line (DTL) only.** Face-on is deferred to v2. Rationale below.

---

## The three constraints that shape everything

1. **Frame rate.** The downswing (top → impact) is ~0.25s, so frame rate decides which metrics are trustworthy:

   | fps | Frames in downswing | Verdict |
   |-----|--------------------:|---------|
   | 30 | ~7 | Unusable |
   | **60** | **~15** | **Usable — P1/P4-anchored metrics are solid; impact-anchored ones degrade** |
   | 120 | ~30 | Good |
   | 240 | ~60 | Ideal |

   What saves 60fps is that most faults here are measured at **address and top of backswing — the slow
   moments.** Only impact-dependent metrics suffer: impact falls *between* frames (±8ms, hands travel ~7cm),
   so tempo ratio is ±7% (catches "way too quick," won't track fine changes) and early extension is
   directional rather than precise.

   **Never hardcode fps.** Read it from `ffprobe`, store per swing, compute everything in *seconds*. Then
   higher-fps footage later just sharpens the same metrics with zero code changes.

   Also: **60fps in bright sun beats 240fps in a dim garage.** Motion blur hurts pose accuracy more than
   frame count does — high frame rates in low light force either a fast shutter (dark, noisy) or a slow one
   (smeared hands).

2. **The club is invisible.** MediaPipe gives 33 *body* joints and nothing about the club. Metrics driven by
   body joints (turn, tilt, sway, tempo, extension) are cheap. Club path / face angle / swing plane require a
   separate detector — out of scope for v1, revisited as optional Phase 6.

3. **Camera angle changes the meaning of every metric.** Face-on and down-the-line (DTL) support *different*
   metrics. Every video must be tagged with its angle at ingest, and only angle-valid metrics computed.
   Baking this in from day one; retrofitting it later is painful.

---

## Why DTL first

**v1 computes DTL metrics only.** Face-on is deferred to v2. Reasons, in order:

- **Footage already exists.** Phase 0 can run today instead of after a range trip.
- **DTL is the harder pose problem** — from behind, the arms cross the torso and the trail leg hides behind
  the lead leg. Far more self-occlusion than face-on. If MediaPipe survives DTL, face-on is easy. That makes
  it a *stronger* GO/NO-GO gate.
- **Early extension is only visible from DTL**, and it's one of the most common amateur faults. Face-on
  cannot see it at all.
- **Loss of posture works cleanly at 60fps** — it's a P1-vs-P4 comparison, both slow moments.

The honest tradeoff: DTL yields **fewer** body-only metrics than face-on, because most of what DTL is famous
for — swing plane, club path, shaft angle — needs the club. Body-only DTL at 60fps gives roughly two solid
faults (loss of posture, head lift) plus early extension directionally. That is thin, but enough to build and
validate the entire pipeline against, and it sharpens considerably at 240fps.

---

## Core primitive: swing event detection

Golf has a standard position framework. We need four frames:

| Event | Name | Detection signal |
|-------|------|------------------|
| P1 | Address | Last sustained low-motion plateau before motion energy crosses threshold |
| P4 | Top of backswing | **Shoulder-line rotation reaches maximum and reverses** |
| P7 | Impact | Shoulder/hip angular velocity peak; hips return through address orientation |
| P10 | Finish | Motion energy returns to near-zero after P7 |

> **Do not detect events from wrist velocity.** Phase 0 on real DTL footage measured
> wrist/arm visibility at 0.62–0.74 mean with **30–42% of frames below 0.5**, while
> shoulders and hips sat at **1.00 with 0% bad frames**. Wrists are the noisiest
> signal available; shoulders and hips are the cleanest. Since the top of the
> backswing is equally well defined by shoulder rotation reversing, key every event
> off the torso, not the hands. This is measured on this project's own footage, not a
> general claim about MediaPipe.

**Every metric is measured at, or between, these frames.** Get this right and the rest is arithmetic.
This is also the highest-risk component — budget the most time here.

Fallback if heuristics prove fragile: the **GolfDB** dataset (McNally et al., 2019) — ~1400 labeled swing
videos with 8 event frames each, plus a baseline model (SwingNet). Worth knowing it exists before hand-rolling
something complicated.

---

## Pipeline

```
video.mp4 (60fps now, 120/240fps later — fps read from ffprobe, never assumed)
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
- **ingest — `nb_frames` lies; frame timing does not.** Measured on real footage: ffprobe advertises 351
  frames but the decoder returns 322, and reports `avg_frame_rate` 55.61 against a nominal 60.00. That gap is
  an unreliable `nb_frames` estimate (`avg_frame_rate = nb_frames / duration`), **not** uneven frame spacing —
  actual decoded intervals are a constant 16.67 ms, and `index / fps` agrees with real timestamps to within
  3 ms across a whole clip. So: **never size an array from `nb_frames`** (the decoded count is the correct
  one — it matches `duration × fps`), but do not expect timing drift either. Real presentation timestamps
  (`CAP_PROP_POS_MSEC`) are still what we store — they cost nothing and stay correct if a clip ever does
  arrive with dropped frames.
- **ingest — `CAP_PROP_POS_MSEC` must be read AFTER `cap.read()`.** Reading it beforehand returns the previous
  frame's time, so frames 0 and 1 both come back as 0.0 and the series is no longer monotonic.
- **ingest — rotation sign is easy to invert.** ffprobe reports the display-matrix angle, so the correction
  is its **negative**: a clip tagged `rotation=-90` needs a 90° **clockwise** rotation. Verify against a
  frame extracted with ffmpeg (which applies rotation correctly) rather than guessing — getting this backwards
  yields an upside-down skeleton whose angles look plausible but are wrong.

---

## Metrics

Key landmarks: 11/12 shoulders, 23/24 hips, 25/26 knees, 27/28 ankles, 15/16 wrists, 0 nose.

### v1 — DTL metrics (build these)

| Metric | How | At 60fps |
|--------|-----|----------|
| Spine tilt | mid-hip→mid-shoulder vector vs vertical, at P1 / P4 / P7 | ✅ at P1, P4 |
| Posture change | spine tilt at P4 minus spine tilt at P1 | ✅ |
| Hip depth | hip horizontal distance from the address butt-line (a vertical line at the hips at P1) | ⚠️ needs P7 |
| Head height | nose vertical position vs P1, normalized by shoulder width | ✅ P1→P4 |
| Head depth | nose horizontal drift toward/away from the ball | ⚠️ partial |
| Knee flex | knee angle at P1 vs P7 | ⚠️ needs P7 |
| Tempo ratio | `(P4−P1) / (P7−P4)` in **seconds**. Tour average ≈ 3:1 | ⚠️ ±7% |

In DTL the camera looks along the target line, so **image-horizontal is the ball-to-golfer axis** — hip
movement toward the ball is horizontal movement in frame. That's what makes early extension measurable.

### v2 — face-on metrics (deferred, do not build yet)

| Metric | How |
|--------|-----|
| Shoulder turn | `arccos(current_shoulder_width / address_shoulder_width)` — foreshortening trick |
| Hip turn | same trick on the hip line |
| X-factor | shoulder turn − hip turn at P4 (good players ~40–50°) |
| Weight shift | mid-hip horizontal position over time, normalized by shoulder width |

**Normalization is mandatory, and the reference must be club-invariant.** Divide every pixel distance by
**shoulder width** (or hip→shoulder distance). Otherwise standing 2 feet closer to the camera silently changes
all your numbers and session-over-session comparison is meaningless.

Do **not** normalize by stance width — stance width changes with club (driver is wider than a 7-iron), so the
identical physical sway would score differently across clubs and cross-club comparison breaks. Shoulder width
does not change with club.

---

## Club is metadata, never inferred

Tag the club at ingest via filename or dropdown (`2026-07-28_dtl_driver.mov`). Do not try to detect it from
pose — MediaPipe can't see the club, and body-pose proxies (stance width, hand height at top) vary more
between golfers than between clubs.

It matters because iron and driver swings are legitimately different, not better or worse:

| | Iron | Driver |
|---|---|---|
| Stance width | Narrower | Wider |
| Ball position | Center-ish | Forward, off lead heel |
| Spine tilt away from target at address | Less | ~5–10° more |
| Attack angle | Descending | Ascending |

That extra driver spine tilt is *correct technique*, but scored against iron thresholds it reads as a posture
fault — the app would confidently tell you to fix something you're doing right. Consequences:

- `thresholds.yaml` needs **per-club sections**, not one global set
- **Trend charts must filter by club** — mixed-club history is noise
- **v1: shoot one club only (7-iron)** until the pipeline works. Add clubs after.

---

## Fault rules

Each fault = a pure function over metrics → `(fired, measured_value, threshold, severity)`.

### v1 — DTL faults (build these)

| Fault | Rough rule | Confidence at 60fps |
|-------|------------|---------------------|
| Loss of posture | spine tilt at P4 differs from P1 by > 8° | **Solid** |
| Head lift | nose rises > 3% of shoulder width, P1→P4 | **Solid** |
| Early extension | hips move ballward > 4% of shoulder width between P4 and P7 | Directional |
| Excessive knee straightening | trail knee angle at P7 exceeds P1 by > 15° | Directional |
| Quick tempo | tempo ratio < 2.2 | Rough |

### v2 — face-on faults (deferred)

| Fault | Rough rule |
|-------|------------|
| Sway | mid-hip drifts away from target > 6% of shoulder width at P4 |
| Slide | excessive lateral hip drift toward target at P7 |
| Reverse pivot | head/weight moves *toward* target during backswing |
| Restricted turn | shoulder turn at P4 < 80° |

Thresholds live in `thresholds.yaml`, never in code, with **per-club sections** — tuning these is the main
activity for weeks 3–4.

**These numbers are invented until calibrated.** They are starting points, not truth.

---

## Data & storage

**`data/raw/` is precious. Everything else is disposable.**

```
data/raw/2026-07-28_dtl_7iron.mov         original — never modified, never auto-deleted
      ├──▶ data/processed/swing_014.npz   keypoints, ~2 MB, regenerable
      ├──▶ outputs/swing_014_annotated.mp4 skeleton overlay, regenerable
      └──▶ swings.db                       metrics + faults + the PATH above
```

The database never contains video — it stores the file path alongside that swing's
metrics. Video-as-blob makes the DB unusable.

Size reality: ~30–40 MB per 10s clip at 1080p/240fps (~125 MB at 4K/120fps), so 100
swings ≈ 5 GB. The entire metrics database is ~2 KB per swing — 1,000 swings is
about 2 MB, smaller than a single clip. Storage will never be the constraint.

**No upload widget — scan the folder instead.** Streamlit's uploader hands you an
in-memory buffer and saves nothing to disk; used naively you'd keep the metrics and
lose the video. It's also pointless locally: the file is already on the same disk.
Instead, drop clips into `data/raw/`, have the app list any video not yet in the
database, and click one to analyze. Makes batch-processing a whole range session
natural too.

**Backup:** keep the project folder in iCloud Drive. `*.sqlite` is gitignored on
purpose (binary DBs bloat history and conflict badly) — if git should be the safety
net, add a small tracked `history.csv` export instead.

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
| **0** | 2–3 hrs | **Use existing 60fps DTL footage.** ffprobe it. Run MediaPipe. Dump annotated video + per-frame visibility scores. **GO / NO-GO on the whole idea.** |
| **1** | Week 1 | ingest + pose + smoothing + persisted keypoints. CLI only. |
| **2** | Week 1–2 | Event detection + a frame-scrubber to hand-label ground truth. Highest risk. |
| **3** | Week 2 | DTL metrics with shoulder-width normalization. |
| **4** | Week 3 | Fault rules + per-club YAML thresholds + tuning. |
| **5** | Week 3–4 | Streamlit UI, SQLite history, trend charts. |
| **6** | +1–3 wks | *Optional.* Shaft-line detection → swing plane. Decide **after** Phase 3. |

---

## Phase 6 (optional): shaft-line detection

**Not committed to.** Revisit after Phase 3, when event detection works and you've seen how the shaft actually
looks against your range's background.

### Why it's gated behind Phase 3

Club tracking is useless without event detection. "Shaft angle at the top of the backswing" requires knowing
*which frame* is the top — every plane metric is anchored to P1/P2/P4/P7. The body pipeline isn't a detour
you'd skip by going straight to the club; it's the prerequisite.

### Approach: detect the shaft, not the clubhead

```
grayscale → Canny edge detect → Hough line transform
   → keep only lines passing within ~20px of the wrist midpoint
   → enforce frame-to-frame angular continuity
   → shaft angle per frame
```

**The wrist constraint is the whole trick.** A naive Hough transform on driving-range footage returns dozens
of straight lines — fence posts, mat edges, net cables, dividers, alignment sticks, the horizon. But wrist
position is already known from MediaPipe, and the shaft *must* emanate from the hands. Filtering to lines
passing near the wrist midpoint eliminates nearly every false positive. This is what makes the difference
between an unusable detector and a working one.

**Deliberately not the clubhead.** A trained detector (YOLO-class) would need 500–2,000 hand-labeled frames —
10–20 hours of labeling before training anything — and a clubhead at 100mph is a smeared streak even at
240fps, smallest and fastest exactly where you most want it. GolfDB has event labels but *no* clubhead boxes,
so it would all be self-labeled. That path is 1–3 months with real risk of never working well. Shaft-line
detection needs **zero training data**.

### What it buys

Shaft angle at address defines the plane line. Extend it and show where the shaft sits relative to it:

| Frame | Reveals |
|-------|---------|
| P1 address | Baseline shaft plane |
| P2 takeaway | On plane, inside, or outside |
| P4 top | Across the line vs laid off |
| Early downswing | **Over the top** vs shallowing |

That's the line instructors draw on video — the most recognizable diagnostic in golf, and the reason DTL
exists as an angle.

### Known failure modes

- **Motion blur through impact** — the shaft smears. Expect this to work at address, takeaway, halfway back,
  top, and early downswing; not at P7. At 60fps, blur starts earlier in the downswing.
- **Cluttered backgrounds** — range fences and dividers are straight lines. Mitigated by the wrist constraint.
- **Low contrast** — a dark shaft against dark background or dark clothing may be undetectable.

---

## The single highest-leverage thing

**Build a hand-labeled test set of your own swings in Phase 2.** 10–20 clips where you've manually marked the
P1/P4/P7/P10 frames.

Without it, every change to the detector is a guess and you'll thrash. With it, every change is measurable
(mean frame error vs. your labels). This is what separates the version that works from the version that's
abandoned in week 3.

---

## Capture setup (do this before writing any code)

- **Down-the-line (v1 angle):** camera sits *on the target line* behind you — extend the ball-to-target line
  backward through you and put the camera on it. **Hand height, ~10–12 ft away.** The common mistake is
  placing it behind your *body* instead of on the *line*, which skews every angle you measure.
- **Slo-Mo mode, 120 or 240fps** when possible. 60fps is workable for v1 (see constraint 1). Bright light
  matters more than frame rate — motion blur destroys pose accuracy.
- **Tripod, fixed height.** Camera height changes measured angles, so *consistency beats correctness*.
  Same spot, same height, every session. Mark it.
- **Fitted clothing**, contrasting with the background. Baggy clothes badly degrade pose estimation, and DTL
  is already the more occluded angle.
- **Same club throughout** (7-iron) for the first session — fewer variables.

---

## Filename convention

```
YYYY-MM-DD_<angle>_<club>_<nn>[_<fault>].mov
```

| Part | Purpose |
|------|---------|
| `2026-07-28` | ISO date **first** so files sort chronologically as plain text. `07-28-2026` sorts wrong forever. |
| `dtl` / `fo` | Angle decides which metrics are valid. Face-on metrics must never be computed on a DTL clip. |
| `7iron` | Selects the per-club threshold set. |
| `01` | Sequence within the session, **zero-padded**. Computers sort text, not numbers — unpadded gives `1, 10, 11, 2`. Resets each session; it is not a global ID. |
| `_earlyext` | **Optional, deliberate-fault clips only.** |

A session:

```
2026-07-28_dtl_7iron_01.mov            normal
2026-07-28_dtl_7iron_08.mov            normal
2026-07-28_dtl_7iron_09_posture.mov    deliberate fault
2026-07-28_dtl_7iron_10_earlyext.mov   deliberate fault
```

**The fault tag is the ground-truth label, not a note.** In Phase 4 threshold tuning
asks "does the early-extension rule fire on the clips where I deliberately stood up?"
The filename encodes the expected output. No tag means "no fault expected."

Tags match the v1 fault names: `_posture`, `_earlyext`, `_headlift`, `_quicktempo`.

**Rules:**

- **Lowercase, no spaces** — spaces force quoting on every shell command forever.
- **No redundant words** — everything in `data/raw/` is already a golf swing of yours.
- **Name once at export, never rename.** The DB stores the file path; renaming after
  ingest orphans that swing's history.

**The filename is a convenience for seeding the DB at ingest, not the source of truth.**
Once ingested, the database owns date, angle, club, and tag. Don't encode ball flight,
conditions, or notes — those are fields, not filename segments.
- *Face-on (v2, later):* perpendicular to target line, hand/belt height, ~10–12 ft away.

### Film deliberate faults, not just your normal swing

Shoot ~15 clips: **8 normal, 6 with one exaggerated fault each** (sway off the ball,
lose posture, stand up through impact, reverse pivot, quick tempo), 1 slow practice swing.

If every clip is your ordinary swing there is nothing to calibrate against — every
threshold in `thresholds.yaml` stays a guess and you can't distinguish a working
detector from a broken one. Exaggerated faults give known-positive examples: if the
early-extension rule doesn't fire on a swing where you deliberately stood up, the rule
is wrong. This turns Phase 4 from guesswork into measurement.

