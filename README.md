# Golf Swing Analyzer

Film your swing at the range. Drop the clips in. Get measurements of what your
body actually did — and how this swing compares with the ones you struck well.

Runs entirely on your laptop. No cloud, no account, no upload. Your video never
leaves your machine.

---

## What it does

- **Finds the four key positions** in every swing automatically — address, top
  of backswing, impact, finish — accurate to under half a frame at 120fps
- **Measures six things** your body did: posture change, hip movement toward the
  ball, head rise at the top and at impact, knee straightening, and tempo
- **Compares each swing against your own history**, not against a textbook —
  "the most hip movement of your 16" is something it can actually prove
- **Draws the skeleton on the key frames**, colouring whatever differed most
- **Writes a plain-English read** of the numbers (optional, needs an API key)

## What it deliberately doesn't do

**It won't tell you a swing was good or bad.** There is no validated standard
for these measurements, so it compares you against yourself instead. That's a
real comparison; a scored verdict would be invented.

**It can't see the club.** Pose estimation gives 33 body joints and nothing
else — so no club path, no face angle, no swing plane, no clubhead speed. Body
motion only.

**It's down-the-line only.** Face-on footage imports but nothing is computed
from it yet.

**One clip per swing.** It doesn't split a continuous range video into
individual swings.

---

## Requirements

- **macOS** (the desktop app is Mac-only; the web UI runs anywhere)
- **Python 3.11+**
- **ffmpeg** — `brew install ffmpeg`
- A phone that shoots **60fps or better**. 120fps is much better.

## Setup

```bash
git clone https://github.com/eurookim/golf-swing-analyzer.git
cd golf-swing-analyzer

python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Optional: a written coaching note on each swing
cp .env.example .env          # then paste an Anthropic API key into it
```

The pose model (~30MB) downloads by itself the first time you process a clip.

### Launching it

```bash
.venv/bin/streamlit run app.py
```

Or build a double-clickable Mac app:

```bash
./install_app.sh              # creates ~/Applications/Golf Swing Analyzer.app
```

---

## Using it

**1. Film.** Camera on the target line behind you, hand height, ~10–12 ft back.
Tripod fixed, one clip per swing, bright light. Full guide in
[`CAPTURE.md`](CAPTURE.md) — the setup matters more than anything in this repo.

**2. Import.** Drop the clips into `data/raw/`, open the app, go to **Add
swings**. Anything not already named to the convention it asks you about once
and renames for you. Roughly 20 seconds a clip to analyse.

**3. Label.** On each swing, mark whether you flushed it or mishit it. Once five
of a club are marked flushed, every comparison switches to measuring against
those — *"more than the ones you struck well"* rather than *"more than usual"*.

**4. Look.** Video, the six measurements with how each compares, key frames with
the skeleton drawn on, and a written note if you set up a key.

---

## Fault detection, and why it's off

There's a rules engine that names faults — loss of posture, early extension,
head lift, knee straightening, quick tempo. **It's disabled, because the
thresholds aren't calibrated.**

Calibrating needs clips where you *deliberately and obviously* exaggerate each
fault, so the app can find a value separating those from your normal swings.
The **Calibration** view shows which rules have enough evidence and which don't,
and names exactly what's missing.

The bar to clear when filming those:

> **If you still hit the ball decently, the fault wasn't big enough.**

Until then the app reports rank within your own swings, which needs no
thresholds and is true today.

---

## How it works

```
video → ffprobe (true fps, rotation) → MediaPipe Pose (33 joints/frame)
      → Savitzky-Golay smoothing → event detection (P1/P4/P7/P10)
      → metrics normalised by torso length → comparison against your history
```

Every distance is divided by **torso length** — the only reference that is both
club-invariant and viewpoint-invariant from down-the-line. Every timing constant
is in seconds rather than frames, so 60/120/240fps footage behaves identically.

Detector accuracy, measured against 15 hand-labelled 120fps swings:

| Event | mean error |
|---|---|
| P4 top of backswing | 0.4 frames |
| P7 impact | 0.4 frames |

Those labels are in the repo, in `data/labels/` — frame numbers only, no video.
They are checked in deliberately: without ground truth every change to the event
detector is a guess, and re-labelling by hand is the expensive part. You will
have labels for swings you do not have footage of. That is expected — they score
the detector, they are not your data, and nothing reads them unless the matching
clip is present.

[`PLAN.md`](PLAN.md) has the full architecture, every metric definition, and the
reasoning behind each decision — including the ones that turned out wrong.

---

## Project layout

```
app.py            the UI            desktop.py    native Mac window
golfswing/        the package       scripts/      one-off tools
tests/            341 tests         data/raw/     your videos (gitignored)
```

```bash
.venv/bin/pytest -q
```

---

## Privacy

Everything is local. Videos, keypoints and the database live in `data/` and are
gitignored. The **only** thing that ever leaves your machine is the coaching
note, if you enable it — and that sends six numbers, never your video.
