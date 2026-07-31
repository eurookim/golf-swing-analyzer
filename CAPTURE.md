# Range session brief

One session. One club. 15 clips. ~25 minutes.

This is the calibration set — the footage that turns invented thresholds into
measured ones. Everything below has a reason behind it; the reasons are at the
bottom if you want them.

---

## If you already shot a normal-swing baseline

**You do not need to repeat it.** The 2026-07-29 session banked 14 verified
normal swings, which is the expensive half. What is still missing is fault clips
that are genuinely exaggerated.

A follow-up session is short — **~10 clips, two per fault**, plus a couple of
normals to confirm the setup matches. Read the exaggeration test below first;
it is the reason a second session is needed at all.

---

## 1. Camera settings

| Setting | Use | Why |
|---------|-----|-----|
| **Mode** | **Slo-Mo** | 60fps works but impact lands between frames |
| **Resolution** | **1080p** | Not 4K — 4× the file size, no accuracy gain, and caps you at 120fps |
| **Frame rate** | **120fps** default · **240fps** in bright sun | 240 needs a fast shutter; in poor light it goes dark and noisy |

**1080p120 is the answer unless it's genuinely bright, then 1080p240.**

Bright light matters more than frame rate. A sharp 120fps frame beats a noisy
240fps one.

---

## 2. Camera position — the thing that matters most

- **ON the target line**, behind you. Extend the ball→target line backward
  through yourself and stand the camera on it. **Not behind your body — behind
  the line.**
- **Hand height** (roughly belt-to-hand).
- **10–12 ft away.**
- **Tripod. Mark the spot.** Note the height setting.

**Set it once and do not move it.** Zooming is fine — it changes nothing that
matters. *Walking the tripod closer is not fine*, because perspective changes and
every angle shifts with it.

Across your last four clips, camera alignment varied by **4.9×**. That is the
single largest source of noise in your data, and it is entirely fixable by not
touching the tripod.

---

## 3. The shot list

**Stop-start: one clip per swing.** Tap before, tap after. ~1 second of stillness
before you take the club back — no need to stand there.

### Clips 1–10: normal swings

Your ordinary swing. **7-iron, every one.** Don't try to swing well; swing
*normally*. These establish your baseline.

### Clips 11–15: one deliberately exaggerated fault each

| Clip | Fault | Do this |
|------|-------|---------|
| 11 | **Loss of posture** | Stand up out of your address angles during the backswing |
| 12 | **Early extension** | Thrust your hips toward the ball coming into impact |
| 13 | **Head lift** | Let your head rise noticeably going back |
| 14 | **Knee straightening** | Snap your trail leg straight through impact |
| 15 | **Quick tempo** | Rush the backswing — snatch it away |

### ⚠️ The exaggeration test — this is where the first attempt failed

> **If you still hit the ball decently, the fault wasn't big enough.**

Apply that standing there, shot by shot. A real early extension leaves you nearly
upright at impact, topping or thinning it. A real loss of posture produces a
genuinely bad strike.

**The fault clips should not look like your swing with a flaw. They should look
like a different, obviously broken swing.** You are not demonstrating a tendency
— you are marking the far end of the scale so a threshold has somewhere to sit.

This is not a stylistic preference. On the first 20-clip session the deliberate
faults measured *inside* the normal range on every metric and every interval
tried, so 4 of 5 rules could not be calibrated at all:

| Rule | Normal swings | Deliberate fault |
|------|---------------|------------------|
| loss_of_posture | −2.3 … 1.11° | **0.54°** (inside) |
| early_extension | 0.162 … 0.276 | **0.148** (below all 14 normals) |
| quick_tempo | 2.16 … 6.29 | **2.47** (inside) |

A subtle fault clip is worse than no fault clip: it costs a swing and yields
nothing, because it cannot be told apart from a normal swing — which is exactly
the question being asked.

### Order matters

Shoot **all ten normal swings first**, then the five faults. Fatigue changes your
swing, and mixing them would leave us unable to tell a threshold effect from a
tiredness effect.

Warm up **before** you start recording. The first few swings of a session are not
representative.

---

## 4. Naming

```
2026-08-02_dtl_7iron_01.mov          normal
2026-08-02_dtl_7iron_11_posture.mov  deliberate fault
```

Fault tags: `_posture` · `_earlyext` · `_headlift` · `_kneestraight` · `_quicktempo`

The tag is not a note — it is the **expected answer**. Calibration asks "does the
early-extension rule fire on the clip where I deliberately did it?"

Zero-pad the numbers (`01`, not `1`) or they sort wrong. `scripts/rename_session.py` does
all of this for you.

---

## 5. Getting them off the phone

**Photos → select clips → File → Export → "Export Unmodified Original"** into
`data/raw/`.

**Not** plain "Export", **not** drag-and-drop. Both re-encode, and for Slo-Mo that
bakes the slow motion in — you get a 30fps file that looks fine and silently
ruins every measurement. `scripts/phase0_check.py` reports true capture fps as its first
output specifically to catch this.

If the clips are only on the phone: AirDrop (which preserves the original), then
move them from `~/Downloads`.

---

## 6. Also worth doing

- **Fitted clothing**, contrasting with the background. Baggy shirts degrade pose
  tracking, and down-the-line is already the more occluded angle.
- **Film a few face-on clips at the end** if you have battery left. Useless for v1,
  but free, and they seed v2.

---

## Why each rule exists

| Rule | Reason |
|------|--------|
| One club | Club changes correct technique. Mixing clubs means a threshold can't be attributed to either. |
| One session | Camera setup varies between sessions more than your swing does. |
| Don't move the tripod | Zoom is magnification; distance changes perspective. Only one of those is harmless. |
| Exaggerate the faults | A known-positive is the only thing that makes a threshold falsifiable. |
| Normal swings first | Otherwise fatigue and fault are confounded. |
| 15 clips | Enough for a distribution; few enough to shoot and label in one sitting. |

If you only remember two things: **don't move the tripod**, and **exaggerate the
five fault swings.**
