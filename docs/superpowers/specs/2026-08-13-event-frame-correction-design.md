# Event Frame Correction — Design

**Date:** 2026-08-13
**Status:** Approved, ready for implementation planning

## Purpose

The detector picks P1, P4, P7 and P10 automatically. P7 is wrong often enough to
matter, and when it is wrong every metric derived from it is wrong too — quietly,
because a misplaced impact frame usually yields a plausible number rather than an
obvious failure.

Correcting a frame today means leaving the app: run `label_swing.py`, look at the
strip, hand-edit `data/labels/<clip>.json`, re-run the pipeline. `label_swing.py`
says as much in its own docstring — it was built as the cheap substitute for "an
interactive scrubber."

This adds the scrubber, in the app, and makes every correction produce verified
ground truth as a side effect of normal use.

## Scope

**In scope:** an in-app scrubber for all four events; recompute and persist the
affected metrics; write a verified label; record that a swing's events were
corrected by hand.

**Out of scope, deliberately:** any feedback into detection logic. No auto-tuning
of thresholds, no fitting parameters to labels. See Deferred below for why, and
for the condition that would change the answer.

## Key decisions

| Decision | Rationale |
|---|---|
| A correction is a manual override, not detector input | 19 verified labels across 26 clips is enough to *measure* a detector change, not to *fit* parameters. Fitting on all available labels would also destroy the ability to score the detector against them. |
| A correction also writes a verified label | Both halves already exist and have never been connected. The expensive part — deciding which frame is right — is already being done by hand. |
| All four events, not only P7 | `labels.EVENT_KEYS` already covers all four. One selector; nearly free. P7 is simply the one most often wrong. |
| Recompute from the cached `.npz`, never re-run pose | `store.load()` is why this is instant. Pose extraction is the slow part and it has already happened. |
| Metrics are recomputed, never hand-edited | `metrics.compute()` is a pure function of sequence and events. A hand-edited metric would be unfalsifiable — it could not be reproduced or checked. |
| Ordering is enforced, not assumed | `p1 < p4 < p7 < p10` is relied on throughout. The detector can already break it (see Related defect), so the correction path must not add a second way in. |

## Architecture

```
        swing page, event frames shown
                    │
        "P7 looks wrong" → open corrector
                    │
        pick event (P1/P4/P7/P10), scrub frames
                    │
        confirm frame N
                    │
   ┌────────────────┴─────────────────┐
   │ store.load(clip.npz)             │  cached landmarks, no mediapipe
   │ replace(events, p7=N)            │  SwingEvents is frozen — copy, don't mutate
   │ validate p1 < p4 < p7 < p10      │  reject rather than persist a broken swing
   │ metrics.compute(sequence, events)│  pure; all metrics recomputed together
   │ db.update_events(...)            │  frames + metrics + events_source
   │ labels.save_labels(verified=True)│  ground truth, for free
   └──────────────────────────────────┘
```

Nothing recomputes pose. The whole operation is a file read, an arithmetic pass,
and two writes.

## What exists and what is new

| Piece | Status |
|---|---|
| Load cached landmarks | exists — `store.load()` |
| Recompute metrics from events | exists — `metrics.compute(sequence, events)`, already pure |
| Draw a skeleton on a frame | exists — `skeleton.draw()`, `skeleton.key_frames()` |
| Read frames from a video | exists — `ingest.read_frames_with_times()` |
| Write a swing row | exists — `db.save_swing()`, but see Named risk |
| Write verified ground truth | exists — `labels.save_labels()` |
| Fetch a single frame by index | **new** — `ingest.frame_at(video, index)`, beside the existing frame reader and reusing its rotation handling |
| The scrubber UI | **new** — Streamlit selector, slider, image, confirm |
| Record that events were corrected | **new** — one column |
| A write path that cannot blank other columns | **new** — see Named risk |

## Data model

`swings` gains one column:

```sql
events_source TEXT NOT NULL DEFAULT 'detected'
```

Values: `'detected'` or `'corrected'`. A text column rather than a boolean so a
third source (for example `'fitted'`, if the deferred work ever happens) does not
require a migration.

`data/labels/<clip>.json` is unchanged — `save_labels` already writes the shape
`evaluate_events.py` expects, including the `verified` flag it filters on.

## Named risk: `INSERT OR REPLACE` blanking columns

`db.save_swing()` carries its own warning:

> `# INSERT OR REPLACE rewrites the whole row, blanking any column not listed.`

`outcome` is already patched back immediately afterward for exactly this reason.
A correction path re-saving a swing would have the same hazard for every
out-of-band column — `outcome`, `fault_tag`, `club`, `date`, and the new
`events_source`. Losing a `fault_tag` while fixing a frame would corrupt the
baseline silently, since tagged clips are excluded from it.

**Mitigation:** do not reuse `save_swing()` for corrections. Add a dedicated
`db.update_events(conn, clip, events, metrics)` that issues an explicit `UPDATE`
touching only the four frame columns, the metric columns, and `events_source`.
An `UPDATE` cannot blank a column it does not name, which removes the failure
mode rather than patching around it.

## User interface

On the swing page, beside the existing key-frame images:

- An expander, collapsed by default — this is a correction path, not the main flow.
- A selector for which event to correct, defaulting to P7.
- A slider over `current ± CORRECTION_WINDOW_FRAMES`, clamped to the clip's
  bounds. `CORRECTION_WINDOW_FRAMES = 15` — at 120fps that is an eighth of a
  second either side, wide enough to contain the detector's error and narrow
  enough to scrub without hunting.
- The selected frame rendered with the skeleton drawn, so the choice is made on
  the same visual evidence the detector used.
- A confirm button naming the frame explicitly.
- After confirming: metrics recomputed, page re-runs, and the swing shows a
  "corrected" marker so a hand-set frame is never mistaken for a detected one.

An ordering violation disables the confirm button and says which constraint fails.

## Verification

Unit, against a fixture clip with cached landmarks:

- Applying a correction recomputes metrics deterministically — same events in,
  same metrics out.
- A correction preserves `outcome`, `fault_tag`, `club` and `date` (the named
  risk above, tested directly).
- A correction writes a label with `verified: true`.
- A correction sets `events_source` to `'corrected'`.
- An out-of-order correction is rejected and nothing is written — no partial state
  where the label moved but the metrics did not.
- Correcting an event back to the detector's original value is not special-cased;
  it still records as corrected, because a human confirmed it.

Manual, once:

- Correct a clip whose P7 is visibly wrong, confirm the tempo ratio changes, and
  confirm `evaluate_events.py` picks up the new label in its score.

## Related defect

The review of `events.py` found that `_refine_impact` computes its window as
`lo = max(0, coarse - half)`, with no floor at `p4`. It can therefore select an
impact frame at or before the top of the backswing, breaking the ordering
invariant. `tempo_ratio` then returns NaN and the failure is laundered into an
ordinary missing metric.

That is a likely contributor to the bad P7s motivating this feature. It should be
fixed separately — it is a detector bug, not a UI gap — but it is the natural
first use of the labels this feature will produce: fix it, run
`evaluate_events.py` before and after, and see whether the P7 error actually drops.

## Deferred: fitting the detector to labels

Not now, and for a specific reason rather than a vague one.

With 19 verified labels from one golfer, tuning `IMPACT_REFINE_SECONDS` or the
peak thresholds against them would fit the parameters to those clips rather than
to golf swings. Worse, a detector tuned on all available labels can no longer be
honestly scored against them — the measurement property that `labels.py` exists to
provide would be spent to buy a fit that cannot be trusted.

The value of this feature is that labelling becomes incidental to reviewing
swings instead of a separate chore, so the label count grows without deliberate
effort. Revisit fitting when there are enough verified labels to hold a
meaningful test split back — on the order of 60, not 19.
