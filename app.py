"""Golf swing analyzer — local Streamlit UI.

    .venv/bin/streamlit run app.py     (or open the app in ~/Applications)

The swing page is the home page: video, the few measurements that stand out,
then everything else. Distributions and Trend are calibration and history
views, reached from the sidebar.
"""

from __future__ import annotations

import math
import subprocess
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from golfswing import (calibrate, coach, db, faults, history, naming,
                       pipeline, pose, preview, skeleton, store, ui)

from golfswing.paths import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
VIDEO_SUFFIXES = (".mov", ".MOV", ".mp4", ".MP4", ".m4v")

METRIC_LABELS = {
    "posture_change": "Posture change (deg)",
    "hip_depth_change": "Hip depth / early extension (x torso)",
    "head_rise_p4": "Head rise to top (x torso)",
    "head_rise_p7": "Head rise to impact (x torso)",
    "head_depth_p7": "Head depth at impact (x torso)",
    "knee_extension_change": "Knee extension (deg)",
    "tempo_ratio": "Tempo (backswing : downswing)",
    "spine_tilt_p1": "Spine tilt at address (deg)",
}

# Which rule each metric feeds, so the current threshold can be drawn on it.
METRIC_TO_RULE = {rule.metric: rule.name for rule in faults.RULES}

st.set_page_config(page_title="Golf Swing Analyzer", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource
def _conn():
    return db.connect()


def find_video(clip: str) -> Path | None:
    """Locate the original clip. Suffix case varies with how it was exported."""
    for suffix in VIDEO_SUFFIXES:
        candidate = RAW_DIR / f"{clip}{suffix}"
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(ui.SURFACE)
    ax.grid(True, color=ui.LINE, linewidth=0.8, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=ui.MUTED, labelsize=9)


def distribution_plot(rows, metric, threshold=None, rule=None, highlight=None):
    """Strip plot: every swing as a dot, this rule's known-positive marked.

    A dot strip rather than a histogram — with 15-20 swings a histogram's bins
    hide exactly the thing being looked for, which is whether the tagged fault
    sits outside the normal cluster.

    **Only the clip tagged for THIS rule is drawn as a fault.** A swing where
    the golfer deliberately lifted their head is not a known-positive for early
    extension, and marking it as one would imply evidence that does not exist.
    Clips tagged for other rules are plotted as ordinary swings, matching how
    calibrate.py groups them.
    """
    normal, tagged, marked = [], [], None
    for row in rows:
        value = row[metric]
        if value is None:
            continue
        if row["clip"] == highlight:
            marked = value
        if rule is not None and row["fault_tag"] == rule:
            tagged.append((value, row["clip"]))
        else:
            normal.append(value)

    fig, ax = plt.subplots(figsize=(9, 1.7), facecolor=ui.SURFACE)
    _style(ax)

    if normal:
        jitter = np.random.default_rng(0).normal(0, 0.045, len(normal))
        ax.scatter(normal, jitter, s=70, color=ui.NORMAL, alpha=0.75,
                   edgecolors=ui.SURFACE, linewidths=1.5, zorder=3)
    for offset, (value, clip) in enumerate(sorted(tagged)):
        ax.scatter([value], [0], s=190, marker="D", color=ui.FAULT,
                   edgecolors=ui.SURFACE, linewidths=2, zorder=5)
        # Stagger labels so adjacent diamonds do not overprint each other.
        ax.annotate(clip.split("_")[-1], (value, 0), textcoords="offset points",
                    xytext=(0, 16 + 13 * (offset % 2)), ha="center", fontsize=9,
                    color=ui.FAULT, fontweight="bold")

    if marked is not None:
        ax.scatter([marked], [0], s=150, facecolors="none",
                   edgecolors=ui.INK, linewidths=2, zorder=6)

    if threshold is not None and math.isfinite(threshold):
        ax.axvline(threshold, color=ui.INK_2, linestyle="--", linewidth=1.5, zorder=2)
        ax.annotate("threshold", (threshold, -0.16), textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color=ui.INK_2)

    ax.set_ylim(-0.25, 0.25)
    ax.set_yticks([])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def page_swing(rows):
    clips = [r["clip"] for r in rows]
    if not clips:
        st.info("No swings in history yet.")
        return

    # The picker lives in the sidebar with the other controls. Unlabelled at the
    # top of the page it read as a caption — especially with the heading below
    # repeating the same name — so it was not discoverable as a control at all.
    chosen = _swing_picker(clips)
    row = next(r for r in rows if r["clip"] == chosen)
    found = coach.standings(rows, chosen)

    st.markdown(
        ui.heading(
            chosen.replace("_", " · "),
            f"{row['fps']:.0f} fps  ·  events P1 {row['p1']} · P4 {row['p4']} · "
            f"P7 {row['p7']} · P10 {row['p10']}"
            + ("  ·  deliberate practice fault" if row["fault_tag"] else ""),
        ),
        unsafe_allow_html=True,
    )

    # Portrait footage needs a narrow column, but at 2:5 the clip was
    # width-limited well below its 82vh height cap. 3:5 lets it reach that cap;
    # the tiles shrink slightly to pay for it (see .metric-tile).
    video_col, stats_col = st.columns([3, 5], gap="large")
    with video_col:
        video = find_video(chosen)
        if video:
            try:
                playable = preview.playable(video)
            except (OSError, subprocess.CalledProcessError) as failure:
                st.warning(f"Could not prepare the video: {failure}")
            else:
                shape = preview.dimensions(playable)
                if shape:
                    # Reserve the right box before the video loads.
                    st.markdown(
                        f"<style>video {{ aspect-ratio: {shape[0]} / {shape[1]}; }}"
                        f"</style>", unsafe_allow_html=True)
                st.video(str(playable))
        else:
            st.caption(f"No video at {RAW_DIR}/{chosen}.*")

    with stats_col:
        _outcome_control(row)
        st.markdown(ui.summary(found), unsafe_allow_html=True)
        # Every measurement, ordered most-unusual first. No section header:
        # showing a subset used to need one explaining the selection, and that
        # line never changed and overclaimed. Ordering says it instead.
        st.markdown(ui.tile_grid(found), unsafe_allow_html=True)
        if not any(s.rank for s in found):
            st.info(
                "Not enough swings on record to rank this one yet — "
                f"{coach.MIN_PEERS_FOR_RANKING} of the same club are needed."
            )

    st.markdown(ui.section("Key frames"), unsafe_allow_html=True)
    _key_frames(chosen, row, found)

    st.markdown(ui.section("Coaching note"), unsafe_allow_html=True)
    _coaching_note(found, chosen)

    # The measurements table used to live here. Every column it carried —
    # value, your median, rank — is now on the tiles themselves, so it was the
    # same six rows printed twice on one page.
    with st.expander("What do these measurements mean?"):
        st.markdown(ui.guidance(), unsafe_allow_html=True)
        st.caption(
            "\"Steadier than usual\" compares this swing with the median of "
            "your own swings using the same club. It is not a verdict on "
            "whether the swing was good — this app has no validated standard "
            "for that, and would be guessing if it claimed one."
        )


def _swing_picker(clips: list[str]) -> str:
    """Sidebar chooser plus prev/next, for flipping through a session."""
    if "clip_index" not in st.session_state:
        st.session_state.clip_index = len(clips) - 1
    st.session_state.clip_index = ui.clamp_index(
        st.session_state.clip_index, len(clips))

    st.sidebar.markdown("**Swing**")
    back, forward = st.sidebar.columns(2)
    if back.button("← Prev", width="stretch",
                   disabled=st.session_state.clip_index == 0):
        st.session_state.clip_index -= 1
    if forward.button("Next →", width="stretch",
                      disabled=st.session_state.clip_index >= len(clips) - 1):
        st.session_state.clip_index += 1

    chosen = st.sidebar.selectbox(
        "Swing", clips,
        index=ui.clamp_index(st.session_state.clip_index, len(clips)),
        label_visibility="collapsed",
    )
    # Keep the arrows in step when the dropdown is used directly.
    st.session_state.clip_index = clips.index(chosen)
    return chosen


OUTCOME_LABELS = {None: "Not recorded", "flushed": "Flushed it",
                  "mishit": "Mishit", "unsure": "Not sure"}


def _outcome_control(row):
    """Record how the shot turned out.

    The camera sits behind the golfer down the target line, so it never sees
    where the ball finished — this is the one thing about a swing that cannot
    be recovered from the footage, only remembered.
    """
    options = [None, "flushed", "mishit", "unsure"]
    current = row["outcome"] if row["outcome"] in options else None
    chosen = st.radio(
        "How was this one?", options, index=options.index(current),
        format_func=lambda v: OUTCOME_LABELS[v], horizontal=True,
    )
    if chosen != current:
        db.set_outcome(_conn(), row["clip"], chosen)
        st.rerun()


def _key_frames(clip, row, found):
    video = find_video(clip)
    if video is None:
        st.caption("No video for this clip, so no key frames.")
        return
    try:
        from golfswing.events import SwingEvents
        sequence = store.load_sequence(PROCESSED_DIR / f"{clip}.npz")
        strip = skeleton.key_frames(
            video, sequence,
            SwingEvents(row["p1"], row["p4"], row["p7"], row["p10"]), found)
    except (OSError, ValueError) as failure:
        st.caption(f"Could not draw the key frames: {failure}")
        return
    if strip is None:
        st.caption("Could not read the video frames.")
        return

    st.image(strip[:, :, ::-1], width="stretch")     # BGR -> RGB
    st.caption(
        "Joints coloured where a measurement differs from your flushed swings — "
        "red for more movement, green for less, plain white where it is typical "
        "or not measured at that point. Address is never coloured: it is the "
        "reference the others are measured from."
    )


def _coaching_note(found, clip):
    if not coach.available():
        st.info(
            "Set an Anthropic API key for a written read of these numbers:\n\n"
            "```\nexport ANTHROPIC_API_KEY=sk-ant-...\n```\n"
            "Everything else works without it."
        )
        return

    if st.button("Explain this swing", key=f"explain_{clip}", type="primary"):
        with st.spinner("Reading the numbers…"):
            try:
                note = coach.explain(found, clip)
            except RuntimeError as failure:
                st.error(str(failure))
            else:
                st.markdown(f'<div class="gs-note">{note}</div>',
                            unsafe_allow_html=True)


SECONDS_PER_CLIP = 22          # measured on a 2.4s 120fps clip


def page_import(conn):
    st.markdown(
        ui.heading("Add swings",
                   f"Drop clips into {RAW_DIR} — from Photos, AirDrop, anywhere."),
        unsafe_allow_html=True,
    )

    # Results are stashed and shown after a rerun. Rendering them inline left
    # the "needs details" form on screen listing a file that had already been
    # renamed and processed.
    result = st.session_state.pop("import_result", None)
    if result:
        st.success(result["summary"])
        for message in result["warnings"]:
            st.warning(message)

    pending = history.pending_clips()
    if not pending:
        if not result:
            st.success("Everything in data/raw has been processed.")
        else:
            st.caption("Open **Swing** in the sidebar to see them.")
        return

    ready = [p for p in pending if naming.follows_convention(p.stem)]
    unnamed = [p for p in pending if not naming.follows_convention(p.stem)]

    if ready:
        st.markdown(ui.section(f"Ready to import — {len(ready)}"),
                    unsafe_allow_html=True)
        for path in ready:
            meta = history.parse_clip(path.stem)
            st.markdown(
                f"<div style='font-size:14px;color:var(--secondary);'>{path.name}"
                f"<span style='color:var(--muted)'> · {meta.date} · {meta.angle}"
                f" · {meta.club}"
                + (f" · {meta.fault_tag}" if meta.fault_tag else "")
                + "</span></div>", unsafe_allow_html=True)

    answers: dict[Path, dict] = {}
    if unnamed:
        st.markdown(ui.section(f"Needs details — {len(unnamed)}"),
                    unsafe_allow_html=True)
        st.caption(
            "The filename is the only record of a clip's club and angle, and a "
            "swing whose club is unknown can never be ranked — ranking compares "
            "against the same club only. So these are asked for, never guessed."
        )
        same = st.checkbox(
            "All of these are from one session (ask once)", value=True)

        for position, path in enumerate(unnamed):
            first = position == 0
            if same and not first:
                answers[path] = dict(answers[unnamed[0]])
                continue
            st.markdown(f"**{path.name}**")
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            answers[path] = {
                "when": c1.date_input("Date", value=date.today(),
                                      key=f"d_{path.name}"),
                "angle": c2.selectbox("Angle", sorted(naming.ANGLES),
                                      key=f"a_{path.name}"),
                "club": c3.text_input("Club", value="7iron",
                                      key=f"c_{path.name}"),
                "fault": c4.selectbox("Deliberate fault",
                                      ["none", *sorted(naming.FAULT_TAGS)],
                                      key=f"f_{path.name}"),
            }

    seconds = len(pending) * SECONDS_PER_CLIP
    estimate = (f"{seconds}s" if seconds < 90
                else f"{seconds / 60:.0f} min")
    plural = "" if len(pending) == 1 else "s"
    st.markdown(ui.section("Import"), unsafe_allow_html=True)
    st.caption(
        f"About {estimate} for {len(pending)} clip{plural} "
        f"(~{SECONDS_PER_CLIP}s each). Keep this window open — pose extraction "
        "runs here, and closing it stops the job."
    )
    if not st.button(f"Import {len(pending)} swing{plural}", type="primary"):
        return

    _run_import(conn, ready, unnamed, answers)


def _run_import(conn, ready, unnamed, answers):
    """Rename what needs it, extract keypoints, then file everything."""
    renamed, failures = list(ready), []

    for path in unnamed:
        choice = answers[path]
        try:
            index = naming.next_index(RAW_DIR, choice["when"],
                                      choice["angle"], choice["club"])
            renamed.append(naming.rename_to_convention(
                path, choice["when"], choice["angle"], choice["club"], index,
                None if choice["fault"] == "none" else choice["fault"],
            ))
        except (ValueError, FileExistsError, OSError) as failure:
            failures.append(f"{path.name}: {failure}")

    if not renamed:
        st.error("Nothing to import.")
        for message in failures:
            st.warning(message)
        return

    pose.ensure_model()
    bar = st.progress(0.0, text="Starting…")
    done = 0
    for position, path in enumerate(renamed, start=1):
        bar.progress((position - 1) / len(renamed), text=f"Analysing {path.name}…")
        try:
            pipeline.process_clip(path, out_dir=PROCESSED_DIR)
            done += 1
        except Exception as failure:          # one bad clip must not stop the rest
            failures.append(f"{path.name}: {failure}")
    bar.progress(1.0, text="Filing results…")

    written, skipped = history.sync(conn)
    bar.empty()

    plural = "" if len(renamed) == 1 else "s"
    st.session_state["import_result"] = {
        "summary": f"{done} of {len(renamed)} clip{plural} analysed · "
                   f"{written} swings on file",
        "warnings": failures + skipped,
    }
    if done:
        st.session_state.pop("clip_index", None)   # land on the newest swing
    st.rerun()


STATUS_STYLE = {
    "ready": ("verdict-better", "ready"),
    "not_separable": ("verdict-worse", "can't separate"),
    "no_example": ("verdict-typical", "no example yet"),
}


def page_calibration(rows, club):
    st.markdown(
        ui.heading("Calibration",
                   "Whether each fault rule can be told apart from a normal "
                   "swing yet. Until a rule is ready, the app reports rank "
                   "within your own swings instead of naming a fault."),
        unsafe_allow_html=True,
    )

    found = calibrate.assess(rows, club=None if club == "all" else club)
    ready = sum(1 for a in found if a.status == "ready")
    st.markdown(
        f'<div class="summary-line"><span class="count">{ready} of '
        f'{len(found)}</span> rules can be calibrated from the swings on file.'
        f'</div>', unsafe_allow_html=True)

    for entry in found:
        style, word = STATUS_STYLE[entry.status]
        st.markdown(
            f'<div class="section-label" style="margin-top:26px">{entry.rule}'
            f'<span class="verdict {style}" style="margin-left:10px;'
            f'text-transform:none;letter-spacing:0">{word}</span></div>',
            unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            span = ("—" if entry.normal_low is None
                    else f"{entry.normal_low:+.3g} … {entry.normal_high:+.3g}"
                         f"  (n={entry.n_normal})")
            faults_seen = (", ".join(f"{v:+.3g}" for v in entry.fault_values)
                           or "none filmed")
            st.markdown(
                f"<div style='font-size:14px;line-height:1.7'>"
                f"normal swings &nbsp;<code>{span}</code><br>"
                f"deliberate fault &nbsp;<code>{faults_seen}</code><br>"
                f"<span style='color:var(--muted)'>{entry.note}</span></div>",
                unsafe_allow_html=True)
        with right:
            current = "—" if entry.current is None else f"{entry.current:.4g}"
            st.markdown(f"<div style='font-size:14px'>current limit "
                        f"<code>{current}</code></div>", unsafe_allow_html=True)
            if entry.status == "ready":
                st.markdown(f"<div style='font-size:14px'>suggested "
                            f"<code>{entry.suggested:.4g}</code></div>",
                            unsafe_allow_html=True)
                st.button("Apply", key=f"apply_{entry.rule}", disabled=True,
                          help="Writing thresholds.yaml is not wired up yet.")


def page_distributions(rows, thresholds, club, highlight=None):
    st.markdown(
        ui.heading("Metric distributions",
                   "Each dot is one swing. Diamonds are deliberately-performed "
                   "faults — they should sit clearly outside the normal cluster. "
                   "If they don't, the threshold cannot be calibrated."),
        unsafe_allow_html=True,
    )
    limits = faults._thresholds_for(thresholds, club)
    for metric, label in METRIC_LABELS.items():
        rule = METRIC_TO_RULE.get(metric)
        st.markdown(f"**{label}**")
        st.pyplot(
            distribution_plot(rows, metric, limits.get(rule) if rule else None,
                              rule=rule, highlight=highlight),
            width="stretch",
        )


def page_trend(rows):
    st.markdown(
        ui.heading("Change over time",
                   "Deliberate-fault clips are excluded — they are calibration "
                   "data, not swings you took, and would read as regressions."),
        unsafe_allow_html=True,
    )
    metric = st.selectbox("Metric", list(METRIC_LABELS),
                          format_func=lambda m: METRIC_LABELS[m])
    usable = [r for r in rows if r[metric] is not None]
    if len(usable) < 2:
        st.info("Need at least two measured swings to show a trend.")
        return

    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=ui.SURFACE)
    _style(ax)
    ax.grid(True, color=ui.LINE, linewidth=0.8, axis="y")
    ax.plot(range(len(usable)), [r[metric] for r in usable],
            color=ui.NORMAL, linewidth=2, marker="o", markersize=6,
            markeredgecolor=ui.SURFACE, markeredgewidth=1.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([r["date"][5:] for r in usable], rotation=45, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric], color=ui.INK_2, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")


def main():
    st.markdown(ui.CSS, unsafe_allow_html=True)

    conn = _conn()
    st.sidebar.markdown("### Golf Swing Analyzer")
    view = st.sidebar.radio(
        "View", ["Swing", "Add swings", "Calibration", "Distributions", "Trend"],
        label_visibility="collapsed")

    all_rows = db.load_swings(conn)
    if not all_rows:
        if view == "Add swings":
            page_import(conn)
        else:
            st.info("No swings yet — open **Add swings** in the sidebar.")
        return

    clubs = sorted({r["club"] for r in all_rows if r["club"]})
    club = st.sidebar.selectbox("Club", ["all", *clubs])
    rows = db.load_swings(conn, club=None if club == "all" else club)

    st.sidebar.divider()
    st.sidebar.caption(
        f"{len(rows)} swings · "
        f"{sum(1 for r in rows if r['fault_tag'])} deliberate faults"
    )
    if st.sidebar.button("Rescan data/processed"):
        written, skipped = history.sync(conn)
        st.sidebar.success(f"{written} synced")
        for message in skipped:
            st.sidebar.warning(message)

    if view == "Add swings":
        page_import(conn)
    elif view == "Calibration":
        page_calibration(rows, club)
    elif view == "Swing":
        page_swing(rows)
    elif view == "Distributions":
        page_distributions(rows, faults.load_thresholds(),
                           None if club == "all" else club)
    else:
        page_trend([r for r in rows if r["fault_tag"] is None])


if __name__ == "__main__":
    main()
