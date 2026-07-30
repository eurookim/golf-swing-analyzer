"""Golf swing analyzer — local Streamlit UI.

    .venv/bin/streamlit run app.py     (or open the app in ~/Applications)

The swing page is the home page: video, the few measurements that stand out,
then everything else. Distributions and Trend are calibration and history
views, reached from the sidebar.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from golfswing import coach, db, faults, history, preview, ui

RAW_DIR = Path("data/raw")
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

    # Portrait footage needs a narrow column; a wide one wastes space on both
    # sides AND caps the height. 2:5 fits a 9:16 clip almost exactly.
    video_col, stats_col = st.columns([2, 5], gap="large")
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

    sheet = Path("outputs") / f"{chosen}_keyframes.jpg"
    if sheet.exists():
        st.markdown(ui.section("Key frames"), unsafe_allow_html=True)
        st.image(str(sheet), width="stretch")

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
    view = st.sidebar.radio("View", ["Swing", "Distributions", "Trend"],
                            label_visibility="collapsed")

    all_rows = db.load_swings(conn)
    if not all_rows:
        st.info("No swings yet. Run `python -m golfswing`, then press Rescan.")
        if st.sidebar.button("Rescan data/processed"):
            st.rerun()
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

    if view == "Swing":
        page_swing(rows)
    elif view == "Distributions":
        page_distributions(rows, faults.load_thresholds(),
                           None if club == "all" else club)
    else:
        page_trend([r for r in rows if r["fault_tag"] is None])


if __name__ == "__main__":
    main()
