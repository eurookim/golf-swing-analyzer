"""Golf swing analyzer — local Streamlit UI.

    .venv/bin/streamlit run app.py

Scoped around calibration rather than polish. The distribution view is the
highest-value screen right now: on the 2026-07-29 session it would have shown at
a glance that the deliberate fault swings sat *inside* the normal cluster, which
otherwise took several rounds of analysis to establish.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from golfswing import db, faults, history

# Validated palette (light surface #fcfcfb): all checks pass, CVD ΔE 24.7.
SURFACE = "#fcfcfb"
NORMAL = "#2a78d6"
FAULT = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

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


st.set_page_config(page_title="Golf Swing Analyzer", layout="wide")


@st.cache_resource
def _conn():
    return db.connect()


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def distribution_plot(rows, metric, threshold=None, rule=None):
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
    normal, tagged = [], []
    for row in rows:
        value = row[metric]
        if value is None:
            continue
        if rule is not None and row["fault_tag"] == rule:
            tagged.append((value, row["clip"]))
        else:
            normal.append(value)

    fig, ax = plt.subplots(figsize=(9, 1.9), facecolor=SURFACE)
    _style(ax)

    if normal:
        jitter = np.random.default_rng(0).normal(0, 0.045, len(normal))
        ax.scatter(normal, jitter, s=70, color=NORMAL, alpha=0.75,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3, label="normal")
    for offset, (value, clip) in enumerate(sorted(tagged)):
        ax.scatter([value], [0], s=190, marker="D", color=FAULT,
                   edgecolors=SURFACE, linewidths=2, zorder=5)
        # Stagger labels so adjacent diamonds do not overprint each other.
        ax.annotate(clip.split("_")[-1], (value, 0), textcoords="offset points",
                    xytext=(0, 16 + 13 * (offset % 2)), ha="center", fontsize=9,
                    color=FAULT, fontweight="bold")

    if threshold is not None and math.isfinite(threshold):
        ax.axvline(threshold, color=INK_2, linestyle="--", linewidth=1.5, zorder=2)
        ax.annotate("threshold", (threshold, -0.16), textcoords="offset points",
                    xytext=(5, 0), fontsize=8, color=INK_2)

    ax.set_ylim(-0.25, 0.25)
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def page_distributions(rows, thresholds, club):
    st.subheader("Metric distributions")
    st.caption(
        "Each dot is one swing. Diamonds are deliberately-performed faults — they "
        "should sit clearly outside the normal cluster. If they don't, the "
        "threshold cannot be calibrated from this set."
    )

    limits = faults._thresholds_for(thresholds, club)
    for metric, label in METRIC_LABELS.items():
        rule = METRIC_TO_RULE.get(metric)
        st.markdown(f"**{label}**")
        st.pyplot(distribution_plot(rows, metric,
                                    limits.get(rule) if rule else None,
                                    rule=rule),
                  use_container_width=True)


def page_swing(rows, thresholds, club):
    st.subheader("Single swing")
    clips = [r["clip"] for r in rows]
    if not clips:
        st.info("No swings in history yet.")
        return
    chosen = st.selectbox("Clip", clips, index=len(clips) - 1)
    row = next(r for r in rows if r["clip"] == chosen)

    sheet = Path("outputs") / f"{chosen}_keyframes.jpg"
    if sheet.exists():
        st.image(str(sheet), use_container_width=True)
    else:
        st.caption(f"No key-frame sheet yet — run render_contact_sheet.py {chosen}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Metrics**")
        st.dataframe(
            {"metric": list(METRIC_LABELS.values()),
             "value": [row[m] for m in METRIC_LABELS]},
            hide_index=True, use_container_width=True,
        )
    with right:
        st.markdown("**Faults**")
        st.warning(
            "Thresholds are **not calibrated** — every value in thresholds.yaml "
            "is invented until a session with genuinely exaggerated faults exists. "
            "Treat any finding below as a hypothesis."
        )
        st.caption(
            f"Detected events — P1 {row['p1']} · P4 {row['p4']} · "
            f"P7 {row['p7']} · P10 {row['p10']}   ({row['fps']:.0f} fps)"
        )


def page_trend(rows):
    st.subheader("Change over time")
    st.caption(
        "Deliberate-fault clips are excluded — they are calibration data, not "
        "swings you took, and would show as phantom regressions."
    )
    metric = st.selectbox("Metric", list(METRIC_LABELS),
                          format_func=lambda m: METRIC_LABELS[m])
    usable = [r for r in rows if r[metric] is not None]
    if len(usable) < 2:
        st.info("Need at least two measured swings to show a trend.")
        return

    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.plot(range(len(usable)), [r[metric] for r in usable],
            color=NORMAL, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([r["date"][5:] for r in usable], rotation=45, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric], color=INK_2, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


def main():
    st.title("Golf Swing Analyzer")

    conn = _conn()
    if st.sidebar.button("Rescan data/processed"):
        written, skipped = history.sync(conn)
        st.sidebar.success(f"{written} swings synced")
        for message in skipped:
            st.sidebar.warning(message)

    all_rows = db.load_swings(conn)
    if not all_rows:
        st.info("No swings yet. Run `python -m golfswing`, then press Rescan.")
        return

    clubs = sorted({r["club"] for r in all_rows if r["club"]})
    club = st.sidebar.selectbox("Club", ["all", *clubs])
    rows = db.load_swings(conn, club=None if club == "all" else club)

    st.sidebar.metric("Swings", len(rows))
    st.sidebar.metric("Deliberate faults", sum(1 for r in rows if r["fault_tag"]))

    thresholds = faults.load_thresholds()
    tab1, tab2, tab3 = st.tabs(["Distributions", "Single swing", "Trend"])
    with tab1:
        page_distributions(rows, thresholds, None if club == "all" else club)
    with tab2:
        page_swing(rows, thresholds, club)
    with tab3:
        page_trend([r for r in rows if r["fault_tag"] is None])


if __name__ == "__main__":
    main()
