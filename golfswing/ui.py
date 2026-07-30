"""Presentation layer for the Streamlit app — CSS and the stat-tile markup.

Kept out of app.py so the page functions read as layout rather than as a wall
of style strings.

The palette is the dataviz-validated one (all six checks pass, CVD ΔE 24.7):
#2a78d6 and #eb6834 on a #fcfcfb surface.
"""

from __future__ import annotations

from golfswing.coach import Standing, extremity, rank_phrase

SURFACE = "#fcfcfb"
RAISED = "#f4f3ee"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
LINE = "#e1e0d9"
NORMAL = "#2a78d6"
FAULT = "#eb6834"

# Above this, a value is far enough from the golfer's median to call out.
NOTABLE_EXTREMITY = 0.6

CSS = f"""
<style>
  /* Streamlit's default top padding wastes most of a laptop's first screen. */
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
  #MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}
  header {{ height: 0 !important; }}

  /* Phone footage is portrait 1080x1920. Width is constrained by a nested
     Streamlit column rather than CSS — targeting Streamlit's internal test-ids
     is brittle, and sizing the <video> with width:auto collapses it to a stub
     before the browser has read the dimensions. */
  video {{
      width: 100%; height: auto; display: block;
      border-radius: 12px; border: 1px solid {LINE};
  }}

  .gs-title {{
      font-size: 1.05rem; font-weight: 650; letter-spacing: -0.01em;
      color: {INK}; margin: 0 0 0.15rem 0;
  }}
  .gs-sub {{ font-size: 0.82rem; color: {MUTED}; margin: 0 0 1.1rem 0; }}

  .gs-section {{
      font-size: 0.72rem; font-weight: 650; letter-spacing: 0.07em;
      text-transform: uppercase; color: {MUTED};
      margin: 1.9rem 0 0.7rem 0;
      padding-bottom: 0.35rem; border-bottom: 1px solid {LINE};
  }}

  .gs-tile {{
      background: {RAISED}; border: 1px solid {LINE}; border-radius: 12px;
      padding: 0.95rem 1.1rem 0.85rem 1.1rem; height: 100%;
  }}
  .gs-tile-label {{
      font-size: 0.74rem; font-weight: 600; color: {INK_2};
      line-height: 1.25; min-height: 2.4em;
  }}
  .gs-tile-value {{
      /* tabular-nums keeps the decimal points aligned across tiles. */
      font-variant-numeric: tabular-nums;
      font-size: 2.35rem; font-weight: 680; letter-spacing: -0.03em;
      color: {INK}; line-height: 1.05; margin: 0.35rem 0 0.1rem 0;
  }}
  .gs-tile-unit {{ font-size: 0.7rem; color: {MUTED}; margin-bottom: 0.6rem; }}

  .gs-chip {{
      display: inline-block; font-size: 0.7rem; font-weight: 650;
      padding: 0.18rem 0.5rem; border-radius: 999px;
      border: 1px solid transparent;
  }}
  /* Colour is never the only signal — the chip always carries its text. */
  .gs-chip-notable {{ background: #fce9e0; color: #8f3410; border-color: #f5c9b4; }}
  .gs-chip-typical {{ background: #eceae3; color: {INK_2}; border-color: {LINE}; }}

  .gs-note {{
      background: {RAISED}; border: 1px solid {LINE};
      border-left: 3px solid {NORMAL};
      border-radius: 10px; padding: 1rem 1.15rem; font-size: 0.9rem;
      line-height: 1.6; color: {INK};
  }}
  .gs-note p:last-child {{ margin-bottom: 0; }}

  div[data-testid="stImage"] img {{ border-radius: 10px; border: 1px solid {LINE}; }}
</style>
"""


def tile(standing: Standing) -> str:
    """One large metric. The chip states rank in words, never colour alone."""
    notable = extremity(standing) >= NOTABLE_EXTREMITY
    chip = "gs-chip-notable" if notable else "gs-chip-typical"
    label = standing.label.split(",")[0]
    return (
        f'<div class="gs-tile">'
        f'  <div class="gs-tile-label">{label}</div>'
        f'  <div class="gs-tile-value">{standing.value:+.2f}</div>'
        f'  <div class="gs-tile-unit">{standing.unit}</div>'
        f'  <span class="gs-chip {chip}">{rank_phrase(standing)}</span>'
        f'</div>'
    )


def section(title: str) -> str:
    return f'<div class="gs-section">{title}</div>'


def heading(title: str, subtitle: str) -> str:
    return f'<div class="gs-title">{title}</div><div class="gs-sub">{subtitle}</div>'
