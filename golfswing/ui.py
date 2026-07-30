"""Presentation layer for the Streamlit app — CSS and the stat-tile markup.

Kept out of app.py so the page functions read as layout rather than as a wall
of style strings.

The design system came from Claude Design. Three things in it had to be adapted
to what Streamlit actually renders, each verified in the browser rather than
assumed:

1. There is no `stVideo` test-id — the `<video>` element sits bare inside a
   generic `stElementContainer`, so the sizing rules target `video` directly.
2. `st.dataframe` renders to a **canvas**, so table CSS cannot reach it at all.
   The measurements table uses `st.table`, which emits real HTML.
3. Inter is not installed on this machine. Naming it alone falls back to
   generic sans (Helvetica on macOS); the stack below falls back to SF Pro,
   which is far closer to the intended look.

The palette is the dataviz-validated one (all six checks pass, CVD ΔE 24.7).
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

CSS = """
<style>
  :root {
    --surface: #fcfcfb;
    --raised: #f4f3ee;
    --ink: #0b0b0b;
    --secondary: #52514e;
    --muted: #898781;
    --hairline: #e1e0d9;
    --primary: #2a78d6;
    --accent: #eb6834;
  }

  .stApp {
    background: var(--surface);
    /* Inter first, then SF Pro — naming Inter alone falls back to Helvetica. */
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }

  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
  #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

  /* Do NOT collapse the header. Streamlit puts the re-open control for a
     collapsed sidebar inside it, so `header { height: 0 }` leaves no way back
     once you hide the sidebar. Make it invisible but keep its box. */
  header { background: transparent !important; }
  [data-testid="stExpandSidebarButton"] {
      visibility: visible !important; opacity: 1 !important; z-index: 1000;
  }

  section[data-testid="stSidebar"] {
    background: var(--raised);
    border-right: 1px solid var(--hairline);
  }
  section[data-testid="stSidebar"] .stRadio label p { font-size: 14px; }
  [data-testid="stSidebar"] [data-baseweb="radio"] div:first-child {
    border-color: var(--muted);
  }
  [data-testid="stSidebar"] [aria-checked="true"] div:first-child {
    border-color: var(--primary); background: var(--primary);
  }

  .gs-title {
    font-size: 20px; font-weight: 600; letter-spacing: -0.01em;
    color: var(--ink); margin: 0 0 2px 0;
  }
  .gs-sub { font-size: 13px; color: var(--muted); margin: 0 0 20px 0; }

  .section-label {
    font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--hairline);
    padding-bottom: 12px; margin: 32px 0 20px 0;
  }

  /* No stVideo test-id exists, so target the element itself. Portrait phone
     footage is 1080x1920; capping the height stops it filling the viewport. */
  video {
    max-height: 66vh; width: auto; max-width: 100%;
    border-radius: 8px; border: 1px solid var(--hairline); display: block;
  }

  .tile-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .metric-tile {
    background: var(--raised);
    border: 1px solid var(--hairline);
    border-radius: 8px;
    padding: 18px 20px;
  }
  .metric-tile .label {
    font-size: 15px; font-weight: 500; color: var(--secondary); margin-bottom: 10px;
  }
  .metric-tile .value {
    font-size: 40px; font-weight: 600; line-height: 1.05;
    font-variant-numeric: tabular-nums; margin-bottom: 8px; color: var(--ink);
  }
  .metric-tile .value .unit {
    font-size: 15px; font-weight: 400; color: var(--muted); margin-left: 6px;
  }
  .rank-pill {
    display: inline-flex; padding: 4px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
    background: var(--raised); border: 1px solid var(--hairline);
    color: var(--secondary);
  }
  /* Exactly one tile per view may carry this — see tile_grid(). */
  .rank-pill.is-notable {
    background: color-mix(in srgb, var(--accent) 10%, transparent);
    border-color: color-mix(in srgb, var(--accent) 25%, transparent);
    color: #b5501f;
  }

  .coaching-note {
    background: var(--raised);
    border-left: 2px solid var(--primary);
    border-radius: 0 6px 6px 0;
    padding: 16px 20px;
    font-size: 15px; line-height: 1.65; color: var(--ink);
  }
  .coaching-note p:last-child { margin-bottom: 0; }

  [data-testid="stTable"] {
    border: 1px solid var(--hairline);
    border-radius: 8px;
    overflow: hidden;
  }
  [data-testid="stTable"] thead tr th {
    background: var(--raised);
    color: var(--muted);
    font-size: 12px; font-weight: 600;
    letter-spacing: 0.03em; text-transform: uppercase;
  }
  [data-testid="stTable"] tbody td {
    font-size: 14px; font-variant-numeric: tabular-nums;
    border-top: 1px solid var(--hairline);
  }

  div[data-testid="stImage"] img {
    border-radius: 8px; border: 1px solid var(--hairline);
  }
</style>
"""


def tile(standing: Standing, notable: bool = False) -> str:
    """One large metric. The pill states rank in words, never colour alone."""
    pill = "rank-pill is-notable" if notable else "rank-pill"
    return (
        f'<div class="metric-tile">'
        f'  <div class="label">{standing.label.split(",")[0]}</div>'
        f'  <div class="value">{standing.value:+.2f}'
        f'    <span class="unit">{standing.unit}</span></div>'
        f'  <span class="{pill}">{rank_phrase(standing)}</span>'
        f'</div>'
    )


def tile_grid(found: list[Standing]) -> str:
    """All the standout tiles, with **at most one** highlighted.

    The design brief calls for a single accent per view, which is a stronger
    rule than "highlight everything past the threshold": two competing
    highlights give the eye nowhere to land. Only the most extreme metric gets
    it, and only if it clears the bar at all — a typical swing highlights
    nothing.
    """
    if not found:
        return ""
    leader = max(range(len(found)), key=lambda i: extremity(found[i]))
    highlight = leader if extremity(found[leader]) >= NOTABLE_EXTREMITY else None
    tiles = "".join(tile(s, notable=(i == highlight)) for i, s in enumerate(found))
    return f'<div class="tile-grid">{tiles}</div>'


def section(title: str) -> str:
    return f'<div class="section-label">{title}</div>'


def heading(title: str, subtitle: str) -> str:
    return f'<div class="gs-title">{title}</div><div class="gs-sub">{subtitle}</div>'
