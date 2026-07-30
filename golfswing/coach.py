"""Turn swing metrics into readable coaching notes.

**Every comparison is against the golfer's own swings, never against a
threshold.** thresholds.yaml is uncalibrated — the 2026-07-29 session showed the
deliberate faults landing inside the normal range, so no rule can currently be
trusted. But "this swing's hip depth is the largest of your 16" needs no
threshold at all: it is a fact about the data, true today.

That distinction is why this module exists in this shape. Handing an uncalibrated
fault verdict to a language model would produce confident, fluent, invented
coaching — the precision-theater failure mode with a friendlier voice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MODEL = "claude-opus-5"

# Below this, a "personal range" is a story rather than a measurement.
MIN_PEERS_FOR_RANKING = 5


@dataclass(frozen=True)
class Metric:
    label: str
    unit: str
    meaning: str      # for the language model: precise, technical
    plain: str        # for the golfer: what it is and why it matters
    aim: str          # what "better" looks like, in one phrase
    short: str        # tile heading — must be unique and fit on one line


# Semantic descriptions, not display labels — the model needs to know what the
# number physically represents in a down-the-line view before it can say
# anything useful about it.
COACHING_METRICS: dict[str, Metric] = {
    "posture_change": Metric(
        "Posture change, address to top", "degrees",
        "how much the forward spine tilt changes during the backswing; "
        "negative means standing up out of the address posture",
        plain="How much your spine angle changes between address and the top of "
              "your backswing. If you stand up out of your posture going back, "
              "you have to find your way back down before impact.",
        aim="closer to 0 — hold the angle you set at address",
        short="Posture change",
    ),
    "hip_depth_change": Metric(
        "Hip movement toward the ball", "torso lengths",
        "how far the hips travel toward the ball between address and impact; "
        "large positive values are the early-extension pattern",
        plain="How far your hips drift toward the ball on the way down — the "
              "move usually called early extension. It steals room for your "
              "arms, so you have to stand up or flip the hands to make contact.",
        aim="closer to 0 — keep your backside where it started",
        short="Hip move to ball",
    ),
    "head_rise_p4": Metric(
        "Head rise, address to top", "torso lengths",
        "positive means the head is higher at the top than at address",
        plain="Whether your head lifts or drops between address and the top. "
              "Moving it changes how far you are from the ball.",
        aim="closer to 0 — steady head height going back",
        short="Head rise at top",
    ),
    "head_rise_p7": Metric(
        "Head rise, address to impact", "torso lengths",
        "positive means the head is higher at impact than at address",
        plain="Whether your head is higher or lower at impact than at address. "
              "This one shows up in your strike: lift and you thin it, drop "
              "and you catch it heavy.",
        aim="closer to 0 — same height at impact as at address",
        short="Head rise at impact",
    ),
    "knee_extension_change": Metric(
        "Knee straightening", "degrees",
        "how much the knees straighten from address to impact",
        plain="How much your knees straighten from address to impact. Some is "
              "normal and powerful; a lot of it pulls you up out of the shot.",
        aim="a smaller change than usual for you",
        short="Knee straightening",
    ),
    "tempo_ratio": Metric(
        "Tempo ratio", "backswing : downswing",
        "time from takeaway to the top divided by time from the top to impact; "
        "tour players are commonly near 3:1, but this measurement is the least "
        "reliable one here because it compounds errors in two event timings",
        plain="How long your backswing takes compared with your downswing. Many "
              "tour players sit near 3:1. Treat this as the least trustworthy "
              "number here — it multiplies small timing errors.",
        aim="nearer 3:1",
        short="Tempo",
    ),
}


@dataclass(frozen=True)
class Standing:
    """Where one metric on one swing sits among the golfer's other swings."""

    metric: str
    label: str
    unit: str
    meaning: str
    value: float
    rank: int | None       # 1 = largest. None when peers are too few to rank.
    n_peers: int
    median: float | None
    # Tile heading. Defaults empty so tests can build a Standing without it;
    # tile() falls back to the long label.
    short: str = ""


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def standings(rows: list[dict], clip: str) -> list[Standing]:
    """Rank one swing's metrics against the golfer's normal swings.

    Deliberate-fault clips are excluded from the baseline — a swing botched on
    purpose is not part of a normal range. A tagged clip can still be the
    subject; it is then measured against the normals, which is exactly the
    comparison that matters for it.
    """
    subject = next((r for r in rows if r["clip"] == clip), None)
    if subject is None:
        raise KeyError(f"no swing named {clip!r}")

    # Same club only. Tempo and hip depth genuinely differ between a driver and
    # a wedge, so a mixed baseline ranks a swing against a different motion —
    # the project's "club is metadata, never inferred" rule applied to ranking.
    baseline = [
        r for r in rows
        if r["fault_tag"] is None and r.get("club") == subject.get("club")
    ]

    found = []
    for name, meta in COACHING_METRICS.items():
        value = subject.get(name)
        if value is None:
            continue

        peers = [r[name] for r in baseline if r.get(name) is not None]

        # A tagged subject is not in its own baseline, so it has to be added to
        # the denominator — otherwise a swing below every normal ranks "17 of 16".
        compared = len(peers) + (0 if subject in baseline else 1)

        if len(peers) >= MIN_PEERS_FOR_RANKING:
            rank = sum(1 for p in peers if p > value) + 1
            median = _median(peers)
        else:
            rank, median = None, None

        found.append(Standing(
            metric=name, label=meta.label, short=meta.short,
            unit=meta.unit, meaning=meta.meaning,
            value=float(value), rank=rank, n_peers=compared, median=median,
        ))
    return found


# What "less of this" means per metric. For most, the coaching goal is simply
# less movement, so zero is the reference. Tempo is the exception: the widely
# cited ideal is roughly 3:1, so that is what "closer" measures against.
#
# This is NOT a calibrated threshold — it says nothing about whether a value is
# acceptable. It only orients the axis, so "steadier than your own typical" can
# be stated in a direction a golfer recognises.
IDEAL: dict[str, float] = {
    "posture_change": 0.0,
    "hip_depth_change": 0.0,
    "head_rise_p4": 0.0,
    "head_rise_p7": 0.0,
    "knee_extension_change": 0.0,
    "tempo_ratio": 3.0,
}

# Below this relative change, the swing is the same as usual within noise.
SAME_AS_USUAL = 0.10


def comparison(standing: Standing) -> str | None:
    """'better' | 'worse' | 'typical' versus this golfer's own median.

    Deliberately not a verdict on the swing. It answers "was this steadier than
    I usually am?", which needs no threshold — only the golfer's own history.
    """
    if standing.median is None or standing.rank is None:
        return None
    target = IDEAL.get(standing.metric)
    if target is None:
        return None

    now = abs(standing.value - target)
    usual = abs(standing.median - target)
    if usual == 0:
        return "typical" if now == 0 else "worse"

    change = (now - usual) / usual
    if abs(change) < SAME_AS_USUAL:
        return "typical"
    return "worse" if change > 0 else "better"


def extremity(standing: Standing) -> float:
    """How unusual this value is for this golfer: 0 at their median, 1 at an end.

    Rank position rather than the raw value, because the metrics are in
    different units and no threshold exists to normalise them against.
    """
    if standing.rank is None or standing.n_peers < 2:
        return 0.0
    middle = (standing.n_peers + 1) / 2
    return abs(standing.rank - middle) / (middle - 1) if middle > 1 else 0.0


def rank_phrase(standing: Standing) -> str:
    """Plain-English position, e.g. 'highest of 17'."""
    if standing.rank is None:
        return f"too few swings ({standing.n_peers})"
    if standing.rank == 1:
        return f"highest of {standing.n_peers}"
    if standing.rank == standing.n_peers:
        return f"lowest of {standing.n_peers}"

    # 11/12/13 are "th" despite ending in 1/2/3; otherwise the last digit decides.
    if standing.rank % 100 in (11, 12, 13):
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(standing.rank % 10, "th")
    return f"{standing.rank}{suffix} of {standing.n_peers}"


def standouts(found: list[Standing], count: int = 3) -> list[Standing]:
    """The few measurements worth showing large.

    Implements the project's "prioritise, don't enumerate" principle: a swing
    sitting at its own median is not news. Unranked metrics are never promoted —
    without enough peers there is no evidence they stand out at all.
    """
    ranked = [s for s in found if s.rank is not None]
    return sorted(ranked, key=extremity, reverse=True)[:count]


SYSTEM_PROMPT = """\
You are helping a golfer read the numbers from their own swing video analysis.

Ground rules, in order of importance:

1. Every measurement you are given is a comparison against THIS GOLFER'S OWN
   swings. There are no validated thresholds — do not say a value is "good",
   "bad", "correct", or "within range" against any external standard, because
   no such standard has been established here.
2. What you CAN say is where a swing sits among their own: "the largest of your
   16", "close to your median". Those are facts about the data.
3. Prioritise. Name the one or two measurements that stand out most, not all of
   them. A list of every metric is not coaching.
4. Be honest about uncertainty. If nothing stands out, say the swing looks
   typical for them — that is a real and useful finding.
5. Never invent a cause you cannot see. These are body-joint measurements from a
   single camera; you cannot see the club, the ball flight, or the strike.
6. Keep it short — a few sentences. Write like a coach talking, not a report.
"""


def build_prompt(found: list[Standing], clip: str) -> str:
    """Assemble the measurement context for one swing."""
    lines = [
        f"Swing: {clip}",
        "",
        "Measurements, each compared against this golfer's own normal swings",
        "(deliberate-practice faults excluded from the baseline):",
        "",
    ]

    for standing in found:
        lines.append(f"- {standing.label}: {standing.value:+.3f} {standing.unit}")
        lines.append(f"    ({standing.meaning})")
        if standing.rank is None:
            lines.append(
                f"    Ranking unavailable — too few swings on record "
                f"({standing.n_peers}) to establish a personal range."
            )
        else:
            lines.append(
                f"    Ranks {standing.rank} of {standing.n_peers} of their swings "
                f"(largest first). Their median is {standing.median:+.3f}."
            )
        lines.append("")

    lines.append(
        "Note: the project's fault thresholds are NOT calibrated — a session of "
        "deliberately exaggerated faults measured inside the normal range, so no "
        "rule can currently separate a fault from a normal swing. Rank within "
        "their own swings is the only trustworthy signal available."
    )
    return "\n".join(lines)


def _has_cli_profile() -> bool:
    """True when `ant auth login` has stored a profile the SDK can use.

    An unset ANTHROPIC_API_KEY does not mean there are no credentials.
    """
    config = Path(os.environ.get("ANTHROPIC_CONFIG_DIR",
                                 Path.home() / ".config" / "anthropic"))
    return (config / "credentials").is_dir() and any(
        (config / "credentials").glob("*.json")
    )


def available() -> bool:
    """Whether a coaching note can be generated at all."""
    return bool(os.environ.get("ANTHROPIC_API_KEY")) or _has_cli_profile()


def explain(found: list[Standing], clip: str) -> str:
    """Ask Claude to read the measurements back as coaching.

    Raises RuntimeError with a readable message rather than a stack trace — this
    runs behind a button in a local app, and an SDK exception surfaced raw is
    not an error message a person can act on.
    """
    try:
        import anthropic
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError(
            "The anthropic package is not installed. Run:\n"
            "    .venv/bin/pip install anthropic"
        ) from exc

    if not available():                              # pragma: no cover
        raise RuntimeError(
            "No Anthropic credentials found. Either export ANTHROPIC_API_KEY, "
            "or run `ant auth login`."
        )

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": build_prompt(found, clip)}],
        )
    except anthropic.AuthenticationError as exc:     # pragma: no cover
        raise RuntimeError("Anthropic rejected the credentials.") from exc
    except anthropic.RateLimitError as exc:          # pragma: no cover
        raise RuntimeError("Rate limited — wait a moment and try again.") from exc
    except anthropic.APIConnectionError as exc:      # pragma: no cover
        raise RuntimeError("Could not reach the API. Check your connection.") from exc
    except anthropic.APIStatusError as exc:          # pragma: no cover
        raise RuntimeError(f"API error {exc.status_code}: {exc.message}") from exc

    if response.stop_reason == "refusal":            # pragma: no cover
        raise RuntimeError("The request was declined by safety classifiers.")

    text = "\n".join(b.text for b in response.content if b.type == "text")
    return text.strip() or "(no response text returned)"
