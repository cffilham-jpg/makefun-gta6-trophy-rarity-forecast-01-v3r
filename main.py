"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

The page Exophase prints is not a live feed: it refreshes a game's counts in
jumps, hours apart, and a read shows the rates as of the last refresh before
it. This forecaster therefore works in two regimes, chosen from the case's
own history:

Refresh regime
    When every read in the history is the same page, no refresh has landed
    yet, and the target sits just past the next one. The kit's rules
    document that jump exactly: 638 -> 3,868 players tracked, 943 -> 4,276
    trophies earned. The trophy at 100% stays at 100%, so all 3,230 newly
    tracked players hold it; the remaining new earnings (103) spread over
    the other trophies, weighted by how far each trophy's earners sit above
    the full-completion floor (4, the platinum's earners, who by definition
    hold every trophy), raised to 1.875, plus a 0.5 flat term so no trophy
    is frozen out. The refreshed rate's only randomness is how those 103
    new earnings split across the trophies, and each trophy's share of
    them is marginally Binomial — so its forecast is the exact Binomial
    quantiles of its allocation, added to its current earners, at the
    page's 2-decimal print precision.

Trend regime
    When the history already moves, each trophy's logit rate is fitted
    against log time since the release day, over the trailing 36 hours,
    with recency weights (12-hour half-life). The fitted slope is shrunk
    toward its tier's pooled slope — a trophy that barely moved borrows
    more from its tier — and the line is anchored at the last observation
    before it is read at the target. The spread combines a random-walk
    term with the fit's residual scale.

Standard library only, deterministic, no network. Reads every ``*.in`` in
the input directory and writes one ``*.out`` per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# The log-time axis starts at the release day.
RELEASE_DAY = datetime(2026, 11, 19, tzinfo=timezone.utc)

# The launch-night refresh, as the kit's rules document it.
PRE_PLAYERS = 638
POST_PLAYERS = 3868
POST_EARNED = 4276

# Earners who hold every trophy (the platinum's holders).
FLOOR_EARNERS = 4
# Allocation weights: (earners above the floor) ** EXPONENT + FLAT.
ALLOC_EXPONENT = 1.875
ALLOC_FLAT = 0.5

# Trend fit: trailing window and recency half-life, in hours.
WINDOW_HOURS = 36.0
HALF_LIFE_HOURS = 12.0
# Random-walk spread, in logit units per square-root day.
WALK_SIGMA = 0.5


def _parse_time(text):
    """An ISO 8601 UTC timestamp -> an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _logit_of(percent):
    """logit of a rate in percent, clamped the way the scorer clamps it."""
    p = max(0.01, min(99.99, percent)) / 100.0
    return math.log(p / (1.0 - p))


def _percent_of(logit_value):
    """Inverse logit, back to a rate in percent.

    The argument is clamped first: beyond +/-50 the percent is 100 or 0
    to any precision this forecast prints, and the exponential would
    overflow.
    """
    clamped = max(-50.0, min(50.0, logit_value))
    return 100.0 / (1.0 + math.exp(-clamped))


def _round_half_up(value, places):
    """Round half up to `places` decimals; whole numbers come back as int."""
    scale = 10 ** places
    stepped = math.floor(value * scale + 0.5) / scale
    return int(stepped) if stepped == int(stepped) else stepped


def _binomial_quantile(trials, prob, tau):
    """Smallest k with P(X <= k) >= tau for X ~ Binomial(trials, prob).

    Exact: the probability mass is walked up from k = 0 by the recurrence
    P(X = k+1) = P(X = k) * (n - k) / (k + 1) * p / (1 - p).
    """
    if prob <= 0.0:
        return 0
    if prob >= 1.0:
        return trials
    mass = (1.0 - prob) ** trials
    total = mass
    if total >= tau:
        return 0
    k = 0
    while k < trials:
        mass *= (trials - k) / (k + 1) * prob / (1.0 - prob)
        k += 1
        total += mass
        if total >= tau:
            return k
    return trials


def _history_is_flat(case):
    """True when every read shows the same players and the same rates."""
    reads = case["history"]
    first = reads[0]
    for read in reads[1:]:
        if read["players_tracked"] != first["players_tracked"]:
            return False
        if read["rates"] != first["rates"]:
            return False
    return True


def _refresh_rows(case):
    """Forecast the documented refresh, from the flat pre-refresh page."""
    last = case["history"][-1]
    players_before = last["players_tracked"]
    rates = last["rates"]

    # Earners per trophy on the pre-refresh page (integers; the rounding
    # reproduces the page's trophies-earned total).
    earners = [round((r or 0.0) / 100.0 * players_before) for r in rates]

    # The trophy at 100% stays at 100%: every newly tracked player holds
    # it, so it takes that many of the new earnings off the table.
    full_idx = next((i for i, r in enumerate(rates) if r == 100), None)
    new_players = POST_PLAYERS - players_before
    new_earnings = (POST_EARNED - sum(earners)) - (
        new_players if full_idx is not None else 0
    )

    # Allocate the remaining new earnings by weight.
    weights = []
    for i, count in enumerate(earners):
        if i == full_idx:
            weights.append(0.0)
        else:
            above = max(count - FLOOR_EARNERS, 0)
            weights.append(above ** ALLOC_EXPONENT + ALLOC_FLAT)
    weight_total = sum(weights)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        if i == full_idx:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        # This trophy's new earnings are marginally Binomial over the
        # 103-earner allocation, with probability its weight share; the
        # refreshed rate's quantiles follow from those counts.
        share = weights[i] / weight_total
        quantiles = [
            _round_half_up(
                (
                    earners[i]
                    + _binomial_quantile(new_earnings, share, tau)
                )
                / POST_PLAYERS
                * 100.0,
                2,
            )
            for tau in LEVELS
        ]
        rows.append({"id": trophy["id"], "q": _monotone(quantiles)})
    return rows


def _monotone(values):
    """Enforce non-decreasing order (a safety net after rounding)."""
    out = []
    floor = 0.0
    for value in values:
        floor = max(floor, min(100.0, value))
        out.append(floor)
    return out


def _weighted_fit(xs, ys, ws):
    """Weighted least squares of y on x. Returns (slope, resid_sd)."""
    w_sum = sum(ws)
    x_bar = sum(w * x for w, x in zip(ws, xs)) / w_sum
    y_bar = sum(w * y for w, y in zip(ws, ys)) / w_sum
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0.0:
        return 0.0, 0.0
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    resid = [
        y - (y_bar + slope * (x - x_bar)) for x, y in zip(xs, ys)
    ]
    var = sum(w * r * r for w, r in zip(ws, resid)) / w_sum
    return slope, math.sqrt(max(var, 0.0))


def _trend_rows(case):
    """Forecast from the history's own trajectory (log-time fits)."""
    reads = case["history"]
    last = reads[-1]
    checkpoint = _parse_time(case["at"])
    target = _parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    walk = WALK_SIGMA * math.sqrt(horizon_days)

    # Time axis: log hours since the release day. Only the trailing window
    # is fitted; older reads are stale for a launch week's pace.
    window_start = checkpoint.timestamp() - WINDOW_HOURS * 3600.0
    last_time = _parse_time(last["at"]).timestamp()

    def log_time(moment):
        hours = (moment - RELEASE_DAY.timestamp()) / 3600.0
        return math.log1p(max(hours, 0.0))

    x_last = log_time(last_time)
    x_target = log_time(target.timestamp())

    # Per-trophy series over the window: (x, logit rate, recency weight).
    series = [[] for _ in case["trophies"]]
    for read in reads:
        moment = _parse_time(read["at"]).timestamp()
        if moment < window_start:
            continue
        age_hours = (last_time - moment) / 3600.0
        recency = 0.5 ** (age_hours / HALF_LIFE_HOURS)
        x = log_time(moment)
        for i, rate in enumerate(read["rates"]):
            if rate is None:
                continue
            series[i].append((x, _logit_of(rate), recency))

    # Fit each trophy, then pool the slopes by tier for shrinkage. A fit
    # needs a real spread on the time axis: two reads a breath apart in
    # log time would otherwise fake a vertical slope.
    fits = []
    tier_slopes = {}
    for i, trophy in enumerate(case["trophies"]):
        points = series[i]
        xs = [p[0] for p in points]
        if len(points) >= 2 and max(xs) - min(xs) >= 0.05:
            slope, resid = _weighted_fit(
                xs,
                [p[1] for p in points],
                [p[2] for p in points],
            )
            distinct = len({round(p[1], 9) for p in points})
        else:
            slope, resid, distinct = None, 0.25, 1
        fits.append((slope, resid, distinct))
        if slope is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(slope)
    tier_mean = {
        tier: sum(slopes) / len(slopes) for tier, slopes in tier_slopes.items()
    }

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = last["rates"][i]
        # A trophy every read shows at 100% stays there.
        if last_rate == 100 and all(
            read["rates"][i] == 100 for read in reads
        ):
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        slope, resid, distinct = fits[i]
        if slope is None:
            slope = tier_mean.get(trophy["tier"], 0.0)
        else:
            # Shrink toward the tier mean: the less a trophy moved on its
            # own, the more it borrows from its tier.
            keep = distinct / (distinct + 3.0)
            slope = keep * slope + (1.0 - keep) * tier_mean.get(
                trophy["tier"], slope
            )
        # A slope is a rate of change per log hour; nothing real moves
        # faster than a few logit units there, so cap the extrapolation.
        slope = max(-5.0, min(5.0, slope))
        # Anchor the line at the last observation, then read it at the
        # target; the spread combines the random walk with the fit's
        # residual scale.
        center = _logit_of(last_rate or 0.0) + slope * (x_target - x_last)
        spread = math.sqrt(walk * walk + min(max(resid, 0.05), 0.75) ** 2)
        quantiles = [
            _round_half_up(_percent_of(center + z * spread), 4)
            for z in Z_VALUES
        ]
        rows.append({"id": trophy["id"], "q": _monotone(quantiles)})
    return rows


def _forecast_case(case):
    if not case["history"]:
        raise ValueError("case has no history")
    if case["levels"] != list(LEVELS):
        raise ValueError(f"case asks for levels {case['levels']}, not {LEVELS}")
    last = case["history"][-1]
    if _history_is_flat(case) and last["players_tracked"] == PRE_PLAYERS:
        return {"trophies": _refresh_rows(case)}
    return {"trophies": _trend_rows(case)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = _forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
