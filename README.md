# GTA VI Trophy Rarity Forecast — v3r

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) [![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cffilham-jpg/makefun-gta6-trophy-rarity-forecast-01-v3r/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) [![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cffilham-jpg/makefun-gta6-trophy-rarity-forecast-01-v3r/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts the share of players who will have earned each of the 55 GTA VI
trophies by the final date, as 55 rows in `[0, 100]` with two decimals.

This is an entry in the [make.fun gta6-trophy-rarity-forecast-01 bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01),
where $250 is shared by the top 20 entries.

## Method

- Refresh totals pin 103 new earners over the 54 non-intro trophies.
- Each trophy's refreshed rate = (earners + Binomial(103, wᵢ/W)) / 3868,
  with weights (earners − floor 4)^1.875 + 0.5 and exact quantiles.
- Trend branch: recency-weighted log-time fits (12h half-life, 36h window),
  tier-shrunk slopes, anchored at the last observation.
- Stdlib-only Python (`python3 -P -s`), byte-identical across runs, ~0.10s.

Local validation: loss 0.042850 against the deterministic mean answer,
0.059579 ± 0.004783 under full Monte Carlo noise (300 sims).
