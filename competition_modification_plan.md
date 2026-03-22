# Competition Modification Plan

## Goal

Keep some flexibility beyond `BTC / ETH / SOL`, but stop the old wide-universe behavior that produced scattered altcoin trades and weak portfolio concentration.

## Chosen Structure

Use a `Core + Satellite + Top-N` approach, then tighten it further based on the real order-history screenshots.

- Core: `BTC/ETH/SOL`
- Satellite: `BNB/LINK/XRP`
- Universe size: 6 symbols
- Regime behavior:
  - `risk_on`: Core + Satellite eligible
  - `neutral`: Core only
  - `risk_off`: no new entries

## Why This Fits Our Situation

- Our real order history showed excessive dispersion across small, medium-cap, and meme-style names.
- Many trades were effectively same-day hot-token rotation with weak persistence and frequent cancel/replace behavior.
- The current leaderboard evidence suggests concentrated exposure in strong majors is working better than broad alt rotation.
- The latest `factor-first` branch already has the right components for this shift:
  - explicit regime filter
  - trend alignment
  - pullback re-entry
  - overextension exit

## Concrete Code Changes

### 1. Narrow the live universe

- Add a new competition profile that explicitly sets the 8-symbol whitelist.
- Avoid relying on the 65-symbol default config.

### 2. Add cross-symbol selection

- Collect buy intents across the whole whitelist for each decision cycle.
- Rank candidates by adjusted entry quality.
- Only allow the top `N` fresh entries each cycle.
- Limit the portfolio to a small number of active positions.

Target behavior:
- `max_active_positions = 2`
- `top_n_entries_per_cycle = 1`
- `satellite_max_active_positions = 1`
- satellites need a stricter entry score than core names

### 3. Enforce Core-vs-Satellite regime gating

- In `neutral`, satellites should not open new positions.
- In `risk_on`, both tiers are allowed.
- In `risk_off`, no new entries.

### 4. Fix factor semantics

- Missing derivatives data must not become a bullish supporting factor.
- `perp_crowding` should return `NEUTRAL` when futures context is unavailable.

### 5. Fix exposure math

- Recompute buy order notional after quantity clamps in risk validation.
- This matters more once we allow larger concentrated positions.

## Initial Profile Direction

Proposed competition profile:
- `base_size_pct`: moderate
- `max_size_pct`: high enough to concentrate into majors
- `neutral_entry_size_multiplier`: very small, effectively Core-only
- `risk_off_entry_size_multiplier`: zero
- `max_portfolio_exposure`: bounded
- `max_single_exposure`: high enough to let majors concentrate

Factor tilt for round two:
- raise `market_regime` and `trend_alignment`
- lower `momentum_impulse`, `breakout_confirmation`, and `perp_crowding`
- tighten `max_taker_ratio`, `max_funding_rate`, and `max_open_interest_change`
- lower `max_volatility`

## Strategy Intent

Primary style:
- trend-following in majors
- buy pullbacks inside valid uptrends
- allow satellites only as secondary trades during strong `risk_on`

Avoid:
- weak mean-reversion entries
- wide altcoin scanning
- many simultaneous small positions
- same-day meme/hot-token chasing

## Expected Outcome

Compared with the previous broad-universe behavior, this should:

- reduce low-quality trades
- increase concentration in the strongest names
- preserve limited flexibility through a small liquid satellite set
- better match the current market structure visible in the competition
