# Web3 Long-Only Quant Trading Framework

Algorithmic trading bot for a 10-day crypto trading competition. Long-only, event-driven, async Python.

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd trading-competition

# 2. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create venv and install all dependencies
uv venv .venv
source .venv/bin/activate
uv sync

# 4. Run in paper mode (no exchange keys needed)
python main.py
```

That's it. The bot will generate synthetic price data and trade against it. Press `Ctrl+C` to stop — it prints a final PnL report on shutdown.

### Docker (alternative)

```bash
# Build image (CPU-only torch, ~1.2 GB)
docker build -t trading-bot .

# Run paper mode
docker run --rm trading-bot

# Run live mode with custom config
docker run --rm -v $(pwd)/config:/app/config trading-bot --mode live --config config/my_config.yaml
```

## Architecture

```
[WSConnector / SimFeed]  ── push_candle ──>  [LiveBuffer]  ── event.set() ──>  [StrategyMonitor]
                                                                                      |
        FeatureExtractor ─┬─ extract()          → rule-based score ─┐                 |
                          └─ extract_sequence() → LSTM forward pass ─┤→ AlphaEngine → Signal
                                                                     |
                                  StrategyLogic → RiskShield → OrderManager → Executor
                                                                                  |
                                  check_stops()                    PortfolioTracker.on_fill()
```

**Producer-consumer pattern:** The data feed pushes candles into a `LiveBuffer`. The `StrategyMonitor` blocks on `asyncio.Event` until a new closed candle arrives — no polling, purely event-driven.

## Project Structure

```
trading-competition/
├── config/
│   └── default.yaml            # All tunable parameters (thresholds, risk limits, fees)
├── core/
│   └── models.py               # Pydantic data models (Tick, OHLCV, Signal, Order, Position)
├── data/
│   ├── buffer.py               # asyncio.Event-driven sliding window for candles
│   ├── connector.py            # Binance WebSocket connector (live mode)
│   ├── sim_feed.py             # Synthetic GBM candle generator (paper mode)
│   └── historical/             # Cached CSV from ccxt (auto-created by training script)
├── features/
│   └── extractor.py            # OHLCV → technical indicators (RSI, EMA, ATR, momentum, vol)
├── models/
│   ├── lstm_model.py           # PyTorch LSTM architecture (2-layer, 64 hidden)
│   ├── model_wrapper.py        # Loads .onnx or .pt model for inference
│   ├── inference.py            # AlphaEngine: rule-based / lstm / ensemble scoring
│   └── train.py                # Offline training script (fetch data → train → export ONNX)
├── strategy/
│   ├── logic.py                # Per-symbol state machine (FLAT → LONG_PENDING → HOLDING)
│   └── monitor.py              # Central event loop / orchestrator
├── risk/
│   ├── risk_shield.py          # Pre-trade validation, trailing stop, ATR stop, circuit breaker
│   └── tracker.py              # Real-time PnL, position tracking, NAV computation
├── execution/
│   ├── executor.py             # BaseExecutor ABC + LiveExecutor (ccxt)
│   ├── sim_executor.py         # Paper trading executor (instant fills + slippage)
│   └── order_manager.py        # Order lifecycle, fill callbacks
├── main.py                     # Entry point — wires everything, runs asyncio loop
├── pyproject.toml              # uv project config + dependencies
└── config/default.yaml         # Tunable parameters
```

## Modes of Operation

### Paper Mode (default)

```bash
python main.py
```

- Generates synthetic candles via geometric Brownian motion
- Simulates order fills with configurable slippage (5bps) and fees (10bps)
- No API keys needed, no network access
- Speed controlled by `paper.speed_multiplier` in config (default 60x = 1 hour per minute)

### Live Mode

```bash
python main.py --mode live
```

- Connects to Binance via WebSocket for real-time kline data
- Executes orders via ccxt async exchange client
- Requires API keys in `config/default.yaml` or a custom config file

```bash
python main.py --config config/my_live_config.yaml --mode live
```

## Design Decisions

### Why long-only?

Competition constraint. The state machine only has three states: `FLAT` (no position), `LONG_PENDING` (buy order submitted, waiting for fill), `HOLDING` (in a position). No short selling, no hedging.

### Why rule-based alpha first, not pure ML?

For a 10-day competition, you need signals you can debug and tune in real-time. The rule-based engine combines four indicators into a composite score:

| Component | Weight | Logic |
|-----------|--------|-------|
| RSI | 0.3 | Oversold (RSI < 30) = bullish, overbought (RSI > 70) = bearish |
| Momentum | 0.3 | Rate of change over 10 candles, normalized |
| EMA Crossover | 0.3 | (EMA12 - EMA26) / EMA26, scaled |
| Volatility | 0.1 | Penalty — high vol reduces conviction |

The LSTM model can be trained offline and swapped in via one config change (`alpha.engine: "lstm"` or `"ensemble"`).

### Why LSTM specifically?

- Crypto price series have temporal dependencies (momentum regimes, mean-reversion cycles)
- 2-layer LSTM with 64 hidden units gives sub-millisecond CPU inference
- Well-understood, debuggable — no black-box risk during competition
- ONNX export decouples inference from PyTorch for faster runtime

### Why asyncio.Event, not a Queue?

The monitor needs the full candle history (last N candles) for feature extraction, not just the latest event. The buffer stores all data and the event simply signals "new data available."

### Why no pandas in the hot path?

Feature extraction runs on every candle. Pure numpy on small arrays (50-500 floats) is faster than constructing DataFrames. Pandas is used in the offline training script but not in the live loop.

### Why ONNX for inference?

- 2-5x faster than PyTorch `torch.no_grad()` for single-sample inference
- No torch dependency in the hot loop
- Deterministic, lighter runtime
- PyTorch `.pt` is kept as a fallback

## Configuration

All parameters live in `config/default.yaml`. Key settings:

```yaml
# Which alpha engine to use
alpha:
  engine: "rule_based"      # "rule_based" | "lstm" | "ensemble"
  entry_threshold: 0.6      # alpha > this → buy
  exit_threshold: -0.2      # alpha < this → sell

# Risk limits
risk:
  trailing_stop_pct: 0.03         # 3% trailing stop from peak
  atr_stop_multiplier: 2.0        # exit if price drops 2x ATR below entry
  daily_drawdown_limit: 0.05      # 5% daily drawdown → circuit breaker halts all trading
  max_portfolio_exposure: 0.50    # max 50% of NAV in positions
  max_single_exposure: 0.15       # max 15% of NAV per symbol

# Position sizing
strategy:
  position_size_pct: 0.10   # 10% of portfolio per trade

# Paper mode settings
paper:
  initial_capital: 100000.0
  slippage_bps: 5
  fee_bps: 10
  speed_multiplier: 60.0    # 60x = 1 hour of candles per minute of wall time
```

## Training the LSTM Model

```bash
# Fetch 90 days of BTC 1m candles from Binance, train, export to ONNX
python -m models.train --symbols BTC/USDT --days 90

# Train on multiple symbols
python -m models.train --symbols BTC/USDT ETH/USDT --days 90 --epochs 50

# Use GPU if available
python -m models.train --symbols BTC/USDT --device auto
```

**What happens:**

1. Fetches historical OHLCV from Binance via ccxt REST API
2. Caches to `data/historical/BTC_USDT_1m.csv` (subsequent runs reuse cache, only fetch new data)
3. Computes feature sequences using `FeatureExtractor`
4. Labels = forward 5-minute return, scaled to [-1, 1] via tanh
5. Trains with MSE loss, chronological 80/20 split (no shuffle — time-series)
6. Saves `config/model.pt` (PyTorch) and `config/model.onnx` (ONNX)

Then switch the config:

```yaml
alpha:
  engine: "lstm"              # or "ensemble" for average of rule-based + LSTM
  model_path: "config/model.onnx"
```

## Risk Management

Three layers of protection, in order of trigger speed:

1. **Trailing Stop (per-position):** If price drops 3% from peak since entry, sell immediately.
2. **ATR Stop (per-position):** If price drops below entry - 2x ATR, sell immediately. Adapts to volatility.
3. **Circuit Breaker (portfolio-wide):** If daily drawdown exceeds 5%, halt ALL new buys and liquidate all positions. Resets daily.

Pre-trade checks also enforce:
- Rate limit: max 10 orders/minute
- Exposure caps: 50% total portfolio, 15% per symbol
- Long-only: rejects any order that would create a short
- Cash check: won't buy more than you can afford

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Package manager | `uv` | Fast, reproducible, lockfile-based |
| Async I/O | `asyncio` + `aiohttp` + `websockets` | Non-blocking WebSocket + HTTP |
| Data models | `pydantic` | Validation, serialization, type safety |
| Exchange API | `ccxt` | Unified interface for 100+ exchanges |
| Indicators | `numpy` (pure) | Fast, no DataFrame overhead in hot path |
| DL training | `torch` | GPU support, flexible, well-documented |
| DL inference | `onnxruntime` | 2-5x faster than torch for single-sample CPU inference |
| Config | `pyyaml` | Simple, human-readable |

## Dependencies

Managed by `uv` via `pyproject.toml`. Key packages:

```
pydantic >= 2.12       # data models
pyyaml >= 6.0          # config
numpy >= 2.2           # feature computation
torch >= 2.10          # LSTM training
onnxruntime < 1.19     # LSTM inference (pinned for Python 3.10 compat)
ccxt >= 4.5            # exchange API
aiohttp >= 3.13        # async HTTP
websockets >= 16.0     # WebSocket connections
pandas >= 2.3          # offline data handling (training script only)
```
