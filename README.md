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

## Scripts

All scripts live in `scripts/` with comments and configurable env vars:

```bash
./scripts/start.sh              # Production runner with auto-restart (up to 50 retries)
./scripts/paper_trade.sh        # Quick paper trading session
./scripts/train.sh              # Train LSTM with synthetic data
./scripts/test.sh               # Run tests with coverage
./scripts/export_model.sh       # Export .pt → .onnx
./scripts/generate_data.sh      # Generate synthetic CSV data
```

Examples with overrides:

```bash
DEVICE=cuda USE_AMP=--amp ./scripts/train.sh    # GPU training with mixed precision
CONFIG=config/fast.yaml ./scripts/paper_trade.sh # Custom config
./scripts/test.sh -k "test_buy"                  # Filter tests by keyword
```

## Testing

```bash
# Run all tests (158 tests, ~27s)
pytest

# Single file or test
pytest tests/test_strategy.py
pytest tests/test_features.py::TestRSI::test_all_gains

# With coverage report
pytest --cov=. --cov-report=term-missing
```

Test modules cover: Pydantic models, feature extraction (RSI/EMA/ATR/momentum/vol), portfolio tracker (cash/PnL/NAV/exposure), strategy state machine, risk shield (stops/circuit breaker/rate limit/exposure caps), alpha engine (rule-based/ensemble), async buffer, sim executor (market/limit fills), order manager lifecycle, simulated feed (GBM/CSV replay), training pipeline (dataset building/synthetic generation), integration (full pipeline end-to-end), and config loading.

## Architecture

```
[WSConnector / SimFeed]  ── push_candle ──>  [LiveBuffer]  ── event.set() ──>  [StrategyMonitor]
                                                  |                                     |
[BinanceSupplementaryFeed] ── push_depth/funding/taker ──>  │                           |
                                                                                        |
        FeatureExtractor ─┬─ extract()          → rule-based score ─┐                 |
         (10 features)    └─ extract_sequence() → LSTM forward pass ─┤→ AlphaEngine → Signal
                                                                     |
                                  StrategyLogic → RiskShield → OrderManager → Executor
                                  (Half-Kelly)   (stops)       (market/limit)     |
                                  check_stops()                    PortfolioTracker.on_fill()
```

**Producer-consumer pattern:** The data feed pushes candles into a `LiveBuffer`. The `StrategyMonitor` blocks on `asyncio.Event` until a new closed candle arrives — no polling, purely event-driven.

## Project Structure

```
trading-competition/
├── config/
│   └── default.yaml            # All tunable parameters (thresholds, risk limits, fees)
├── artifacts/                  # Model checkpoints (.pt, .onnx) — gitignored
├── logs/                       # Rotating log files — gitignored
├── scripts/                    # Shell scripts for common operations
├── core/
│   └── models.py               # Pydantic data models (Tick, OHLCV, Signal, Order, Position)
├── data/
│   ├── buffer.py               # asyncio.Event-driven sliding window for candles + supplementary data
│   ├── connector.py            # Binance WebSocket connector (live) + BinanceSupplementaryFeed
│   ├── sim_feed.py             # Synthetic GBM candle generator (paper mode)
│   └── historical/             # Cached CSV from ccxt (auto-created by training script)
├── features/
│   └── extractor.py            # OHLCV → 10 features (RSI, EMA, ATR, momentum, vol, OB imbalance, vol ratio, funding, taker ratio)
├── models/
│   ├── lstm_model.py           # PyTorch LSTM architecture (2-layer, 128 hidden, 10 features)
│   ├── model_wrapper.py        # Loads .onnx or .pt model for inference
│   ├── inference.py            # AlphaEngine: rule-based / lstm / ensemble scoring
│   └── train.py                # Offline training script (walk-forward validation, recency weighting)
├── strategy/
│   ├── logic.py                # Per-symbol state machine + Half-Kelly sizing + market/limit order selection
│   └── monitor.py              # Central event loop / orchestrator
├── risk/
│   ├── risk_shield.py          # Pre-trade validation, trailing stop, ATR stop, circuit breaker
│   └── tracker.py              # Real-time PnL, position tracking, NAV computation
├── execution/
│   ├── executor.py             # BaseExecutor ABC + LiveExecutor (ccxt)
│   ├── sim_executor.py         # Paper trading executor (instant fills + slippage)
│   └── order_manager.py        # Order lifecycle, fill callbacks
├── main.py                     # Entry point — wires everything, runs asyncio loop
└── pyproject.toml              # uv project config + dependencies
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

## Training the LSTM Model

```bash
# Train with synthetic data (no API needed — works anywhere)
python -m models.train --synthetic --symbols BTC/USDT --device auto

# Train with real data from Binance
python -m models.train --symbols BTC/USDT --days 90 --device auto

# GPU training with mixed precision and torch.compile
python -m models.train --synthetic --symbols BTC/USDT ETH/USDT --device cuda --amp --compile

# Walk-forward validation with recency weighting
python -m models.train --symbols BTC/USDT --days 90 --walk-forward --recency-half-life 35 --device cuda --amp

# With wandb experiment tracking
python -m models.train --synthetic --device cuda --amp --compile --wandb --wandb-tags test

# Or use the script
./scripts/train.sh
```

**What happens:**

1. Generates synthetic GBM candles (or fetches from Binance via ccxt with CSV caching)
2. Vectorized feature extraction — computes all 10 features in one pass (~0.5s for 10k candles)
3. Labels = forward return (configurable via `--forward-window`), scaled to [-1, 1] via tanh
4. Trains with MSE loss (optionally recency-weighted), chronological split (no shuffle — time-series)
5. Walk-forward validation available: train 0-60%, val 60-80%, test 80-100%
6. Saves `artifacts/model.pt` (PyTorch) and `artifacts/model.onnx` (ONNX)
7. Logs to wandb if `--wandb` is set (entity: `Base-Work-Space`, project: `trading-lstm`)

Then switch the config:

```yaml
alpha:
  engine: "lstm"              # or "ensemble" for average of rule-based + LSTM
  model_path: "artifacts/model.onnx"
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
- 2-layer LSTM with 128 hidden units gives sub-millisecond CPU inference
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
  model_path: "artifacts/model.onnx"
  seq_len: 60               # LSTM lookback window (60 × 1m = 1 hour)

# Position sizing (Half-Kelly)
strategy:
  base_size_pct: 0.05       # minimum allocation (5% of NAV)
  max_size_pct: 0.15        # maximum allocation (15% of NAV)
  kelly_fraction: 0.5       # half-Kelly multiplier
  estimated_win_rate: 0.55   # conservative prior
  estimated_payoff: 1.5      # avg_win / avg_loss
  urgent_alpha_threshold: 0.85  # alpha above this → market order (below → limit order)

# Risk limits
risk:
  trailing_stop_pct: 0.03         # 3% trailing stop from peak
  atr_stop_multiplier: 2.0        # exit if price drops 2x ATR below entry
  daily_drawdown_limit: 0.05      # 5% daily drawdown → circuit breaker halts all trading
  max_portfolio_exposure: 0.50    # max 50% of NAV in positions
  max_single_exposure: 0.15       # max 15% of NAV per symbol
  max_orders_per_minute: 60       # rate limit (supports 66 symbols)

# Execution fees
execution:
  fee_market_bps: 10        # 0.1% taker fee (market orders)
  fee_limit_bps: 5          # 0.05% maker fee (limit orders)

# Paper mode settings
paper:
  initial_capital: 100000.0
  slippage_bps: 5
  fee_bps: 10
  speed_multiplier: 60.0    # 60x = 1 hour of candles per minute of wall time
```

## Position Sizing

Uses Half-Kelly criterion scaled by alpha signal conviction:

| Alpha Score | Position Size |
|-------------|--------------|
| 0.61 (barely above threshold) | ~5.0% of NAV |
| 0.80 (moderate conviction) | ~6.3% of NAV |
| 1.00 (maximum conviction) | ~7.5% of NAV |

## Order Type Selection

- **Alpha > 0.85** or risk/stop exits: Market order (0.1% fee, guaranteed fill)
- **Alpha 0.6-0.85**: Limit order (0.05% fee, saves costs on non-urgent entries)

## Supplementary Data Feeds

In addition to the competition exchange price data, the system collects free public Binance data:

| Feature | Source | Update Frequency |
|---------|--------|-----------------|
| Order book imbalance | WS `{symbol}@depth20@100ms` | 100ms |
| Volume ratio | Computed from candle data | Per candle |
| Funding rate | WS `{symbol}@markPrice@1s` | 1s |
| Taker buy/sell ratio | REST `/futures/data/takerlongshortRatio` | 5m poll |

All endpoints are free, require no API key, and fail gracefully (features default to neutral values).

## Risk Management

Three layers of protection, in order of trigger speed:

1. **Trailing Stop (per-position):** If price drops 3% from peak since entry, sell immediately.
2. **ATR Stop (per-position):** If price drops below entry - 2x ATR, sell immediately. Adapts to volatility.
3. **Circuit Breaker (portfolio-wide):** If daily drawdown exceeds 5%, halt ALL new buys and liquidate all positions. Resets daily.

Pre-trade checks also enforce:
- Rate limit: max 60 orders/minute
- Exposure caps: 50% total portfolio, 15% per symbol
- Long-only: rejects any order that would create a short
- Cash check: won't buy more than you can afford

## Logging

Both trading and training sessions log to console + rotating files in `logs/`:
- Trading: `logs/trading_{mode}_{timestamp}.log`
- Training: `logs/train_{timestamp}.log`

Files rotate at 10MB with 5 backups.

## Experiment Tracking

Training integrates with [Weights & Biases](https://wandb.ai) via `--wandb`:

```bash
python -m models.train --synthetic --device cuda --amp --compile --wandb --wandb-tags test
```

Runs are auto-grouped by experiment config like: `cuda_e50_10k_amp_compile_0314`. Tags distinguish `test` vs `deploy` runs. Metrics logged per epoch: train/val loss, learning rate, epoch time. Override defaults with `--wandb-entity`, `--wandb-project`, `--wandb-group`, `--wandb-name`.

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Package manager | `uv` | Fast, reproducible, lockfile-based |
| Async I/O | `asyncio` + `aiohttp` + `websockets` | Non-blocking WebSocket + HTTP |
| Data models | `pydantic` | Validation, serialization, type safety |
| Exchange API | `ccxt` | Unified interface for 100+ exchanges |
| Indicators | `numpy` (pure) | Fast, no DataFrame overhead in hot path |
| DL training | `torch` | GPU support, `torch.compile`, AMP mixed precision |
| DL inference | `onnxruntime` | 2-5x faster than torch for single-sample CPU inference |
| Experiment tracking | `wandb` | Loss curves, run grouping, hyperparameter comparison |
| Config | `pyyaml` | Simple, human-readable |
