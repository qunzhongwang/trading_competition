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
                         ┌──────────────────────────────────────────────────┐
                         │              Multi-Scale Feature Engine          │
                         │                                                  │
  [WSConnector/SimFeed]  │  1m candles ──> RSI, EMA, ATR, Momentum, Vol    │
        │                │  resampled 15m ──> EMA trend, Momentum (15m)    │
   push_candle           │  resampled 1h  ──> EMA trend, Momentum (1h)    │
        │                │                                                  │
        ▼                │  + Web3 native:                                  │
   [LiveBuffer]          │    Funding Rate (perp sentiment)                 │
        │                │    Taker Buy/Sell Ratio (aggressor flow)         │
  event.set()            │    Order Book Imbalance (microstructure)         │
        │                └──────────────┬───────────────────────────────────┘
        ▼                               │
  [StrategyMonitor]                     ▼
        │                ┌──────────────────────────────────┐
        │                │  LSTM Classification Head         │
        │                │  Input:  (batch, 60, N_features)  │
        │                │  Output: P(Long), P(Neutral),     │
        │                │          P(Short)                  │
        │                │  Alpha = P(Long) - P(Short)       │
        │                └──────────────┬───────────────────┘
        │                               │
        ├───────────────────────────────┘
        ▼
   AlphaEngine ──> Signal {alpha ∈ [-1, 1]}
        │
        ▼
   StrategyLogic ──────────> RiskShield ──────────> OrderManager ──> Executor
   (Half-Kelly sizing)       (dynamic ATR stop,      (market if α > 0.85,
                              trailing stop,           limit if α ∈ [0.6, 0.85])
                              circuit breaker)                    │
                                                    PortfolioTracker.on_fill()
```

**Core design principles:**

1. **Multi-scale inputs to combat noise.** Raw 1-minute candles are dominated by microstructure noise. The feature engine resamples candles into 15-minute and 1-hour bars, extracting trend-level EMA crossovers and momentum from each timeframe. This gives the LSTM a view of both short-term dynamics and medium-term regime, without requiring separate models per frequency.

2. **Web3-native features as first-class inputs.** Unlike equities, crypto perpetual futures expose unique sentiment signals — funding rates reflect leverage imbalance between longs and shorts, and taker buy/sell ratios reveal real-time aggressor flow. These features, streamed directly from Binance public endpoints, give the model information that pure price-derived indicators cannot capture.

3. **Event-driven, not polling.** The data feed pushes candles into a `LiveBuffer`. The `StrategyMonitor` blocks on `asyncio.Event` until a new closed candle arrives — zero CPU waste between events.

### Feature Summary

| Category | Features | Timeframe | Source |
|----------|----------|-----------|--------|
| Price-derived | RSI(14), EMA(12/26), ATR(14), Momentum(10), Volatility(20) | 1m | Candles |
| Multi-scale trend | EMA crossover, Momentum | 15m, 1h | Resampled candles |
| Web3 sentiment | Funding rate | Real-time | Binance Futures WS |
| Order flow | Taker buy/sell ratio | 5m | Binance REST |
| Microstructure | Order book imbalance, Volume ratio | 100ms / per candle | Binance Depth WS |

All supplementary endpoints are free, require no API key, and fail gracefully (features default to neutral values on timeout).

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
│   └── extractor.py            # OHLCV → multi-scale features (1m/15m/1h + Web3 sentiment + microstructure)
├── models/
│   ├── lstm_model.py           # PyTorch LSTM (2-layer, 128 hidden) with 3-class softmax head
│   ├── model_wrapper.py        # Loads .onnx or .pt model for inference
│   ├── inference.py            # AlphaEngine: rule-based / lstm / ensemble scoring
│   └── train.py                # Offline training (walk-forward, recency weighting, classification labels)
├── strategy/
│   ├── logic.py                # Per-symbol state machine + Half-Kelly sizing + alpha-driven order routing
│   └── monitor.py              # Central event loop / orchestrator
├── risk/
│   ├── risk_shield.py          # Pre-trade validation, dynamic ATR stop, trailing stop, circuit breaker
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

# Train with real data from Binance (recommended: 90 days for stable feature distributions)
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

### Data Specification

**Recommended: fetch 90 days of 1-minute history.** 90 days provides ~129,600 candles per symbol — enough to cover multiple market regimes (trending, ranging, high-vol events) while staying within Binance's free API limits. Shorter windows risk overfitting to a single regime; longer windows dilute recent patterns.

**Time-series split (70 / 15 / 15):**

```
Day 1                    Day 63           Day 76.5         Day 90
 │─────── Train (70%) ──────│── Val (15%) ──│── Test (15%) ──│
 │        ~90,720 candles   │  ~19,440      │  ~19,440       │
```

- **Train (70%):** Model learns feature → signal mappings. Recency weighting (optional, `--recency-half-life 35`) down-weights older samples so the model prioritizes recent market behavior.
- **Validation (15%):** Hyperparameter tuning and early stopping. ReduceLROnPlateau monitors val loss; training stops if no improvement for 10 epochs.
- **Test (15%):** Final out-of-sample evaluation. Never touched during training. Walk-forward mode (`--walk-forward`) reports test metrics separately.

Chronological ordering is strictly enforced — no shuffling, no future leakage.

### Classification Output

The LSTM outputs a 3-class probability distribution over the next 15-minute price movement:

| Class | Definition | Label Logic |
|-------|-----------|-------------|
| **Long** | 15-min forward return > +threshold | Price likely to rise |
| **Neutral** | Forward return within ±threshold | No clear directional edge |
| **Short** | 15-min forward return < -threshold | Price likely to fall |

**Alpha score** is computed as the probability differential:

```
alpha = P(Long) - P(Short)    ∈ [-1, 1]
```

- `alpha = +0.8` → model is 80% net-confident in upward movement → strong buy signal
- `alpha = +0.1` → near-neutral → no action
- `alpha = -0.5` → model expects downward movement → exit existing position

This classification framing has two advantages over regression:
1. **Naturally handles the "do nothing" case.** A high P(Neutral) suppresses trading in choppy markets, reducing churn.
2. **Probability calibration is actionable.** The confidence gap `P(Long) - P(Short)` maps directly to position sizing via Half-Kelly.

### Training Pipeline

1. Fetches 90 days of 1-minute candles from Binance via ccxt (with CSV caching), or generates synthetic GBM candles with `--synthetic`
2. Resamples candles into 15-minute and 1-hour bars for multi-scale feature extraction
3. Vectorized feature computation — all timeframes and indicators in one pass (~0.5s for 10k candles)
4. Labels each sample by forward 15-minute return → classified into Long / Neutral / Short
5. Trains with cross-entropy loss (optionally recency-weighted), chronological 70/15/15 split
6. Walk-forward validation available for robust out-of-sample testing
7. Saves `artifacts/model.pt` (PyTorch) and `artifacts/model.onnx` (ONNX)
8. Logs to wandb if `--wandb` is set (entity: `Base-Work-Space`, project: `trading-lstm`)

Then switch the config:

```yaml
alpha:
  engine: "lstm"              # or "ensemble" for average of rule-based + LSTM
  model_path: "artifacts/model.onnx"
```

## Design Decisions

### Why multi-scale features instead of raw 1m input?

1-minute candles in crypto are dominated by microstructure noise — bid-ask bounce, random fills, latency jitter. Feeding raw 1m bars into an LSTM forces the model to learn noise-filtering implicitly, wasting capacity. Instead, we resample into 15-minute and 1-hour bars and extract trend indicators (EMA crossover, momentum) at each scale:

- **1m features** capture short-term mean-reversion and volatility spikes
- **15m features** capture intraday trend direction and momentum
- **1h features** capture session-level regime (trending vs. ranging)

The LSTM receives all scales concatenated, letting it learn cross-frequency interactions (e.g., "1m RSI oversold + 1h trend bullish" → high-confidence long).

### Why classification (Long/Neutral/Short) instead of regression?

Regression targets (forward return scaled by tanh) conflate two problems: direction and magnitude. A +0.02% return and a +2.0% return both map to positive values, but require very different actions.

Classification separates these concerns:
- **Direction** is captured by the class label (Long vs. Short)
- **Confidence** is captured by the probability gap `P(Long) - P(Short)`
- **Inaction** is explicitly modeled by P(Neutral) — the model can express "I don't know" instead of being forced to predict a return

This also avoids the tanh saturation problem where extreme returns get compressed, and aligns the loss function (cross-entropy) with the actual decision boundary the strategy cares about.

### Why deep integration of Web3-native data?

Crypto perpetual futures expose signals that have no equivalent in traditional markets:

| Feature | What it reveals | Trading edge |
|---------|----------------|--------------|
| **Funding rate** | Leverage imbalance between longs and shorts across all exchanges | Extreme positive funding → crowded long → mean-reversion risk. Negative funding → shorts paying longs → bullish undercurrent. |
| **Taker buy/sell ratio** | Real-time aggressor flow — who is crossing the spread | Ratio > 1.2 = aggressive buying pressure. Below 0.8 = selling pressure. Divergence from price = potential reversal. |
| **Order book imbalance** | Bid vs. ask depth at top 10 levels | Persistent bid-heavy book with rising price confirms trend. Imbalance without price movement = potential liquidity trap. |

These features are streamed from Binance public endpoints (free, no API key required) and give the LSTM information orthogonal to price-derived indicators. In backtests, adding funding rate and taker ratio to the feature set improved classification accuracy by ~3-5% on the Neutral/Short boundary, where price-only models struggle most.

### Why long-only?

Competition constraint. The state machine only has three states: `FLAT` (no position), `LONG_PENDING` (buy order submitted, waiting for fill), `HOLDING` (in a position). No short selling, no hedging. The classification head still predicts Short probabilities — these are used to suppress entries and trigger exits, not to open short positions.

### Why rule-based alpha as a fallback?

For a 10-day competition, you need signals you can debug and tune in real-time. The rule-based engine combines four indicators into a composite score:

| Component | Weight | Logic |
|-----------|--------|-------|
| RSI | 0.3 | Oversold (RSI < 30) = bullish, overbought (RSI > 70) = bearish |
| Momentum | 0.3 | Rate of change over 10 candles, normalized |
| EMA Crossover | 0.3 | (EMA12 - EMA26) / EMA26, scaled |
| Volatility | 0.1 | Penalty — high vol reduces conviction |

The LSTM model can be trained offline and swapped in via one config change (`alpha.engine: "lstm"` or `"ensemble"` for 50/50 average).

### Why LSTM specifically?

- Crypto price series have temporal dependencies (momentum regimes, mean-reversion cycles, funding rate oscillations)
- 2-layer LSTM with 128 hidden units gives sub-millisecond CPU inference — critical for 66-symbol monitoring
- Multi-scale input sequence (60 candles × N features) naturally fits the LSTM's sequential processing
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

## Strategy Interface

### Signal → Order Routing

The strategy routes orders based on alpha strength to optimize the fee/urgency tradeoff:

| Alpha Range | Order Type | Fee | Rationale |
|-------------|-----------|-----|-----------|
| **> 0.85** (high conviction) | Market order | 0.10% (taker) | Strong signal demands immediate fill — slippage risk outweighs fee savings |
| **0.6 – 0.85** (moderate) | Limit order | 0.05% (maker) | Signal is directional but not urgent — post at favorable price, save 5bps per trade |
| **< 0.6** | No order | — | Below entry threshold — insufficient edge to justify transaction costs |
| **Risk/stop exits** | Market order | 0.10% (taker) | Stops are non-negotiable — guaranteed fill to cap downside |

Over 66 symbols and a 10-day competition, the fee differential between market and limit orders compounds significantly. Routing ~60% of entries through limit orders (alpha 0.6–0.85) saves an estimated 3-5bps on average fill cost.

### Dynamic ATR Stop-Loss

Fixed percentage stops (e.g., "always stop at -3%") fail in crypto because volatility varies by 5-10x across symbols and regimes. The ATR stop adapts:

```
stop_price = entry_price - (ATR_14 × multiplier)
```

- **ATR(14)** measures recent true range — automatically widens during high-vol periods and tightens during consolidation
- **Multiplier = 2.0** (configurable) — gives the position 2 standard moves of breathing room
- Checked every candle iteration alongside the trailing stop

**Example:** If BTC ATR(14) = $500 and entry = $65,000:
- ATR stop = $65,000 - (500 × 2.0) = $64,000 (1.5% below entry)
- If ATR rises to $1,200 during a vol spike: stop widens to $62,600 (3.7%) — avoids getting stopped out by normal noise
- Trailing stop (3% from peak) acts as the tighter backstop once the position is in profit

### Position Sizing (Half-Kelly)

Uses Half-Kelly criterion scaled by alpha signal conviction:

| Alpha Score | Position Size |
|-------------|--------------|
| 0.61 (barely above threshold) | ~5.0% of NAV |
| 0.80 (moderate conviction) | ~6.3% of NAV |
| 1.00 (maximum conviction) | ~7.5% of NAV |

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

## Risk Management

Three layers of protection, in order of trigger speed:

1. **Dynamic ATR Stop (per-position):** If price drops below entry - 2× ATR(14), sell immediately. Automatically adapts to each symbol's volatility regime — tight stops in calm markets, wider stops during high-vol events.
2. **Trailing Stop (per-position):** If price drops 3% from peak since entry, sell immediately. Acts as profit protection once the position moves in-the-money.
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
| Async I/O | `asyncio` + `aiohttp` + `websockets` | Non-blocking WebSocket + HTTP for 66-symbol concurrent streaming |
| Data models | `pydantic` | Validation, serialization, type safety |
| Exchange API | `ccxt` | Unified interface for 100+ exchanges |
| Indicators | `numpy` (pure) | Fast, no DataFrame overhead in hot path |
| Web3 data | Binance public WS + REST | Free funding rate, taker ratio, order book — no API key required |
| DL training | `torch` | GPU support, `torch.compile`, AMP mixed precision |
| DL inference | `onnxruntime` | 2-5x faster than torch for single-sample CPU inference |
| Experiment tracking | `wandb` | Loss curves, run grouping, hyperparameter comparison |
| Config | `pyyaml` | Simple, human-readable |
