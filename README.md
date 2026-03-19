# Web3 Long-Only Quant Trading Framework

Algorithmic trading bot for the Roostoo 10-day crypto trading competition. Long-only, event-driven, async Python.

**Architecture:** Binance WebSocket for real-time OHLCV data (high quality, free). Roostoo REST API for order execution and balance tracking on the mock exchange.

## Competition Quick Start (5 minutes)

Get the bot running in Roostoo competition mode as fast as possible:

```bash
# 1. Clone, install, activate
git clone <repo-url> && cd trading-competition
curl -LsSf https://astral.sh/uv/install.sh | sh  # skip if you have uv
uv venv .venv && source .venv/bin/activate && uv sync

# 2. Set up API credentials
cp .env.example .env
# Edit .env — fill in your Roostoo API key and secret:
#   ROOSTOO_API_KEY=your_key_here
#   ROOSTOO_API_SECRET=your_secret_here
# For the actual competition, use ROOSTOO_COMP_API_KEY / ROOSTOO_COMP_API_SECRET instead

# 3. (Optional) Train LSTM model — skip to use rule-based alpha engine
python -m models.train --synthetic --symbols BTC/USDT --device auto
# Then set alpha.engine to "lstm" or "ensemble" in config/default.yaml

# 4. Verify everything works (dry run with paper mode)
python main.py  # Ctrl+C after a few minutes — check for errors

# 5. Start competition mode
python main.py --mode roostoo
# Or use the auto-restart script:
./scripts/start_competition.sh
```

### Deploy to AWS EC2 (recommended for 10-day competition)

```bash
# One-command deploy: installs Python 3.11, uv, clones repo, sets up systemd
./scripts/deploy_aws.sh <EC2_HOST> [~/.ssh/your-key.pem]

# SSH in and set API keys
ssh -i ~/.ssh/your-key.pem ubuntu@<EC2_HOST>
cat > /home/ubuntu/trading_competition/.env << 'EOF'
ROOSTOO_COMP_API_KEY=your_competition_key
ROOSTOO_COMP_API_SECRET=your_competition_secret
EOF

# Start (systemd — auto-restarts on crash)
sudo systemctl start trading-bot

# Monitor
journalctl -u trading-bot -f                          # system logs
tail -f /home/ubuntu/trading_competition/logs/trades_*.jsonl  # trade events
```

### Pre-Competition Checklist

| Step | Command / Action | Status |
|------|-----------------|--------|
| Install dependencies | `uv sync` | |
| Set API credentials in `.env` | Use `ROOSTOO_COMP_*` keys for competition | |
| Choose alpha engine | Set `alpha.engine` in `config/default.yaml` (`rule_based` / `lstm` / `ensemble`) | |
| Train model (if using LSTM) | `python -m models.train --synthetic --device auto` | |
| Paper test run (1+ hours) | `python main.py` — verify no crashes, check logs | |
| Roostoo test run | `python main.py --mode roostoo` — verify orders execute | |
| Run tests | `pytest` — all 210 tests should pass | |
| Deploy to EC2 | `./scripts/deploy_aws.sh <host>` | |
| Set EC2 API keys | SSH in, create `.env` with competition keys | |
| Start bot | `sudo systemctl start trading-bot` | |
| Monitor first hour | `journalctl -u trading-bot -f` — watch for fills and risk metrics | |

### Credential Priority

The `.env` file supports two sets of Roostoo credentials. Competition keys take priority if both are set:

1. **Competition API** (`ROOSTOO_COMP_API_KEY` / `ROOSTOO_COMP_API_SECRET`): Use these during the actual competition (Mar 21-31)
2. **Testing API** (`ROOSTOO_API_KEY` / `ROOSTOO_API_SECRET`): For pre-competition testing against the mock exchange

## Paper Mode Quick Start

```bash
# No API keys needed — generates synthetic price data
uv venv .venv && source .venv/bin/activate && uv sync
python main.py
```

Press `Ctrl+C` to stop — prints a final PnL report with risk metrics on shutdown.

### Docker (alternative)

```bash
# Build image (CPU-only torch, ~1.2 GB)
docker build -t trading-bot -f docker/Dockerfile .

# Run paper mode
docker run --rm trading-bot

# Run Roostoo competition mode
docker run --rm --env-file .env trading-bot --mode roostoo

# Run live (Binance) mode
docker run --rm -e BINANCE_API_KEY=... -e BINANCE_API_SECRET=... trading-bot --mode live
```

## Roostoo Competition Mode

### How Roostoo Mode Works

```
Binance WebSocket ──(real-time 1m candles)──> LiveBuffer ──> StrategyMonitor
                                                                    │
                                              Feature extraction + Alpha scoring
                                                                    │
                                              Order decision + Risk validation
                                                                    │
                                    Roostoo REST API <──(BTC/USDT → BTC/USD mapping)
                                              │
                            POST /v3/place_order (HMAC SHA256 signed)
```

- **Data source:** Binance WebSocket (65 symbols × 1m candles). Free, high quality, already implemented.
- **Execution:** Roostoo REST API. Symbol mapping converts internal `BTC/USDT` → `BTC/USD` at the executor boundary.
- **Auth:** HMAC SHA256 signature on sorted query params. Auto-validates server time drift.
- **Trade logging:** All orders, signals, and API calls logged to `logs/trades_{date}.jsonl`.

### Competition Details

| Parameter | Value |
|-----------|-------|
| Starting capital | $1,000,000 |
| Trading pairs | 65 (see `config/default.yaml`) |
| Fees | 0.1% taker (market), 0.05% maker (limit) |
| Constraints | Spot only, no short selling, no HFT/arbitrage |
| Duration | Mar 21-31 (at least 8 active trading days) |

**Scoring:**
- Screen 2: Portfolio return (qualifier)
- Screen 3 (40%): `0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar`
- Screen 4 (60%): Code quality, strategy clarity, runnability

### AWS EC2 Deployment

See [Competition Quick Start > Deploy to AWS EC2](#deploy-to-aws-ec2-recommended-for-10-day-competition) above for full deployment instructions.

## Scripts

All scripts live in `scripts/` with comments and configurable env vars:

```bash
./scripts/start.sh              # Production runner with auto-restart (up to 50 retries)
./scripts/start_competition.sh  # Roostoo competition mode runner (auto-restart, env validation)
./scripts/deploy_aws.sh         # Deploy to AWS EC2 (system setup + systemd service)
./scripts/paper_trade.sh        # Quick paper trading session
./scripts/train.sh              # Train LSTM with synthetic data
./scripts/test.sh               # Run tests with coverage
./scripts/export_model.sh       # Export .pt → .onnx
python scripts/upload_model_to_hf.py --repo-id USER/REPO  # Push artifacts to Hugging Face
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
# Run all tests (210 tests, ~12s)
pytest

# Single file or test
pytest tests/test_strategy.py
pytest tests/test_features.py::TestRSI::test_all_gains

# With coverage report
pytest --cov=. --cov-report=term-missing
```

Test modules cover: Pydantic models, feature extraction (RSI/EMA/ATR/momentum/vol), portfolio tracker (cash/PnL/NAV/exposure), strategy state machine, risk shield (stops/circuit breaker/rate limit/exposure caps), alpha engine (rule-based/ensemble), async buffer, sim executor (market/limit fills), order manager lifecycle, simulated feed (GBM/CSV replay), training pipeline (dataset building/synthetic generation), integration (full pipeline end-to-end), config loading, signal confirmation, multi-timeframe resampling, ICIR tracker, trade tracker, graduated exits, and alpha decay.

## Architecture

```
                         ┌──────────────────────────────────────────────────┐
                         │              Feature Engine (10 features)        │
                         │                                                  │
  [WSConnector/SimFeed]  │  1m candles ──> RSI, EMA, ATR, Momentum, Vol    │
        │                │                                                  │
   push_candle           │  + Web3 native:                                  │
        │                │    Funding Rate (perp sentiment)                 │
        ▼                │    Taker Buy/Sell Ratio (aggressor flow)         │
   [LiveBuffer]          │    Order Book Imbalance (microstructure)         │
        │                │    Volume Ratio                                  │
  event.set()            │                                                  │
        │                │  + Multi-Timeframe (15m, 1h):                   │
        ▼                │    15m EMA crossover → trend filter              │
  [StrategyMonitor]      │    1h momentum(10) → regime filter               │
        │                └──────────────┬───────────────────────────────────┘
        │                               │
        │                               ▼
        │                ┌──────────────────────────────────┐
        │                │  LSTM Regression Head              │
        │                │  Input:  (batch, 60, 10)           │
        │                │  Output: Tanh → alpha ∈ [-1, 1]    │
        │                │  (planned: 3-class classification) │
        │                └──────────────┬───────────────────┘
        │                               │
        ├───────────────────────────────┘
        ▼
   AlphaEngine ──> Signal {alpha ∈ [-1, 1]}
        │               │
        │          Multi-TF filter (dampen/boost)
        │          Per-symbol ICIR weights (Bayesian shrinkage)
        │          Alpha decay (configurable half-life)
        │
        ▼
   StrategyLogic ──────────> RiskShield ──────────> OrderManager ──> Executor
   (Signal confirmation,     (dynamic ATR stop,      (market if α > 0.85,
    Half-Kelly sizing,        trailing stop,           limit if α ∈ [0.6, 0.85],
    Graduated exit tiers,     circuit breaker)          order timeout 30s)
    Adaptive Kelly)                                             │
                                                   PortfolioTracker.on_fill()
                                                   TradeLogger (JSONL)
                                                   RiskMetrics (Sortino/Sharpe/Calmar)
```

**Executors:**
- **Paper mode:** `SimExecutor` — instant fills with slippage simulation
- **Live mode:** `LiveExecutor` — real orders via ccxt (Binance)
- **Roostoo mode:** `RoostooExecutor` — HMAC-signed REST API to Roostoo mock exchange

**Core design principles:**

1. **Web3-native features as first-class inputs.** Unlike equities, crypto perpetual futures expose unique sentiment signals — funding rates reflect leverage imbalance between longs and shorts, and taker buy/sell ratios reveal real-time aggressor flow. These features, streamed directly from Binance public endpoints, give the model information that pure price-derived indicators cannot capture.

2. **Event-driven, not polling.** The data feed pushes candles into a `LiveBuffer`. The `StrategyMonitor` blocks on `asyncio.Event` until a new closed candle arrives — zero CPU waste between events.

3. **Multi-scale inputs.** The `MultiResampler` resamples 1m candles into 15-minute and 1-hour bars, extracting trend-level EMA crossovers and momentum from each timeframe to filter out microstructure noise. Higher-TF signals dampen or boost the rule-based alpha without affecting the LSTM path.

### Feature Summary

| Category | Features | Timeframe | Source |
|----------|----------|-----------|--------|
| Price-derived | RSI(14), EMA(12/26), ATR(14), Momentum(10), Volatility(20) | 1m | Candles |
| Web3 sentiment | Funding rate | Real-time (per-candle history) | Binance Futures WS |
| Order flow | Taker buy/sell ratio | 5m (per-candle history) | Binance REST |
| Microstructure | Order book imbalance, Volume ratio | 100ms / per candle (per-candle history) | Binance Depth WS |

All supplementary endpoints are free, require no API key, and fail gracefully (features default to neutral values on timeout). Supplementary features (funding rate, taker ratio, order book imbalance) are stored as per-candle time series in `LiveBuffer`, providing natural variation for LSTM z-score normalization at inference time.

## Project Structure

```
trading-competition/
├── config/
│   └── default.yaml            # All tunable parameters (65 symbols, thresholds, risk limits, fees)
├── artifacts/                  # Model checkpoints (.pt, .onnx) — gitignored
├── logs/                       # Rotating log files + trade JSONL — gitignored
├── scripts/
│   ├── start.sh                # Production runner with auto-restart
│   ├── start_competition.sh    # Roostoo competition runner
│   ├── deploy_aws.sh           # AWS EC2 deployment script
│   ├── paper_trade.sh          # Quick paper trading session
│   ├── train.sh / test.sh      # Training and testing
│   ├── export_model.sh         # .pt → .onnx export
│   ├── upload_model_to_hf.py   # Push model.pt / .onnx to Hugging Face Hub
│   └── generate_data.sh        # Synthetic CSV data generation
├── core/
│   └── models.py               # Pydantic models (Tick, OHLCV, Signal, Order, Position, RiskMetrics)
├── data/
│   ├── buffer.py               # asyncio.Event-driven sliding window for candles + supplementary + resampled data
│   ├── connector.py            # Binance WebSocket connector (live) + BinanceSupplementaryFeed
│   ├── sim_feed.py             # Synthetic GBM candle generator (paper mode)
│   ├── resampler.py            # CandleResampler (1m→Nm) + MultiResampler (multi-timeframe)
│   ├── roostoo_auth.py         # HMAC SHA256 authentication for Roostoo API
│   └── historical/             # Cached CSV from ccxt (auto-created by training script)
├── features/
│   └── extractor.py            # OHLCV → 10 features (vectorized + iterative fallback)
├── models/
│   ├── lstm_model.py           # PyTorch LSTM architecture (2-layer, 128 hidden, tanh regression head)
│   ├── model_wrapper.py        # Loads .onnx or .pt model for inference
│   ├── inference.py            # AlphaEngine: rule-based / lstm / ensemble scoring + multi-TF filter
│   ├── icir_tracker.py         # Per-symbol Bayesian ICIR tracker with online shrinkage
│   └── train.py                # Offline training (walk-forward validation, recency weighting)
├── strategy/
│   ├── logic.py                # Per-symbol state machine + signal confirmation + graduated exits
│   ├── monitor.py              # Central event loop / orchestrator (multi-TF, ICIR, adaptive Kelly)
│   └── trade_tracker.py        # Rolling trade outcomes for adaptive Kelly sizing
├── risk/
│   ├── risk_shield.py          # Pre-trade validation, dynamic ATR stop, trailing stop, circuit breaker
│   └── tracker.py              # Real-time PnL, positions, NAV, Sortino/Sharpe/Calmar computation
├── execution/
│   ├── executor.py             # BaseExecutor ABC + LiveExecutor (ccxt)
│   ├── roostoo_executor.py     # RoostooExecutor — REST API with HMAC auth, symbol mapping
│   ├── sim_executor.py         # Paper trading executor (instant fills + slippage)
│   ├── order_manager.py        # Order lifecycle, fill callbacks, timeout cancellation
│   └── trade_logger.py         # Structured JSONL trade/signal/API event logging
├── docker/
│   └── Dockerfile              # Multi-stage build (Python 3.11, CPU torch)
├── main.py                     # Entry point — wires everything, runs asyncio loop
├── .env.example                # Template for API credentials
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

### Roostoo Competition Mode

```bash
# Set API keys (one-time)
cp .env.example .env
# Edit .env with your credentials

# Run
python main.py --mode roostoo
```

- Connects to **Binance WebSocket** for real-time 1m candle data (65 symbols)
- Executes orders on **Roostoo mock exchange** via signed REST API
- Symbol mapping: internal `BTC/USDT` → Roostoo `BTC/USD` at executor boundary
- Syncs balance from Roostoo on startup
- Logs all trades to `logs/trades_{date}.jsonl`
- Computes risk-adjusted metrics (Sortino, Sharpe, Calmar) continuously

### Live Mode

```bash
python main.py --mode live
```

- Connects to Binance via WebSocket for real-time kline data
- Executes orders via ccxt async exchange client
- Requires Binance API keys in `.env` or config

## Risk Metrics

Competition score is 40% risk-adjusted performance. The tracker computes these continuously:

| Metric | Formula | Weight |
|--------|---------|--------|
| Sortino Ratio | `mean_excess_return / downside_std × √365` | 40% |
| Sharpe Ratio | `mean_excess_return / total_std × √365` | 30% |
| Calmar Ratio | `annualized_return / max_drawdown` | 30% |

**Composite = 0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar**

Metrics are logged periodically during trading and in the final shutdown report.

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

**Time-series split (default 80/20, walk-forward 60/20/20):**

```
Default mode:
Day 1                                      Day 72           Day 90
 │────────────── Train (80%) ──────────────│── Val (20%) ────│
 │              ~103,680 candles           │  ~25,920        │

Walk-forward mode (--walk-forward):
Day 1                    Day 54           Day 72           Day 90
 │─────── Train (60%) ──────│── Val (20%) ──│── Test (20%) ──│
```

- **Train:** Model learns feature → signal mappings. Recency weighting (optional, `--recency-half-life 35`) down-weights older samples so the model prioritizes recent market behavior.
- **Validation:** Hyperparameter tuning and early stopping. ReduceLROnPlateau monitors val loss; training stops if no improvement for 10 epochs.
- **Test (walk-forward only):** Final out-of-sample evaluation. Never touched during training.

Each symbol is split chronologically before concatenation — no cross-symbol temporal leakage. No shuffling, no future leakage.

### Model Output

The LSTM outputs a scalar alpha score via tanh activation:

```
alpha = tanh(linear(LSTM_hidden)) ∈ [-1, 1]
```

- `alpha = +0.8` → strong bullish signal → buy (market order if > 0.85)
- `alpha = +0.1` → weak signal → no action (below entry threshold 0.6)
- `alpha = -0.3` → bearish signal → exit existing position (below exit threshold -0.2)

Labels are constructed from forward returns: `y = tanh(forward_return × 100)`, scaled to [-1, 1]. Trained with MSE loss (optionally recency-weighted).

**Planned upgrade:** Convert to 3-class classification (Long/Neutral/Short) with cross-entropy loss, where alpha = P(Long) - P(Short). This would naturally handle the "do nothing" case via P(Neutral) and provide better probability calibration for position sizing.

### Training Pipeline

1. Fetches 90 days of 1-minute candles from Binance via ccxt (with CSV caching), or generates synthetic GBM candles with `--synthetic`
2. Vectorized feature computation — all 10 features in one pass (~0.5s for 10k candles)
3. Labels = forward return (configurable via `--forward-window`), scaled to [-1, 1] via tanh
4. Trains with MSE loss (optionally recency-weighted), chronological split (80/20 or 60/20/20 walk-forward)
5. Per-symbol chronological split before concatenation — no cross-symbol temporal leakage
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

### Multi-scale features

1-minute candles in crypto are dominated by microstructure noise — bid-ask bounce, random fills, latency jitter. The `MultiResampler` resamples candles into 15-minute and 1-hour bars and extracts trend indicators (EMA crossover, momentum) at each scale:

- **1m features** capture short-term mean-reversion and volatility spikes
- **15m EMA(12)/EMA(26) crossover** captures intraday trend direction
- **1h momentum(10)** captures session-level regime (trending vs. ranging)

The multi-TF filter dampens or boosts the rule-based alpha score:
- Bearish higher-TF context (filter < 0) dampens bullish alpha: `alpha *= max(0, 1 + filter)`
- Bullish higher-TF context (filter > 0) slightly boosts alpha: `alpha *= min(1.5, 1 + 0.2 * filter)`
- LSTM path is unchanged — multi-TF filtering only applies to rule-based and ensemble modes

### Why regression output (with planned classification upgrade)?

The current model uses tanh regression, predicting forward returns scaled to [-1, 1]. This is simple and works well with MSE loss.

A planned upgrade will convert to 3-class classification (Long/Neutral/Short):
- **Direction** captured by class label (Long vs. Short)
- **Confidence** captured by probability gap `P(Long) - P(Short)`
- **Inaction** explicitly modeled by P(Neutral)

This would avoid tanh saturation on extreme returns and align the loss function (cross-entropy) with the strategy's actual decision boundary.

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

| Component | Default Weight | Logic |
|-----------|--------|-------|
| RSI | 0.3 | Oversold (RSI < 30) = bullish, overbought (RSI > 70) = bearish |
| Momentum | 0.3 | Rate of change over 10 candles, normalized |
| EMA Crossover | 0.3 | (EMA12 - EMA26) / EMA26, scaled |
| Volatility | 0.1 | Penalty — high vol reduces conviction |

**Per-symbol adaptive weights (ICIR):** When enabled, the `BayesianICIRTracker` replaces these default weights with per-symbol Bayesian-shrunk weights. Offline priors are loaded from `artifacts/icir_priors.json`; during the competition, Bayesian shrinkage (`λ = min_lambda + (1-min_lambda) × e^(-n/τ)`) continuously adapts toward online IC observations while always retaining 30% of the prior.

The LSTM model can be trained offline and swapped in via one config change (`alpha.engine: "lstm"` or `"ensemble"` for 50/50 average).

### Why LSTM specifically?

- Crypto price series have temporal dependencies (momentum regimes, mean-reversion cycles, funding rate oscillations)
- 2-layer LSTM with 128 hidden units gives sub-millisecond CPU inference — critical for 65-symbol monitoring
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

The strategy routes orders based on alpha strength to optimize the fee/urgency tradeoff. **Signal confirmation** requires N consecutive bars (default 2) above the entry threshold before emitting a buy order — this filters out single-bar noise spikes.

| Alpha Range | Order Type | Fee | Rationale |
|-------------|-----------|-----|-----------|
| **> 0.85** (high conviction) | Market order | 0.10% (taker) | Strong signal demands immediate fill — slippage risk outweighs fee savings |
| **0.6 – 0.85** (moderate) | Limit order | 0.05% (maker) | Signal is directional but not urgent — post at favorable price, save 5bps per trade |
| **< 0.6** | No order | — | Below entry threshold — insufficient edge to justify transaction costs |
| **Risk/stop exits** | Market order | 0.10% (taker) | Stops are non-negotiable — guaranteed fill to cap downside |

**Limit order timeout:** Pending limit orders are automatically cancelled after `order_timeout_seconds` (default 30s) to prevent capital from being trapped indefinitely in unfilled orders.

**Graduated exits:** Instead of all-or-nothing exits, configurable exit tiers allow partial position reduction:
```yaml
exit_tiers:
  - threshold: -0.1   # first tier: sell 50% when alpha drops to -0.1
    sell_pct: 0.5
  - threshold: -0.3   # second tier: sell remaining when alpha drops to -0.3
    sell_pct: 1.0
```

Over 65 symbols and a 10-day competition, the fee differential between market and limit orders compounds significantly. Routing ~60% of entries through limit orders (alpha 0.6–0.85) saves an estimated 3-5bps on average fill cost.

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

**Adaptive Kelly (optional):** When `adaptive_kelly: true`, the `TradeTracker` replaces static win rate / payoff ratio priors with blended estimates from rolling trade outcomes. The blending coefficient `alpha = min(1, n_trades / min_trades)` smoothly transitions from pure priors to observed statistics as more trades are recorded.

## Configuration

All parameters live in `config/default.yaml`. Key settings:

```yaml
# Mode: "paper" (default), "live" (Binance), "roostoo" (competition)
mode: "paper"

# 65 trading pairs (internal format: XXX/USDT for Binance data)
# Automatically mapped to XXX/USD for Roostoo execution
symbols:
  - "BTC/USDT"
  - "ETH/USDT"
  # ... 63 more pairs

# Roostoo mock exchange (credentials via .env or env vars)
roostoo:
  base_url: "https://mock-api.roostoo.com"

# Which alpha engine to use
alpha:
  engine: "rule_based"      # "rule_based" | "lstm" | "ensemble"
  entry_threshold: 0.6      # alpha > this → buy
  exit_threshold: -0.2      # alpha < this → sell
  model_path: "artifacts/model.onnx"
  seq_len: 60               # LSTM lookback window (60 × 1m = 1 hour)
  resample_minutes: 5       # gate alpha scoring on 5-min bar completion
  multi_timeframes: [15, 60] # higher-TF filters (15m EMA crossover, 1h momentum)
  decay_half_life_s: 999999  # alpha signal decay (set 150 to enable)
  icir_prior_path: "artifacts/icir_priors.json"  # per-symbol factor weights
  icir_window: 100           # rolling IC window for online learning
  icir_min_lambda: 0.3       # floor shrinkage (always keep 30% prior)

# Position sizing (Half-Kelly)
strategy:
  base_size_pct: 0.05       # minimum allocation (5% of NAV)
  max_size_pct: 0.15        # maximum allocation (15% of NAV)
  kelly_fraction: 0.5       # half-Kelly multiplier
  estimated_win_rate: 0.55   # conservative prior
  estimated_payoff: 1.5      # avg_win / avg_loss
  urgent_alpha_threshold: 0.85  # alpha above this → market order (below → limit order)
  confirmation_bars: 2       # require N consecutive bars above threshold
  adaptive_kelly: false      # enable rolling Kelly from trade outcomes
  exit_tiers: []             # graduated exits (see Signal → Order Routing)

# Risk limits
risk:
  trailing_stop_pct: 0.03         # 3% trailing stop from peak
  atr_stop_multiplier: 2.0        # exit if price drops 2x ATR below entry
  daily_drawdown_limit: 0.05      # 5% daily drawdown → circuit breaker halts all trading
  max_portfolio_exposure: 0.50    # max 50% of NAV in positions
  max_single_exposure: 0.15       # max 15% of NAV per symbol
  max_orders_per_minute: 60       # rate limit (supports 65 symbols)

# Execution fees
execution:
  fee_market_bps: 10        # 0.1% taker fee (market orders)
  fee_limit_bps: 5          # 0.05% maker fee (limit orders)

# Paper mode settings
paper:
  initial_capital: 1000000.0  # $1M (matches competition)
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
- Trade events: `logs/trades_{date}.jsonl` (structured JSONL — orders, signals, API calls)
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
| Async I/O | `asyncio` + `aiohttp` + `websockets` | Non-blocking WebSocket + HTTP for 65-symbol concurrent streaming |
| Data models | `pydantic` | Validation, serialization, type safety |
| Exchange API | `ccxt` (Binance) + `aiohttp` (Roostoo) | Binance WS for data, Roostoo REST for competition execution |
| Indicators | `numpy` (pure) | Fast, no DataFrame overhead in hot path |
| Web3 data | Binance public WS + REST | Free funding rate, taker ratio, order book — no API key required |
| DL training | `torch` | GPU support, `torch.compile`, AMP mixed precision |
| DL inference | `onnxruntime` | 2-5x faster than torch for single-sample CPU inference |
| Experiment tracking | `wandb` | Loss curves, run grouping, hyperparameter comparison |
| Config | `pyyaml` | Simple, human-readable |
