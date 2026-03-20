# Web3 Long-Only Quant Trading Framework

This repository is now a factor-first, event-driven trading system for the Roostoo competition and related paper/live workflows.

The primary decision chain is:

`market data -> features -> factor observations -> strategy intent -> trade instruction -> risk validation -> execution`

The model layer is optional. It is no longer the primary strategy trigger. When enabled, it acts only as an overlay for filtering, confidence calibration, and size adjustment.

For a Chinese architecture map with diagrams and module I/O contracts, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Core Architecture

```text
Binance WS / SimFeed
        |
        v
   LiveBuffer
   - 1m OHLCV
   - order-book imbalance
   - funding rate
   - taker ratio
   - open interest
   - resampled 5m / 15m / 1h bars
        |
        v
 StrategyMonitor
        |
        +--> FeatureExtractor
        |     - RSI / EMA / ATR / momentum / volatility / volume ratio
        |
        +--> FactorEngine
        |     - trend_alignment
        |     - momentum_impulse
        |     - volume_confirmation
        |     - liquidity_balance
        |     - perp_crowding
        |     - volatility_regime
        |
        +--> optional AlphaEngine overlay
        |     - filter
        |     - confidence adjustment
        |     - size nudging
        |
        v
   StrategyLogic
   - emits StrategyIntent
   - converts intent into TradeInstruction
        |
        v
    RiskShield
    - exposure limits
    - cash checks
    - daily drawdown breaker
    - ATR / trailing stops
        |
        v
   OrderManager
        |
        v
 Executor (paper / live / roostoo)
        |
        +--> PortfolioTracker
        +--> TradeLogger
```

## Why This Architecture

- Strategy comes before model. The system trades explicit market structure and flow observations, not a black-box score.
- Signals must be explainable. Every factor has a bias, strength, horizon, thesis, and invalidation rule.
- The system outputs executable commands, not vague opinions.
- Risk and execution timing are realistic enough for research handoff to production-style runtime.
- The architecture is designed so new web3 factors can be added at the factor layer without rewriting the execution stack.

## Runtime Contracts

The system is built around explicit typed objects in `core/models.py`.

| Object | Role |
| --- | --- |
| `OHLCV` | Market bar input |
| `FeatureVector` | Engineered technical and supplementary features |
| `FactorObservation` | One named factor with bias, strength, horizon, thesis, and invalidation |
| `FactorSnapshot` | Aggregate factor state for one symbol at one timestamp |
| `StrategyIntent` | Human-readable trading decision before risk/execution |
| `TradeInstruction` | Normalized execution command with expiry and order fields |
| `Order` | Executor-facing order object |
| `PortfolioSnapshot` | Current NAV, positions, PnL, drawdown state |

This separation matters:

- `FactorSnapshot` is about market interpretation.
- `StrategyIntent` is about trading decision.
- `TradeInstruction` is about executable command shape.
- `Order` is about exchange submission.

Each layer narrows ambiguity instead of mixing reasoning, sizing, and execution concerns together.

## Runtime Flow

### 1. Market Data Ingestion

- `data/connector.py` streams 1-minute candles from Binance or uses the simulated feed in paper mode.
- `BinanceSupplementaryFeed` collects order-book imbalance, funding rate, taker ratio, and open interest.
- `data/buffer.py` stores recent candles, resampled bars, and supplementary history in `LiveBuffer`.

### 2. Bar Gating And Resampling

- `data/resampler.py` converts 1-minute candles into 5m and higher-timeframe bars.
- `strategy/monitor.py` only runs the decision pipeline when the primary resampled bar closes.
- `main.py` replays prefetched 1-minute history through the resampler on startup so 5m / 15m / 1h buffers are warm before the first live trading decision.
- Price updates and stop checks still happen on the 1-minute stream.

This gives two useful properties:

- entry logic is evaluated on a cleaner strategic timeframe
- risk logic still reacts on the native 1-minute stream

### 3. Feature Extraction

- `features/extractor.py` turns candle history into a `FeatureVector`
- the hot path stays in numpy, without pandas
- the same extractor also builds normalized feature sequences for neural inference

Core runtime features include:

- RSI
- fast and slow EMA
- ATR
- momentum
- realized volatility
- volume ratio
- order-book imbalance
- funding rate
- taker ratio

### 4. Factor Construction

- `strategy/factor_engine.py` converts raw features plus supplementary context into explicit `FactorObservation` objects
- those observations are aggregated into one `FactorSnapshot`

Current factor pack:

- `trend_alignment`
- `momentum_impulse`
- `volume_confirmation`
- `liquidity_balance`
- `perp_crowding`
- `volatility_regime`

The snapshot exposes:

- `entry_score`
- `blocker_score`
- `exit_score`
- `confidence`
- `supporting_factors`
- `blocking_factors`
- `summary`

### 5. Optional Model Overlay

- `models/inference.py` provides `AlphaEngine`
- `models/model_wrapper.py` loads `.onnx` or `.pt` checkpoints
- `strategy/monitor.py` only uses the model overlay when `strategy.use_model_overlay=true`
- the overlay now scores on the same primary strategy timeframe as the factor engine, not on mismatched raw 1-minute candles
- supplementary history passed into the model is aligned to that same strategy cadence

The overlay is not allowed to replace the factor strategy.

It may only:

- filter weak entries
- slightly boost conviction when aligned
- nudge size and exit urgency

Important:

- `models/lstm_model.py` and `models/transformer_model.py` define architectures only
- they are not trained checkpoints by themselves
- inference-ready artifacts must be loaded separately through `ModelWrapper`
- the runtime works without a trained model

### 6. Strategy Decision Layer

- `strategy/logic.py` consumes a `FactorSnapshot` and emits a `StrategyIntent`
- the same module converts that intent into a `TradeInstruction`
- long entries now require breadth by default:
  - at least `min_supporting_factors=2`
  - at least `min_supporting_categories=2`
  - `trend_alignment` present unless explicitly disabled

This is the point where the system decides:

- whether to buy or sell
- whether to use market or limit entry
- how much capital to allocate
- how urgent the instruction is
- what invalidates the thesis
- what stop-loss and take-profit framing to use

The strategy state machine remains explicit:

- `FLAT`
- `LONG_PENDING`
- `HOLDING`
- `EXIT_PENDING`

That state machine now treats live partial fills explicitly:

- partial entry fills keep the strategy in `LONG_PENDING` until completion or cancellation
- factor-driven exits move to `EXIT_PENDING` instead of pretending the position is already flat
- cancelled exit orders return from `EXIT_PENDING` to `HOLDING`

### 7. Risk Layer

- `risk/risk_shield.py` validates orders before submission
- `risk/tracker.py` maintains portfolio state, NAV, exposure, and drawdown

Pre-trade checks include:

- circuit breaker state
- long-only enforcement
- per-minute order rate limit
- single-name exposure
- total portfolio exposure
- cash sufficiency

Post-trade and runtime checks include:

- trailing stop
- ATR stop
- daily drawdown circuit breaker

The circuit breaker now uses true daily drawdown, not lifetime drawdown.

### 8. Order Routing And Execution

- `execution/order_manager.py` manages submission, pending state, cancellation, and fill callbacks
- executors sit behind a clean interface:
  - `execution/sim_executor.py`
  - `execution/executor.py`
  - `execution/roostoo_executor.py`

Paper execution is no longer same-bar optimistic.

- market orders fill on the next candle open with slippage
- limit orders fill only if the next bar touches the limit price
- cumulative exchange fill updates are converted into incremental deltas before they hit `PortfolioTracker`, so partial fills do not double-count position or cash changes

That timing discipline makes offline behavior less misleading.

### 9. Audit And Logging

- `execution/trade_logger.py` writes append-only JSONL logs
- `strategy/monitor.py` logs the chain end to end

Structured events now include:

- `factor_snapshot`
- `strategy_intent`
- `trade_instruction`
- `order`
- `api`

This makes the strategy explainable after the fact instead of only during code reading.

## Module Map

| Path | Responsibility |
| --- | --- |
| `main.py` | Composition root, mode selection, startup, shutdown |
| `core/models.py` | Shared schemas and typed contracts |
| `data/connector.py` | Binance and supplementary data ingestion |
| `data/buffer.py` | In-memory market state and history |
| `data/resampler.py` | Multi-timeframe bar construction |
| `features/extractor.py` | Technical feature engineering |
| `strategy/factor_engine.py` | Explicit factor generation |
| `strategy/logic.py` | Strategy state machine and intent generation |
| `strategy/monitor.py` | Runtime orchestrator for the full decision pipeline |
| `models/inference.py` | Rule-based alpha and model overlay engine |
| `models/model_wrapper.py` | Checkpoint loading for ONNX and PyTorch |
| `models/train.py` | Offline model training and export |
| `risk/risk_shield.py` | Pre-trade and runtime risk controls |
| `risk/tracker.py` | Portfolio state, PnL, drawdown, risk metrics |
| `execution/order_manager.py` | Order lifecycle management |
| `execution/trade_logger.py` | Structured audit logs |

## Research And Training Pipeline

The repo has two connected but distinct loops:

### Online Runtime Loop

`data -> features -> factors -> intent -> instruction -> risk -> execution -> portfolio update`

This is the production-style decision loop.

### Offline Research Loop

`historical data -> feature sequences -> labels -> model training -> checkpoint export -> optional runtime overlay`

`models/train.py` is responsible for:

1. loading historical OHLCV
2. computing features with the same extractor family
3. building sequence and forward-return labels
4. training LSTM or Transformer models
5. exporting the best model to ONNX or checkpoint artifacts

Research hardening in this branch includes:

- label normalization fitted on the training split only
- removal of fabricated random supplementary features
- paper execution aligned to next-bar fills
- model kept secondary to explicit factors

## Extending The Strategy With New Web3 Factors

The preferred extension point is the factor layer, not the executor and not the model architecture.

Recommended integration path:

`external source -> normalized adapter -> FactorObservation -> FactorSnapshot -> StrategyIntent`

Examples of factor families that fit naturally here:

- on-chain valuation and profit state
- exchange flow and stablecoin flow
- perpetual funding and open-interest crowding
- protocol revenue and usage trends
- bridge and wallet cohort flows
- microstructure imbalance from perp order books

This keeps the architecture stable even as the research surface expands.

## Modes

### Paper

```bash
.venv/bin/python3 main.py
```

- uses `SimulatedFeed`
- uses `SimExecutor`
- no API keys required

### Roostoo Competition

```bash
.venv/bin/python3 main.py --mode roostoo
```

- uses Binance for market data
- uses Roostoo for execution
- recovers positions on restart
- backfills resampled strategy history on startup so factor generation is warm immediately instead of waiting for fresh 5m / 15m / 1h bars
- keeps sell-side strategy state in `EXIT_PENDING` while Roostoo orders are only partially filled
- writes structured JSONL logs

Roostoo credentials can come from either:

- `ROOSTOO_COMP_API_KEY` / `ROOSTOO_COMP_API_SECRET`
- `ROOSTOO_API_KEY` / `ROOSTOO_API_SECRET`

### Live

```bash
.venv/bin/python3 main.py --mode live
```

- uses Binance data
- uses exchange execution
- requires credentials

## Quick Start

```bash
git clone https://github.com/qunzhongwang/trading_competition.git
cd trading_competition

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv
source .venv/bin/activate
uv sync

cp .env.example .env
# add ROOSTOO_COMP_API_KEY and ROOSTOO_COMP_API_SECRET if needed
```

Run paper mode:

```bash
.venv/bin/python3 main.py
```

Run competition mode:

```bash
.venv/bin/python3 main.py --mode roostoo
```

Wrapper script:

```bash
./scripts/start_competition.sh
```

## Roostoo Smoke Tests

The repository includes opt-in live smoke tests in `tests/test_roostoo_live_smoke.py`.

They are gated on environment variables so the default test suite never sends
real API requests.

Balance and ticker smoke:

```bash
export RUN_ROOSTOO_SMOKE=1
export ROOSTOO_COMP_API_KEY=...
export ROOSTOO_COMP_API_SECRET=...
.venv/bin/pytest -q tests/test_roostoo_live_smoke.py -k balance_and_ticker
```

Optional live order smoke:

```bash
export RUN_ROOSTOO_SMOKE=1
export RUN_ROOSTOO_ORDER_SMOKE=1
export ROOSTOO_COMP_API_KEY=...
export ROOSTOO_COMP_API_SECRET=...
export ROOSTOO_ORDER_SMOKE_QTY=0.00005
export ROOSTOO_ORDER_SMOKE_PRICE=1000
.venv/bin/pytest -q tests/test_roostoo_live_smoke.py -k live_order
```

Notes:

- the balance/ticker smoke is the safer default
- the order smoke is intentionally disabled unless explicitly enabled
- if the environment variables are missing, these tests will be skipped
- `ROOSTOO_BASE_URL` and `ROOSTOO_SMOKE_SYMBOL` are optional overrides

## Optional Model Training

Train an overlay model:

```bash
.venv/bin/python3 -m models.train --synthetic --symbols BTC/USDT --device auto
.venv/bin/python3 -m models.train --symbols BTC/USDT --days 90 --walk-forward --device auto
```

Run with overlay enabled:

```bash
.venv/bin/python3 main.py --mode roostoo --use-model-overlay --engine lstm
.venv/bin/python3 main.py --mode roostoo --use-model-overlay --engine ensemble
```

`--engine` selects the overlay engine only. It does not replace the factor strategy.
