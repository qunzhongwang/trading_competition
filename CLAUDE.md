# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run paper trading (synthetic data, no API keys needed)
source .venv/bin/activate && python main.py

# Run with custom config or mode override
python main.py --config config/my_config.yaml
python main.py --mode live

# Production runner with auto-restart
./scripts/start.sh

# Train LSTM model with synthetic data (no API needed)
python -m models.train --synthetic --symbols BTC/USDT --device auto

# Train with real data from Binance (needs network access)
python -m models.train --symbols BTC/USDT --days 90 --device auto

# Train with GPU optimizations
python -m models.train --synthetic --device cuda --amp --compile

# Train with wandb experiment tracking
python -m models.train --synthetic --device cuda --amp --compile --wandb --wandb-tags test

# Generate synthetic CSV data for offline use
./scripts/generate_data.sh

# Export .pt checkpoint to .onnx
./scripts/export_model.sh artifacts/model.pt

# Run all tests
pytest

# Run a single test file
pytest tests/test_strategy.py

# Run a single test class or method
pytest tests/test_tracker.py::TestBuyFill
pytest tests/test_features.py::TestRSI::test_all_gains

# Run with coverage
pytest --cov=. --cov-report=term-missing
```

## Architecture

Async event-driven trading bot. All modules connect through a producer-consumer pattern orchestrated by `main.py` via dependency injection.

**Data flow per candle:**
```
Feed → LiveBuffer → [event.set()] → StrategyMonitor → FeatureExtractor → AlphaEngine → StrategyLogic → RiskShield → OrderManager → Executor
```

**Key async mechanism:** `LiveBuffer` uses `asyncio.Event` — the feed calls `push_candle()` which sets the event, and `StrategyMonitor` blocks on `wait_for_update()`. This is event-driven, not polling.

### Module dependency graph

- **`core/models.py`** — Pydantic schemas imported by every other module. All data flows through these types: `OHLCV`, `Signal`, `Order`, `Position`, `PortfolioSnapshot`. Enums: `StrategyState` (FLAT/LONG_PENDING/HOLDING).
- **`data/`** — `LiveBuffer` is the central data store (asyncio.Lock-guarded deques). `WSConnector` (live) and `SimulatedFeed` (paper) are interchangeable producers.
- **`features/extractor.py`** — Stateless. Pure numpy, no pandas. Has two paths: `extract()` for single snapshot (rule-based), `extract_sequence()` for LSTM input `(seq_len, 6)` array with z-score normalization.
- **`models/inference.py`** — `AlphaEngine` routes to rule-based scorer, LSTM via `ModelWrapper`, or ensemble. Mode selected by `alpha.engine` config value. Logs inference timing at DEBUG level.
- **`strategy/logic.py`** — One `StrategyLogic` instance per symbol. State machine transitions driven by alpha signals and order fills. The monitor creates these and wires fill callbacks.
- **`strategy/monitor.py`** — The orchestrator. Runs the full pipeline each time a closed candle arrives. Also runs `RiskShield.check_stops()` and circuit breaker checks every iteration.
- **`risk/`** — `PortfolioTracker` is the single source of truth for cash, positions, NAV. `RiskShield` does pre-trade validation and post-trade stop monitoring.
- **`execution/`** — `SimExecutor` and `LiveExecutor` implement the same `BaseExecutor` ABC. `OrderManager` handles lifecycle and dispatches fill callbacks to both `PortfolioTracker` and `StrategyLogic`.

### Paper vs Live

The system is mode-agnostic after initialization. `main.py` selects the data feed and executor based on `config["mode"]`:
- Paper: `SimulatedFeed` + `SimExecutor` (GBM candles, instant fills with slippage)
- Live: `WSConnector` + `LiveExecutor` (Binance WebSocket, ccxt orders)

### Alpha engine modes

Set `alpha.engine` in `config/default.yaml`:
- `"rule_based"` — Composite of RSI + momentum + EMA crossover - volatility penalty
- `"lstm"` — ONNX inference on 30-candle feature sequence
- `"ensemble"` — 50/50 average of both

### Risk layers (checked every iteration)

1. **Pre-trade** (`RiskShield.validate`): circuit breaker, long-only enforcement, rate limit (10/min), exposure caps (50% portfolio, 15% per symbol), cash check
2. **Trailing stop**: sells if price drops 3% from peak
3. **ATR stop**: sells if price drops 2x ATR below entry
4. **Circuit breaker**: 5% daily drawdown halts all trading and liquidates

### Training pipeline

`models/train.py` supports two data sources:
- `--synthetic` — GBM-generated candles (no network needed, works anywhere)
- Default — fetches from Binance via ccxt (geo-blocked in some locations)

GPU flags: `--amp` (mixed precision via `torch.amp`), `--compile` (`torch.compile`). Both degrade gracefully on CPU.

Dataset building uses vectorized feature extraction (computes all features once, then slices windows). 10k candles builds in ~0.5s vs ~13 min with the naive approach.

Model artifacts save to `artifacts/model.pt` and `artifacts/model.onnx`.

### Experiment tracking

`--wandb` enables [Weights & Biases](https://wandb.ai) logging. Defaults: entity=`Base-Work-Space`, project=`trading-lstm`.

- **Group**: auto-generated as `{device}_e{epochs}_{candles}{k|d}_{flags}_{MMDD}` (one group per day per experiment)
- **Tags**: defaults to `test`; use `--wandb-tags deploy` for production runs
- **Metrics logged**: `train/loss`, `val/loss`, `train/lr`, `train/epoch_time_s` per epoch, plus summary stats

Override with `--wandb-entity`, `--wandb-project`, `--wandb-group`, `--wandb-name`, `--wandb-tags`.

### Logging

`main.py` and `models/train.py` both log to console + rotating files in `logs/`:
- Trading: `logs/trading_{mode}_{timestamp}.log`
- Training: `logs/train_{timestamp}.log`

### Directory layout

```
config/default.yaml    — all tunable parameters (thresholds, risk limits, fees)
artifacts/             — model checkpoints (.pt, .onnx) — gitignored
logs/                  — rotating log files — gitignored
scripts/               — shell scripts (train, paper_trade, test, export_model, generate_data, start)
data/historical/       — cached CSV data — gitignored
```

## Configuration

All tunable parameters are in `config/default.yaml`. When modifying thresholds or risk limits, change this file — no code changes needed. The config dict is passed to every component constructor.

Key: `alpha.model_path` points to `artifacts/model.onnx`. Switch `alpha.engine` between `rule_based`, `lstm`, or `ensemble`.
