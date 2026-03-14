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

# Train LSTM model (fetches historical data from Binance, trains, exports ONNX)
python -m models.train --symbols BTC/USDT --days 90 --device auto

# No test suite or linter is configured yet
```

## Architecture

This is an async event-driven trading bot. All modules connect through a producer-consumer pattern orchestrated by `main.py` via dependency injection.

**Data flow per candle:**
```
Feed → LiveBuffer → [event.set()] → StrategyMonitor → FeatureExtractor → AlphaEngine → StrategyLogic → RiskShield → OrderManager → Executor
```

**Key async mechanism:** `LiveBuffer` uses `asyncio.Event` — the feed calls `push_candle()` which sets the event, and `StrategyMonitor` blocks on `wait_for_update()`. This is event-driven, not polling.

### Module dependency graph

- **`core/models.py`** — Pydantic schemas imported by every other module. All data flows through these types: `OHLCV`, `Signal`, `Order`, `Position`, `PortfolioSnapshot`. Enums: `StrategyState` (FLAT/LONG_PENDING/HOLDING).
- **`data/`** — `LiveBuffer` is the central data store (asyncio.Lock-guarded deques). `WSConnector` (live) and `SimulatedFeed` (paper) are interchangeable producers.
- **`features/extractor.py`** — Stateless. Pure numpy, no pandas. Has two paths: `extract()` for single snapshot (rule-based), `extract_sequence()` for LSTM input `(seq_len, 6)` array with z-score normalization.
- **`models/inference.py`** — `AlphaEngine` routes to rule-based scorer, LSTM via `ModelWrapper`, or ensemble. Mode selected by `alpha.engine` config value.
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

## Configuration

All tunable parameters are in `config/default.yaml`. When modifying thresholds or risk limits, change this file — no code changes needed. The config dict is passed to every component constructor.
