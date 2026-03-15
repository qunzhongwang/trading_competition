Web3 Long-Only Quant Trading Framework
A modular, high-performance algorithmic trading framework designed for Web3 markets. This system is optimized for long-only strategies utilizing deep learning inference, real-time WebSocket data ingestion, and event-driven risk management.

🏗 System Architecture
The framework is decoupled into five core modules to ensure low latency and high maintainability during high-stakes competition windows.

1. Data Engine (/data)
connector.py: Handles asynchronous WebSocket connections to Web3 exchanges. Includes auto-reconnect logic and heartbeats. Also contains BinanceSupplementaryFeed for free public Binance data (order book depth, funding rate, taker ratio).

buffer.py: A thread-safe, in-memory sliding window (Buffer) that stores the most recent N ticks/K-lines and supplementary market data for real-time feature calculation.

2. Alpha & Feature Engine (/features & /models)
extractor.py: Transforms raw OHLCV data + supplementary data into 10 features (RSI, EMA fast/slow, ATR, momentum, volatility, order book imbalance, volume ratio, funding rate, taker buy/sell ratio).

model_wrapper.py: Loads pre-trained Deep Learning models (ONNX, PyTorch, or Scikit-learn).

inference.py: Runs real-time forward passes to generate Alpha scores (expected return or buy/sell probabilities). Routes supplementary data through to feature extraction.

3. Strategy & Monitor (/strategy)
monitor.py: The central event loop. It listens to data updates and triggers the Alpha engine. Passes supplementary data from buffer to feature extraction and alpha scoring.

logic.py: Implements the "Decision State Machine" with Half-Kelly position sizing and alpha-based order type selection.

States: FLAT, LONG_PENDING, HOLDING.

Trigger: Only executes entry when Alpha signal strength exceeds a defined threshold.

Position Sizing: Half-Kelly formula scales allocation by alpha conviction (5-7.5% of NAV).

Order Type: High conviction (alpha > 0.85) uses market orders; moderate conviction uses limit orders to save on fees.

4. Risk Manager (/risk)
risk_shield.py: Validates all orders against pre-defined constraints.

Dynamic Stop-Loss: Implements Trailing Stops and ATR-based exits.

Market Circuit Breaker: Halts trading if daily drawdown exceeds a specific limit.

tracker.py: Maintains real-time PnL and position exposure.

5. Execution Engine (/execution)
order_manager.py: Manages the lifecycle of an order (Create, Cancel, Status Poll).

executor.py: Determines order type usage.

Market Orders: Used for emergency exits and high-conviction entries.

Limit Orders: Used for Maker-rebate optimization in stable regimes.

🚀 Getting Started
Prerequisites
Python 3.10+

asyncio for non-blocking I/O

pandas & numpy for data manipulation

ccxt or exchange-specific SDKs

Core Loop Logic
The system operates on an asynchronous producer-consumer pattern:

Producer: WebSocket task pushes new ticks to the LiveBuffer.

Consumer: The StrategyMonitor polls the buffer, runs Inference, and checks RiskShield.

Executor: Dispatches orders to the exchange API.

📈 Competition Focus
Long-Only Constraint: Optimized for "Buy and Hold-until-decay" logic.

10-Day Window: Focuses on high-frequency Alpha signals and rapid risk-off triggers during regime shifts.

Scalability: Built to handle multiple symbols concurrently using Python's asyncio event loop.