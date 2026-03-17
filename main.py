"""Trading Competition Framework — Entry Point.

Usage:
    uv run python main.py                        # paper mode (default config)
    uv run python main.py --config config/live.yaml  # custom config
    uv run python main.py --mode live             # override mode
    uv run python main.py --mode roostoo          # Roostoo competition mode
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import yaml

from data.buffer import LiveBuffer
from data.connector import WSConnector, BinanceSupplementaryFeed
from data.sim_feed import SimulatedFeed
from execution.executor import LiveExecutor
from execution.order_manager import OrderManager
from execution.roostoo_executor import RoostooExecutor
from execution.sim_executor import SimExecutor
from execution.trade_logger import TradeLogger
from features.extractor import FeatureExtractor
from models.inference import AlphaEngine
from models.model_wrapper import ModelWrapper
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from strategy.monitor import StrategyMonitor

logger = logging.getLogger(__name__)


def setup_logging(mode: str = "paper") -> None:
    """Configure console + rotating file logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"trading_{mode}_{ts}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)

    logger.info("Logging to %s", log_file)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_env_overrides(config: dict) -> None:
    """Override config values from environment variables (never commit secrets).

    Competition keys (ROOSTOO_COMP_*) take priority over testing keys (ROOSTOO_*).
    Key and secret are treated as a pair — both must be set for either tier.
    """
    roostoo_cfg = config.setdefault("roostoo", {})

    comp_key = os.environ.get("ROOSTOO_COMP_API_KEY", "")
    comp_secret = os.environ.get("ROOSTOO_COMP_API_SECRET", "")
    test_key = os.environ.get("ROOSTOO_API_KEY", "")
    test_secret = os.environ.get("ROOSTOO_API_SECRET", "")

    if comp_key and comp_secret:
        roostoo_cfg["api_key"] = comp_key
        roostoo_cfg["api_secret"] = comp_secret
        logger.info("Using competition Roostoo API credentials")
    elif comp_key or comp_secret:
        logger.warning("ROOSTOO_COMP_API_KEY and ROOSTOO_COMP_API_SECRET must both be set; ignoring partial comp credentials")
        if test_key and test_secret:
            roostoo_cfg["api_key"] = test_key
            roostoo_cfg["api_secret"] = test_secret
            logger.info("Falling back to testing Roostoo API credentials")
    elif test_key and test_secret:
        roostoo_cfg["api_key"] = test_key
        roostoo_cfg["api_secret"] = test_secret
        logger.info("Using testing Roostoo API credentials")

    exchange_cfg = config.setdefault("exchange", {})
    if os.environ.get("BINANCE_API_KEY"):
        exchange_cfg["api_key"] = os.environ["BINANCE_API_KEY"]
    if os.environ.get("BINANCE_API_SECRET"):
        exchange_cfg["api_secret"] = os.environ["BINANCE_API_SECRET"]


async def main(config: dict) -> None:
    mode = config.get("mode", "paper")
    logger.info("=== Trading Competition Framework ===")
    logger.info("Mode: %s", mode)
    logger.info("Symbols: %d pairs", len(config.get("symbols", [])))

    # ── Build Components ──

    # 1. Data buffer
    data_cfg = config.get("data", {})
    buffer = LiveBuffer(
        max_candles=data_cfg.get("buffer_size", 500),
        max_ticks=data_cfg.get("tick_buffer_size", 5000),
    )

    # 2. Data source (producer) — always Binance WS for candle data
    if mode == "paper":
        feed = SimulatedFeed(config, buffer)
    else:
        # Both "live" and "roostoo" modes use Binance WS for market data
        feed = WSConnector(config, buffer)

    # 3. Trade logger
    trade_logger = TradeLogger()

    # 4. Executor
    paper_cfg = config.get("paper", {})
    if mode == "paper":
        executor = SimExecutor(paper_cfg, buffer)
    elif mode == "roostoo":
        roostoo_cfg = config.get("roostoo", {})
        roostoo_exec = RoostooExecutor(roostoo_cfg)
        roostoo_exec.set_trade_logger(trade_logger)
        await roostoo_exec.start()

        # Sync initial balance from Roostoo
        balances = await roostoo_exec.get_balance()
        usd_balance = balances.get("USD", 0)
        if usd_balance > 0:
            logger.info("Roostoo USD balance: $%.2f", usd_balance)
        else:
            logger.warning("Could not fetch Roostoo balance, using config initial_capital")

        executor = roostoo_exec
    else:
        live_exec = LiveExecutor(config.get("exchange", {}))
        await live_exec.start()
        executor = live_exec

    # 5. Feature extractor
    extractor = FeatureExtractor(config.get("features", {}))

    # 6. Alpha engine (with optional LSTM model)
    model: Optional[ModelWrapper] = None
    alpha_cfg = config.get("alpha", {})
    engine_type = alpha_cfg.get("engine", "rule_based")

    if engine_type in ("lstm", "transformer", "ensemble"):
        model_path = alpha_cfg.get("model_path", "")
        model_type = alpha_cfg.get("model_type", "lstm")
        if model_path and Path(model_path).exists():
            model = ModelWrapper(model_path, n_features=extractor.N_FEATURES,
                                 model_type=model_type)
            model.load()
            logger.info("Loaded %s model from %s", model_type, model_path)
        else:
            logger.warning("Model path '%s' not found, falling back to rule_based", model_path)
            config["alpha"]["engine"] = "rule_based"

    alpha_engine = AlphaEngine(config, extractor, model)

    # 7. Portfolio tracker
    initial_capital = paper_cfg.get("initial_capital", 1000000.0)
    fee_bps = paper_cfg.get("fee_bps", 10.0)
    tracker = PortfolioTracker(initial_capital, fee_bps)

    # 8. Risk shield
    risk_shield = RiskShield(config)

    # 9. Order manager
    order_manager = OrderManager(executor, tracker)

    # 10. Strategy monitor (orchestrator)
    monitor = StrategyMonitor(
        config=config,
        buffer=buffer,
        extractor=extractor,
        alpha_engine=alpha_engine,
        risk_shield=risk_shield,
        tracker=tracker,
        order_manager=order_manager,
    )

    # ── Graceful Shutdown ──
    shutdown_event = asyncio.Event()

    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Run ──
    logger.info("Starting trading loop...")

    # 11. Supplementary Binance data feed (order book, funding, taker ratio)
    supp_feed = BinanceSupplementaryFeed(config.get("symbols", []), buffer)

    # NAV snapshot interval for risk metrics (every 60 iterations ~= 1 hour at 1m candles)
    NAV_SNAPSHOT_INTERVAL = 60
    METRICS_LOG_INTERVAL = 120

    async def run_with_shutdown():
        feed_task = asyncio.create_task(feed.start())
        monitor_task = asyncio.create_task(monitor.run())
        supp_task = asyncio.create_task(supp_feed.start())

        # Periodic NAV snapshot task
        async def nav_snapshot_loop():
            iteration = 0
            while not shutdown_event.is_set():
                await asyncio.sleep(60)  # every minute
                iteration += 1
                tracker.record_nav_snapshot()

                if iteration % METRICS_LOG_INTERVAL == 0:
                    metrics = tracker.compute_risk_metrics()
                    logger.info(
                        "Risk Metrics | Sharpe=%.3f | Sortino=%.3f | Calmar=%.3f | "
                        "Composite=%.3f | Return=%.2f%% | MaxDD=%.2f%%",
                        metrics.sharpe_ratio, metrics.sortino_ratio,
                        metrics.calmar_ratio, metrics.composite_score,
                        metrics.total_return_pct, metrics.max_drawdown * 100,
                    )

        nav_task = asyncio.create_task(nav_snapshot_loop())

        # Wait for shutdown signal or task completion
        done, pending = await asyncio.wait(
            [feed_task, monitor_task, supp_task, nav_task,
             asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cleanup
        logger.info("Shutting down...")
        await order_manager.cancel_all()
        await monitor.stop()
        await feed.stop()
        if supp_feed is not None:
            await supp_feed.stop()

        if mode == "roostoo" and isinstance(executor, RoostooExecutor):
            await executor.stop()

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Final report with risk metrics
        snap = tracker.snapshot()
        total_return = ((snap.nav - initial_capital) / initial_capital) * 100
        metrics = tracker.compute_risk_metrics()

        logger.info("=== Final Report ===")
        logger.info("NAV: $%.2f (started: $%.2f)", snap.nav, initial_capital)
        logger.info("Return: %.2f%%", total_return)
        logger.info("Max Drawdown: %.2f%%", snap.drawdown * 100)
        logger.info("Sharpe Ratio: %.3f", metrics.sharpe_ratio)
        logger.info("Sortino Ratio: %.3f", metrics.sortino_ratio)
        logger.info("Calmar Ratio: %.3f", metrics.calmar_ratio)
        logger.info("Composite Score (0.4S+0.3Sh+0.3C): %.3f", metrics.composite_score)
        logger.info("Cash: $%.2f", snap.cash)
        n_positions = sum(1 for pos in snap.positions if pos.quantity > 0)
        logger.info("Active Positions: %d", n_positions)
        for pos in snap.positions:
            if pos.quantity > 0:
                logger.info(
                    "  Position: %s qty=%.6f entry=%.2f current=%.2f pnl=%.2f",
                    pos.symbol, pos.quantity, pos.entry_price, pos.current_price,
                    pos.unrealized_pnl,
                )

    await run_with_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Competition Framework")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--mode", default=None, help="Override mode: paper, live, or roostoo")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode

    _apply_env_overrides(config)
    setup_logging(config.get("mode", "paper"))
    asyncio.run(main(config))
