"""Trading Competition Framework — Entry Point.

Usage:
    uv run python main.py                        # paper mode (default config)
    uv run python main.py --config config/live.yaml  # custom config
    uv run python main.py --mode live             # override mode
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import yaml

from data.buffer import LiveBuffer
from data.connector import WSConnector
from data.sim_feed import SimulatedFeed
from execution.executor import LiveExecutor
from execution.order_manager import OrderManager
from execution.sim_executor import SimExecutor
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


async def main(config: dict) -> None:
    mode = config.get("mode", "paper")
    logger.info("=== Trading Competition Framework ===")
    logger.info("Mode: %s", mode)
    logger.info("Symbols: %s", config.get("symbols", []))

    # ── Build Components ──

    # 1. Data buffer
    data_cfg = config.get("data", {})
    buffer = LiveBuffer(
        max_candles=data_cfg.get("buffer_size", 500),
        max_ticks=data_cfg.get("tick_buffer_size", 5000),
    )

    # 2. Data source (producer)
    if mode == "paper":
        feed = SimulatedFeed(config, buffer)
    else:
        feed = WSConnector(config, buffer)

    # 3. Executor
    paper_cfg = config.get("paper", {})
    if mode == "paper":
        executor = SimExecutor(paper_cfg, buffer)
    else:
        live_exec = LiveExecutor(config.get("exchange", {}))
        await live_exec.start()
        executor = live_exec

    # 4. Feature extractor
    extractor = FeatureExtractor(config.get("features", {}))

    # 5. Alpha engine (with optional LSTM model)
    model: Optional[ModelWrapper] = None
    alpha_cfg = config.get("alpha", {})
    engine_type = alpha_cfg.get("engine", "rule_based")

    if engine_type in ("lstm", "ensemble"):
        model_path = alpha_cfg.get("model_path", "")
        if model_path and Path(model_path).exists():
            model = ModelWrapper(model_path, n_features=extractor.N_FEATURES)
            model.load()
            logger.info("Loaded ML model from %s", model_path)
        else:
            logger.warning("Model path '%s' not found, falling back to rule_based", model_path)
            config["alpha"]["engine"] = "rule_based"

    alpha_engine = AlphaEngine(config, extractor, model)

    # 6. Portfolio tracker
    initial_capital = paper_cfg.get("initial_capital", 100000.0)
    fee_bps = paper_cfg.get("fee_bps", 10.0)
    tracker = PortfolioTracker(initial_capital, fee_bps)

    # 7. Risk shield
    risk_shield = RiskShield(config)

    # 8. Order manager
    order_manager = OrderManager(executor, tracker)

    # 9. Strategy monitor (orchestrator)
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

    async def run_with_shutdown():
        feed_task = asyncio.create_task(feed.start())
        monitor_task = asyncio.create_task(monitor.run())

        # Wait for shutdown signal or task completion
        done, pending = await asyncio.wait(
            [feed_task, monitor_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cleanup
        logger.info("Shutting down...")
        await monitor.stop()
        await feed.stop()

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Final report
        snap = tracker.snapshot()
        total_return = ((snap.nav - initial_capital) / initial_capital) * 100
        logger.info("=== Final Report ===")
        logger.info("NAV: $%.2f (started: $%.2f)", snap.nav, initial_capital)
        logger.info("Return: %.2f%%", total_return)
        logger.info("Max Drawdown: %.2f%%", snap.drawdown * 100)
        logger.info("Cash: $%.2f", snap.cash)
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
    parser.add_argument("--mode", default=None, help="Override mode: paper or live")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode

    setup_logging(config.get("mode", "paper"))
    asyncio.run(main(config))
