from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from core.models import OHLCV, Order, OrderStatus, StrategyState
from data.buffer import LiveBuffer
from execution.order_manager import OrderManager
from features.extractor import FeatureExtractor
from models.inference import AlphaEngine
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from strategy.logic import StrategyLogic

logger = logging.getLogger(__name__)


class StrategyMonitor:
    """Central event loop / orchestrator.

    Consumes candles from the buffer, runs the feature → alpha → strategy → risk → execution pipeline.
    Event-driven: blocks on buffer.wait_for_update(), triggered by new closed candles.
    """

    def __init__(
        self,
        config: dict,
        buffer: LiveBuffer,
        extractor: FeatureExtractor,
        alpha_engine: AlphaEngine,
        risk_shield: RiskShield,
        tracker: PortfolioTracker,
        order_manager: OrderManager,
    ):
        self._config = config
        self._buffer = buffer
        self._extractor = extractor
        self._alpha_engine = alpha_engine
        self._risk_shield = risk_shield
        self._tracker = tracker
        self._order_manager = order_manager
        self._running = False

        # Per-symbol strategy state machines
        symbols = config.get("symbols", [])
        self._strategies: Dict[str, StrategyLogic] = {
            sym: StrategyLogic(sym, config) for sym in symbols
        }

        # Register fill callback for strategy logic
        self._order_manager.register_fill_callback(self._on_order_event)

        # Min candles before we start trading
        self._warmup_candles = extractor.min_candles + config.get("alpha", {}).get("seq_len", 30)

    async def run(self) -> None:
        """Main event loop."""
        self._running = True
        logger.info("StrategyMonitor started, warmup=%d candles", self._warmup_candles)

        iteration = 0
        while self._running:
            got_update = await self._buffer.wait_for_update(timeout=5.0)
            if not got_update:
                continue

            iteration += 1
            await self._process_iteration(iteration)

    async def stop(self) -> None:
        self._running = False
        logger.info("StrategyMonitor stopped")

    async def _process_iteration(self, iteration: int) -> None:
        """Process one iteration: for each symbol, run the full pipeline."""
        snapshot = self._tracker.snapshot()
        latest_candles: Dict[str, OHLCV] = {}
        atr_values: Dict[str, float] = {}

        for symbol, strategy in self._strategies.items():
            candles = await self._buffer.get_candles(symbol)
            if not candles:
                continue

            latest_candles[symbol] = candles[-1]

            # Update position prices
            self._tracker.update_prices(symbol, candles[-1].close)

            # Check warmup
            if len(candles) < self._warmup_candles:
                if iteration % 50 == 1:
                    logger.info(
                        "[%s] warming up: %d/%d candles",
                        symbol, len(candles), self._warmup_candles,
                    )
                continue

            # ── Feature Extraction ──
            features = self._extractor.extract(candles)
            atr_values[symbol] = features.atr

            # ── Alpha Scoring ──
            signal = self._alpha_engine.score(candles)

            # ── Strategy Decision ──
            snapshot = self._tracker.snapshot()
            order = strategy.on_signal(signal, snapshot, current_price=candles[-1].close)

            if order is not None:
                # ── Risk Validation ──
                validated = self._risk_shield.validate(order, self._tracker)
                if validated is not None:
                    await self._order_manager.submit(validated)

        # ── Post-trade Risk Checks ──
        # Trailing stops and ATR stops
        stop_orders = self._risk_shield.check_stops(self._tracker, latest_candles, atr_values)
        for stop_order in stop_orders:
            validated = self._risk_shield.validate(stop_order, self._tracker)
            if validated is not None:
                await self._order_manager.submit(validated)
                # Update strategy state
                if stop_order.symbol in self._strategies:
                    self._strategies[stop_order.symbol].force_flat()

        # Circuit breaker check
        if self._risk_shield.check_circuit_breaker(self._tracker):
            await self._liquidate_all()

        # Periodic logging
        if iteration % 10 == 0:
            snap = self._tracker.snapshot()
            holdings = [
                f"{p.symbol}:{p.quantity:.4f}@{p.current_price:.2f}"
                for p in snap.positions
                if p.state == StrategyState.HOLDING
            ]
            logger.info(
                "Iter %d | NAV=%.2f | Cash=%.2f | PnL=%.2f | DD=%.2f%% | Holdings=%s",
                iteration, snap.nav, snap.cash, snap.daily_pnl,
                snap.drawdown * 100, holdings or "none",
            )

    async def _liquidate_all(self) -> None:
        """Emergency liquidation: sell all positions."""
        logger.critical("LIQUIDATING ALL POSITIONS")
        snapshot = self._tracker.snapshot()

        for pos in snapshot.positions:
            if pos.quantity > 0:
                from core.models import OrderType, Side

                order = Order(
                    symbol=pos.symbol,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                )
                await self._order_manager.submit(order)
                if pos.symbol in self._strategies:
                    self._strategies[pos.symbol].force_flat()

    def _on_order_event(self, order: Order) -> None:
        """Callback for order fill/cancel events."""
        if order.symbol in self._strategies:
            if order.status == OrderStatus.FILLED:
                self._strategies[order.symbol].on_fill(order)
            elif order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                self._strategies[order.symbol].on_cancel(order)
