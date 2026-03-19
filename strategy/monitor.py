from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Set

from core.models import OHLCV, Order, OrderStatus, StrategyState
from data.buffer import LiveBuffer
from data.resampler import CandleResampler, MultiResampler
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
        resampler: Optional[CandleResampler] = None,
        multi_resampler: Optional[MultiResampler] = None,
        trade_tracker=None,
        icir_tracker=None,
    ):
        self._config = config
        self._buffer = buffer
        self._extractor = extractor
        self._alpha_engine = alpha_engine
        self._risk_shield = risk_shield
        self._tracker = tracker
        self._order_manager = order_manager
        self._running = False
        self._resampler = resampler
        self._multi_resampler = multi_resampler
        self._trade_tracker = trade_tracker
        self._icir_tracker = icir_tracker

        # Multi-TF timeframes to fetch for alpha filter
        alpha_cfg = config.get("alpha", {})
        self._multi_timeframes = alpha_cfg.get("multi_timeframes", [])

        # Per-symbol strategy state machines
        symbols = config.get("symbols", [])
        self._strategies: Dict[str, StrategyLogic] = {
            sym: StrategyLogic(sym, config) for sym in symbols
        }

        # Inject trade tracker into each strategy for adaptive Kelly
        if trade_tracker is not None:
            for strategy in self._strategies.values():
                strategy.set_trade_tracker(trade_tracker)

        # Register fill callback for strategy logic
        self._order_manager.register_fill_callback(self._on_order_event)

        # Min candles before we start trading
        self._warmup_candles = extractor.min_candles + config.get("alpha", {}).get("seq_len", 30)

        # ICIR tracking: store previous factors per symbol for online learning
        self._prev_factors: Dict[str, list] = {}
        self._prev_prices: Dict[str, float] = {}

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
        """Process one iteration: for each symbol, run the full pipeline.

        Price updates and stop checks run every 1-min candle.
        Alpha scoring and trade decisions only run when the resampler emits
        a completed N-min bar (or every candle if no resampler).
        """
        # Check pending limit orders for fills
        await self._order_manager.check_pending()

        snapshot = self._tracker.snapshot()
        latest_candles: Dict[str, OHLCV] = {}
        atr_values: Dict[str, float] = {}

        # Track which symbols have a completed resampled bar this iteration
        alpha_ready: Set[str] = set()

        for symbol, strategy in self._strategies.items():
            candles = await self._buffer.get_candles(symbol)
            if not candles:
                continue

            latest_candles[symbol] = candles[-1]

            # Update position prices (every 1-min candle)
            self._tracker.update_prices(symbol, candles[-1].close)

            # Gate alpha on resampled bar completion
            if self._multi_resampler is not None:
                resampled_bars = self._multi_resampler.push(candles[-1])
                # Store higher-TF completed bars in the buffer
                for period, bar in resampled_bars.items():
                    if bar is not None:
                        await self._buffer.push_resampled(period, bar)
                # Alpha gating on primary (smallest) period
                primary = self._multi_resampler.primary_minutes
                if resampled_bars.get(primary) is not None:
                    alpha_ready.add(symbol)
            elif self._resampler is not None:
                resampled = self._resampler.push(candles[-1])
                if resampled is not None:
                    alpha_ready.add(symbol)
            else:
                alpha_ready.add(symbol)

            # Check warmup
            if len(candles) < self._warmup_candles:
                if iteration % 50 == 1:
                    logger.info(
                        "[%s] warming up: %d/%d candles",
                        symbol, len(candles), self._warmup_candles,
                    )
                continue

            # ── Feature Extraction (always, for ATR stops) ──
            supplementary = await self._buffer.get_supplementary(symbol)
            features = self._extractor.extract(candles, supplementary=supplementary)
            atr_values[symbol] = features.atr

            # ── Alpha Scoring (only on completed resampled bars) ──
            if symbol not in alpha_ready:
                continue

            # ICIR online learning: record previous factors vs realized return
            if self._icir_tracker is not None and symbol in self._prev_factors:
                prev_price = self._prev_prices.get(symbol, 0.0)
                if prev_price > 0:
                    realized_return = (candles[-1].close - prev_price) / prev_price
                    self._icir_tracker.record(symbol, self._prev_factors[symbol], realized_return)

            # Store current factors for ICIR next iteration
            if self._icir_tracker is not None:
                self._prev_factors[symbol] = [
                    (50.0 - features.rsi) / 50.0,
                    max(-1.0, min(1.0, features.momentum * 20.0)),
                    max(-1.0, min(1.0, (features.ema_fast - features.ema_slow) / features.ema_slow * 100.0)) if features.ema_slow > 0 else 0.0,
                    min(1.0, features.volatility * 50.0),
                ]
                self._prev_prices[symbol] = candles[-1].close

            supplementary_history = await self._buffer.get_supplementary_history(
                symbol, self._alpha_engine._seq_len)

            # Fetch higher-TF candles for multi-TF filter
            candles_15m = None
            candles_1h = None
            if self._multi_timeframes:
                if 15 in self._multi_timeframes:
                    candles_15m = await self._buffer.get_resampled_candles(symbol, 15, n=50)
                if 60 in self._multi_timeframes:
                    candles_1h = await self._buffer.get_resampled_candles(symbol, 60, n=50)

            signal = self._alpha_engine.score(
                candles, supplementary=supplementary,
                supplementary_history=supplementary_history,
                candles_15m=candles_15m, candles_1h=candles_1h,
            )

            # ── Strategy Decision ──
            snapshot = self._tracker.snapshot()
            order = strategy.on_signal(signal, snapshot, current_price=candles[-1].close)

            if order is not None:
                # ── Risk Validation ──
                validated = self._risk_shield.validate(order, self._tracker)
                if validated is not None:
                    result = await self._order_manager.submit(validated)
                else:
                    # Risk rejected the order — reset strategy state
                    strategy.on_cancel(order)

        # ── Post-trade Risk Checks ──
        # Trailing stops and ATR stops
        stop_orders = self._risk_shield.check_stops(self._tracker, latest_candles, atr_values)
        for stop_order in stop_orders:
            validated = self._risk_shield.validate(stop_order, self._tracker, is_stop=True)
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
                # Record trade for adaptive Kelly
                if self._trade_tracker is not None and order.side.value == "SELL" and order.filled_price:
                    # Find entry price from tracker positions
                    pos = self._tracker.get_position(order.symbol)
                    if pos and pos.entry_price > 0:
                        self._trade_tracker.record_trade(pos.entry_price, order.filled_price)
            elif order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                self._strategies[order.symbol].on_cancel(order)
