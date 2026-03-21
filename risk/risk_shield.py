from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional

from core.models import OHLCV, Order, OrderType, Side, StrategyState
from risk.tracker import PortfolioTracker

logger = logging.getLogger(__name__)


class RiskShield:
    """Pre-trade validation and real-time stop management.

    Responsibilities:
    - Pre-trade: validate orders against exposure limits, rate limits, circuit breaker
    - Post-trade: check trailing stops, ATR stops, circuit breaker on every candle
    """

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        self._max_portfolio_exposure: float = risk_cfg.get(
            "max_portfolio_exposure", 0.50
        )
        self._max_single_exposure: float = risk_cfg.get("max_single_exposure", 0.15)
        self._trailing_stop_pct: float = risk_cfg.get("trailing_stop_pct", 0.03)
        self._atr_stop_multiplier: float = risk_cfg.get("atr_stop_multiplier", 2.0)
        self._daily_drawdown_limit: float = risk_cfg.get("daily_drawdown_limit", 0.05)
        self._max_orders_per_minute: int = risk_cfg.get("max_orders_per_minute", 10)
        self._break_even_trigger_pct: float = max(
            0.0, risk_cfg.get("break_even_trigger_pct", 0.0)
        )
        self._break_even_buffer_pct: float = max(
            0.0, risk_cfg.get("break_even_buffer_pct", 0.0)
        )

        self._order_timestamps: deque = deque(maxlen=100)
        self._circuit_breaker_active: bool = False

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    def validate(
        self,
        order: Order,
        tracker: PortfolioTracker,
        market_price: float = 0.0,
        is_stop: bool = False,
    ) -> Optional[Order]:
        """Pre-trade validation. Returns order (possibly adjusted) or None if rejected.

        Args:
            market_price: Latest observable market price for this symbol. Required for
                realistic market-order exposure checks.
            is_stop: If True, skip rate limit — stop-loss orders must never be dropped.
        """
        snapshot = tracker.snapshot()

        # 1. Circuit breaker
        if self._circuit_breaker_active:
            if order.side == Side.BUY:
                logger.warning("REJECTED: circuit breaker active, no new buys")
                return None
            # Allow sells during circuit breaker (liquidation)

        # 2. Long-only enforcement
        if order.side == Side.SELL:
            pos = tracker.get_position(order.symbol)
            if pos.quantity < order.quantity:
                logger.warning(
                    "ADJUSTED: sell qty %.6f > position %.6f, clamping",
                    order.quantity,
                    pos.quantity,
                )
                order.quantity = pos.quantity
            if order.quantity <= 0:
                logger.warning("REJECTED: no position to sell for %s", order.symbol)
                return None

        # 3. Rate limit (skip for stop-loss orders — stops must never be dropped)
        if not is_stop:
            now = time.time()
            # Remove timestamps older than 60s
            while self._order_timestamps and self._order_timestamps[0] < now - 60:
                self._order_timestamps.popleft()
            if len(self._order_timestamps) >= self._max_orders_per_minute:
                logger.warning(
                    "REJECTED: rate limit exceeded (%d/min)",
                    self._max_orders_per_minute,
                )
                return None
            self._order_timestamps.append(now)

        # 4. Exposure checks (only for buys)
        if order.side == Side.BUY:
            price = order.price or market_price or 0.0
            if price <= 0:
                # Estimate with current position price or use a conservative estimate
                pos = tracker.get_position(order.symbol)
                price = pos.current_price if pos.current_price > 0 else 1.0

            order_value = price * order.quantity

            # Single exposure check
            current_exposure = tracker.get_exposure(order.symbol)
            new_single_exposure = current_exposure + (
                order_value / snapshot.nav if snapshot.nav > 0 else 1.0
            )
            if new_single_exposure > self._max_single_exposure:
                max_value = (
                    self._max_single_exposure - current_exposure
                ) * snapshot.nav
                if max_value <= 0:
                    logger.warning(
                        "REJECTED: single exposure limit (%.1f%%)",
                        current_exposure * 100,
                    )
                    return None
                order.quantity = max_value / price
                logger.info(
                    "ADJUSTED: clamped to single exposure limit, qty=%.6f",
                    order.quantity,
                )

            # Portfolio exposure check
            total_exposure = tracker.get_total_exposure()
            new_total = total_exposure + (
                order_value / snapshot.nav if snapshot.nav > 0 else 1.0
            )
            if new_total > self._max_portfolio_exposure:
                max_value = (
                    self._max_portfolio_exposure - total_exposure
                ) * snapshot.nav
                if max_value <= 0:
                    logger.warning(
                        "REJECTED: portfolio exposure limit (%.1f%%)",
                        total_exposure * 100,
                    )
                    return None
                order.quantity = min(order.quantity, max_value / price)
                logger.info(
                    "ADJUSTED: clamped to portfolio exposure limit, qty=%.6f",
                    order.quantity,
                )

            # Cash check
            total_cost = price * order.quantity * 1.001  # include rough fee estimate
            if total_cost > snapshot.cash:
                order.quantity = (
                    snapshot.cash * 0.999
                ) / price  # leave margin for fees
                if order.quantity <= 0:
                    logger.warning("REJECTED: insufficient cash (%.2f)", snapshot.cash)
                    return None
                logger.info(
                    "ADJUSTED: clamped to available cash, qty=%.6f", order.quantity
                )

        return order

    def check_stops(
        self,
        tracker: PortfolioTracker,
        latest_candles: Dict[str, OHLCV],
        atr_values: Dict[str, float],
    ) -> List[Order]:
        """Check all positions for stop conditions. Returns SELL orders to execute."""
        orders = []
        snapshot = tracker.snapshot()

        for pos in snapshot.positions:
            if pos.state != StrategyState.HOLDING or pos.quantity <= 0:
                continue

            symbol = pos.symbol
            candle = latest_candles.get(symbol)
            if candle is None:
                continue

            current_price = candle.close
            tracker.update_prices(symbol, current_price)

            triggered = False
            reason = ""

            # Trailing stop
            if pos.peak_price > 0:
                stop_price = pos.peak_price * (1 - self._trailing_stop_pct)
                reason = (
                    f"trailing stop (peak={pos.peak_price:.2f}, stop={stop_price:.2f})"
                )

                # If a trade already moved far enough in our favor, tighten the
                # floor so a winner is less likely to round-trip into a loser.
                if (
                    self._break_even_trigger_pct > 0
                    and pos.entry_price > 0
                    and pos.peak_price
                    >= pos.entry_price * (1 + self._break_even_trigger_pct)
                ):
                    break_even_price = pos.entry_price * (
                        1 + self._break_even_buffer_pct
                    )
                    if break_even_price > stop_price:
                        stop_price = break_even_price
                        reason = (
                            "break-even lock "
                            f"(entry={pos.entry_price:.2f}, peak={pos.peak_price:.2f}, "
                            f"floor={stop_price:.2f})"
                        )

                if current_price <= stop_price:
                    triggered = True

            # ATR stop
            if not triggered and symbol in atr_values:
                atr = atr_values[symbol]
                atr_stop_price = pos.entry_price - (atr * self._atr_stop_multiplier)
                if current_price <= atr_stop_price:
                    triggered = True
                    reason = f"ATR stop (entry={pos.entry_price:.2f}, atr={atr:.2f}, stop={atr_stop_price:.2f})"

            if triggered:
                logger.warning(
                    "STOP triggered for %s: %s @ %.2f", symbol, reason, current_price
                )
                orders.append(
                    Order(
                        symbol=symbol,
                        side=Side.SELL,
                        order_type=OrderType.MARKET,
                        quantity=pos.quantity,
                    )
                )

        return orders

    def check_circuit_breaker(self, tracker: PortfolioTracker) -> bool:
        """Check daily drawdown circuit breaker.

        Returns True if breaker just activated (caller should liquidate all positions).
        """
        if self._circuit_breaker_active:
            return False  # already active

        snapshot = tracker.snapshot()
        if snapshot.daily_drawdown >= self._daily_drawdown_limit:
            self._circuit_breaker_active = True
            logger.critical(
                "CIRCUIT BREAKER ACTIVATED: daily_drawdown=%.2f%% >= limit=%.2f%%",
                snapshot.daily_drawdown * 100,
                self._daily_drawdown_limit * 100,
            )
            return True

        return False

    def reset_daily(self) -> None:
        """Reset circuit breaker at start of new trading day."""
        self._circuit_breaker_active = False
        self._order_timestamps.clear()
        logger.info("Risk shield daily reset")
