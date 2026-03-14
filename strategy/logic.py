from __future__ import annotations

import logging
from typing import Optional

from core.models import (
    Order,
    OrderType,
    PortfolioSnapshot,
    Side,
    Signal,
    StrategyState,
)

logger = logging.getLogger(__name__)


class StrategyLogic:
    """Per-symbol state machine for the long-only strategy.

    State transitions:
        FLAT + alpha > entry_threshold → emit BUY → LONG_PENDING
        LONG_PENDING + fill → HOLDING
        LONG_PENDING + cancel/reject → FLAT
        HOLDING + alpha < exit_threshold → emit SELL → FLAT
        HOLDING + risk stop → emit SELL → FLAT (handled externally)
    """

    def __init__(self, symbol: str, config: dict):
        self._symbol = symbol
        self._state: StrategyState = StrategyState.FLAT

        alpha_cfg = config.get("alpha", {})
        strategy_cfg = config.get("strategy", {})
        self._entry_threshold: float = alpha_cfg.get("entry_threshold", 0.6)
        self._exit_threshold: float = alpha_cfg.get("exit_threshold", -0.2)
        self._position_size_pct: float = strategy_cfg.get("position_size_pct", 0.10)

    @property
    def state(self) -> StrategyState:
        return self._state

    @property
    def symbol(self) -> str:
        return self._symbol

    def on_signal(self, signal: Signal, portfolio: PortfolioSnapshot, current_price: float = 0.0) -> Optional[Order]:
        """Process alpha signal and decide whether to trade.

        Args:
            signal: Alpha signal
            portfolio: Current portfolio state
            current_price: Latest market price for this symbol

        Returns an Order if action is needed, None otherwise.
        """
        if self._state == StrategyState.FLAT:
            if signal.alpha_score > self._entry_threshold:
                qty = self._compute_buy_quantity(portfolio, current_price)
                if qty <= 0:
                    return None

                self._state = StrategyState.LONG_PENDING
                logger.info(
                    "[%s] FLAT → LONG_PENDING: alpha=%.3f > %.3f, qty=%.6f",
                    self._symbol, signal.alpha_score, self._entry_threshold, qty,
                )
                return Order(
                    symbol=self._symbol,
                    side=Side.BUY,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                )

        elif self._state == StrategyState.HOLDING:
            if signal.alpha_score < self._exit_threshold:
                # Find current position quantity
                pos_qty = 0.0
                for pos in portfolio.positions:
                    if pos.symbol == self._symbol:
                        pos_qty = pos.quantity
                        break

                if pos_qty <= 0:
                    self._state = StrategyState.FLAT
                    return None

                logger.info(
                    "[%s] HOLDING → FLAT: alpha=%.3f < %.3f, selling %.6f",
                    self._symbol, signal.alpha_score, self._exit_threshold, pos_qty,
                )
                self._state = StrategyState.FLAT
                return Order(
                    symbol=self._symbol,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos_qty,
                )

        elif self._state == StrategyState.LONG_PENDING:
            # Waiting for fill — no action
            pass

        return None

    def on_fill(self, order: Order) -> None:
        """Called when an order is filled."""
        if order.symbol != self._symbol:
            return

        if order.side == Side.BUY and self._state == StrategyState.LONG_PENDING:
            self._state = StrategyState.HOLDING
            logger.info("[%s] LONG_PENDING → HOLDING (filled @ %.2f)", self._symbol, order.filled_price)

        elif order.side == Side.SELL:
            self._state = StrategyState.FLAT
            logger.info("[%s] → FLAT (sold @ %.2f)", self._symbol, order.filled_price)

    def on_cancel(self, order: Order) -> None:
        """Called when an order is cancelled or rejected."""
        if order.symbol != self._symbol:
            return

        if self._state == StrategyState.LONG_PENDING:
            self._state = StrategyState.FLAT
            logger.info("[%s] LONG_PENDING → FLAT (order cancelled)", self._symbol)

    def force_flat(self) -> None:
        """Force state to FLAT (used by circuit breaker)."""
        self._state = StrategyState.FLAT

    def _compute_buy_quantity(self, portfolio: PortfolioSnapshot, current_price: float) -> float:
        """Compute buy quantity based on position sizing."""
        if portfolio.cash <= 0 or current_price <= 0:
            return 0.0

        # Allocate position_size_pct of total portfolio
        allocation = portfolio.nav * self._position_size_pct
        # Don't exceed available cash
        allocation = min(allocation, portfolio.cash * 0.99)

        qty = allocation / current_price
        return qty
