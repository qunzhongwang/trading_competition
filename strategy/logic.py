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

        # Half-Kelly sizing parameters
        self._base_size_pct: float = strategy_cfg.get("base_size_pct", 0.05)
        self._max_size_pct: float = strategy_cfg.get("max_size_pct", 0.15)
        self._kelly_fraction: float = strategy_cfg.get("kelly_fraction", 0.5)
        self._win_rate: float = strategy_cfg.get("estimated_win_rate", 0.55)
        self._payoff_ratio: float = strategy_cfg.get("estimated_payoff", 1.5)
        self._urgent_alpha_threshold: float = strategy_cfg.get("urgent_alpha_threshold", 0.85)
        exec_cfg = config.get("execution", {})
        self._limit_offset_bps: float = exec_cfg.get("limit_offset_bps", 5)

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
                qty = self._compute_buy_quantity(portfolio, current_price, signal.alpha_score)
                if qty <= 0:
                    return None

                # Use MARKET for urgent alpha, LIMIT otherwise to save on fees
                if signal.alpha_score > self._urgent_alpha_threshold:
                    order_type = OrderType.MARKET
                    price = None
                    order_type_label = "MARKET"
                else:
                    order_type = OrderType.LIMIT
                    price = round(current_price * (1 - self._limit_offset_bps / 10000), 8)
                    order_type_label = "LIMIT"

                self._state = StrategyState.LONG_PENDING
                logger.info(
                    "[%s] FLAT → LONG_PENDING: alpha=%.3f > %.3f, qty=%.6f, order=%s",
                    self._symbol, signal.alpha_score, self._entry_threshold, qty, order_type_label,
                )
                order = Order(
                    symbol=self._symbol,
                    side=Side.BUY,
                    order_type=order_type,
                    quantity=qty,
                )
                if price is not None:
                    order.price = price
                return order

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

    def _compute_buy_quantity(self, portfolio: PortfolioSnapshot, current_price: float, alpha_score: float) -> float:
        """Compute buy quantity using Half-Kelly position sizing."""
        if portfolio.cash <= 0 or current_price <= 0:
            return 0.0

        # Half-Kelly scaling
        raw_kelly = self._win_rate - (1 - self._win_rate) / self._payoff_ratio
        alpha_intensity = (alpha_score - self._entry_threshold) / (1.0 - self._entry_threshold)
        alpha_intensity = max(0.0, min(1.0, alpha_intensity))

        scaled = self._kelly_fraction * raw_kelly * alpha_intensity
        position_pct = self._base_size_pct + scaled * (self._max_size_pct - self._base_size_pct)
        position_pct = max(self._base_size_pct, min(self._max_size_pct, position_pct))

        logger.info(
            "[%s] Half-Kelly sizing: raw_kelly=%.4f, alpha_intensity=%.4f, position_pct=%.4f",
            self._symbol, raw_kelly, alpha_intensity, position_pct,
        )

        allocation = portfolio.nav * position_pct
        allocation = min(allocation, portfolio.cash * 0.99)
        return allocation / current_price
