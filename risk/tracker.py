from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List

from core.models import (
    Order,
    Position,
    PortfolioSnapshot,
    Side,
    StrategyState,
)

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Maintains live portfolio state. Single source of truth for positions and PnL."""

    def __init__(self, initial_capital: float, fee_bps: float = 10.0):
        self._cash: float = initial_capital
        self._initial_nav: float = initial_capital
        self._peak_nav: float = initial_capital
        self._daily_start_nav: float = initial_capital
        self._positions: Dict[str, Position] = {}
        self._fee_rate: float = fee_bps / 10000.0

    def snapshot(self) -> PortfolioSnapshot:
        nav = self._compute_nav()
        drawdown = (self._peak_nav - nav) / self._peak_nav if self._peak_nav > 0 else 0.0
        daily_pnl = nav - self._daily_start_nav

        return PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            cash=self._cash,
            positions=list(self._positions.values()),
            nav=nav,
            daily_pnl=daily_pnl,
            peak_nav=self._peak_nav,
            drawdown=drawdown,
        )

    def on_fill(self, order: Order) -> None:
        """Update positions and cash on order fill."""
        if order.filled_price is None or order.filled_quantity <= 0:
            logger.warning("Invalid fill: %s", order)
            return

        symbol = order.symbol
        pos = self._get_or_create_position(symbol)
        cost = order.filled_price * order.filled_quantity
        fee = cost * self._fee_rate

        if order.side == Side.BUY:
            # Update weighted average entry price
            old_value = pos.entry_price * pos.quantity
            new_value = order.filled_price * order.filled_quantity
            pos.quantity += order.filled_quantity
            pos.entry_price = (old_value + new_value) / pos.quantity if pos.quantity > 0 else 0.0
            pos.peak_price = max(pos.peak_price, order.filled_price)
            pos.current_price = order.filled_price
            pos.state = StrategyState.HOLDING
            self._cash -= (cost + fee)

            logger.info(
                "BUY filled: %s qty=%.6f @ %.2f, fee=%.2f, cash=%.2f",
                symbol, order.filled_quantity, order.filled_price, fee, self._cash,
            )

        elif order.side == Side.SELL:
            # Realize PnL
            pnl = (order.filled_price - pos.entry_price) * order.filled_quantity
            pos.realized_pnl += pnl
            pos.quantity -= order.filled_quantity
            pos.current_price = order.filled_price
            self._cash += (cost - fee)

            if pos.quantity <= 1e-10:
                pos.quantity = 0.0
                pos.entry_price = 0.0
                pos.peak_price = 0.0
                pos.state = StrategyState.FLAT

            logger.info(
                "SELL filled: %s qty=%.6f @ %.2f, pnl=%.2f, fee=%.2f, cash=%.2f",
                symbol, order.filled_quantity, order.filled_price, pnl, fee, self._cash,
            )

        # Update peak NAV
        nav = self._compute_nav()
        if nav > self._peak_nav:
            self._peak_nav = nav

    def update_prices(self, symbol: str, price: float) -> None:
        """Update current price and unrealized PnL for a position."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = price
            if pos.quantity > 0:
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
                if price > pos.peak_price:
                    pos.peak_price = price

    def get_position(self, symbol: str) -> Position:
        return self._get_or_create_position(symbol)

    def get_exposure(self, symbol: str) -> float:
        """Get position value as fraction of NAV."""
        nav = self._compute_nav()
        if nav <= 0:
            return 0.0
        pos = self._positions.get(symbol)
        if pos is None or pos.quantity <= 0:
            return 0.0
        return (pos.current_price * pos.quantity) / nav

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure as fraction of NAV."""
        nav = self._compute_nav()
        if nav <= 0:
            return 0.0
        total = sum(
            p.current_price * p.quantity
            for p in self._positions.values()
            if p.quantity > 0
        )
        return total / nav

    def reset_daily(self) -> None:
        """Call at start of each trading day."""
        self._daily_start_nav = self._compute_nav()
        logger.info("Daily reset: NAV=%.2f", self._daily_start_nav)

    def _compute_nav(self) -> float:
        positions_value = sum(
            p.current_price * p.quantity
            for p in self._positions.values()
            if p.quantity > 0
        )
        return self._cash + positions_value

    def _get_or_create_position(self, symbol: str) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]
