from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from core.models import (
    Order,
    Position,
    PortfolioSnapshot,
    RiskMetrics,
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
        # NAV history for risk-adjusted metrics
        self._nav_history: List[Tuple[datetime, float]] = [
            (datetime.utcnow(), initial_capital)
        ]

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

        # Update peak NAV and record history
        nav = self._compute_nav()
        if nav > self._peak_nav:
            self._peak_nav = nav
        self._nav_history.append((datetime.utcnow(), nav))

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

    # ── Risk-Adjusted Metrics ──

    def record_nav_snapshot(self) -> None:
        """Record current NAV for metrics computation. Call periodically."""
        nav = self._compute_nav()
        self._nav_history.append((datetime.utcnow(), nav))
        if nav > self._peak_nav:
            self._peak_nav = nav

    def _daily_returns(self) -> np.ndarray:
        """Compute daily returns from NAV history.

        Groups NAV snapshots by date and computes return between
        last NAV of each day.
        """
        if len(self._nav_history) < 2:
            return np.array([], dtype=np.float64)

        # Group by date, take last NAV per day
        daily_navs: Dict[str, float] = {}
        for ts, nav in self._nav_history:
            date_key = ts.strftime("%Y-%m-%d")
            daily_navs[date_key] = nav  # last value wins

        sorted_navs = [daily_navs[k] for k in sorted(daily_navs.keys())]
        if len(sorted_navs) < 2:
            return np.array([], dtype=np.float64)

        arr = np.array(sorted_navs, dtype=np.float64)
        return np.diff(arr) / arr[:-1]

    def compute_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe Ratio."""
        returns = self._daily_returns()
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / 365.0
        std = float(np.std(excess, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess)) / std * math.sqrt(365)

    def compute_sortino(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sortino Ratio (only downside deviation in denominator)."""
        returns = self._daily_returns()
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / 365.0
        downside = excess[excess < 0]
        if len(downside) < 1:
            return float("inf") if float(np.mean(excess)) > 0 else 0.0
        downside_std = float(np.std(downside, ddof=1))
        if downside_std < 1e-10:
            return 0.0
        return float(np.mean(excess)) / downside_std * math.sqrt(365)

    def compute_calmar(self) -> float:
        """Calmar Ratio = annualized return / max drawdown."""
        returns = self._daily_returns()
        if len(returns) < 2:
            return 0.0
        nav = self._compute_nav()
        total_return = (nav - self._initial_nav) / self._initial_nav
        n_days = len(returns)
        annualized_return = total_return * (365.0 / max(n_days, 1))

        # Max drawdown from NAV history
        max_dd = self._max_drawdown_from_history()
        if max_dd < 1e-10:
            return float("inf") if annualized_return > 0 else 0.0
        return annualized_return / max_dd

    def _max_drawdown_from_history(self) -> float:
        """Compute max drawdown from recorded NAV history."""
        if len(self._nav_history) < 2:
            return 0.0
        navs = np.array([nav for _, nav in self._nav_history], dtype=np.float64)
        peak = np.maximum.accumulate(navs)
        drawdowns = (peak - navs) / np.where(peak > 0, peak, 1.0)
        return float(np.max(drawdowns))

    def compute_composite_score(self) -> float:
        """Competition composite: 0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar."""
        sortino = self.compute_sortino()
        sharpe = self.compute_sharpe()
        calmar = self.compute_calmar()
        return 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    def compute_risk_metrics(self) -> RiskMetrics:
        """Compute all risk-adjusted metrics at once."""
        returns = self._daily_returns()
        nav = self._compute_nav()
        total_return_pct = ((nav - self._initial_nav) / self._initial_nav) * 100
        n_days = len(returns)
        annualized_return = (total_return_pct / 100) * (365.0 / max(n_days, 1))

        sharpe = self.compute_sharpe()
        sortino = self.compute_sortino()
        calmar = self.compute_calmar()
        composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            composite_score=composite,
            annualized_return=annualized_return,
            max_drawdown=self._max_drawdown_from_history(),
            total_return_pct=total_return_pct,
            n_trading_days=n_days,
        )
