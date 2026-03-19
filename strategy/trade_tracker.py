"""Rolling trade outcome tracker for adaptive Kelly position sizing.

Tracks recent trades and computes blended win rate / payoff ratio
that smoothly transition from static priors to observed outcomes.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Tuple

logger = logging.getLogger(__name__)


class TradeTracker:
    """Tracks rolling trade outcomes for adaptive Kelly parameters."""

    def __init__(
        self,
        window: int = 50,
        min_trades: int = 10,
        prior_win_rate: float = 0.55,
        prior_payoff: float = 1.5,
    ):
        self._window = window
        self._min_trades = min_trades
        self._prior_win_rate = prior_win_rate
        self._prior_payoff = prior_payoff
        self._trades: deque = deque(maxlen=window)  # (entry_price, exit_price)
        self._n_total = 0

    def record_trade(self, entry_price: float, exit_price: float) -> None:
        """Record a completed trade (entry → exit)."""
        if entry_price <= 0:
            return
        self._trades.append((entry_price, exit_price))
        self._n_total += 1
        ret = (exit_price - entry_price) / entry_price
        logger.debug("Trade recorded: entry=%.2f exit=%.2f ret=%.4f (total=%d)",
                      entry_price, exit_price, ret, self._n_total)

    def get_kelly_params(self) -> Tuple[float, float]:
        """Return blended (win_rate, payoff_ratio).

        alpha = min(1.0, n_trades / min_trades)
        param = alpha * observed + (1 - alpha) * prior
        """
        if not self._trades:
            return self._prior_win_rate, self._prior_payoff

        wins = []
        losses = []
        for entry, exit_ in self._trades:
            ret = (exit_ - entry) / entry
            if ret > 0:
                wins.append(ret)
            else:
                losses.append(abs(ret))

        n = len(self._trades)
        alpha = min(1.0, n / self._min_trades)

        # Observed win rate
        obs_win_rate = len(wins) / n if n > 0 else self._prior_win_rate

        # Observed payoff ratio (avg win / avg loss)
        avg_win = sum(wins) / len(wins) if wins else 0.01
        avg_loss = sum(losses) / len(losses) if losses else 0.01
        obs_payoff = avg_win / avg_loss if avg_loss > 1e-10 else self._prior_payoff

        # Blend with priors
        blended_win = alpha * obs_win_rate + (1 - alpha) * self._prior_win_rate
        blended_payoff = alpha * obs_payoff + (1 - alpha) * self._prior_payoff

        return blended_win, blended_payoff

    @property
    def n_trades(self) -> int:
        return self._n_total
