"""Tests for signal confirmation, graduated exits, and alpha decay in strategy/logic.py."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.models import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    PortfolioSnapshot,
    Side,
    Signal,
    StrategyState,
)
from strategy.logic import StrategyLogic
from tests.conftest import make_filled_buy, make_signal


def _make_signal_now(alpha=0.7, symbol="BTC/USDT"):
    """Make a signal with a recent timestamp to avoid decay issues."""
    return Signal(
        symbol=symbol, alpha_score=alpha, confidence=abs(alpha),
        timestamp=datetime.utcnow(), source="rule_based",
    )


@pytest.fixture
def config_confirm2():
    return {
        "alpha": {"entry_threshold": 0.6, "exit_threshold": -0.2},
        "strategy": {"confirmation_bars": 2},
    }


@pytest.fixture
def config_confirm1():
    return {
        "alpha": {"entry_threshold": 0.6, "exit_threshold": -0.2},
        "strategy": {"confirmation_bars": 1},
    }


@pytest.fixture
def snap_100k():
    return PortfolioSnapshot(
        timestamp=datetime(2025, 1, 1),
        cash=100000.0, positions=[], nav=100000.0,
        daily_pnl=0.0, peak_nav=100000.0, drawdown=0.0,
    )


@pytest.fixture
def portfolio_with_btc():
    pos = Position(
        symbol="BTC/USDT", quantity=1.0, entry_price=100.0,
        current_price=105.0, unrealized_pnl=5.0,
        peak_price=107.0, state=StrategyState.HOLDING,
    )
    return PortfolioSnapshot(
        timestamp=datetime(2025, 1, 1),
        cash=90000.0, positions=[pos], nav=90105.0,
        daily_pnl=105.0, peak_nav=90107.0, drawdown=0.0,
    )


class TestSignalConfirmation:
    def test_single_bar_no_trigger_with_confirm_2(self, config_confirm2, snap_100k):
        logic = StrategyLogic("BTC/USDT", config_confirm2)
        order = logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        assert order is None
        assert logic.state == StrategyState.FLAT

    def test_two_consecutive_triggers(self, config_confirm2, snap_100k):
        logic = StrategyLogic("BTC/USDT", config_confirm2)
        # First bar — no trigger
        order1 = logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        assert order1 is None

        # Second bar — confirmed
        order2 = logic.on_signal(_make_signal_now(alpha=0.75), snap_100k, current_price=100.0)
        assert order2 is not None
        assert order2.side == Side.BUY
        assert logic.state == StrategyState.LONG_PENDING

    def test_gap_resets_confirmation(self, config_confirm2, snap_100k):
        logic = StrategyLogic("BTC/USDT", config_confirm2)
        # First bar above threshold
        logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        # Second bar BELOW threshold — resets
        logic.on_signal(_make_signal_now(alpha=0.3), snap_100k, current_price=100.0)
        # Third bar above threshold — only 1 consecutive, no trigger
        order = logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        assert order is None

    def test_confirm_1_matches_old_behavior(self, config_confirm1, snap_100k):
        logic = StrategyLogic("BTC/USDT", config_confirm1)
        order = logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        assert order is not None
        assert order.side == Side.BUY

    def test_exit_still_immediate(self, config_confirm2, snap_100k, portfolio_with_btc):
        logic = StrategyLogic("BTC/USDT", config_confirm2)
        # Get to HOLDING state
        logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_fill(make_filled_buy(price=100.0, qty=10.0))
        assert logic.state == StrategyState.HOLDING

        # Single negative alpha → immediate exit
        order = logic.on_signal(_make_signal_now(alpha=-0.5), portfolio_with_btc, current_price=100.0)
        assert order is not None
        assert order.side == Side.SELL


class TestGraduatedExits:
    def _config_with_tiers(self):
        return {
            "alpha": {"entry_threshold": 0.6, "exit_threshold": -0.2},
            "strategy": {
                "confirmation_bars": 1,
                "exit_tiers": [
                    {"threshold": -0.1, "sell_pct": 0.5},
                    {"threshold": -0.3, "sell_pct": 1.0},
                ],
            },
        }

    def test_partial_exit_tier1(self, snap_100k, portfolio_with_btc):
        logic = StrategyLogic("BTC/USDT", self._config_with_tiers())
        logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_fill(make_filled_buy(price=100.0, qty=1.0))
        assert logic.state == StrategyState.HOLDING

        order = logic.on_signal(_make_signal_now(alpha=-0.15), portfolio_with_btc, current_price=100.0)
        assert order is not None
        assert order.side == Side.SELL
        assert order.quantity == pytest.approx(0.5)
        # Still HOLDING after partial exit
        assert logic.state == StrategyState.HOLDING

    def test_full_exit_tier2(self, snap_100k, portfolio_with_btc):
        logic = StrategyLogic("BTC/USDT", self._config_with_tiers())
        logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_fill(make_filled_buy(price=100.0, qty=1.0))

        # Trigger tier 1
        logic.on_signal(_make_signal_now(alpha=-0.15), portfolio_with_btc, current_price=100.0)
        # Trigger tier 2 — should sell rest
        order = logic.on_signal(_make_signal_now(alpha=-0.35), portfolio_with_btc, current_price=100.0)
        assert order is not None
        assert order.side == Side.SELL
        assert logic.state == StrategyState.FLAT

    def test_no_tiers_falls_back_to_threshold(self, snap_100k, portfolio_with_btc):
        config = {
            "alpha": {"entry_threshold": 0.6, "exit_threshold": -0.2},
            "strategy": {"confirmation_bars": 1, "exit_tiers": []},
        }
        logic = StrategyLogic("BTC/USDT", config)
        logic.on_signal(_make_signal_now(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_fill(make_filled_buy(price=100.0, qty=1.0))

        order = logic.on_signal(_make_signal_now(alpha=-0.5), portfolio_with_btc, current_price=100.0)
        assert order is not None
        assert order.side == Side.SELL


class TestAlphaDecay:
    def test_fresh_signal_no_decay(self):
        now = datetime(2025, 1, 1, 0, 0, 0)
        sig = Signal(symbol="BTC/USDT", alpha_score=0.8, timestamp=now)
        assert sig.decayed_alpha(now, half_life_s=150) == pytest.approx(0.8)

    def test_decay_halves_at_half_life(self):
        t0 = datetime(2025, 1, 1, 0, 0, 0)
        t1 = t0 + timedelta(seconds=150)
        sig = Signal(symbol="BTC/USDT", alpha_score=0.8, timestamp=t0)
        assert sig.decayed_alpha(t1, half_life_s=150) == pytest.approx(0.4)

    def test_large_half_life_effectively_disabled(self):
        t0 = datetime(2025, 1, 1, 0, 0, 0)
        t1 = t0 + timedelta(seconds=300)
        sig = Signal(symbol="BTC/USDT", alpha_score=0.8, timestamp=t0)
        result = sig.decayed_alpha(t1, half_life_s=999999)
        assert result > 0.799  # basically no decay
