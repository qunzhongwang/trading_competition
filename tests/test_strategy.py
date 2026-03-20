"""Tests for strategy/logic.py — StrategyLogic state machine."""

import copy
from datetime import datetime, timedelta

import pytest

from core.models import (
    FactorBias,
    FactorObservation,
    FactorSnapshot,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Side,
    StrategyState,
)
from strategy.logic import StrategyLogic
from tests.conftest import make_filled_buy, make_signal


@pytest.fixture
def logic(default_config):
    return StrategyLogic("BTC/USDT", default_config)


@pytest.fixture
def snap_100k(portfolio_snapshot):
    return portfolio_snapshot


def _bullish_observation(
    name: str,
    category: str,
    *,
    ts: datetime | None = None,
    strength: float = 0.8,
) -> FactorObservation:
    return FactorObservation(
        symbol="BTC/USDT",
        name=name,
        category=category,
        timestamp=ts or datetime(2025, 1, 1),
        bias=FactorBias.BULLISH,
        strength=strength,
        value=strength,
        threshold=0.0,
        horizon_minutes=60,
        expected_move_bps=80.0,
        thesis=f"{name} supports the long",
        invalidate_condition=f"{name} fades",
    )


def _bearish_observation(
    name: str,
    category: str,
    *,
    ts: datetime | None = None,
    strength: float = 0.8,
) -> FactorObservation:
    return FactorObservation(
        symbol="BTC/USDT",
        name=name,
        category=category,
        timestamp=ts or datetime(2025, 1, 1),
        bias=FactorBias.BEARISH,
        strength=strength,
        value=-strength,
        threshold=0.0,
        horizon_minutes=60,
        expected_move_bps=80.0,
        thesis=f"{name} blocks the long",
        invalidate_condition=f"{name} clears",
    )


def _factor_snapshot(
    *,
    ts: datetime | None = None,
    entry_score: float = 0.75,
    exit_score: float = 0.0,
    blocker_score: float = 0.1,
    observations: list[FactorObservation] | None = None,
    supporting_factors: list[str] | None = None,
    blocking_factors: list[str] | None = None,
) -> FactorSnapshot:
    observations = observations or []
    return FactorSnapshot(
        symbol="BTC/USDT",
        timestamp=ts or datetime(2025, 1, 1),
        regime="trend_following",
        entry_score=entry_score,
        exit_score=exit_score,
        blocker_score=blocker_score,
        confidence=max(entry_score, exit_score, 0.5),
        observations=observations,
        supporting_factors=supporting_factors or [],
        blocking_factors=blocking_factors or [],
        summary="test snapshot",
    )


class TestFlatState:
    def test_buy_on_high_alpha(self, logic, snap_100k):
        signal = make_signal(alpha=0.8)
        order = logic.on_signal(signal, snap_100k, current_price=100.0)
        assert order is not None
        assert order.side == Side.BUY
        assert order.symbol == "BTC/USDT"
        assert logic.state == StrategyState.LONG_PENDING

    def test_no_action_below_threshold(self, logic, snap_100k):
        signal = make_signal(alpha=0.3)
        order = logic.on_signal(signal, snap_100k, current_price=100.0)
        assert order is None
        assert logic.state == StrategyState.FLAT

    def test_no_action_at_threshold(self, logic, snap_100k):
        signal = make_signal(alpha=0.6)  # equal to threshold, not above
        order = logic.on_signal(signal, snap_100k, current_price=100.0)
        assert order is None

    def test_no_buy_zero_price(self, logic, snap_100k):
        signal = make_signal(alpha=0.8)
        order = logic.on_signal(signal, snap_100k, current_price=0.0)
        assert order is None


class TestLongPendingState:
    def test_ignores_signals_while_pending(self, logic, snap_100k):
        # First signal → LONG_PENDING
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        assert logic.state == StrategyState.LONG_PENDING

        # Second signal → still LONG_PENDING, no new order
        order = logic.on_signal(make_signal(alpha=0.9), snap_100k, current_price=100.0)
        assert order is None
        assert logic.state == StrategyState.LONG_PENDING

    def test_fill_transitions_to_holding(self, logic, snap_100k):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        fill = make_filled_buy(price=100.0, qty=10.0)
        logic.on_fill(fill)
        assert logic.state == StrategyState.HOLDING

    def test_cancel_returns_to_flat(self, logic, snap_100k):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        cancel = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            status=OrderStatus.CANCELLED,
        )
        logic.on_cancel(cancel)
        assert logic.state == StrategyState.FLAT


class TestHoldingState:
    def _enter_holding(self, logic, snap_100k):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_fill(make_filled_buy(price=100.0, qty=10.0))
        assert logic.state == StrategyState.HOLDING

    def test_sell_on_negative_alpha(self, logic, snap_100k, portfolio_with_position):
        self._enter_holding(logic, snap_100k)
        signal = make_signal(alpha=-0.5)
        order = logic.on_signal(signal, portfolio_with_position, current_price=100.0)
        assert order is not None
        assert order.side == Side.SELL
        assert logic.state == StrategyState.FLAT

    def test_no_sell_above_exit_threshold(
        self, logic, snap_100k, portfolio_with_position
    ):
        self._enter_holding(logic, snap_100k)
        signal = make_signal(alpha=0.0)  # above exit threshold -0.2
        order = logic.on_signal(signal, portfolio_with_position, current_price=100.0)
        assert order is None
        assert logic.state == StrategyState.HOLDING

    def test_sell_quantity_matches_position(
        self, logic, snap_100k, portfolio_with_position
    ):
        self._enter_holding(logic, snap_100k)
        signal = make_signal(alpha=-0.5)
        order = logic.on_signal(signal, portfolio_with_position, current_price=100.0)
        # Should sell the full position qty from the snapshot
        assert order.quantity == 1.0  # from portfolio_with_position fixture


class TestForceFlat:
    def test_force_flat_from_holding(self, logic, snap_100k):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_fill(make_filled_buy(price=100.0, qty=10.0))
        logic.force_flat()
        assert logic.state == StrategyState.FLAT

    def test_force_flat_from_pending(self, logic, snap_100k):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        logic.force_flat()
        assert logic.state == StrategyState.FLAT


class TestPositionSizing:
    def test_sizing_is_percentage_of_nav(self, logic, snap_100k):
        signal = make_signal(alpha=0.8)
        order = logic.on_signal(signal, snap_100k, current_price=100.0)
        # Half-Kelly sizing with alpha=0.8, entry_threshold=0.6:
        # raw_kelly = 0.55 - 0.45/1.5 = 0.25
        # alpha_intensity = (0.8 - 0.6) / (1.0 - 0.6) = 0.5
        # scaled = 0.5 * 0.25 * 0.5 = 0.0625
        # position_pct = 0.05 + 0.0625 * 0.10 = 0.05625
        # allocation = 100000 * 0.05625 = 5625.0, capped by cash*0.99 = 99000
        expected_qty = 5625.0 / 100.0
        assert order.quantity == pytest.approx(expected_qty)

    def test_sizing_capped_by_cash(self):
        """When cash is low, allocation is limited."""
        config = {
            "alpha": {"entry_threshold": 0.6, "exit_threshold": -0.2},
            "strategy": {"position_size_pct": 0.50, "confirmation_bars": 1},
        }
        logic = StrategyLogic("BTC/USDT", config)
        snap = PortfolioSnapshot(
            timestamp=__import__("datetime").datetime(2025, 1, 1),
            cash=5000.0,
            positions=[],
            nav=100000.0,
            daily_pnl=0.0,
            peak_nav=100000.0,
            drawdown=0.0,
        )
        order = logic.on_signal(make_signal(alpha=0.8), snap, current_price=100.0)
        # 50% of 100k = 50000, but cash only 5000*0.99 = 4950
        assert order.quantity == pytest.approx(4950.0 / 100.0)


class TestWrongSymbol:
    def test_fill_wrong_symbol_ignored(self, logic, snap_100k):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        fill = make_filled_buy(symbol="ETH/USDT", price=100.0, qty=10.0)
        logic.on_fill(fill)
        assert logic.state == StrategyState.LONG_PENDING  # unchanged

    def test_cancel_wrong_symbol_ignored(self, logic, snap_100k):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        cancel = Order(
            symbol="ETH/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            status=OrderStatus.CANCELLED,
        )
        logic.on_cancel(cancel)
        assert logic.state == StrategyState.LONG_PENDING


class TestFactorFirstStateMachine:
    def test_failed_breadth_does_not_count_toward_confirmation(
        self, default_config, snap_100k
    ):
        config = copy.deepcopy(default_config)
        config["strategy"].update(
            {
                "confirmation_bars": 2,
                "min_supporting_factors": 2,
                "min_supporting_categories": 2,
                "require_trend_alignment": True,
            }
        )
        logic = StrategyLogic("BTC/USDT", config)
        t0 = datetime(2025, 1, 1, 0, 0)

        weak_snapshot = _factor_snapshot(
            ts=t0,
            observations=[_bullish_observation("trend_alignment", "price_structure", ts=t0)],
            supporting_factors=["trend_alignment"],
        )
        good_observations = [
            _bullish_observation("trend_alignment", "price_structure", ts=t0 + timedelta(minutes=5)),
            _bullish_observation("volume_confirmation", "flow", ts=t0 + timedelta(minutes=5)),
        ]
        good_snapshot = _factor_snapshot(
            ts=t0 + timedelta(minutes=5),
            observations=good_observations,
            supporting_factors=["trend_alignment", "volume_confirmation"],
        )
        second_good_snapshot = _factor_snapshot(
            ts=t0 + timedelta(minutes=10),
            observations=[
                _bullish_observation("trend_alignment", "price_structure", ts=t0 + timedelta(minutes=10)),
                _bullish_observation("volume_confirmation", "flow", ts=t0 + timedelta(minutes=10)),
            ],
            supporting_factors=["trend_alignment", "volume_confirmation"],
        )

        assert logic.on_factors(weak_snapshot, snap_100k, current_price=100.0) is None
        assert logic.on_factors(good_snapshot, snap_100k, current_price=100.0) is None

        intent = logic.on_factors(
            second_good_snapshot, snap_100k, current_price=100.0
        )
        assert intent is not None
        assert intent.direction == Side.BUY
        assert logic.state == StrategyState.LONG_PENDING

    def test_factor_exit_uses_exit_pending_and_cancel_recovers_holding(
        self, logic, snap_100k, portfolio_with_position
    ):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_fill(make_filled_buy(price=100.0, qty=1.0))
        assert logic.state == StrategyState.HOLDING

        snapshot = _factor_snapshot(
            exit_score=0.8,
            blocker_score=0.7,
            observations=[_bearish_observation("perp_crowding", "derivatives")],
            blocking_factors=["perp_crowding"],
        )
        intent = logic.on_factors(snapshot, portfolio_with_position, current_price=110.0)
        assert intent is not None
        assert intent.direction == Side.SELL
        assert logic.state == StrategyState.EXIT_PENDING

        partial_sell = Order(
            symbol="BTC/USDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_price=110.0,
            filled_quantity=0.4,
        )
        logic.on_fill(partial_sell)
        assert logic.state == StrategyState.EXIT_PENDING

        cancel = Order(
            symbol="BTC/USDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.CANCELLED,
            filled_price=110.0,
            filled_quantity=0.4,
        )
        logic.on_cancel(cancel)
        assert logic.state == StrategyState.HOLDING

    def test_partial_entry_cancel_promotes_to_holding(self, logic, snap_100k):
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        assert logic.state == StrategyState.LONG_PENDING

        partial_buy = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_price=100.0,
            filled_quantity=0.4,
        )
        logic.on_fill(partial_buy)
        assert logic.state == StrategyState.LONG_PENDING

        cancel = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.CANCELLED,
            filled_price=100.0,
            filled_quantity=0.4,
        )
        logic.on_cancel(cancel)
        assert logic.state == StrategyState.HOLDING

    def test_partial_exit_records_one_weighted_trade(self, logic, snap_100k, portfolio_with_position):
        class _TradeTracker:
            def __init__(self):
                self.trades = []

            def get_kelly_params(self):
                return 0.55, 1.5

            def record_trade(self, entry_price: float, exit_price: float) -> None:
                self.trades.append((entry_price, exit_price))

        tracker = _TradeTracker()
        logic.set_trade_tracker(tracker)
        logic.on_signal(make_signal(alpha=0.8), snap_100k, current_price=100.0)
        logic.on_fill(make_filled_buy(price=100.0, qty=1.0))

        snapshot = _factor_snapshot(
            exit_score=0.8,
            blocker_score=0.7,
            observations=[_bearish_observation("perp_crowding", "derivatives")],
            blocking_factors=["perp_crowding"],
        )
        logic.on_factors(snapshot, portfolio_with_position, current_price=110.0)

        logic.on_fill(
            Order(
                symbol="BTC/USDT",
                side=Side.SELL,
                order_type=OrderType.MARKET,
                quantity=1.0,
                status=OrderStatus.PARTIALLY_FILLED,
                filled_price=110.0,
                filled_quantity=0.4,
            )
        )
        logic.on_fill(
            Order(
                symbol="BTC/USDT",
                side=Side.SELL,
                order_type=OrderType.MARKET,
                quantity=1.0,
                status=OrderStatus.FILLED,
                filled_price=112.0,
                filled_quantity=0.6,
            )
        )

        assert logic.state == StrategyState.FLAT
        assert len(tracker.trades) == 1
        assert tracker.trades[0][0] == pytest.approx(100.0)
        assert tracker.trades[0][1] == pytest.approx(111.2)
