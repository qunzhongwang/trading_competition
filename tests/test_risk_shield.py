"""Tests for risk/risk_shield.py — pre-trade validation, stops, circuit breaker."""

import pytest

from core.models import Order, OrderType, Side, StrategyState
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from tests.conftest import make_candle, make_filled_buy


@pytest.fixture
def shield(default_config):
    return RiskShield(default_config)


@pytest.fixture
def tracker():
    return PortfolioTracker(initial_capital=100_000.0, fee_bps=10.0)


def _buy_order(symbol="BTC/USDT", qty=1.0, price=100.0):
    return Order(
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=qty,
        price=price,
    )


def _sell_order(symbol="BTC/USDT", qty=1.0):
    return Order(
        symbol=symbol,
        side=Side.SELL,
        order_type=OrderType.MARKET,
        quantity=qty,
    )


class TestCircuitBreaker:
    def test_blocks_buys_when_active(self, shield, tracker):
        shield._circuit_breaker_active = True
        order = _buy_order()
        result = shield.validate(order, tracker)
        assert result is None

    def test_allows_sells_when_active(self, shield, tracker):
        # Need a position to sell
        tracker.on_fill(make_filled_buy(price=100.0, qty=5.0))
        shield._circuit_breaker_active = True
        order = _sell_order(qty=5.0)
        result = shield.validate(order, tracker)
        assert result is not None

    def test_activates_on_drawdown(self, shield, tracker):
        # Buy a large position then crash the price
        tracker.on_fill(make_filled_buy(price=100.0, qty=500.0))
        tracker.update_prices("BTC/USDT", 80.0)  # -20% → big drawdown
        activated = shield.check_circuit_breaker(tracker)
        assert activated is True
        assert shield.circuit_breaker_active is True

    def test_no_double_activation(self, shield, tracker):
        shield._circuit_breaker_active = True
        result = shield.check_circuit_breaker(tracker)
        assert result is False  # already active

    def test_reset_clears_breaker(self, shield):
        shield._circuit_breaker_active = True
        shield.reset_daily()
        assert shield.circuit_breaker_active is False


class TestLongOnlyEnforcement:
    def test_clamps_sell_to_position(self, shield, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=3.0))
        order = _sell_order(qty=10.0)
        result = shield.validate(order, tracker)
        assert result is not None
        assert result.quantity == pytest.approx(3.0)

    def test_rejects_sell_with_no_position(self, shield, tracker):
        order = _sell_order(qty=1.0)
        result = shield.validate(order, tracker)
        assert result is None


class TestRateLimit:
    def test_allows_within_limit(self, shield, tracker):
        for _ in range(10):
            order = _buy_order(qty=0.01, price=100.0)
            result = shield.validate(order, tracker)
            assert result is not None

    def test_rejects_over_limit(self, shield, tracker):
        for _ in range(10):
            shield.validate(_buy_order(qty=0.01, price=100.0), tracker)
        # 11th should be rejected
        result = shield.validate(_buy_order(qty=0.01, price=100.0), tracker)
        assert result is None


class TestExposureLimits:
    def test_rejects_when_portfolio_exposure_exceeded(self, shield, tracker):
        # Buy up to near the 50% limit
        tracker.on_fill(make_filled_buy(price=100.0, qty=490.0))
        tracker.update_prices("BTC/USDT", 100.0)
        # Try to buy more
        order = _buy_order(qty=200.0, price=100.0)
        result = shield.validate(order, tracker)
        # Should be clamped or rejected
        if result is not None:
            assert result.quantity < 200.0

    def test_rejects_when_single_exposure_exceeded(self, shield, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=140.0))
        tracker.update_prices("BTC/USDT", 100.0)
        order = _buy_order(qty=100.0, price=100.0)
        result = shield.validate(order, tracker)
        if result is not None:
            assert result.quantity < 100.0

    def test_cash_check(self, shield, tracker):
        # Try to buy more than cash allows
        order = _buy_order(qty=2000.0, price=100.0)  # cost = 200k > 100k cash
        result = shield.validate(order, tracker)
        if result is not None:
            # Quantity should be clamped to what cash allows
            assert result.quantity * 100.0 < 100_000

    def test_market_buy_uses_reference_market_price(self, shield, tracker):
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )
        result = shield.validate(order, tracker, market_price=1000.0)
        assert result is not None
        assert result.quantity < 100.0  # clamped by exposure/cash using market price


class TestTrailingStop:
    def test_triggers_on_drop_from_peak(self, shield, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        pos = tracker.get_position("BTC/USDT")
        pos.peak_price = 110.0
        pos.state = StrategyState.HOLDING

        # 3% trailing stop from peak 110 → stop at 106.7
        candle = make_candle(close=106.0)
        orders = shield.check_stops(tracker, {"BTC/USDT": candle}, {})
        assert len(orders) == 1
        assert orders[0].side == Side.SELL
        assert orders[0].quantity == 1.0

    def test_no_trigger_above_stop(self, shield, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        pos = tracker.get_position("BTC/USDT")
        pos.peak_price = 110.0
        pos.state = StrategyState.HOLDING

        candle = make_candle(close=108.0)  # above 106.7 stop
        orders = shield.check_stops(tracker, {"BTC/USDT": candle}, {})
        assert len(orders) == 0


class TestBreakEvenLock:
    def test_triggers_after_profitable_retrace(self, default_config, tracker):
        config = {
            **default_config,
            "risk": {
                **default_config["risk"],
                "break_even_trigger_pct": 0.01,
                "break_even_buffer_pct": 0.002,
            },
        }
        shield = RiskShield(config)
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        pos = tracker.get_position("BTC/USDT")
        pos.peak_price = 102.0
        pos.state = StrategyState.HOLDING

        candle = make_candle(close=100.10)
        orders = shield.check_stops(tracker, {"BTC/USDT": candle}, {})
        assert len(orders) == 1
        assert orders[0].side == Side.SELL

    def test_does_not_trigger_before_profit_threshold(self, default_config, tracker):
        config = {
            **default_config,
            "risk": {
                **default_config["risk"],
                "break_even_trigger_pct": 0.01,
                "break_even_buffer_pct": 0.002,
            },
        }
        shield = RiskShield(config)
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        pos = tracker.get_position("BTC/USDT")
        pos.peak_price = 100.8
        pos.state = StrategyState.HOLDING

        candle = make_candle(close=100.10)
        orders = shield.check_stops(tracker, {"BTC/USDT": candle}, {})
        assert len(orders) == 0


class TestATRStop:
    def test_triggers_below_atr_stop(self, shield, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        pos = tracker.get_position("BTC/USDT")
        pos.state = StrategyState.HOLDING
        pos.peak_price = 100.0

        # ATR stop = entry - 2*ATR = 100 - 2*5 = 90
        candle = make_candle(close=89.0)
        orders = shield.check_stops(tracker, {"BTC/USDT": candle}, {"BTC/USDT": 5.0})
        assert len(orders) == 1

    def test_no_trigger_above_atr_stop(self, shield, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        pos = tracker.get_position("BTC/USDT")
        pos.state = StrategyState.HOLDING
        pos.peak_price = 100.0

        # Price 98 is above ATR stop (90) AND above trailing stop (97)
        candle = make_candle(close=98.0)
        orders = shield.check_stops(tracker, {"BTC/USDT": candle}, {"BTC/USDT": 5.0})
        assert len(orders) == 0


class TestNoPositionStops:
    def test_no_stops_for_flat_position(self, shield, tracker):
        candle = make_candle(close=50.0)
        orders = shield.check_stops(tracker, {"BTC/USDT": candle}, {"BTC/USDT": 5.0})
        assert len(orders) == 0
