"""Tests for core/models.py — Pydantic schemas, enums, defaults."""

from datetime import datetime


from core.models import (
    OHLCV,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PortfolioSnapshot,
    Side,
    Signal,
    StrategyState,
)


class TestEnums:
    def test_side_values(self):
        assert Side.BUY == "BUY"
        assert Side.SELL == "SELL"

    def test_order_type_values(self):
        assert OrderType.MARKET == "MARKET"
        assert OrderType.LIMIT == "LIMIT"

    def test_order_status_values(self):
        statuses = [s.value for s in OrderStatus]
        assert "PENDING" in statuses
        assert "FILLED" in statuses
        assert "CANCELLED" in statuses

    def test_strategy_state(self):
        assert StrategyState.FLAT == "FLAT"
        assert StrategyState.LONG_PENDING == "LONG_PENDING"
        assert StrategyState.HOLDING == "HOLDING"
        assert StrategyState.EXIT_PENDING == "EXIT_PENDING"


class TestOHLCV:
    def test_create(self):
        c = OHLCV(
            symbol="BTC/USDT",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=50.0,
            timestamp=datetime(2025, 1, 1),
        )
        assert c.is_closed is True
        assert c.close == 102.0

    def test_not_closed(self):
        c = OHLCV(
            symbol="BTC/USDT",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=50.0,
            timestamp=datetime(2025, 1, 1),
            is_closed=False,
        )
        assert c.is_closed is False


class TestOrder:
    def test_defaults(self):
        o = Order(
            symbol="BTC/USDT", side=Side.BUY, order_type=OrderType.MARKET, quantity=1.0
        )
        assert o.status == OrderStatus.PENDING
        assert o.filled_price is None
        assert o.filled_quantity == 0.0
        assert len(o.order_id) == 12

    def test_unique_ids(self):
        o1 = Order(
            symbol="BTC/USDT", side=Side.BUY, order_type=OrderType.MARKET, quantity=1.0
        )
        o2 = Order(
            symbol="BTC/USDT", side=Side.BUY, order_type=OrderType.MARKET, quantity=1.0
        )
        assert o1.order_id != o2.order_id


class TestPosition:
    def test_defaults(self):
        p = Position(symbol="BTC/USDT")
        assert p.quantity == 0.0
        assert p.state == StrategyState.FLAT
        assert p.unrealized_pnl == 0.0
        assert p.realized_pnl == 0.0


class TestSignal:
    def test_create(self):
        s = Signal(symbol="BTC/USDT", alpha_score=0.8, timestamp=datetime(2025, 1, 1))
        assert s.confidence == 1.0
        assert s.source == "rule_based"


class TestPortfolioSnapshot:
    def test_create(self):
        snap = PortfolioSnapshot(
            timestamp=datetime(2025, 1, 1),
            cash=100_000.0,
            nav=100_000.0,
            peak_nav=100_000.0,
        )
        assert snap.drawdown == 0.0
        assert snap.daily_pnl == 0.0
        assert len(snap.positions) == 0
