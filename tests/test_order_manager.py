"""Tests for execution/order_manager.py — OrderManager lifecycle and callbacks."""

import pytest

from core.models import Order, OrderStatus, OrderType, Side
from data.buffer import LiveBuffer
from execution.order_manager import OrderManager
from execution.sim_executor import SimExecutor
from risk.tracker import PortfolioTracker
from tests.conftest import make_candle


@pytest.fixture
def buffer():
    return LiveBuffer(max_candles=10)


@pytest.fixture
def tracker():
    return PortfolioTracker(initial_capital=100_000.0, fee_bps=10.0)


@pytest.fixture
def manager(buffer, tracker):
    config = {"slippage_bps": 5, "fee_bps": 10}
    executor = SimExecutor(config, buffer)
    return OrderManager(executor, tracker)


@pytest.mark.asyncio
class TestSubmit:
    async def test_market_order_fills(self, manager, buffer, tracker):
        await buffer.push_candle(make_candle(close=100.0))
        order = Order(
            symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=1.0,
        )
        result = await manager.submit(order)
        assert result.status == OrderStatus.FILLED
        # Portfolio should reflect the fill
        pos = tracker.get_position("BTC/USDT")
        assert pos.quantity == 1.0

    async def test_limit_order_tracked(self, manager, buffer):
        await buffer.push_candle(make_candle(close=110.0))
        order = Order(
            symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, quantity=1.0, price=100.0,
        )
        result = await manager.submit(order)
        assert result.status == OrderStatus.SUBMITTED
        assert manager.has_pending is True
        assert order.order_id in manager.active_orders


@pytest.mark.asyncio
class TestCallbacks:
    async def test_fill_callback_invoked(self, manager, buffer):
        await buffer.push_candle(make_candle(close=100.0))
        fills = []
        manager.register_fill_callback(lambda o: fills.append(o))

        order = Order(
            symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=1.0,
        )
        await manager.submit(order)
        assert len(fills) == 1
        assert fills[0].status == OrderStatus.FILLED

    async def test_multiple_callbacks(self, manager, buffer):
        await buffer.push_candle(make_candle(close=100.0))
        results_a = []
        results_b = []
        manager.register_fill_callback(lambda o: results_a.append(o.symbol))
        manager.register_fill_callback(lambda o: results_b.append(o.symbol))

        order = Order(
            symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=1.0,
        )
        await manager.submit(order)
        assert results_a == ["BTC/USDT"]
        assert results_b == ["BTC/USDT"]


@pytest.mark.asyncio
class TestCancel:
    async def test_cancel_removes_from_active(self, manager, buffer):
        await buffer.push_candle(make_candle(close=110.0))
        order = Order(
            symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.LIMIT, quantity=1.0, price=100.0,
        )
        submitted = await manager.submit(order)
        assert manager.has_pending

        await manager.cancel(submitted.order_id)
        assert not manager.has_pending

    async def test_cancel_nonexistent(self, manager):
        # Should not raise
        await manager.cancel("nonexistent-id")
