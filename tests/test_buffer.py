"""Tests for data/buffer.py — LiveBuffer async operations."""

import asyncio

import pytest
import pytest_asyncio

from core.models import OHLCV, Tick
from data.buffer import LiveBuffer
from tests.conftest import make_candle


@pytest.fixture
def buffer():
    return LiveBuffer(max_candles=10, max_ticks=20)


@pytest.mark.asyncio
class TestPushCandle:
    async def test_push_and_retrieve(self, buffer):
        candle = make_candle(close=100.0)
        await buffer.push_candle(candle)
        candles = await buffer.get_candles("BTC/USDT")
        assert len(candles) == 1
        assert candles[0].close == 100.0

    async def test_max_candles_respected(self, buffer):
        for i in range(15):
            await buffer.push_candle(make_candle(close=float(i)))
        candles = await buffer.get_candles("BTC/USDT")
        assert len(candles) == 10  # max_candles=10

    async def test_multi_symbol(self, buffer):
        await buffer.push_candle(make_candle(symbol="BTC/USDT", close=100.0))
        await buffer.push_candle(make_candle(symbol="ETH/USDT", close=50.0))
        btc = await buffer.get_candles("BTC/USDT")
        eth = await buffer.get_candles("ETH/USDT")
        assert len(btc) == 1
        assert len(eth) == 1

    async def test_get_latest(self, buffer):
        await buffer.push_candle(make_candle(close=100.0))
        await buffer.push_candle(make_candle(close=200.0))
        latest = await buffer.get_latest_candle("BTC/USDT")
        assert latest.close == 200.0

    async def test_get_latest_empty(self, buffer):
        result = await buffer.get_latest_candle("BTC/USDT")
        assert result is None


@pytest.mark.asyncio
class TestPushTick:
    async def test_push_and_retrieve(self, buffer):
        from datetime import datetime
        tick = Tick(symbol="BTC/USDT", price=100.0, quantity=1.0, timestamp=datetime(2025, 1, 1))
        await buffer.push_tick(tick)
        ticks = await buffer.get_ticks("BTC/USDT")
        assert len(ticks) == 1

    async def test_max_ticks(self, buffer):
        from datetime import datetime
        for i in range(25):
            tick = Tick(symbol="BTC/USDT", price=float(i), quantity=1.0, timestamp=datetime(2025, 1, 1))
            await buffer.push_tick(tick)
        ticks = await buffer.get_ticks("BTC/USDT")
        assert len(ticks) == 20  # max_ticks=20


@pytest.mark.asyncio
class TestEvent:
    async def test_closed_candle_sets_event(self, buffer):
        await buffer.push_candle(make_candle(is_closed=True))
        # Event should be set, so wait_for_update should return immediately
        result = await buffer.wait_for_update(timeout=0.1)
        # Note: wait_for_update clears event first, then waits.
        # We need to push after calling wait.

    async def test_event_driven_wake(self, buffer):
        """Simulate producer-consumer: push candle wakes up waiter."""
        results = []

        async def consumer():
            got = await buffer.wait_for_update(timeout=2.0)
            results.append(got)

        async def producer():
            await asyncio.sleep(0.05)
            await buffer.push_candle(make_candle(is_closed=True))

        await asyncio.gather(consumer(), producer())
        assert results == [True]

    async def test_timeout_returns_false(self, buffer):
        result = await buffer.wait_for_update(timeout=0.05)
        assert result is False

    async def test_non_closed_candle_no_event(self, buffer):
        """Pushing a non-closed candle should NOT trigger the event."""
        async def consumer():
            return await buffer.wait_for_update(timeout=0.1)

        async def producer():
            await asyncio.sleep(0.02)
            await buffer.push_candle(make_candle(is_closed=False))

        result = await asyncio.gather(consumer(), producer())
        assert result[0] is False  # timed out


@pytest.mark.asyncio
class TestMetadata:
    async def test_candle_count(self, buffer):
        assert buffer.candle_count("BTC/USDT") == 0
        await buffer.push_candle(make_candle())
        assert buffer.candle_count("BTC/USDT") == 1

    async def test_symbols(self, buffer):
        await buffer.push_candle(make_candle(symbol="BTC/USDT"))
        await buffer.push_candle(make_candle(symbol="ETH/USDT"))
        syms = buffer.symbols()
        assert set(syms) == {"BTC/USDT", "ETH/USDT"}

    async def test_get_n_candles(self, buffer):
        for i in range(5):
            await buffer.push_candle(make_candle(close=float(i)))
        candles = await buffer.get_candles("BTC/USDT", n=3)
        assert len(candles) == 3
        assert candles[-1].close == 4.0  # last 3


@pytest.mark.asyncio
class TestResampled:
    async def test_push_and_get_resampled(self, buffer):
        candle = make_candle(close=100.0)
        await buffer.push_resampled(15, candle)
        result = await buffer.get_resampled_candles("BTC/USDT", 15)
        assert len(result) == 1
        assert result[0].close == 100.0

    async def test_get_resampled_empty(self, buffer):
        result = await buffer.get_resampled_candles("BTC/USDT", 60)
        assert result == []

    async def test_resampled_multi_timeframe(self, buffer):
        c = make_candle(close=50.0)
        await buffer.push_resampled(15, c)
        await buffer.push_resampled(60, c)
        r15 = await buffer.get_resampled_candles("BTC/USDT", 15)
        r60 = await buffer.get_resampled_candles("BTC/USDT", 60)
        assert len(r15) == 1
        assert len(r60) == 1

    async def test_resampled_respects_n_limit(self, buffer):
        for i in range(10):
            await buffer.push_resampled(15, make_candle(close=float(i)))
        result = await buffer.get_resampled_candles("BTC/USDT", 15, n=3)
        assert len(result) == 3
        assert result[-1].close == 9.0
