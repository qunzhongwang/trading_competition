"""Tests for SimulatedFeed — GBM generation and CSV replay."""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime


from data.buffer import LiveBuffer
from data.sim_feed import SimulatedFeed


class TestGBMGeneration:
    """Test synthetic candle generation via _generate_synthetic."""

    async def test_generates_candles_to_buffer(self):
        config = {
            "symbols": ["BTC/USDT"],
            "paper": {"speed_multiplier": 100000.0},
        }
        buffer = LiveBuffer(max_candles=100)
        feed = SimulatedFeed(config, buffer)

        task = asyncio.create_task(feed.start())
        await asyncio.sleep(0.05)
        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        candles = await buffer.get_candles("BTC/USDT")
        assert len(candles) > 0

    async def test_multi_symbol(self):
        config = {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "paper": {"speed_multiplier": 100000.0},
        }
        buffer = LiveBuffer(max_candles=100)
        feed = SimulatedFeed(config, buffer)

        task = asyncio.create_task(feed.start())
        await asyncio.sleep(0.05)
        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        btc = await buffer.get_candles("BTC/USDT")
        eth = await buffer.get_candles("ETH/USDT")
        assert len(btc) > 0
        assert len(eth) > 0

    async def test_candle_prices_positive(self):
        config = {
            "symbols": ["BTC/USDT"],
            "paper": {"speed_multiplier": 100000.0},
        }
        buffer = LiveBuffer(max_candles=50)
        feed = SimulatedFeed(config, buffer)

        task = asyncio.create_task(feed.start())
        await asyncio.sleep(0.05)
        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        for c in await buffer.get_candles("BTC/USDT"):
            assert c.open > 0
            assert c.high > 0
            assert c.low > 0
            assert c.close > 0
            assert c.volume > 0

    async def test_unknown_symbol_uses_defaults(self):
        config = {
            "symbols": ["DOGE/USDT"],
            "paper": {"speed_multiplier": 100000.0},
        }
        buffer = LiveBuffer(max_candles=50)
        feed = SimulatedFeed(config, buffer)

        task = asyncio.create_task(feed.start())
        await asyncio.sleep(0.05)
        await feed.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        candles = await buffer.get_candles("DOGE/USDT")
        assert len(candles) > 0


class TestCSVReplay:
    """Test CSV file replay."""

    async def test_replay_csv(self, tmp_path):
        csv_file = tmp_path / "replay.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "symbol",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            writer.writeheader()
            for i in range(5):
                writer.writerow(
                    {
                        "symbol": "BTC/USDT",
                        "timestamp": datetime(2025, 1, 1, 0, i).isoformat(),
                        "open": 100 + i,
                        "high": 102 + i,
                        "low": 99 + i,
                        "close": 101 + i,
                        "volume": 10,
                    }
                )

        config = {
            "symbols": ["BTC/USDT"],
            "paper": {"replay_file": str(csv_file), "speed_multiplier": 100000.0},
        }
        buffer = LiveBuffer(max_candles=50)
        feed = SimulatedFeed(config, buffer)
        await feed.start()

        candles = await buffer.get_candles("BTC/USDT")
        assert len(candles) == 5

    async def test_stop_during_replay(self, tmp_path):
        csv_file = tmp_path / "big_replay.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "symbol",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            writer.writeheader()
            for i in range(1000):
                writer.writerow(
                    {
                        "symbol": "BTC/USDT",
                        "timestamp": datetime(2025, 1, 1, i // 60, i % 60).isoformat(),
                        "open": 100,
                        "high": 102,
                        "low": 99,
                        "close": 101,
                        "volume": 10,
                    }
                )

        config = {
            "symbols": ["BTC/USDT"],
            "paper": {"replay_file": str(csv_file), "speed_multiplier": 10.0},
        }
        buffer = LiveBuffer(max_candles=2000)
        feed = SimulatedFeed(config, buffer)

        async def stop_soon():
            await asyncio.sleep(0.05)
            await feed.stop()

        await asyncio.gather(feed.start(), stop_soon())
        candles = await buffer.get_candles("BTC/USDT")
        assert len(candles) < 1000
