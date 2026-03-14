from __future__ import annotations

import asyncio
import csv
import logging
import math
import random
from datetime import datetime, timedelta
from typing import List, Optional

from core.models import OHLCV
from data.buffer import LiveBuffer

logger = logging.getLogger(__name__)


class SimulatedFeed:
    """Replays historical CSV data or generates synthetic GBM candles.

    Drop-in replacement for WSConnector in paper mode. Pushes OHLCV candles
    to the LiveBuffer at accelerated speed.
    """

    def __init__(self, config: dict, buffer: LiveBuffer):
        self._buffer = buffer
        self._symbols: List[str] = config.get("symbols", ["BTC/USDT"])
        paper_cfg = config.get("paper", {})
        self._replay_file: str = paper_cfg.get("replay_file", "")
        self._speed_multiplier: float = paper_cfg.get("speed_multiplier", 60.0)
        self._running = False

        # GBM params per symbol (realistic crypto defaults)
        self._sim_params = {
            "BTC/USDT": {"price": 65000.0, "drift": 0.0001, "vol": 0.002, "base_volume": 50.0},
            "ETH/USDT": {"price": 3500.0, "drift": 0.00015, "vol": 0.003, "base_volume": 500.0},
        }

    async def start(self) -> None:
        self._running = True
        if self._replay_file:
            await self._replay_csv()
        else:
            await self._generate_synthetic()

    async def stop(self) -> None:
        self._running = False

    async def _replay_csv(self) -> None:
        """Replay candles from a CSV file with columns: symbol,timestamp,open,high,low,close,volume"""
        logger.info("Replaying data from %s", self._replay_file)
        interval = 60.0 / self._speed_multiplier  # real seconds per candle

        with open(self._replay_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not self._running:
                    break
                candle = OHLCV(
                    symbol=row["symbol"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    is_closed=True,
                )
                await self._buffer.push_candle(candle)
                await asyncio.sleep(interval)

        logger.info("CSV replay complete")

    async def _generate_synthetic(self) -> None:
        """Generate synthetic candles using geometric Brownian motion."""
        logger.info(
            "Generating synthetic candles for %s at %.0fx speed",
            self._symbols,
            self._speed_multiplier,
        )
        interval = 60.0 / self._speed_multiplier  # real seconds per 1m candle
        now = datetime.utcnow()

        # Initialize prices
        prices = {}
        for sym in self._symbols:
            params = self._sim_params.get(sym, {"price": 1000.0, "drift": 0.0001, "vol": 0.002, "base_volume": 100.0})
            prices[sym] = params["price"]

        candle_idx = 0
        while self._running:
            ts = now + timedelta(minutes=candle_idx)

            for sym in self._symbols:
                params = self._sim_params.get(sym, {"price": 1000.0, "drift": 0.0001, "vol": 0.002, "base_volume": 100.0})
                price = prices[sym]

                # GBM: dS = S * (mu*dt + sigma*dW)
                dt = 1.0 / 1440.0  # 1 minute as fraction of day
                dw = random.gauss(0, 1)
                returns = params["drift"] * dt + params["vol"] * math.sqrt(dt) * dw

                # Generate intra-candle OHLC
                open_price = price
                close_price = price * (1 + returns)

                # High/low with some noise
                intra_vol = abs(returns) + params["vol"] * math.sqrt(dt) * 0.5
                high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, intra_vol)))
                low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, intra_vol)))

                volume = params["base_volume"] * (1 + abs(random.gauss(0, 0.5)))

                candle = OHLCV(
                    symbol=sym,
                    open=round(open_price, 2),
                    high=round(high_price, 2),
                    low=round(low_price, 2),
                    close=round(close_price, 2),
                    volume=round(volume, 4),
                    timestamp=ts,
                    is_closed=True,
                )
                await self._buffer.push_candle(candle)

                prices[sym] = close_price

            candle_idx += 1
            await asyncio.sleep(interval)
