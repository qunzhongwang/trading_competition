from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Dict, List, Optional

from core.models import OHLCV, Tick

logger = logging.getLogger(__name__)


class LiveBuffer:
    """Thread-safe sliding window for OHLCV candles and raw ticks.

    Uses asyncio.Event to signal new closed candles to consumers (StrategyMonitor).
    The connector calls push_candle(), which sets the event.
    The monitor calls wait_for_update(), which blocks until the event is set.
    """

    def __init__(self, max_candles: int = 500, max_ticks: int = 5000):
        self._max_candles = max_candles
        self._max_ticks = max_ticks
        self._candles: Dict[str, deque] = {}  # symbol → deque[OHLCV]
        self._ticks: Dict[str, deque] = {}  # symbol → deque[Tick]
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    async def push_tick(self, tick: Tick) -> None:
        async with self._lock:
            if tick.symbol not in self._ticks:
                self._ticks[tick.symbol] = deque(maxlen=self._max_ticks)
            self._ticks[tick.symbol].append(tick)

    async def push_candle(self, candle: OHLCV) -> None:
        async with self._lock:
            if candle.symbol not in self._candles:
                self._candles[candle.symbol] = deque(maxlen=self._max_candles)
            self._candles[candle.symbol].append(candle)
        if candle.is_closed:
            self._event.set()
            logger.debug(
                "Candle pushed: %s close=%.2f vol=%.2f",
                candle.symbol,
                candle.close,
                candle.volume,
            )

    async def get_candles(self, symbol: str, n: int = 0) -> List[OHLCV]:
        """Return last n candles for symbol. If n=0, return all."""
        async with self._lock:
            buf = self._candles.get(symbol, deque())
            if n <= 0 or n >= len(buf):
                return list(buf)
            return list(buf)[-n:]

    async def get_latest_candle(self, symbol: str) -> Optional[OHLCV]:
        async with self._lock:
            buf = self._candles.get(symbol, deque())
            return buf[-1] if buf else None

    async def get_ticks(self, symbol: str, n: int = 0) -> List[Tick]:
        async with self._lock:
            buf = self._ticks.get(symbol, deque())
            if n <= 0 or n >= len(buf):
                return list(buf)
            return list(buf)[-n:]

    async def wait_for_update(self, timeout: float = 5.0) -> bool:
        """Block until a new closed candle arrives. Returns False on timeout."""
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def candle_count(self, symbol: str) -> int:
        return len(self._candles.get(symbol, deque()))

    def symbols(self) -> List[str]:
        return list(self._candles.keys())
