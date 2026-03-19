from __future__ import annotations

import asyncio
import logging
import time
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
        self._depth_data: Dict[str, dict] = {}  # latest order book imbalance per symbol
        self._funding_data: Dict[str, float] = {}  # latest funding rate per symbol
        self._taker_data: Dict[str, float] = {}  # latest taker ratio per symbol
        # Per-candle supplementary history (mirrors candle deque size)
        self._obi_history: Dict[str, deque] = {}
        self._funding_history: Dict[str, deque] = {}
        self._taker_history: Dict[str, deque] = {}
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()
        # Resampled candle storage: {minutes: {symbol: deque[OHLCV]}}
        self._resampled: Dict[int, Dict[str, deque]] = {}
        # Staleness tracking: symbol → monotonic timestamp of last candle push
        self._last_candle_time: Dict[str, float] = {}
        # Supplementary data staleness
        self._supp_last_update: Dict[str, float] = {}
        self._supp_stale_threshold: float = 300.0  # 5 minutes

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
            self._last_candle_time[candle.symbol] = time.monotonic()
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
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            self._event.clear()  # clear AFTER waking — avoids losing signals set during processing
            return True
        except asyncio.TimeoutError:
            self._event.clear()
            return False

    async def push_depth(self, symbol: str, bids: list, asks: list) -> None:
        """Store order book imbalance from depth snapshot."""
        async with self._lock:
            bid_volume = sum(float(b[1]) for b in bids[:10]) if bids else 0.0
            ask_volume = sum(float(a[1]) for a in asks[:10]) if asks else 0.0
            imbalance = bid_volume / ask_volume if ask_volume > 1e-10 else 1.0
            self._depth_data[symbol] = {"order_book_imbalance": imbalance}
            if symbol not in self._obi_history:
                self._obi_history[symbol] = deque(maxlen=self._max_candles)
            self._obi_history[symbol].append(imbalance)
            self._supp_last_update[f"{symbol}:depth"] = time.monotonic()

    async def push_funding(self, symbol: str, rate: float) -> None:
        """Store latest funding rate."""
        async with self._lock:
            self._funding_data[symbol] = rate
            if symbol not in self._funding_history:
                self._funding_history[symbol] = deque(maxlen=self._max_candles)
            self._funding_history[symbol].append(rate)
            self._supp_last_update[f"{symbol}:funding"] = time.monotonic()

    async def push_taker_ratio(self, symbol: str, ratio: float) -> None:
        """Store latest taker buy/sell ratio."""
        async with self._lock:
            self._taker_data[symbol] = ratio
            if symbol not in self._taker_history:
                self._taker_history[symbol] = deque(maxlen=self._max_candles)
            self._taker_history[symbol].append(ratio)
            self._supp_last_update[f"{symbol}:taker"] = time.monotonic()

    async def get_supplementary(self, symbol: str) -> dict:
        """Get all supplementary data for a symbol."""
        async with self._lock:
            # Warn if supplementary data is stale
            now = time.monotonic()
            for key_suffix in ("depth", "funding", "taker"):
                key = f"{symbol}:{key_suffix}"
                last = self._supp_last_update.get(key)
                if last is not None and (now - last) > self._supp_stale_threshold:
                    logger.warning(
                        "Stale supplementary data for %s: %s last updated %.0fs ago",
                        symbol,
                        key_suffix,
                        now - last,
                    )
            depth = self._depth_data.get(symbol, {})
            return {
                "order_book_imbalance": depth.get("order_book_imbalance", 0.0),
                "funding_rate": self._funding_data.get(symbol, 0.0),
                "taker_ratio": self._taker_data.get(symbol, 0.0),
            }

    async def get_supplementary_history(self, symbol: str, n: int) -> dict:
        """Get last n values of each supplementary feature as lists."""
        async with self._lock:
            obi = list(self._obi_history.get(symbol, deque()))[-n:]
            funding = list(self._funding_history.get(symbol, deque()))[-n:]
            taker = list(self._taker_history.get(symbol, deque()))[-n:]
            return {
                "order_book_imbalance": obi,
                "funding_rate": funding,
                "taker_ratio": taker,
            }

    async def push_resampled(self, minutes: int, candle: OHLCV) -> None:
        """Store a completed resampled candle (e.g. 15m or 1h bar)."""
        async with self._lock:
            if minutes not in self._resampled:
                self._resampled[minutes] = {}
            if candle.symbol not in self._resampled[minutes]:
                self._resampled[minutes][candle.symbol] = deque(
                    maxlen=self._max_candles
                )
            self._resampled[minutes][candle.symbol].append(candle)

    async def get_resampled_candles(
        self, symbol: str, minutes: int, n: int = 50
    ) -> List[OHLCV]:
        """Return last n resampled candles for a symbol at given timeframe."""
        async with self._lock:
            buf = self._resampled.get(minutes, {}).get(symbol, deque())
            if n <= 0 or n >= len(buf):
                return list(buf)
            return list(buf)[-n:]

    def candle_count(self, symbol: str) -> int:
        return len(self._candles.get(symbol, deque()))

    def symbols(self) -> List[str]:
        return list(self._candles.keys())

    def seconds_since_last_candle(self, symbol: str) -> float:
        """Return seconds since the last candle was pushed for this symbol.

        Returns float('inf') if no candle has been received for the symbol.
        """
        last = self._last_candle_time.get(symbol)
        if last is None:
            return float("inf")
        return time.monotonic() - last
