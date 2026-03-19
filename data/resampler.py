"""Live candle resampler for converting 1-min candles to N-min bars.

Used during live/paper trading to gate alpha scoring on completed
N-minute bars while still processing 1-min candles for price updates
and stop checks.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from core.models import OHLCV

logger = logging.getLogger(__name__)


class CandleResampler:
    """Accumulates 1-min candles and emits completed N-min bars."""

    def __init__(self, minutes: int = 5):
        if minutes < 1:
            raise ValueError(f"minutes must be >= 1, got {minutes}")
        self._minutes = minutes
        self._pending: Dict[str, List[OHLCV]] = {}

    @property
    def minutes(self) -> int:
        return self._minutes

    def _floor_ts(self, ts) -> int:
        """Floor timestamp to nearest N-minute boundary. Returns total minutes since midnight."""
        total_min = ts.hour * 60 + ts.minute
        return (total_min // self._minutes) * self._minutes

    def push(self, candle: OHLCV) -> Optional[OHLCV]:
        """Push a 1-min candle. Returns a completed N-min candle or None.

        If minutes=1, passes through directly (no buffering).
        """
        if self._minutes <= 1:
            return candle

        sym = candle.symbol
        bucket = self._pending.get(sym, [])

        if bucket:
            # Check if this candle belongs to a new bucket
            prev_floor = self._floor_ts(bucket[0].timestamp)
            cur_floor = self._floor_ts(candle.timestamp)
            if cur_floor != prev_floor:
                # New bucket — emit old if complete, start new
                result = self._emit(bucket) if len(bucket) == self._minutes else None
                self._pending[sym] = [candle]
                return result

        bucket.append(candle)
        self._pending[sym] = bucket

        # Emit if bucket is full
        if len(bucket) == self._minutes:
            self._pending[sym] = []
            return self._emit(bucket)

        return None

    def _emit(self, bucket: List[OHLCV]) -> OHLCV:
        """Aggregate a full bucket into one N-min candle."""
        return OHLCV(
            symbol=bucket[0].symbol,
            open=bucket[0].open,
            high=max(c.high for c in bucket),
            low=min(c.low for c in bucket),
            close=bucket[-1].close,
            volume=sum(c.volume for c in bucket),
            timestamp=bucket[-1].timestamp,
            is_closed=True,
        )


class MultiResampler:
    """Wraps multiple CandleResamplers for multi-timeframe support.

    Push a 1-min candle once, get back completed bars for each configured period.
    """

    def __init__(self, periods: List[int]):
        if not periods:
            raise ValueError("periods must be non-empty")
        self._resamplers: Dict[int, CandleResampler] = {
            p: CandleResampler(p) for p in periods
        }

    def push(self, candle: OHLCV) -> Dict[int, Optional[OHLCV]]:
        """Push a 1-min candle. Returns {period: completed_bar_or_None} for each period."""
        return {p: r.push(candle) for p, r in self._resamplers.items()}

    @property
    def primary_minutes(self) -> int:
        """The smallest period (used for alpha gating)."""
        return min(self._resamplers.keys())

    @property
    def periods(self) -> List[int]:
        return sorted(self._resamplers.keys())
