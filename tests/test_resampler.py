"""Tests for data/resampler.py — live CandleResampler."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.models import OHLCV
from data.resampler import CandleResampler


def _candle(symbol: str, minute_offset: int, close: float = 100.0) -> OHLCV:
    """Helper to create a candle at a specific minute offset."""
    return OHLCV(
        symbol=symbol,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=10.0,
        timestamp=datetime(2025, 1, 1, 0, 0) + timedelta(minutes=minute_offset),
        is_closed=True,
    )


class TestCandleResampler:
    def test_passthrough_1min(self):
        r = CandleResampler(minutes=1)
        c = _candle("BTC/USDT", 0)
        result = r.push(c)
        assert result is not None
        assert result.close == c.close

    def test_emits_after_n_candles(self):
        r = CandleResampler(minutes=5)
        results = []
        for i in range(10):
            result = r.push(_candle("BTC/USDT", i, close=100.0 + i))
            if result is not None:
                results.append(result)
        assert len(results) == 2

    def test_ohlcv_correctness(self):
        r = CandleResampler(minutes=5)
        candles = [_candle("BTC/USDT", i, close=100.0 + i) for i in range(5)]
        result = None
        for c in candles:
            result = r.push(c)
        assert result is not None
        assert result.open == candles[0].open
        assert result.close == candles[-1].close
        assert result.high == max(c.high for c in candles)
        assert result.low == min(c.low for c in candles)
        assert abs(result.volume - sum(c.volume for c in candles)) < 1e-6

    def test_multi_symbol(self):
        r = CandleResampler(minutes=5)
        btc_results = []
        eth_results = []
        for i in range(5):
            btc = r.push(_candle("BTC/USDT", i))
            eth = r.push(_candle("ETH/USDT", i))
            if btc:
                btc_results.append(btc)
            if eth:
                eth_results.append(eth)
        assert len(btc_results) == 1
        assert len(eth_results) == 1
        assert btc_results[0].symbol == "BTC/USDT"
        assert eth_results[0].symbol == "ETH/USDT"

    def test_incomplete_bucket_not_emitted(self):
        r = CandleResampler(minutes=5)
        for i in range(3):
            result = r.push(_candle("BTC/USDT", i))
            assert result is None

    def test_invalid_minutes_raises(self):
        with pytest.raises(ValueError):
            CandleResampler(minutes=0)
