"""Tests for multi-timeframe resampler and alpha filter."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.models import OHLCV
from data.resampler import CandleResampler, MultiResampler


def _candle(symbol: str, minute_offset: int, close: float = 100.0) -> OHLCV:
    return OHLCV(
        symbol=symbol,
        open=close - 1, high=close + 1, low=close - 2, close=close,
        volume=10.0,
        timestamp=datetime(2025, 1, 1, 0, 0) + timedelta(minutes=minute_offset),
        is_closed=True,
    )


class TestMultiResampler:
    def test_returns_dict_for_all_periods(self):
        mr = MultiResampler([5, 15, 60])
        result = mr.push(_candle("BTC/USDT", 0))
        assert isinstance(result, dict)
        assert set(result.keys()) == {5, 15, 60}

    def test_primary_is_smallest(self):
        mr = MultiResampler([15, 5, 60])
        assert mr.primary_minutes == 5

    def test_5min_bar_emits_before_15min(self):
        mr = MultiResampler([5, 15])
        five_min_count = 0
        fifteen_min_count = 0
        for i in range(15):
            result = mr.push(_candle("BTC/USDT", i, close=100.0 + i))
            if result[5] is not None:
                five_min_count += 1
            if result[15] is not None:
                fifteen_min_count += 1

        assert five_min_count == 3  # bars at 5, 10, 15 minutes
        assert fifteen_min_count == 1  # bar at 15 minutes

    def test_empty_periods_raises(self):
        with pytest.raises(ValueError):
            MultiResampler([])

    def test_periods_sorted(self):
        mr = MultiResampler([60, 5, 15])
        assert mr.periods == [5, 15, 60]

    def test_multi_symbol_independence(self):
        mr = MultiResampler([5])
        btc_emitted = 0
        eth_emitted = 0
        for i in range(5):
            r1 = mr.push(_candle("BTC/USDT", i))
            r2 = mr.push(_candle("ETH/USDT", i))
            if r1[5] is not None:
                btc_emitted += 1
            if r2[5] is not None:
                eth_emitted += 1
        assert btc_emitted == 1
        assert eth_emitted == 1


class TestMultiTFFilter:
    """Test the multi-TF filter in AlphaEngine."""

    def test_filter_with_no_data(self):
        from features.extractor import FeatureExtractor
        from models.inference import AlphaEngine
        from tests.conftest import make_candle_series

        config = {"alpha": {"engine": "rule_based"}, "features": {}}
        extractor = FeatureExtractor({})
        engine = AlphaEngine(config, extractor)

        candles = make_candle_series(60)
        # No multi-TF data — should still work
        signal = engine.score(candles, candles_15m=None, candles_1h=None)
        assert -1.0 <= signal.alpha_score <= 1.0

    def test_bullish_filter_boosts(self):
        from models.inference import AlphaEngine
        from features.extractor import FeatureExtractor
        from tests.conftest import make_candle_series

        config = {"alpha": {"engine": "rule_based"}, "features": {}}
        extractor = FeatureExtractor({})
        engine = AlphaEngine(config, extractor)

        # Create strong uptrend 15m candles (EMA fast > slow)
        candles_15m = make_candle_series(30, start_price=100.0, trend=2.0, noise=0.1)
        # Create strong uptrend 1h candles (positive momentum)
        candles_1h = make_candle_series(15, start_price=100.0, trend=5.0, noise=0.1)

        filter_val = engine._multi_tf_filter(candles_15m, candles_1h)
        assert filter_val > 0  # bullish filter
