"""Tests for models/train.py — dataset building, synthetic generation, conversions."""
from __future__ import annotations

import numpy as np
import pytest

from features.extractor import FeatureExtractor
from models.train import build_dataset, generate_synthetic_ohlcv, raw_to_ohlcv


@pytest.fixture
def extractor():
    return FeatureExtractor({
        "rsi_period": 14, "ema_fast": 12, "ema_slow": 26,
        "atr_period": 14, "volatility_window": 20, "momentum_window": 10,
    })


class TestRawToOhlcv:
    def test_converts_raw_lists(self):
        raw = [[1704067200000, 100.0, 105.0, 95.0, 102.0, 50.0]]
        candles = raw_to_ohlcv(raw, "BTC/USDT")
        assert len(candles) == 1
        assert candles[0].symbol == "BTC/USDT"
        assert candles[0].close == 102.0
        assert candles[0].is_closed is True

    def test_multiple_rows(self):
        raw = [
            [1704067200000, 100.0, 105.0, 95.0, 102.0, 50.0],
            [1704067260000, 102.0, 106.0, 101.0, 104.0, 60.0],
        ]
        candles = raw_to_ohlcv(raw, "ETH/USDT")
        assert len(candles) == 2
        assert candles[1].open == 102.0

    def test_empty_input(self):
        candles = raw_to_ohlcv([], "BTC/USDT")
        assert candles == []


class TestBuildDataset:
    def test_insufficient_candles_raises(self, extractor):
        candles = generate_synthetic_ohlcv("BTC/USDT", n_candles=10, seed=1)
        with pytest.raises(ValueError, match="Not enough candles"):
            build_dataset(candles, extractor, seq_len=30, forward_window=5)

    def test_labels_in_valid_range(self, extractor):
        candles = generate_synthetic_ohlcv("BTC/USDT", n_candles=200, seed=42)
        X, y = build_dataset(candles, extractor)
        assert np.all(y >= -1.0) and np.all(y <= 1.0)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_output_shapes(self, extractor):
        candles = generate_synthetic_ohlcv("BTC/USDT", n_candles=200, seed=42)
        X, y = build_dataset(candles, extractor, seq_len=30, forward_window=5)
        assert X.ndim == 3
        assert X.shape[1] == 30  # seq_len
        assert X.shape[2] == 10  # n_features
        assert y.shape == (len(X), 1)

    def test_custom_forward_window(self, extractor):
        candles = generate_synthetic_ohlcv("BTC/USDT", n_candles=200, seed=42)
        X1, _ = build_dataset(candles, extractor, forward_window=3)
        X2, _ = build_dataset(candles, extractor, forward_window=10)
        # More forward window = fewer samples
        assert len(X1) > len(X2)


class TestGenerateSyntheticOhlcv:
    def test_correct_count(self):
        candles = generate_synthetic_ohlcv("BTC/USDT", n_candles=500, seed=1)
        assert len(candles) == 500

    def test_symbol_preserved(self):
        candles = generate_synthetic_ohlcv("ETH/USDT", n_candles=10, seed=1)
        assert all(c.symbol == "ETH/USDT" for c in candles)

    def test_all_closed(self):
        candles = generate_synthetic_ohlcv("BTC/USDT", n_candles=10, seed=1)
        assert all(c.is_closed for c in candles)

    def test_high_gte_low(self):
        candles = generate_synthetic_ohlcv("BTC/USDT", n_candles=1000, seed=42)
        assert all(c.high >= c.low for c in candles)

    def test_timestamps_increasing(self):
        candles = generate_synthetic_ohlcv("BTC/USDT", n_candles=100, seed=1)
        for i in range(1, len(candles)):
            assert candles[i].timestamp > candles[i - 1].timestamp
