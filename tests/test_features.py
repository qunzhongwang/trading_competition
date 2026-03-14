"""Tests for features/extractor.py — RSI, EMA, ATR, momentum, volatility, sequences."""

import numpy as np
import pytest

from features.extractor import FeatureExtractor
from tests.conftest import make_candle, make_candle_series


@pytest.fixture
def extractor(default_config):
    return FeatureExtractor(default_config["features"])


class TestRSI:
    def test_all_gains(self, extractor):
        closes = [float(i) for i in range(1, 20)]  # strictly rising
        rsi = extractor.compute_rsi(closes, 14)
        assert rsi == 100.0

    def test_all_losses(self, extractor):
        closes = [float(20 - i) for i in range(20)]  # strictly falling
        rsi = extractor.compute_rsi(closes, 14)
        assert rsi == pytest.approx(0.0, abs=0.01)

    def test_neutral(self, extractor):
        # Alternating +1 / -1 → roughly 50
        closes = [100 + (1 if i % 2 == 0 else -1) for i in range(20)]
        rsi = extractor.compute_rsi(closes, 14)
        assert 40 < rsi < 60

    def test_too_few_candles(self, extractor):
        assert extractor.compute_rsi([100, 101], 14) == 50.0


class TestEMA:
    def test_constant_series(self, extractor):
        closes = [50.0] * 30
        ema = extractor.compute_ema(closes, 12)
        assert ema == pytest.approx(50.0, abs=0.01)

    def test_rising_series(self, extractor):
        closes = [float(i) for i in range(1, 30)]
        ema = extractor.compute_ema(closes, 12)
        # EMA should lag behind the latest close but be above the midpoint
        assert closes[-1] > ema > np.mean(closes)

    def test_too_few_values(self, extractor):
        assert extractor.compute_ema([42.0], 12) == 42.0
        assert extractor.compute_ema([], 12) == 0.0


class TestATR:
    def test_constant_candles(self, extractor):
        candles = [make_candle(close=100, high=100, low=100, open_=100) for _ in range(20)]
        atr = extractor.compute_atr(candles, 14)
        assert atr == pytest.approx(0.0, abs=0.01)

    def test_volatile_candles(self, extractor):
        candles = [make_candle(close=100, high=110, low=90) for _ in range(20)]
        atr = extractor.compute_atr(candles, 14)
        assert atr > 0

    def test_too_few_candles(self, extractor):
        candles = [make_candle() for _ in range(3)]
        assert extractor.compute_atr(candles, 14) == 0.0


class TestMomentum:
    def test_rising(self, extractor):
        closes = [100.0] * 10 + [110.0]
        mom = extractor.compute_momentum(closes, 10)
        assert mom == pytest.approx(0.1, abs=0.001)

    def test_flat(self, extractor):
        closes = [100.0] * 15
        assert extractor.compute_momentum(closes, 10) == pytest.approx(0.0)

    def test_zero_denominator(self, extractor):
        closes = [0.0] * 15
        assert extractor.compute_momentum(closes, 10) == 0.0


class TestVolatility:
    def test_constant(self, extractor):
        closes = [100.0] * 25
        vol = extractor.compute_volatility(closes, 20)
        assert vol == pytest.approx(0.0, abs=1e-9)

    def test_positive(self, extractor):
        closes = [100 + (i % 5) * 2 for i in range(25)]
        vol = extractor.compute_volatility(closes, 20)
        assert vol > 0


class TestExtract:
    def test_returns_feature_vector(self, extractor, candles_60):
        fv = extractor.extract(candles_60)
        assert fv.symbol == "BTC/USDT"
        assert 0 <= fv.rsi <= 100
        assert fv.ema_fast > 0
        assert fv.ema_slow > 0
        assert fv.atr >= 0
        assert fv.volatility >= 0

    def test_too_few_candles_returns_zeros(self, extractor):
        candles = [make_candle() for _ in range(3)]
        fv = extractor.extract(candles)
        assert fv.rsi == 0.0
        assert fv.atr == 0.0

    def test_min_candles_property(self, extractor):
        assert extractor.min_candles >= 14  # at least max(rsi, ema_slow, atr) + 2


class TestExtractSequence:
    def test_shape(self, extractor, candles_120):
        seq = extractor.extract_sequence(candles_120, seq_len=30)
        assert seq.shape == (30, 6)
        assert seq.dtype == np.float32

    def test_z_normalized(self, extractor, candles_120):
        seq = extractor.extract_sequence(candles_120, seq_len=30)
        # Each column should have mean ≈ 0, std ≈ 1
        for col in range(seq.shape[1]):
            assert abs(seq[:, col].mean()) < 0.5
            assert 0.5 < seq[:, col].std() < 2.0

    def test_too_few_candles_returns_zeros(self, extractor):
        candles = [make_candle() for _ in range(5)]
        seq = extractor.extract_sequence(candles, seq_len=30)
        assert seq.shape == (30, 6)
        assert np.all(seq == 0)
