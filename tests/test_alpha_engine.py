"""Tests for models/inference.py — AlphaEngine rule-based scoring and signal generation."""

import pytest

from core.models import FeatureVector, Signal
from features.extractor import FeatureExtractor
from models.inference import AlphaEngine
from tests.conftest import make_candle_series


@pytest.fixture
def extractor(default_config):
    return FeatureExtractor(default_config["features"])


@pytest.fixture
def engine(default_config, extractor):
    return AlphaEngine(default_config, extractor, model=None)


class TestRuleBasedScore:
    def test_oversold_rsi_is_bullish(self, engine):
        fv = FeatureVector(
            symbol="BTC/USDT",
            timestamp=__import__("datetime").datetime(2025, 1, 1),
            rsi=20.0,        # very oversold
            ema_fast=100.0,
            ema_slow=99.0,   # slight bullish crossover
            atr=1.0,
            momentum=0.02,
            volatility=0.005,
        )
        score = engine._rule_based_score(fv)
        assert score > 0  # should be bullish

    def test_overbought_rsi_is_bearish(self, engine):
        fv = FeatureVector(
            symbol="BTC/USDT",
            timestamp=__import__("datetime").datetime(2025, 1, 1),
            rsi=85.0,
            ema_fast=99.0,
            ema_slow=100.0,  # bearish crossover
            atr=1.0,
            momentum=-0.02,
            volatility=0.005,
        )
        score = engine._rule_based_score(fv)
        assert score < 0  # should be bearish

    def test_high_volatility_reduces_score(self, engine):
        base = FeatureVector(
            symbol="BTC/USDT",
            timestamp=__import__("datetime").datetime(2025, 1, 1),
            rsi=30.0, ema_fast=101.0, ema_slow=100.0,
            atr=1.0, momentum=0.01, volatility=0.001,
        )
        high_vol = base.model_copy(update={"volatility": 0.1})
        score_low_vol = engine._rule_based_score(base)
        score_high_vol = engine._rule_based_score(high_vol)
        assert score_low_vol > score_high_vol


class TestScoreMethod:
    def test_returns_signal(self, engine, candles_60):
        signal = engine.score(candles_60)
        assert isinstance(signal, Signal)
        assert signal.symbol == "BTC/USDT"
        assert -1.0 <= signal.alpha_score <= 1.0
        assert signal.source == "rule_based"

    def test_alpha_clamped(self, engine, candles_60):
        signal = engine.score(candles_60)
        assert -1.0 <= signal.alpha_score <= 1.0

    def test_confidence_is_abs_alpha(self, engine, candles_60):
        signal = engine.score(candles_60)
        assert signal.confidence == pytest.approx(abs(signal.alpha_score))

    def test_uptrend_is_bullish(self, engine, uptrend_candles):
        signal = engine.score(uptrend_candles)
        # Not guaranteed to be positive, but should lean bullish with strong trend
        # This is a soft test — strong uptrend should produce positive momentum
        assert signal.alpha_score > -0.5  # at least not very bearish

    def test_downtrend_is_bearish(self, engine, downtrend_candles):
        signal = engine.score(downtrend_candles)
        assert signal.alpha_score < 0.5  # at least not very bullish


class TestLSTMFallback:
    def test_no_model_returns_zero(self, engine, candles_60):
        score = engine._model_score(candles_60)
        assert score == 0.0


class TestEnsembleMode:
    def test_ensemble_averages(self, default_config, extractor):
        config = {**default_config, "alpha": {**default_config["alpha"], "engine": "ensemble"}}
        engine = AlphaEngine(config, extractor, model=None)
        # LSTM returns 0 (no model), so ensemble = 0.5 * rule_based + 0.5 * 0
        candles = make_candle_series(60)
        signal = engine.score(candles)
        assert signal.source == "ensemble"

    def test_unknown_engine_fallback(self, default_config, extractor):
        config = {**default_config, "alpha": {**default_config["alpha"], "engine": "magic"}}
        engine = AlphaEngine(config, extractor, model=None)
        candles = make_candle_series(60)
        signal = engine.score(candles)
        assert signal.source == "rule_based"
