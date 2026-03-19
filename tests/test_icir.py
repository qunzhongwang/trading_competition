"""Tests for models/icir_tracker.py — BayesianICIRTracker."""
from __future__ import annotations

import pytest

from models.icir_tracker import BayesianICIRTracker, _pearson_correlation


class TestICIRFallback:
    def test_no_priors_returns_default_weights(self):
        tracker = BayesianICIRTracker(prior_weights={}, n_factors=4)
        weights = tracker.get_weights("BTC/USDT")
        assert weights == [0.3, 0.3, 0.3, 0.1]

    def test_missing_symbol_returns_default(self):
        priors = {"ETH/USDT": {"rsi": 0.4, "momentum": 0.3, "ema": 0.2, "vol": 0.1}}
        tracker = BayesianICIRTracker(prior_weights=priors)
        weights = tracker.get_weights("BTC/USDT")
        assert weights == [0.3, 0.3, 0.3, 0.1]

    def test_with_priors_returns_prior_weights(self):
        priors = {"BTC/USDT": {"rsi": 0.4, "momentum": 0.2, "ema": 0.3, "vol": 0.1}}
        tracker = BayesianICIRTracker(prior_weights=priors)
        weights = tracker.get_weights("BTC/USDT")
        assert weights == [0.4, 0.2, 0.3, 0.1]


class TestICIRPerSymbolIsolation:
    def test_different_symbols_independent(self):
        tracker = BayesianICIRTracker(prior_weights={})
        # Record data only for BTC
        for i in range(50):
            tracker.record("BTC/USDT", [0.5, 0.3, 0.1, 0.1], 0.01 * (i % 3 - 1))

        btc_weights = tracker.get_weights("BTC/USDT")
        eth_weights = tracker.get_weights("ETH/USDT")

        # ETH should still be at default (no data recorded)
        assert eth_weights == [0.3, 0.3, 0.3, 0.1]
        # BTC might have evolved
        assert len(btc_weights) == 4


class TestICIRShrinkageDecay:
    def test_few_samples_returns_prior(self):
        priors = {"BTC/USDT": {"rsi": 0.4, "momentum": 0.2, "ema": 0.3, "vol": 0.1}}
        tracker = BayesianICIRTracker(
            prior_weights=priors, min_samples=30, min_lambda=0.3,
        )
        # Only 5 samples — should return pure prior
        for i in range(5):
            tracker.record("BTC/USDT", [0.5, 0.3, 0.1, 0.1], 0.01)

        weights = tracker.get_weights("BTC/USDT")
        assert weights == [0.4, 0.2, 0.3, 0.1]

    def test_many_samples_shifts_from_prior(self):
        priors = {"BTC/USDT": {"rsi": 0.25, "momentum": 0.25, "ema": 0.25, "vol": 0.25}}
        tracker = BayesianICIRTracker(
            prior_weights=priors, min_samples=10, min_lambda=0.3, tau=10.0,
        )
        # Record many samples where factor 0 (RSI) dominates
        import random
        random.seed(42)
        for i in range(200):
            # RSI signal strongly correlated with returns
            rsi = random.gauss(0, 1)
            ret = rsi * 0.1 + random.gauss(0, 0.01)
            tracker.record("BTC/USDT", [rsi, random.gauss(0, 0.1), random.gauss(0, 0.1), random.gauss(0, 0.1)], ret)

        weights = tracker.get_weights("BTC/USDT")
        # RSI weight should be highest
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        assert _pearson_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert _pearson_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_zero_variance(self):
        assert _pearson_correlation([1, 1, 1], [1, 2, 3]) == pytest.approx(0.0)
