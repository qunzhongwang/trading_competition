"""Per-symbol Bayesian ICIR tracker with online shrinkage updates.

Each of the 65+ symbols gets its own weight vector for the rule-based
alpha factors (RSI, momentum, EMA, volatility). Offline priors are
loaded from a JSON file; during the competition, Bayesian shrinkage
continuously adapts toward online observations.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default fallback weights matching the hardcoded rule-based scorer
DEFAULT_WEIGHTS = [0.3, 0.3, 0.3, 0.1]


class _SymbolICIR:
    """Tracks rolling IC for a single symbol's factors."""

    def __init__(
        self,
        prior: List[float],
        window: int,
        min_samples: int,
        min_lambda: float,
        tau: float,
        n_factors: int,
    ):
        self._prior = list(prior)
        self._window = window
        self._min_samples = min_samples
        self._min_lambda = min_lambda
        self._tau = tau
        self._n_factors = n_factors
        # Rolling buffers: each entry is (factor_scores, forward_return)
        self._factor_history: deque = deque(maxlen=window)
        self._return_history: deque = deque(maxlen=window)
        self._n_samples = 0

    def record(self, factor_scores: List[float], forward_return: float) -> None:
        self._factor_history.append(list(factor_scores))
        self._return_history.append(forward_return)
        self._n_samples += 1

    def get_weights(self) -> List[float]:
        """Return Bayesian-shrunk weights."""
        n = self._n_samples

        # Not enough samples — pure prior
        if n < self._min_samples or len(self._factor_history) < self._min_samples:
            return list(self._prior)

        # Compute online IC (Spearman rank correlation approximation using Pearson on ranks)
        online_weights = self._compute_online_icir()
        if online_weights is None:
            return list(self._prior)

        # Bayesian shrinkage: lambda decays toward min_lambda
        lam = self._min_lambda + (1.0 - self._min_lambda) * math.exp(-n / self._tau)

        # Blend: lambda * prior + (1 - lambda) * online
        weights = [
            lam * p + (1.0 - lam) * o for p, o in zip(self._prior, online_weights)
        ]

        # Normalize to sum to 1 (absolute values, since vol is a penalty)
        total = sum(abs(w) for w in weights)
        if total > 1e-10:
            weights = [abs(w) / total for w in weights]

        return weights

    def _compute_online_icir(self) -> Optional[List[float]]:
        """Compute ICIR-based weights from rolling factor/return history."""
        n = len(self._factor_history)
        if n < 2:
            return None

        factors = list(self._factor_history)
        returns = list(self._return_history)

        # Compute IC (correlation) per factor
        ics = []
        for f_idx in range(self._n_factors):
            f_vals = [factors[i][f_idx] for i in range(n)]
            ic = _pearson_correlation(f_vals, returns)
            ics.append(ic)

        # Compute ICIR = mean(IC) / std(IC) using rolling windows
        # For simplicity with a single IC estimate, use IC magnitude as weight proxy
        abs_ics = [abs(ic) for ic in ics]
        total = sum(abs_ics)
        if total < 1e-10:
            return None

        return [ic / total for ic in abs_ics]


class BayesianICIRTracker:
    """Per-symbol Bayesian ICIR tracker with online shrinkage updates."""

    def __init__(
        self,
        prior_weights: Dict[str, dict],
        n_factors: int = 4,
        window: int = 100,
        min_samples: int = 30,
        min_lambda: float = 0.3,
        tau: float = 50.0,
    ):
        self._prior_weights = prior_weights
        self._n_factors = n_factors
        self._window = window
        self._min_samples = min_samples
        self._min_lambda = min_lambda
        self._tau = tau
        self._trackers: Dict[str, _SymbolICIR] = {}

    def _get_tracker(self, symbol: str) -> _SymbolICIR:
        if symbol not in self._trackers:
            # Load prior from file, or use default [0.3, 0.3, 0.3, 0.1]
            prior_dict = self._prior_weights.get(symbol, {})
            if prior_dict:
                prior = [
                    prior_dict.get("rsi", DEFAULT_WEIGHTS[0]),
                    prior_dict.get("momentum", DEFAULT_WEIGHTS[1]),
                    prior_dict.get("ema", DEFAULT_WEIGHTS[2]),
                    prior_dict.get("vol", DEFAULT_WEIGHTS[3]),
                ]
            else:
                prior = list(DEFAULT_WEIGHTS)

            self._trackers[symbol] = _SymbolICIR(
                prior=prior,
                window=self._window,
                min_samples=self._min_samples,
                min_lambda=self._min_lambda,
                tau=self._tau,
                n_factors=self._n_factors,
            )
        return self._trackers[symbol]

    def record(
        self, symbol: str, factor_scores: List[float], forward_return: float
    ) -> None:
        """Record one observation for a specific symbol."""
        self._get_tracker(symbol).record(factor_scores, forward_return)

    def get_weights(self, symbol: str) -> List[float]:
        """Return Bayesian-shrunk weights for a symbol."""
        return self._get_tracker(symbol).get_weights()


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-15:
        return 0.0

    return cov / denom
