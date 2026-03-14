from __future__ import annotations

import logging
import math
from typing import List

import numpy as np

from core.models import OHLCV, FeatureVector

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Stateless feature extractor: list[OHLCV] → FeatureVector.

    All computations use pure numpy for speed. No pandas in the hot path.
    """

    # Feature names in the order used by extract_sequence()
    FEATURE_NAMES = ["rsi", "ema_fast", "ema_slow", "atr", "momentum", "volatility"]
    N_FEATURES = len(FEATURE_NAMES)

    def __init__(self, config: dict):
        self._rsi_period: int = config.get("rsi_period", 14)
        self._ema_fast: int = config.get("ema_fast", 12)
        self._ema_slow: int = config.get("ema_slow", 26)
        self._atr_period: int = config.get("atr_period", 14)
        self._momentum_window: int = config.get("momentum_window", 10)
        self._volatility_window: int = config.get("volatility_window", 20)

    @property
    def min_candles(self) -> int:
        """Minimum candles needed to compute all features."""
        return max(self._rsi_period, self._ema_slow, self._atr_period,
                   self._momentum_window, self._volatility_window) + 2

    def extract(self, candles: List[OHLCV]) -> FeatureVector:
        """Compute features for the most recent candle."""
        if len(candles) < self.min_candles:
            logger.warning(
                "Not enough candles (%d < %d), returning zeros",
                len(candles), self.min_candles,
            )
            return FeatureVector(
                symbol=candles[-1].symbol if candles else "",
                timestamp=candles[-1].timestamp if candles else __import__("datetime").datetime.utcnow(),
            )

        closes = [c.close for c in candles]

        return FeatureVector(
            symbol=candles[-1].symbol,
            timestamp=candles[-1].timestamp,
            rsi=self.compute_rsi(closes, self._rsi_period),
            ema_fast=self.compute_ema(closes, self._ema_fast),
            ema_slow=self.compute_ema(closes, self._ema_slow),
            atr=self.compute_atr(candles, self._atr_period),
            momentum=self.compute_momentum(closes, self._momentum_window),
            volatility=self.compute_volatility(closes, self._volatility_window),
        )

    def extract_sequence(self, candles: List[OHLCV], seq_len: int = 30) -> np.ndarray:
        """Extract a (seq_len, n_features) normalized array for LSTM input.

        Computes features at each timestep in the window, then z-score normalizes
        across the sequence.
        """
        total_needed = self.min_candles + seq_len
        if len(candles) < total_needed:
            logger.warning(
                "Not enough candles for sequence (%d < %d), padding with zeros",
                len(candles), total_needed,
            )
            # Return zero-padded array
            return np.zeros((seq_len, self.N_FEATURES), dtype=np.float32)

        features = []
        for i in range(seq_len):
            # Window ending at candle[-(seq_len - i)]
            end_idx = len(candles) - (seq_len - 1 - i)
            window = candles[:end_idx]
            fv = self.extract(window)
            features.append([
                fv.rsi, fv.ema_fast, fv.ema_slow,
                fv.atr, fv.momentum, fv.volatility,
            ])

        arr = np.array(features, dtype=np.float32)

        # Z-score normalize each feature across the sequence
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # avoid div by zero
        arr = (arr - mean) / std

        return arr

    @staticmethod
    def compute_rsi(closes: List[float], period: int) -> float:
        """Relative Strength Index."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss < 1e-10:
            return 100.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def compute_ema(values: List[float], period: int) -> float:
        """Exponential Moving Average (last value)."""
        if len(values) < period:
            return values[-1] if values else 0.0

        arr = np.array(values, dtype=np.float64)
        multiplier = 2.0 / (period + 1)
        ema = arr[0]
        for val in arr[1:]:
            ema = val * multiplier + ema * (1 - multiplier)
        return float(ema)

    @staticmethod
    def compute_atr(candles: List[OHLCV], period: int) -> float:
        """Average True Range."""
        if len(candles) < period + 1:
            return 0.0

        recent = candles[-(period + 1):]
        true_ranges = []
        for i in range(1, len(recent)):
            high = recent[i].high
            low = recent[i].low
            prev_close = recent[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        return float(np.mean(true_ranges))

    @staticmethod
    def compute_momentum(closes: List[float], window: int) -> float:
        """Rate of change: (close[-1] / close[-window]) - 1."""
        if len(closes) < window + 1:
            return 0.0
        if closes[-(window + 1)] == 0:
            return 0.0
        return (closes[-1] / closes[-(window + 1)]) - 1.0

    @staticmethod
    def compute_volatility(closes: List[float], window: int) -> float:
        """Standard deviation of log returns over window."""
        if len(closes) < window + 1:
            return 0.0

        prices = np.array(closes[-(window + 1):], dtype=np.float64)
        prices = np.where(prices <= 0, 1e-10, prices)  # safety
        log_returns = np.diff(np.log(prices))
        return float(np.std(log_returns))
