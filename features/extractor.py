from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

from core.models import OHLCV, FeatureVector

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Stateless feature extractor: list[OHLCV] → FeatureVector.

    All computations use pure numpy for speed. No pandas in the hot path.
    """

    # Feature names in the order used by extract_sequence()
    FEATURE_NAMES = ["rsi", "ema_fast", "ema_slow", "atr", "momentum", "volatility",
                     "order_book_imbalance", "volume_ratio", "funding_rate", "taker_ratio"]
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

    def extract(self, candles: List[OHLCV], supplementary: Optional[dict] = None) -> FeatureVector:
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
        volume_ratio = self.compute_volume_ratio(candles)
        supp = supplementary or {}

        return FeatureVector(
            symbol=candles[-1].symbol,
            timestamp=candles[-1].timestamp,
            rsi=self.compute_rsi(closes, self._rsi_period),
            ema_fast=self.compute_ema(closes, self._ema_fast),
            ema_slow=self.compute_ema(closes, self._ema_slow),
            atr=self.compute_atr(candles, self._atr_period),
            momentum=self.compute_momentum(closes, self._momentum_window),
            volatility=self.compute_volatility(closes, self._volatility_window),
            order_book_imbalance=supp.get("order_book_imbalance", 0.0),
            volume_ratio=volume_ratio,
            funding_rate=supp.get("funding_rate", 0.0),
            taker_ratio=supp.get("taker_ratio", 0.0),
        )

    def extract_sequence(self, candles: List[OHLCV], seq_len: int = 30,
                         supplementary: Optional[dict] = None,
                         supplementary_history: Optional[dict] = None) -> np.ndarray:
        """Extract a (seq_len, n_features) normalized array for LSTM input.

        Delegates to the vectorized implementation for performance.
        Falls back to the iterative method on error.
        """
        try:
            return self.extract_sequence_vectorized(candles, seq_len, supplementary,
                                                    supplementary_history=supplementary_history)
        except Exception as e:
            logger.warning("Vectorized extraction failed (%s), falling back to iterative", e)
            return self._extract_sequence_iterative(candles, seq_len, supplementary,
                                                    supplementary_history=supplementary_history)

    def _extract_sequence_iterative(self, candles: List[OHLCV], seq_len: int = 30,
                                    supplementary: Optional[dict] = None,
                                    supplementary_history: Optional[dict] = None) -> np.ndarray:
        """Original iterative extraction (fallback)."""
        total_needed = self.min_candles + seq_len
        if len(candles) < total_needed:
            return np.zeros((seq_len, self.N_FEATURES), dtype=np.float32)

        # Build per-timestep supplementary arrays from history
        obi_seq, funding_seq, taker_seq = self._resolve_supplementary_history(
            supplementary, supplementary_history, seq_len)

        features = []
        for i in range(seq_len):
            end_idx = len(candles) - (seq_len - 1 - i)
            window = candles[:end_idx]
            fv = self.extract(window, supplementary=supplementary)
            features.append([
                fv.rsi, fv.ema_fast, fv.ema_slow,
                fv.atr, fv.momentum, fv.volatility,
                obi_seq[i], fv.volume_ratio,
                funding_seq[i], taker_seq[i],
            ])

        arr = np.array(features, dtype=np.float32)
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        arr = (arr - mean) / std
        return arr

    def extract_sequence_vectorized(self, candles: List[OHLCV], seq_len: int = 30,
                                    supplementary: Optional[dict] = None,
                                    supplementary_history: Optional[dict] = None) -> np.ndarray:
        """Vectorized extraction: compute all features in single numpy passes, then slice.

        ~60x faster than iterative for seq_len=60 since each indicator is computed once
        over the full history instead of once per timestep.
        """
        total_needed = self.min_candles + seq_len
        if len(candles) < total_needed:
            return np.zeros((seq_len, self.N_FEATURES), dtype=np.float32)

        n = len(candles)
        closes = np.array([c.close for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.float64)

        # RSI array
        rsi_arr = self._vectorized_rsi(closes, self._rsi_period)
        # EMA arrays
        ema_fast_arr = self._vectorized_ema(closes, self._ema_fast)
        ema_slow_arr = self._vectorized_ema(closes, self._ema_slow)
        # ATR array
        atr_arr = self._vectorized_atr(highs, lows, closes, self._atr_period)
        # Momentum array
        momentum_arr = self._vectorized_momentum(closes, self._momentum_window)
        # Volatility array
        volatility_arr = self._vectorized_volatility(closes, self._volatility_window)
        # Volume ratio array
        volume_ratio_arr = self._vectorized_volume_ratio(volumes, window=24)

        # Supplementary: use per-timestep history if available, else constant
        obi_seq, funding_seq, taker_seq = self._resolve_supplementary_history(
            supplementary, supplementary_history, seq_len)

        # Stack: take last seq_len values from each array
        # All arrays are length n; we take indices [n-seq_len : n]
        sl = slice(n - seq_len, n)
        seq = np.column_stack([
            rsi_arr[sl],
            ema_fast_arr[sl],
            ema_slow_arr[sl],
            atr_arr[sl],
            momentum_arr[sl],
            volatility_arr[sl],
            np.array(obi_seq, dtype=np.float32),
            volume_ratio_arr[sl],
            np.array(funding_seq, dtype=np.float32),
            np.array(taker_seq, dtype=np.float32),
        ]).astype(np.float32)

        # Z-score normalize
        mean = seq.mean(axis=0, keepdims=True)
        std = seq.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        seq = (seq - mean) / std
        return seq

    @staticmethod
    def _resolve_supplementary_history(
        supplementary: Optional[dict],
        supplementary_history: Optional[dict],
        seq_len: int,
    ) -> tuple:
        """Return (obi_seq, funding_seq, taker_seq) lists of length seq_len.

        If history is provided and long enough, use per-timestep values.
        If history is shorter than seq_len, pad the beginning with the earliest value.
        If no history, fall back to broadcasting the scalar from supplementary.
        """
        supp = supplementary or {}
        hist = supplementary_history or {}

        def _resolve(hist_key: str, scalar_key: str) -> list:
            values = hist.get(hist_key, [])
            if values:
                if len(values) >= seq_len:
                    return list(values[-seq_len:])
                # Pad beginning with earliest available value
                pad_val = values[0]
                return [pad_val] * (seq_len - len(values)) + list(values)
            # No history — broadcast scalar
            return [supp.get(scalar_key, 0.0)] * seq_len

        obi_seq = _resolve("order_book_imbalance", "order_book_imbalance")
        funding_seq = _resolve("funding_rate", "funding_rate")
        taker_seq = _resolve("taker_ratio", "taker_ratio")
        return obi_seq, funding_seq, taker_seq

    @staticmethod
    def _vectorized_rsi(closes: np.ndarray, period: int) -> np.ndarray:
        """Compute RSI for every position in the array."""
        n = len(closes)
        rsi = np.full(n, 50.0)
        if n < period + 1:
            return rsi
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        # Simple moving average for initial window, then rolling
        for i in range(period, len(deltas)):
            avg_gain = np.mean(gains[i - period:i])
            avg_loss = np.mean(losses[i - period:i])
            if avg_loss < 1e-10:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
        return rsi

    @staticmethod
    def _vectorized_ema(values: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA for every position in the array."""
        n = len(values)
        ema = np.zeros(n, dtype=np.float64)
        ema[0] = values[0]
        multiplier = 2.0 / (period + 1)
        for i in range(1, n):
            ema[i] = values[i] * multiplier + ema[i - 1] * (1 - multiplier)
        return ema

    @staticmethod
    def _vectorized_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                        period: int) -> np.ndarray:
        """Compute ATR for every position in the array."""
        n = len(closes)
        atr = np.zeros(n, dtype=np.float64)
        if n < 2:
            return atr
        # True ranges
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        # Rolling mean of true range
        for i in range(period, len(tr)):
            atr[i + 1] = np.mean(tr[i - period:i])
        return atr

    @staticmethod
    def _vectorized_momentum(closes: np.ndarray, window: int) -> np.ndarray:
        """Compute momentum (rate of change) for every position."""
        n = len(closes)
        mom = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            if closes[i - window] != 0:
                mom[i] = (closes[i] / closes[i - window]) - 1.0
        return mom

    @staticmethod
    def _vectorized_volatility(closes: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling log-return volatility for every position."""
        n = len(closes)
        vol = np.zeros(n, dtype=np.float64)
        safe_closes = np.where(closes <= 0, 1e-10, closes)
        log_returns = np.diff(np.log(safe_closes))
        for i in range(window, len(log_returns)):
            vol[i + 1] = np.std(log_returns[i - window:i])
        return vol

    @staticmethod
    def _vectorized_volume_ratio(volumes: np.ndarray, window: int = 24) -> np.ndarray:
        """Compute volume ratio for every position."""
        n = len(volumes)
        ratio = np.ones(n, dtype=np.float64)
        for i in range(window, n):
            avg = np.mean(volumes[i - window:i])
            if avg > 1e-10:
                ratio[i] = volumes[i] / avg
        return ratio

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

    @staticmethod
    def compute_volume_ratio(candles: List[OHLCV], window: int = 24) -> float:
        """Current candle volume / rolling average volume."""
        if len(candles) < window + 1:
            return 1.0
        volumes = [c.volume for c in candles[-(window + 1):-1]]
        avg = sum(volumes) / len(volumes) if volumes else 1.0
        if avg < 1e-10:
            return 1.0
        return candles[-1].volume / avg
