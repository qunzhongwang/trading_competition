from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional

import numpy as np

from core.models import OHLCV, FeatureVector, Signal
from features.extractor import FeatureExtractor
from models.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


class AlphaEngine:
    """Produces Signal from features. Supports rule-based, LSTM, or ensemble modes.

    Modes (set via config alpha.engine):
        - "rule_based": composite score from RSI, momentum, EMA crossover, vol penalty
        - "lstm": ONNX/PyTorch LSTM forward pass on feature sequence
        - "ensemble": average of rule-based and LSTM scores
    """

    def __init__(
        self,
        config: dict,
        extractor: FeatureExtractor,
        model: Optional[ModelWrapper] = None,
    ):
        alpha_cfg = config.get("alpha", {})
        self._engine_type: str = alpha_cfg.get("engine", "rule_based")
        self._entry_threshold: float = alpha_cfg.get("entry_threshold", 0.6)
        self._exit_threshold: float = alpha_cfg.get("exit_threshold", -0.2)
        self._seq_len: int = alpha_cfg.get("seq_len", 30)
        self._extractor = extractor
        self._model = model

        # Rule-based weights
        self._w_rsi = 0.3
        self._w_momentum = 0.3
        self._w_ema = 0.3
        self._w_vol_penalty = 0.1

    def score(self, candles: List[OHLCV]) -> Signal:
        """Generate alpha signal from candle history."""
        t0 = time.perf_counter()
        features = self._extractor.extract(candles)

        if self._engine_type == "rule_based":
            alpha = self._rule_based_score(features)
            source = "rule_based"
        elif self._engine_type == "lstm":
            alpha = self._lstm_score(candles)
            source = "lstm"
        elif self._engine_type == "ensemble":
            rule_alpha = self._rule_based_score(features)
            lstm_alpha = self._lstm_score(candles)
            alpha = 0.5 * rule_alpha + 0.5 * lstm_alpha
            source = "ensemble"
        else:
            logger.warning("Unknown engine type '%s', falling back to rule_based", self._engine_type)
            alpha = self._rule_based_score(features)
            source = "rule_based"

        # Clamp to [-1, 1]
        alpha = max(-1.0, min(1.0, alpha))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Alpha %s: %.4f (%.1fms, %s)", features.symbol, alpha, elapsed_ms, source
        )

        return Signal(
            symbol=features.symbol,
            alpha_score=alpha,
            confidence=abs(alpha),
            timestamp=features.timestamp,
            source=source,
        )

    def _rule_based_score(self, features: FeatureVector) -> float:
        """Composite alpha score from technical indicators.

        Components:
            - RSI: oversold (< 30) → positive, overbought (> 70) → negative
            - Momentum: positive = bullish
            - EMA crossover: fast > slow → bullish
            - Volatility: high vol → penalty (reduce conviction)
        """
        # RSI signal: (50 - RSI) / 50, so RSI=30 → +0.4, RSI=70 → -0.4
        rsi_signal = (50.0 - features.rsi) / 50.0

        # Momentum signal: clamp to [-1, 1]
        mom_signal = max(-1.0, min(1.0, features.momentum * 20.0))

        # EMA crossover signal
        if features.ema_slow > 0:
            ema_signal = (features.ema_fast - features.ema_slow) / features.ema_slow
            ema_signal = max(-1.0, min(1.0, ema_signal * 100.0))
        else:
            ema_signal = 0.0

        # Volatility penalty: higher vol → lower score magnitude
        vol_penalty = min(1.0, features.volatility * 50.0)

        alpha = (
            self._w_rsi * rsi_signal
            + self._w_momentum * mom_signal
            + self._w_ema * ema_signal
            - self._w_vol_penalty * vol_penalty
        )

        return alpha

    def _lstm_score(self, candles: List[OHLCV]) -> float:
        """Run LSTM inference on feature sequence."""
        if self._model is None or not self._model.is_loaded:
            logger.warning("LSTM model not loaded, returning 0.0")
            return 0.0

        seq = self._extractor.extract_sequence(candles, seq_len=self._seq_len)
        return self._model.predict(seq)

    @property
    def entry_threshold(self) -> float:
        return self._entry_threshold

    @property
    def exit_threshold(self) -> float:
        return self._exit_threshold
