from __future__ import annotations

import logging
import time
from typing import List, Optional

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
        icir_tracker=None,
    ):
        alpha_cfg = config.get("alpha", {})
        self._engine_type: str = alpha_cfg.get("engine", "rule_based")
        self._entry_threshold: float = alpha_cfg.get("entry_threshold", 0.6)
        self._exit_threshold: float = alpha_cfg.get("exit_threshold", -0.2)
        self._seq_len: int = alpha_cfg.get("seq_len", 30)
        self._extractor = extractor
        self._model = model

        # Optional ICIR tracker for per-symbol adaptive weights
        self._icir = icir_tracker

        # Default rule-based weights (used when no ICIR tracker)
        self._w_rsi = 0.3
        self._w_momentum = 0.3
        self._w_ema = 0.3
        self._w_vol_penalty = 0.1

        # Alpha decay config
        self._decay_half_life_s: float = alpha_cfg.get("decay_half_life_s", 999999)

    def score(
        self,
        candles: List[OHLCV],
        supplementary: Optional[dict] = None,
        supplementary_history: Optional[dict] = None,
        candles_15m: Optional[List[OHLCV]] = None,
        candles_1h: Optional[List[OHLCV]] = None,
    ) -> Signal:
        """Generate alpha signal from candle history."""
        t0 = time.perf_counter()
        features = self._extractor.extract(candles, supplementary=supplementary)

        if self._engine_type == "rule_based":
            alpha = self._rule_based_score(features)
            source = "rule_based"
        elif self._engine_type in ("lstm", "transformer"):
            alpha = self._model_score(candles, supplementary, supplementary_history)
            source = self._engine_type
        elif self._engine_type == "ensemble":
            rule_alpha = self._rule_based_score(features)
            model_alpha = self._model_score(
                candles, supplementary, supplementary_history
            )
            alpha = 0.5 * rule_alpha + 0.5 * model_alpha
            source = "ensemble"
        else:
            logger.warning(
                "Unknown engine type '%s', falling back to rule_based",
                self._engine_type,
            )
            alpha = self._rule_based_score(features)
            source = "rule_based"

        # Apply multi-timeframe filter (dampens/boosts rule-based and ensemble alpha)
        if candles_15m or candles_1h:
            tf_filter = self._multi_tf_filter(candles_15m, candles_1h)
            if tf_filter != 0.0:
                old_alpha = alpha
                if tf_filter < 0 and alpha > 0:
                    alpha *= max(0.0, 1.0 + tf_filter)
                elif tf_filter > 0 and alpha > 0:
                    alpha *= min(1.5, 1.0 + 0.2 * tf_filter)
                if abs(old_alpha - alpha) > 0.01:
                    logger.debug(
                        "multi-TF filter %.3f: alpha %.4f → %.4f (%s)",
                        tf_filter,
                        old_alpha,
                        alpha,
                        features.symbol,
                    )

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
        """Composite alpha score from technical indicators."""
        # Get per-symbol weights from ICIR tracker if available
        if self._icir is not None:
            weights = self._icir.get_weights(features.symbol)
            w_rsi, w_momentum, w_ema, w_vol = weights
        else:
            w_rsi = self._w_rsi
            w_momentum = self._w_momentum
            w_ema = self._w_ema
            w_vol = self._w_vol_penalty

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
            w_rsi * rsi_signal
            + w_momentum * mom_signal
            + w_ema * ema_signal
            - w_vol * vol_penalty
        )

        return alpha

    def _multi_tf_filter(
        self, candles_15m: Optional[List[OHLCV]], candles_1h: Optional[List[OHLCV]]
    ) -> float:
        """Compute multi-timeframe trend filter from 15m and 1h bars.

        Returns a value in [-1, 1]:
            > 0: bullish higher-TF context (boost long signals)
            < 0: bearish higher-TF context (dampen long signals)
        """
        ema_15m_score = 0.0
        momentum_1h_score = 0.0

        # 15m: EMA(12) vs EMA(26) crossover direction
        if candles_15m and len(candles_15m) >= 26:
            closes = [c.close for c in candles_15m[-30:]]
            ema_fast = self._ema(closes, 12)
            ema_slow = self._ema(closes, 26)
            if ema_slow > 0:
                diff = (ema_fast - ema_slow) / ema_slow
                if diff > 0.001:
                    ema_15m_score = 1.0
                elif diff < -0.001:
                    ema_15m_score = -1.0

        # 1h: momentum(10) — clamped [-1, 1]
        if candles_1h and len(candles_1h) >= 11:
            closes = [c.close for c in candles_1h[-11:]]
            mom = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] > 0 else 0.0
            momentum_1h_score = max(-1.0, min(1.0, mom * 20.0))

        return 0.5 * ema_15m_score + 0.5 * momentum_1h_score

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        """Compute EMA of the last `period` values."""
        if not values or period <= 0:
            return 0.0
        multiplier = 2.0 / (period + 1)
        ema = values[0]
        for v in values[1:]:
            ema = (v - ema) * multiplier + ema
        return ema

    def _model_score(
        self,
        candles: List[OHLCV],
        supplementary: Optional[dict] = None,
        supplementary_history: Optional[dict] = None,
    ) -> float:
        """Run neural model (LSTM or Transformer) inference on feature sequence."""
        if self._model is None or not self._model.is_loaded:
            logger.warning("Model not loaded, returning 0.0")
            return 0.0

        seq = self._extractor.extract_sequence(
            candles,
            seq_len=self._seq_len,
            supplementary=supplementary,
            supplementary_history=supplementary_history,
        )
        return self._model.predict(seq)

    @property
    def entry_threshold(self) -> float:
        return self._entry_threshold

    @property
    def exit_threshold(self) -> float:
        return self._exit_threshold
