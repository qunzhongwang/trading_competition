from __future__ import annotations

from typing import Dict, List, Optional

from core.models import (
    FactorBias,
    FactorObservation,
    FactorSnapshot,
    FeatureVector,
    OHLCV,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class FactorEngine:
    """Translate raw features and market context into explicit strategy factors."""

    def __init__(self, config: dict):
        strategy_cfg = config.get("strategy", {})
        regime_cfg = config.get("regime", {})
        trend_cfg = config.get("trend", {})
        self._factor_weights: Dict[str, float] = {
            "market_regime": 0.20,
            "trend_alignment": 0.28,
            "momentum_impulse": 0.22,
            "breakout_confirmation": 0.12,
            "volume_confirmation": 0.15,
            "liquidity_balance": 0.10,
            "perp_crowding": 0.15,
            "volatility_regime": 0.10,
            **strategy_cfg.get("factor_weights", {}),
        }
        self._min_volume_ratio: float = strategy_cfg.get("min_volume_ratio", 1.10)
        self._min_order_book_imbalance: float = strategy_cfg.get(
            "min_order_book_imbalance", 1.02
        )
        self._max_funding_rate: float = strategy_cfg.get("max_funding_rate", 0.0005)
        self._max_taker_ratio: float = strategy_cfg.get("max_taker_ratio", 1.20)
        self._max_open_interest_change: float = strategy_cfg.get(
            "max_open_interest_change", 0.03
        )
        self._open_interest_lookback_samples: int = max(
            2, int(strategy_cfg.get("open_interest_lookback_samples", 60))
        )
        self._max_volatility: float = strategy_cfg.get("max_volatility", 0.025)
        self._regime_enabled: bool = regime_cfg.get("enabled", False)
        self._risk_on_threshold: float = regime_cfg.get("risk_on_threshold", 0.25)
        self._neutral_threshold: float = regime_cfg.get("neutral_threshold", 0.05)
        self._breakout_min_distance: float = trend_cfg.get(
            "breakout_min_distance", 0.10
        )
        self._breakout_overheat_distance: float = trend_cfg.get(
            "breakout_overheat_distance", 1.75
        )
        self._min_trend_slope: float = trend_cfg.get("min_trend_slope", 0.0005)
        self._min_volume_zscore: float = trend_cfg.get("min_volume_zscore", -0.25)

    def evaluate(
        self,
        features: FeatureVector,
        supplementary: Optional[dict] = None,
        supplementary_history: Optional[dict] = None,
        candles_15m: Optional[List[OHLCV]] = None,
        candles_1h: Optional[List[OHLCV]] = None,
        market_context: Optional[dict] = None,
    ) -> FactorSnapshot:
        supp = supplementary or {}
        hist = supplementary_history or {}
        observations = [
            self._market_regime(features, market_context),
            self._trend_alignment(features, candles_15m, candles_1h),
            self._momentum_impulse(features),
            self._breakout_confirmation(features),
            self._volume_confirmation(features),
            self._liquidity_balance(features, supp),
            self._perp_crowding(features, supp, hist),
            self._volatility_regime(features),
        ]

        support_weight = 0.0
        support_score = 0.0
        blocker_weight = 0.0
        blocker_score = 0.0
        exit_weight = 0.0
        exit_score = 0.0

        supporting_factors = []
        blocking_factors = []
        for obs in observations:
            weight = self._factor_weights.get(obs.name, 0.0)
            if obs.bias == FactorBias.BULLISH:
                support_weight += weight
                support_score += weight * obs.strength
                supporting_factors.append(obs.name)
            elif obs.bias == FactorBias.BEARISH:
                blocker_weight += weight
                blocker_score += weight * obs.strength
                exit_weight += weight
                exit_score += weight * obs.strength
                blocking_factors.append(obs.name)

        entry_score = support_score / support_weight if support_weight > 0 else 0.0
        blocker = blocker_score / blocker_weight if blocker_weight > 0 else 0.0
        exit_score = exit_score / exit_weight if exit_weight > 0 else 0.0
        confidence = _clamp(entry_score * (1.0 - 0.5 * blocker))

        if market_context is not None:
            regime = market_context.get("regime", "neutral")
        elif entry_score >= 0.65 and blocker < 0.35:
            regime = "risk_on"
        elif blocker >= 0.60:
            regime = "risk_off"
        else:
            regime = "neutral"

        summary_parts = [obs.thesis for obs in observations if obs.bias != FactorBias.NEUTRAL]
        summary = "; ".join(summary_parts[:3]) if summary_parts else "No dominant factor signal"

        return FactorSnapshot(
            symbol=features.symbol,
            timestamp=features.timestamp,
            regime=regime,
            entry_score=_clamp(entry_score),
            exit_score=_clamp(exit_score),
            blocker_score=_clamp(blocker),
            confidence=confidence,
            observations=observations,
            supporting_factors=supporting_factors,
            blocking_factors=blocking_factors,
            summary=summary,
        )

    @property
    def supplementary_history_window(self) -> int:
        """Minimum raw supplementary history required by the factor engine."""
        return self._open_interest_lookback_samples

    def _market_regime(
        self, features: FeatureVector, market_context: Optional[dict]
    ) -> FactorObservation:
        if not self._regime_enabled or market_context is None:
            return FactorObservation(
                symbol=features.symbol,
                name="market_regime",
                category="market",
                timestamp=features.timestamp,
                bias=FactorBias.NEUTRAL,
                strength=0.0,
                value=0.0,
                threshold=self._neutral_threshold,
                horizon_minutes=240,
                expected_move_bps=0.0,
                thesis="Market regime filter disabled",
                invalidate_condition="Regime filter becomes active",
                metadata={},
            )

        regime = market_context.get("regime", "neutral")
        score = float(market_context.get("score", 0.0))
        breadth = float(market_context.get("breadth", 0.5))
        bias = FactorBias.NEUTRAL
        strength = 0.0
        if regime == "risk_on":
            denom = max(self._risk_on_threshold - self._neutral_threshold, 1e-6)
            strength = _clamp((score - self._neutral_threshold) / denom)
            bias = FactorBias.BULLISH
        elif regime == "risk_off":
            denom = max(self._neutral_threshold + 1.0, 1e-6)
            strength = _clamp((self._neutral_threshold - score) / denom)
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="market_regime",
            category="market",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength,
            value=score,
            threshold=self._neutral_threshold,
            horizon_minutes=240,
            expected_move_bps=60.0 + 120.0 * strength,
            thesis=(
                f"Market regime {regime} with score {score:.3f} and breadth {breadth:.2f}"
            ),
            invalidate_condition="Benchmark trend and market breadth revert toward neutral",
            metadata=dict(market_context),
        )

    def _trend_alignment(
        self,
        features: FeatureVector,
        candles_15m: Optional[List[OHLCV]],
        candles_1h: Optional[List[OHLCV]],
    ) -> FactorObservation:
        primary_diff = 0.0
        if features.ema_slow > 0:
            primary_diff = (features.ema_fast - features.ema_slow) / features.ema_slow

        tf15 = 0.0
        if candles_15m and len(candles_15m) >= 2:
            prev = candles_15m[-2].close
            tf15 = (candles_15m[-1].close - prev) / prev if prev > 0 else 0.0

        tf1h = 0.0
        if candles_1h and len(candles_1h) >= 4:
            prev = candles_1h[-4].close
            tf1h = (candles_1h[-1].close - prev) / prev if prev > 0 else 0.0

        trend_strength = _clamp(max(primary_diff * 180.0, 0.0) + max(tf15 * 40.0, 0.0) + max(tf1h * 20.0, 0.0))
        bias = FactorBias.NEUTRAL
        if primary_diff > 0 and (tf15 >= -0.001) and (tf1h >= -0.002):
            bias = FactorBias.BULLISH
        elif primary_diff < 0 and tf15 < 0 and tf1h < 0:
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="trend_alignment",
            category="price_structure",
            timestamp=features.timestamp,
            bias=bias,
            strength=trend_strength if bias != FactorBias.NEUTRAL else 0.0,
            value=primary_diff,
            threshold=0.0,
            horizon_minutes=240,
            expected_move_bps=80.0 + 180.0 * trend_strength,
            thesis=(
                f"Trend aligned with EMA spread {primary_diff:.4f}, 15m drift {tf15:.4f}, 1h drift {tf1h:.4f}"
            ),
            invalidate_condition="Primary EMA spread flips negative or higher-timeframe drift turns down",
            metadata={"ema_spread": primary_diff, "tf15": tf15, "tf1h": tf1h},
        )

    def _momentum_impulse(self, features: FeatureVector) -> FactorObservation:
        rsi_penalty = _clamp((features.rsi - 68.0) / 20.0)
        momentum_strength = _clamp(
            max(features.momentum * 25.0, 0.0) * (1.0 - 0.5 * rsi_penalty)
        )

        bias = FactorBias.NEUTRAL
        if features.momentum > 0 and features.rsi < 75:
            bias = FactorBias.BULLISH
        elif features.momentum < 0:
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="momentum_impulse",
            category="price_structure",
            timestamp=features.timestamp,
            bias=bias,
            strength=momentum_strength if bias != FactorBias.NEUTRAL else 0.0,
            value=features.momentum,
            threshold=0.0,
            horizon_minutes=180,
            expected_move_bps=60.0 + 160.0 * momentum_strength,
            thesis=f"Momentum {features.momentum:.4f} with RSI {features.rsi:.1f}",
            invalidate_condition="Momentum turns negative or RSI closes back below neutral after breakout failure",
            metadata={"momentum": features.momentum, "rsi": features.rsi},
        )

    def _volume_confirmation(self, features: FeatureVector) -> FactorObservation:
        vol_strength = _clamp((features.volume_ratio - self._min_volume_ratio) / 1.2)
        bias = FactorBias.BULLISH if features.volume_ratio >= self._min_volume_ratio else FactorBias.NEUTRAL

        return FactorObservation(
            symbol=features.symbol,
            name="volume_confirmation",
            category="flow",
            timestamp=features.timestamp,
            bias=bias,
            strength=vol_strength if bias == FactorBias.BULLISH else 0.0,
            value=features.volume_ratio,
            threshold=self._min_volume_ratio,
            horizon_minutes=120,
            expected_move_bps=40.0 + 100.0 * vol_strength,
            thesis=f"Volume ratio {features.volume_ratio:.2f} vs trigger {self._min_volume_ratio:.2f}",
            invalidate_condition="Volume ratio falls back below confirmation threshold",
            metadata={"volume_ratio": features.volume_ratio},
        )

    def _breakout_confirmation(self, features: FeatureVector) -> FactorObservation:
        breakout_distance = float(features.raw.get("breakout_distance", 0.0))
        trend_slope = float(features.raw.get("trend_slope", 0.0))
        volume_zscore = float(features.raw.get("volume_zscore", 0.0))

        bias = FactorBias.NEUTRAL
        strength = 0.0
        if (
            breakout_distance >= self._breakout_min_distance
            and breakout_distance <= self._breakout_overheat_distance
            and trend_slope >= self._min_trend_slope
            and volume_zscore >= self._min_volume_zscore
        ):
            distance_component = breakout_distance / max(
                self._breakout_overheat_distance, 1e-6
            )
            slope_component = trend_slope / max(self._min_trend_slope * 4.0, 1e-6)
            volume_component = (volume_zscore + 1.0) / 3.0
            strength = _clamp(
                0.5 * distance_component
                + 0.3 * slope_component
                + 0.2 * volume_component
            )
            bias = FactorBias.BULLISH
        elif breakout_distance > self._breakout_overheat_distance:
            strength = _clamp(
                (breakout_distance - self._breakout_overheat_distance)
                / max(self._breakout_overheat_distance, 1e-6)
            )
            bias = FactorBias.BEARISH
        elif breakout_distance < -0.50 and trend_slope < 0:
            strength = _clamp(
                min(abs(breakout_distance), 2.0) / 2.0
                + min(abs(trend_slope) / max(self._min_trend_slope * 4.0, 1e-6), 1.0)
                * 0.25
            )
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="breakout_confirmation",
            category="price_structure",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias != FactorBias.NEUTRAL else 0.0,
            value=breakout_distance,
            threshold=self._breakout_min_distance,
            horizon_minutes=180,
            expected_move_bps=30.0 + 130.0 * strength,
            thesis=(
                f"Breakout distance {breakout_distance:.2f} ATR, trend slope {trend_slope:.4f}, "
                f"volume z-score {volume_zscore:.2f}"
            ),
            invalidate_condition="Breakout distance compresses back into the prior range or price becomes overextended",
            metadata={
                "breakout_distance": breakout_distance,
                "trend_slope": trend_slope,
                "volume_zscore": volume_zscore,
            },
        )

    def _liquidity_balance(
        self,
        features: FeatureVector,
        supplementary: dict,
    ) -> FactorObservation:
        imbalance = supplementary.get(
            "order_book_imbalance", features.order_book_imbalance
        )
        strength = _clamp(abs(imbalance - 1.0) / 0.25)
        inverse_threshold = 1.0 / max(self._min_order_book_imbalance, 1e-6)

        bias = FactorBias.NEUTRAL
        if imbalance >= self._min_order_book_imbalance:
            bias = FactorBias.BULLISH
        elif 0.0 < imbalance <= inverse_threshold:
            bias = FactorBias.BEARISH

        return FactorObservation(
            symbol=features.symbol,
            name="liquidity_balance",
            category="microstructure",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias != FactorBias.NEUTRAL else 0.0,
            value=imbalance,
            threshold=self._min_order_book_imbalance,
            horizon_minutes=30,
            expected_move_bps=20.0 + 70.0 * strength,
            thesis=f"Order-book imbalance {imbalance:.3f}",
            invalidate_condition="Top-of-book imbalance mean reverts back to neutral",
            metadata={"order_book_imbalance": imbalance},
        )

    def _perp_crowding(
        self,
        features: FeatureVector,
        supplementary: dict,
        supplementary_history: dict,
    ) -> FactorObservation:
        funding = supplementary.get("funding_rate", features.funding_rate)
        taker_ratio = supplementary.get("taker_ratio", features.taker_ratio)
        open_interest = supplementary.get("open_interest", 0.0)

        oi_hist = supplementary_history.get("open_interest", [])
        oi_window = oi_hist[-self._open_interest_lookback_samples :]
        oi_change = 0.0
        if len(oi_window) >= 2 and oi_window[0] > 0:
            oi_change = (oi_window[-1] - oi_window[0]) / oi_window[0]

        crowded = max(
            0.0,
            max(funding - self._max_funding_rate, 0.0) / max(self._max_funding_rate, 1e-6),
            max(taker_ratio - self._max_taker_ratio, 0.0) / max(self._max_taker_ratio, 1e-6),
            max(oi_change - self._max_open_interest_change, 0.0)
            / max(self._max_open_interest_change, 1e-6),
        )
        strength = _clamp(crowded)

        bias = FactorBias.NEUTRAL
        if strength > 0:
            bias = FactorBias.BEARISH
        elif 0.0 < funding < self._max_funding_rate * 0.5 and taker_ratio <= 1.05:
            bias = FactorBias.BULLISH
            strength = 0.15

        return FactorObservation(
            symbol=features.symbol,
            name="perp_crowding",
            category="derivatives",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength,
            value=funding,
            threshold=self._max_funding_rate,
            horizon_minutes=180,
            expected_move_bps=40.0 + 120.0 * strength,
            thesis=(
                f"Funding {funding:.6f}, taker ratio {taker_ratio:.3f}, open-interest change {oi_change:.3f}"
            ),
            invalidate_condition="Funding and taker pressure normalize back toward neutral",
            metadata={
                "funding_rate": funding,
                "taker_ratio": taker_ratio,
                "open_interest": open_interest,
                "open_interest_change": oi_change,
            },
        )

    def _volatility_regime(self, features: FeatureVector) -> FactorObservation:
        excess_vol = max(features.volatility - self._max_volatility, 0.0)
        strength = _clamp(excess_vol / max(self._max_volatility, 1e-6))
        bias = FactorBias.BEARISH if features.volatility > self._max_volatility else FactorBias.NEUTRAL

        return FactorObservation(
            symbol=features.symbol,
            name="volatility_regime",
            category="risk",
            timestamp=features.timestamp,
            bias=bias,
            strength=strength if bias == FactorBias.BEARISH else 0.0,
            value=features.volatility,
            threshold=self._max_volatility,
            horizon_minutes=60,
            expected_move_bps=0.0,
            thesis=f"Realized volatility {features.volatility:.4f} vs cap {self._max_volatility:.4f}",
            invalidate_condition="Volatility cools back into the strategy operating range",
            metadata={"volatility": features.volatility},
        )
