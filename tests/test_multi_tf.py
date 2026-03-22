"""Tests for multi-timeframe resampler and alpha filter."""

from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from core.models import FactorSnapshot, FeatureVector, OHLCV, Order, OrderType, Side, StrategyState
from data.buffer import LiveBuffer
from features.extractor import FeatureExtractor
from data.resampler import MultiResampler
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from strategy.monitor import StrategyMonitor


def _candle(symbol: str, minute_offset: int, close: float = 100.0) -> OHLCV:
    return OHLCV(
        symbol=symbol,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=10.0,
        timestamp=datetime(2025, 1, 1, 0, 0) + timedelta(minutes=minute_offset),
        is_closed=True,
    )


def _feature(
    symbol: str,
    *,
    momentum: float,
    ema_fast: float,
    ema_slow: float = 100.0,
    volume_ratio: float = 1.3,
    volatility: float = 0.01,
) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        timestamp=datetime(2025, 1, 1, 0, 0),
        rsi=55.0,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        atr=1.0,
        momentum=momentum,
        volatility=volatility,
        order_book_imbalance=1.05,
        volume_ratio=volume_ratio,
        funding_rate=0.0001,
        taker_ratio=1.0,
        raw={},
    )


class TestMultiResampler:
    def test_returns_dict_for_all_periods(self):
        mr = MultiResampler([5, 15, 60])
        result = mr.push(_candle("BTC/USDT", 0))
        assert isinstance(result, dict)
        assert set(result.keys()) == {5, 15, 60}

    def test_primary_is_smallest(self):
        mr = MultiResampler([15, 5, 60])
        assert mr.primary_minutes == 5

    def test_5min_bar_emits_before_15min(self):
        mr = MultiResampler([5, 15])
        five_min_count = 0
        fifteen_min_count = 0
        for i in range(15):
            result = mr.push(_candle("BTC/USDT", i, close=100.0 + i))
            if result[5] is not None:
                five_min_count += 1
            if result[15] is not None:
                fifteen_min_count += 1

        assert five_min_count == 3  # bars at 5, 10, 15 minutes
        assert fifteen_min_count == 1  # bar at 15 minutes

    def test_empty_periods_raises(self):
        with pytest.raises(ValueError):
            MultiResampler([])

    def test_periods_sorted(self):
        mr = MultiResampler([60, 5, 15])
        assert mr.periods == [5, 15, 60]

    def test_multi_symbol_independence(self):
        mr = MultiResampler([5])
        btc_emitted = 0
        eth_emitted = 0
        for i in range(5):
            r1 = mr.push(_candle("BTC/USDT", i))
            r2 = mr.push(_candle("ETH/USDT", i))
            if r1[5] is not None:
                btc_emitted += 1
            if r2[5] is not None:
                eth_emitted += 1
        assert btc_emitted == 1
        assert eth_emitted == 1


class TestMultiTFFilter:
    """Test the multi-TF filter in AlphaEngine."""

    def test_filter_with_no_data(self):
        from features.extractor import FeatureExtractor
        from models.inference import AlphaEngine
        from tests.conftest import make_candle_series

        config = {"alpha": {"engine": "rule_based"}, "features": {}}
        extractor = FeatureExtractor({})
        engine = AlphaEngine(config, extractor)

        candles = make_candle_series(60)
        # No multi-TF data — should still work
        signal = engine.score(candles, candles_15m=None, candles_1h=None)
        assert -1.0 <= signal.alpha_score <= 1.0

    def test_bullish_filter_boosts(self):
        from models.inference import AlphaEngine
        from features.extractor import FeatureExtractor
        from tests.conftest import make_candle_series

        config = {"alpha": {"engine": "rule_based"}, "features": {}}
        extractor = FeatureExtractor({})
        engine = AlphaEngine(config, extractor)

        # Create strong uptrend 15m candles (EMA fast > slow)
        candles_15m = make_candle_series(30, start_price=100.0, trend=2.0, noise=0.1)
        # Create strong uptrend 1h candles (positive momentum)
        candles_1h = make_candle_series(15, start_price=100.0, trend=5.0, noise=0.1)

        filter_val = engine._multi_tf_filter(candles_15m, candles_1h)
        assert filter_val > 0  # bullish filter


class TestModelOverlayTimeframe:
    @pytest.mark.asyncio
    async def test_overlay_uses_resampled_strategy_candles_and_aligned_history(
        self, default_config
    ):
        class _CapturingAlphaEngine:
            def __init__(self):
                self._seq_len = 4
                self.calls = []

            def score(
                self,
                candles,
                supplementary=None,
                supplementary_history=None,
                candles_15m=None,
                candles_1h=None,
            ):
                self.calls.append(
                    {
                        "candles": candles,
                        "supplementary_history": supplementary_history or {},
                    }
                )
                from core.models import Signal

                return Signal(
                    symbol="BTC/USDT",
                    alpha_score=0.1,
                    confidence=0.1,
                    timestamp=candles[-1].timestamp,
                    source="lstm",
                )

        class _StaticFactorEngine:
            def evaluate(self, features, **kwargs):
                return FactorSnapshot(
                    symbol=features.symbol,
                    timestamp=features.timestamp,
                    regime="neutral",
                    entry_score=0.0,
                    exit_score=0.0,
                    blocker_score=0.0,
                    confidence=0.0,
                    observations=[],
                    supporting_factors=[],
                    blocking_factors=[],
                    summary="neutral",
                )

        class _OrderManagerStub:
            def register_fill_callback(self, cb):
                self._callback = cb

            async def check_pending(self):
                return None

        config = {
            **default_config,
            "alpha": {
                **default_config["alpha"],
                "engine": "lstm",
                "seq_len": 4,
            },
            "strategy": {
                **default_config["strategy"],
                "use_model_overlay": True,
            },
        }
        buffer = LiveBuffer(max_candles=300)
        base = datetime(2025, 1, 1, 0, 0)
        for i in range(200):
            await buffer.push_candle(
                _candle("BTC/USDT", i, close=100.0 + i * 0.1)
            )
        for i in range(40):
            await buffer.push_resampled(
                5,
                OHLCV(
                    symbol="BTC/USDT",
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.5 + i,
                    volume=50.0,
                    timestamp=base + timedelta(minutes=5 * i + 4),
                    is_closed=True,
                ),
            )
        for value in range(1, 21):
            await buffer.push_taker_ratio("BTC/USDT", float(value))

        extractor = FeatureExtractor(config["features"])
        tracker = PortfolioTracker(100_000.0)
        shield = RiskShield(config)
        alpha_engine = _CapturingAlphaEngine()
        order_manager = _OrderManagerStub()

        monitor = StrategyMonitor(
            config=config,
            buffer=buffer,
            extractor=extractor,
            alpha_engine=alpha_engine,
            risk_shield=shield,
            tracker=tracker,
            order_manager=order_manager,
            factor_engine=_StaticFactorEngine(),
        )
        monitor._primary_minutes = 5

        await monitor._process_iteration(1)

        assert len(alpha_engine.calls) == 1
        call = alpha_engine.calls[0]
        assert len(call["candles"]) == 40
        assert call["candles"][-1].timestamp == base + timedelta(minutes=199)
        assert call["supplementary_history"]["taker_ratio"] == [5.0, 10.0, 15.0, 20.0]


class TestMonitorStatusFormatting:
    def test_formats_tiny_quantities_without_rounding_to_zero(self):
        assert StrategyMonitor._format_quantity(0.00003) == "0.00003000"
        assert StrategyMonitor._format_balance_item("BTC", 0.00003) == "BTC:0.00003000"

    def test_formats_regime_status_with_score_and_breadth(self):
        status = StrategyMonitor._format_regime_status(
            {"regime": "risk_off", "score": -0.214, "breadth": 0.23}
        )
        assert status == "Regime=risk_off(score=-0.214, breadth=0.23)"


class TestMarketContext:
    def test_breadth_floor_caps_risk_on_regime(self, default_config):
        class _OrderManagerStub:
            def register_fill_callback(self, cb):
                self._callback = cb

        class _AlphaStub:
            _seq_len = 1

        config = {
            **default_config,
            "symbols": [
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "LINK/USDT",
                "DOGE/USDT",
            ],
            "regime": {
                "enabled": True,
                "benchmark_symbols": ["BTC/USDT", "ETH/USDT"],
                "risk_on_threshold": 0.10,
                "neutral_threshold": 0.0,
                "breadth_min_symbols": 4,
                "volatility_ceiling": 0.02,
            },
        }

        monitor = StrategyMonitor(
            config=config,
            buffer=LiveBuffer(max_candles=10),
            extractor=FeatureExtractor(config["features"]),
            alpha_engine=_AlphaStub(),
            risk_shield=RiskShield(config),
            tracker=PortfolioTracker(100_000.0),
            order_manager=_OrderManagerStub(),
        )

        symbol_state = {
            "BTC/USDT": {
                "features": _feature("BTC/USDT", momentum=0.02, ema_fast=101.0),
                "supplementary": {},
                "candles_1h": None,
            },
            "ETH/USDT": {
                "features": _feature("ETH/USDT", momentum=0.018, ema_fast=100.9),
                "supplementary": {},
                "candles_1h": None,
            },
            "SOL/USDT": {
                "features": _feature("SOL/USDT", momentum=0.01, ema_fast=100.4),
                "supplementary": {},
                "candles_1h": None,
            },
            "LINK/USDT": {
                "features": _feature("LINK/USDT", momentum=-0.01, ema_fast=99.7),
                "supplementary": {},
                "candles_1h": None,
            },
            "DOGE/USDT": {
                "features": _feature("DOGE/USDT", momentum=-0.008, ema_fast=99.6),
                "supplementary": {},
                "candles_1h": None,
            },
        }

        context = monitor._build_market_context(symbol_state)
        assert context is not None
        assert context["score"] > 0.10
        assert context["breadth"] == pytest.approx(0.6)
        assert context["positive_symbols"] == 3
        assert context["breadth_ok"] is False
        assert context["regime"] == "neutral"


class TestMonitorCandidateSelection:
    class _OrderManagerStub:
        def register_fill_callback(self, cb):
            self._callback = cb

    class _AlphaStub:
        _seq_len = 1

    class _StrategyStub:
        def __init__(self):
            self.cancelled_orders = []

        def build_instruction(self, intent, current_price):
            class _Instruction:
                def __init__(self, symbol):
                    self._symbol = symbol

                def to_order(self):
                    return Order(
                        symbol=self._symbol,
                        side=Side.BUY,
                        order_type=OrderType.MARKET,
                        quantity=1.0,
                    )

            return _Instruction(intent["symbol"])

        def on_cancel(self, order):
            self.cancelled_orders.append(order)

    def _build_monitor(self, default_config, strategy_overrides=None):
        config = {
            **default_config,
            "symbols": ["BTC/USDT", "ETH/USDT", "LINK/USDT"],
            "strategy": {
                **default_config["strategy"],
                "min_entry_score": 0.74,
                "core_symbols": ["BTC/USDT", "ETH/USDT"],
                "satellite_symbols": ["LINK/USDT"],
                "allow_satellite_in_neutral": False,
                "satellite_max_active_positions": 1,
                "satellite_min_entry_score_bonus": 0.04,
                "core_priority_bonus": 0.03,
                **(strategy_overrides or {}),
            },
        }
        return StrategyMonitor(
            config=config,
            buffer=LiveBuffer(max_candles=10),
            extractor=FeatureExtractor(config["features"]),
            alpha_engine=self._AlphaStub(),
            risk_shield=RiskShield(config),
            tracker=PortfolioTracker(100_000.0),
            order_manager=self._OrderManagerStub(),
        )

    def _candidate(self, symbol: str, regime: str, entry: float = 0.78):
        return {
            "symbol": symbol,
            "factor_snapshot": FactorSnapshot(
                symbol=symbol,
                timestamp=datetime(2025, 1, 1),
                regime=regime,
                entry_score=entry,
                exit_score=0.0,
                blocker_score=0.02,
                confidence=0.8,
                observations=[],
                supporting_factors=[],
                blocking_factors=[],
                summary="test",
            ),
        }

    def test_core_symbol_gets_priority_bonus_in_ranking(self, default_config):
        monitor = self._build_monitor(default_config)
        ranked = monitor._rank_buy_candidates(
            [
                self._candidate("LINK/USDT", "risk_on"),
                self._candidate("BTC/USDT", "risk_on"),
            ]
        )
        assert [candidate["symbol"] for candidate in ranked] == [
            "BTC/USDT",
            "LINK/USDT",
        ]

    def test_neutral_regime_blocks_satellite_candidate(self, default_config):
        monitor = self._build_monitor(default_config)
        strategy = self._StrategyStub()
        ranked = monitor._rank_buy_candidates(
            [
                {
                    **self._candidate("LINK/USDT", "neutral"),
                    "strategy": strategy,
                    "intent": {"symbol": "LINK/USDT"},
                    "current_price": 100.0,
                }
            ]
        )
        assert ranked == []
        assert len(strategy.cancelled_orders) == 1

    def test_satellite_requires_higher_entry_score_than_core(self, default_config):
        monitor = self._build_monitor(default_config)
        strategy = self._StrategyStub()
        ranked = monitor._rank_buy_candidates(
            [
                {
                    **self._candidate("LINK/USDT", "risk_on", entry=0.76),
                    "strategy": strategy,
                    "intent": {"symbol": "LINK/USDT"},
                    "current_price": 100.0,
                }
            ]
        )
        assert ranked == []
        assert len(strategy.cancelled_orders) == 1

    def test_active_satellite_count_ignores_excluded_symbols(self, default_config):
        monitor = self._build_monitor(default_config)
        monitor.strategies["LINK/USDT"]._state = StrategyState.HOLDING

        assert monitor._active_satellite_count() == 1
        assert monitor._active_satellite_count(exclude_symbols={"LINK/USDT"}) == 0

    def test_active_symbol_count_can_ignore_unsubmitted_candidates(self, default_config):
        monitor = self._build_monitor(default_config)
        monitor.strategies["BTC/USDT"]._state = StrategyState.LONG_PENDING
        monitor.strategies["ETH/USDT"]._state = StrategyState.HOLDING

        assert monitor._active_symbol_count() == 2
        assert monitor._active_symbol_count(exclude_symbols={"BTC/USDT"}) == 1
