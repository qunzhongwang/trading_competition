"""Tests for bug fixes and production hardening (Phase 1-3).

Covers: circuit breaker reset, monitor exception handling, position recovery,
NaN guards, negative qty, inf metrics, partial fills, stuck orders, staleness,
unbounded NAV, signal confirmation clearing, config validation.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import (
    OHLCV,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
    Signal,
    StrategyState,
)
from data.buffer import LiveBuffer
from execution.order_manager import OrderManager
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from strategy.logic import StrategyLogic
from tests.conftest import make_candle_series, make_filled_buy, make_signal


# ── Phase 1: Critical Bugs ──


class TestCircuitBreakerDailyReset:
    """Fix 1: Circuit breaker resets after reset_daily()."""

    def test_circuit_breaker_clears_after_reset(self):
        config = {"risk": {"daily_drawdown_limit": 0.05}}
        shield = RiskShield(config)
        tracker = PortfolioTracker(100_000.0, fee_bps=10.0)

        # Force a large drawdown to trigger circuit breaker
        order = make_filled_buy(price=100.0, qty=500.0)
        tracker.on_fill(order)
        tracker.update_prices("BTC/USDT", 50.0)  # 50% drop

        triggered = shield.check_circuit_breaker(tracker)
        assert triggered
        assert shield.circuit_breaker_active

        # Now reset
        shield.reset_daily()
        assert not shield.circuit_breaker_active

    def test_tracker_daily_pnl_resets(self):
        tracker = PortfolioTracker(100_000.0, fee_bps=10.0)
        order = make_filled_buy(price=100.0, qty=10.0)
        tracker.on_fill(order)
        tracker.update_prices("BTC/USDT", 90.0)  # loss

        snap = tracker.snapshot()
        assert snap.daily_pnl < 0

        tracker.reset_daily()
        snap = tracker.snapshot()
        assert snap.daily_pnl == pytest.approx(0.0, abs=0.1)


class TestMonitorExceptionHandler:
    """Fix 2: Monitor continues after exception in _process_iteration."""

    @pytest.mark.asyncio
    async def test_monitor_continues_after_error(self, default_config):
        from strategy.monitor import StrategyMonitor
        from features.extractor import FeatureExtractor
        from models.inference import AlphaEngine

        buffer = LiveBuffer()
        extractor = FeatureExtractor(default_config.get("features", {}))
        alpha_engine = AlphaEngine(default_config, extractor)
        shield = RiskShield(default_config)
        tracker = PortfolioTracker(100_000.0)
        executor = AsyncMock()
        order_manager = OrderManager(executor, tracker)

        monitor = StrategyMonitor(
            config=default_config,
            buffer=buffer,
            extractor=extractor,
            alpha_engine=alpha_engine,
            risk_shield=shield,
            tracker=tracker,
            order_manager=order_manager,
        )

        # Patch _process_iteration to raise on first call, succeed on second
        call_count = 0
        original_process = monitor._process_iteration

        async def mock_process(iteration):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated API timeout")
            # Stop the loop after second iteration
            await monitor.stop()

        monitor._process_iteration = mock_process

        # Push a candle to trigger the event
        candle = make_candle_series(1)[0]
        await buffer.push_candle(candle)

        # Create a task that pushes another candle after a delay
        async def push_delayed():
            await asyncio.sleep(0.1)
            c2 = make_candle_series(1, seed=99)[0]
            await buffer.push_candle(c2)

        asyncio.create_task(push_delayed())

        # Run monitor — should not crash despite first iteration error
        await asyncio.wait_for(monitor.run(), timeout=5.0)
        assert call_count == 2


class TestPositionRecovery:
    """Fix 3: restore_position rebuilds state without modifying cash."""

    def test_restore_position(self):
        tracker = PortfolioTracker(100_000.0, fee_bps=10.0)
        tracker.restore_position("BTC/USDT", quantity=0.5, entry_price=60000.0)
        tracker.restore_position("ETH/USDT", quantity=10.0, entry_price=3000.0)

        snap = tracker.snapshot()
        # Cash should remain unchanged
        assert snap.cash == 100_000.0
        # Positions should be created
        assert len(snap.positions) == 2

        btc = tracker.get_position("BTC/USDT")
        assert btc.quantity == 0.5
        assert btc.entry_price == 60000.0
        assert btc.state == StrategyState.HOLDING

        eth = tracker.get_position("ETH/USDT")
        assert eth.quantity == 10.0
        assert eth.entry_price == 3000.0

    def test_restored_positions_in_nav(self):
        tracker = PortfolioTracker(100_000.0, fee_bps=10.0)
        tracker.restore_position("BTC/USDT", quantity=1.0, entry_price=50000.0)

        snap = tracker.snapshot()
        # NAV = cash (100k) + position value (1 * 50000)
        assert snap.nav == pytest.approx(150_000.0)


class TestStartupValidation:
    """Fix 4: Empty Roostoo API keys rejected at startup."""

    def test_empty_keys_rejected(self):
        from main import _validate_roostoo_config

        config = {"roostoo": {"api_key": "", "api_secret": ""}}
        with pytest.raises(SystemExit):
            _validate_roostoo_config(config)

    def test_valid_keys_pass(self):
        from main import _validate_roostoo_config

        config = {"roostoo": {"api_key": "abc123", "api_secret": "secret456"}}
        _validate_roostoo_config(config)  # Should not raise


class TestNaNPropagation:
    """Fix 5: NaN from alpha engine defaults to 0.0."""

    def test_nan_alpha_defaults_to_zero(self, default_config):
        from models.inference import AlphaEngine
        from features.extractor import FeatureExtractor

        extractor = FeatureExtractor(default_config.get("features", {}))
        model = MagicMock()
        model.is_loaded = True
        model.predict = MagicMock(return_value=float("nan"))

        default_config["alpha"]["engine"] = "lstm"
        engine = AlphaEngine(default_config, extractor, model=model)

        candles = make_candle_series(60)
        signal = engine.score(candles)
        assert not math.isnan(signal.alpha_score)
        assert signal.alpha_score == 0.0

    def test_inf_alpha_defaults_to_zero(self, default_config):
        from models.inference import AlphaEngine
        from features.extractor import FeatureExtractor

        extractor = FeatureExtractor(default_config.get("features", {}))
        model = MagicMock()
        model.is_loaded = True
        model.predict = MagicMock(return_value=float("inf"))

        default_config["alpha"]["engine"] = "lstm"
        engine = AlphaEngine(default_config, extractor, model=model)

        candles = make_candle_series(60)
        signal = engine.score(candles)
        assert not math.isinf(signal.alpha_score)
        assert signal.alpha_score == 0.0


class TestNegativePositionQuantity:
    """Fix 6: Negative cash doesn't produce negative buy quantity."""

    def test_negative_cash_returns_zero_qty(self, default_config):
        strategy = StrategyLogic("BTC/USDT", default_config)
        snapshot = PortfolioSnapshot(
            timestamp=datetime(2025, 1, 1),
            cash=-500.0,  # negative cash
            positions=[],
            nav=99500.0,
            daily_pnl=-500.0,
            peak_nav=100000.0,
            drawdown=0.005,
        )
        qty = strategy._compute_buy_quantity(snapshot, current_price=100.0, alpha_score=0.8)
        assert qty == 0.0


class TestInfMetricPropagation:
    """Fix 7: Calmar/Sortino inf capped to MAX_METRIC in composite score."""

    def test_composite_score_finite_with_zero_drawdown(self):
        tracker = PortfolioTracker(100_000.0, fee_bps=10.0)
        # Record two days of nav history with positive return (no drawdown)
        tracker._nav_history.append((datetime(2025, 1, 1), 100_000.0))
        tracker._nav_history.append((datetime(2025, 1, 2), 101_000.0))
        tracker._nav_history.append((datetime(2025, 1, 3), 102_000.0))

        score = tracker.compute_composite_score()
        assert math.isfinite(score)

    def test_risk_metrics_finite(self):
        tracker = PortfolioTracker(100_000.0, fee_bps=10.0)
        tracker._nav_history.append((datetime(2025, 1, 1), 100_000.0))
        tracker._nav_history.append((datetime(2025, 1, 2), 101_000.0))
        tracker._nav_history.append((datetime(2025, 1, 3), 102_000.0))

        metrics = tracker.compute_risk_metrics()
        assert math.isfinite(metrics.composite_score)
        assert math.isfinite(metrics.calmar_ratio)
        assert math.isfinite(metrics.sortino_ratio)


# ── Phase 2: Engineering Fixes ──


class TestPartialFills:
    """Fix 9: Partial fill detection in roostoo executor."""

    @pytest.mark.asyncio
    async def test_partially_filled_order_stays_active(self):
        executor = AsyncMock()
        tracker = PortfolioTracker(100_000.0)
        order_manager = OrderManager(executor, tracker)

        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            status=OrderStatus.PARTIALLY_FILLED,
        )
        executor.execute = AsyncMock(return_value=order)

        result = await order_manager.submit(order)
        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert order.order_id in order_manager.active_orders


class TestStuckOrders:
    """Fix 10: Orders removed after repeated get_status failures."""

    @pytest.mark.asyncio
    async def test_order_removed_after_max_errors(self):
        executor = AsyncMock()
        executor.get_status = AsyncMock(side_effect=RuntimeError("API down"))
        tracker = PortfolioTracker(100_000.0)
        order_manager = OrderManager(executor, tracker, timeout_seconds=0)

        # Manually add an active order
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            status=OrderStatus.SUBMITTED,
        )
        order_manager._active_orders[order.order_id] = order

        # Call check_pending 5 times (max_errors = 5)
        for _ in range(5):
            await order_manager.check_pending()

        # Order should be removed
        assert order.order_id not in order_manager.active_orders
        assert order.status == OrderStatus.CANCELLED


class TestWebSocketStaleness:
    """Fix 11: Buffer staleness tracking."""

    @pytest.mark.asyncio
    async def test_staleness_tracking(self):
        buf = LiveBuffer()
        candle = make_candle_series(1)[0]
        await buf.push_candle(candle)

        staleness = buf.seconds_since_last_candle("BTC/USDT")
        assert staleness < 1.0  # just pushed

    def test_unknown_symbol_returns_inf(self):
        buf = LiveBuffer()
        staleness = buf.seconds_since_last_candle("UNKNOWN/USDT")
        assert staleness == float("inf")


class TestUnboundedNavHistory:
    """Fix 13: NAV history capped at 20000."""

    def test_nav_history_bounded(self):
        tracker = PortfolioTracker(100_000.0, fee_bps=10.0)
        for i in range(25000):
            tracker.record_nav_snapshot()
        assert len(tracker._nav_history) == 20000


# ── Phase 3: Medium Priority ──


class TestConfigValidation:
    """Fix 14: Invalid config values rejected at startup."""

    def test_invalid_entry_threshold(self):
        from main import _validate_config

        config = {
            "alpha": {"entry_threshold": 1.5, "exit_threshold": -0.2},
            "strategy": {"base_size_pct": 0.05, "max_size_pct": 0.15},
            "risk": {
                "daily_drawdown_limit": 0.05,
                "max_portfolio_exposure": 0.5,
                "max_single_exposure": 0.15,
            },
        }
        with pytest.raises(SystemExit):
            _validate_config(config)

    def test_exit_ge_entry_threshold(self):
        from main import _validate_config

        config = {
            "alpha": {"entry_threshold": 0.6, "exit_threshold": 0.7},
            "strategy": {"base_size_pct": 0.05, "max_size_pct": 0.15},
            "risk": {
                "daily_drawdown_limit": 0.05,
                "max_portfolio_exposure": 0.5,
                "max_single_exposure": 0.15,
            },
        }
        with pytest.raises(SystemExit):
            _validate_config(config)

    def test_valid_config_passes(self, default_config):
        from main import _validate_config

        _validate_config(default_config)  # Should not raise


class TestSignalConfirmationClearing:
    """Fix 18: Broken streak resets confirmation counter."""

    def test_broken_streak_resets(self, default_config):
        default_config["strategy"]["confirmation_bars"] = 3
        strategy = StrategyLogic("BTC/USDT", default_config)

        snapshot = PortfolioSnapshot(
            timestamp=datetime(2025, 1, 1),
            cash=100000.0,
            positions=[],
            nav=100000.0,
            daily_pnl=0.0,
            peak_nav=100000.0,
            drawdown=0.0,
        )

        # Two bars above threshold
        signal_high = make_signal(alpha=0.8)
        strategy.on_signal(signal_high, snapshot, current_price=100.0)
        strategy.on_signal(signal_high, snapshot, current_price=100.0)

        # One bar below threshold — should break the streak
        signal_low = make_signal(alpha=0.3)
        strategy.on_signal(signal_low, snapshot, current_price=100.0)

        # Alpha history should be cleared after the broken streak
        assert len(strategy._alpha_history) == 0


class TestInitialHoldQty:
    """Fix 17: _initial_hold_qty initialized in __init__."""

    def test_initial_hold_qty_exists(self, default_config):
        strategy = StrategyLogic("BTC/USDT", default_config)
        assert hasattr(strategy, "_initial_hold_qty")
        assert strategy._initial_hold_qty == 0.0


class TestDayBoundaryReset:
    """Fix 1 (integration): Day boundary triggers reset in monitor."""

    @pytest.mark.asyncio
    async def test_day_boundary_detection(self, default_config):
        from strategy.monitor import StrategyMonitor
        from features.extractor import FeatureExtractor
        from models.inference import AlphaEngine

        buffer = LiveBuffer()
        extractor = FeatureExtractor(default_config.get("features", {}))
        alpha_engine = AlphaEngine(default_config, extractor)
        shield = RiskShield(default_config)
        tracker = PortfolioTracker(100_000.0)
        executor = AsyncMock()
        order_manager = OrderManager(executor, tracker)

        monitor = StrategyMonitor(
            config=default_config,
            buffer=buffer,
            extractor=extractor,
            alpha_engine=alpha_engine,
            risk_shield=shield,
            tracker=tracker,
            order_manager=order_manager,
        )

        # Set last date to yesterday
        monitor._last_trading_date = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        # Process iteration should detect day boundary
        with patch.object(shield, "reset_daily") as mock_reset_shield, \
             patch.object(tracker, "reset_daily") as mock_reset_tracker:
            await monitor._process_iteration(1)
            mock_reset_shield.assert_called_once()
            mock_reset_tracker.assert_called_once()
