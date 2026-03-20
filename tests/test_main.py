"""Tests for main.py — config loading and component initialization."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from data.buffer import LiveBuffer
from data.resampler import CandleResampler, MultiResampler
from main import (
    _backfill_resampled_history,
    _resolve_roostoo_starting_capital,
    load_config,
)
from tests.conftest import make_candle


class TestLoadConfig:
    def test_loads_default_config(self):
        config = load_config("config/default.yaml")
        assert config["mode"] == "paper"
        assert "symbols" in config
        assert "alpha" in config

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_config_has_required_sections(self):
        config = load_config("config/default.yaml")
        required = [
            "mode",
            "symbols",
            "exchange",
            "data",
            "features",
            "alpha",
            "strategy",
            "risk",
            "execution",
            "paper",
        ]
        for key in required:
            assert key in config, f"Missing config section: {key}"

    def test_model_path_points_to_artifacts(self):
        config = load_config("config/default.yaml")
        model_path = config["alpha"]["model_path"]
        assert "artifacts/" in model_path

    def test_symbols_are_list(self):
        config = load_config("config/default.yaml")
        assert isinstance(config["symbols"], list)
        assert len(config["symbols"]) > 0

    def test_risk_limits_present(self):
        config = load_config("config/default.yaml")
        risk = config["risk"]
        assert "max_portfolio_exposure" in risk
        assert "trailing_stop_pct" in risk
        assert "daily_drawdown_limit" in risk


class TestRoostooStartingCapital:
    def test_uses_free_usd_balance_when_available(self):
        starting = _resolve_roostoo_starting_capital(
            1_000_000.0, {"USD": 1250.5, "BTC": 0.1}
        )
        assert starting == pytest.approx(1250.5)

    def test_allows_zero_usd_when_balance_snapshot_exists(self):
        starting = _resolve_roostoo_starting_capital(
            1_000_000.0, {"USD": 0.0, "BTC": 0.5}
        )
        assert starting == 0.0

    def test_falls_back_to_config_capital_when_balance_fetch_failed(self):
        starting = _resolve_roostoo_starting_capital(1_000_000.0, {})
        assert starting == 1_000_000.0


class TestResampledHistoryBackfill:
    @pytest.mark.asyncio
    async def test_backfills_single_resampler_from_prefetched_1m_history(self):
        buffer = LiveBuffer(max_candles=50)
        base = datetime(2025, 1, 1, 0, 0)
        for i in range(10):
            await buffer.push_candle(
                make_candle(
                    symbol="BTC/USDT",
                    close=100.0 + i,
                    ts=base + timedelta(minutes=i),
                )
            )

        await _backfill_resampled_history(
            buffer,
            ["BTC/USDT"],
            resampler=CandleResampler(5),
        )

        resampled = await buffer.get_resampled_candles("BTC/USDT", 5, n=10)
        assert len(resampled) == 2
        assert resampled[0].timestamp == base + timedelta(minutes=4)
        assert resampled[1].timestamp == base + timedelta(minutes=9)

    @pytest.mark.asyncio
    async def test_backfills_all_multi_timeframe_buffers(self):
        buffer = LiveBuffer(max_candles=80)
        base = datetime(2025, 1, 1, 0, 0)
        for i in range(15):
            await buffer.push_candle(
                make_candle(
                    symbol="BTC/USDT",
                    close=200.0 + i,
                    ts=base + timedelta(minutes=i),
                )
            )

        await _backfill_resampled_history(
            buffer,
            ["BTC/USDT"],
            multi_resampler=MultiResampler([5, 15]),
        )

        candles_5m = await buffer.get_resampled_candles("BTC/USDT", 5, n=10)
        candles_15m = await buffer.get_resampled_candles("BTC/USDT", 15, n=10)
        assert len(candles_5m) == 3
        assert len(candles_15m) == 1
        assert candles_15m[0].timestamp == base + timedelta(minutes=14)
