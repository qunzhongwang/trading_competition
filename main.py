"""Trading Competition Framework — Entry Point.

Usage:
    uv run python main.py                        # paper mode (default config)
    uv run python main.py --config config/live.yaml  # custom config
    uv run python main.py --mode live             # override mode
    uv run python main.py --mode roostoo          # Roostoo competition mode
    uv run python main.py --mode roostoo --strategy-profile core_satellite_rotation_v1
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import signal
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import yaml

from core.models import Order, OrderType, Side, StrategyState
from data.buffer import LiveBuffer
from data.connector import WSConnector, BinanceSupplementaryFeed, prefetch_candles
from data.sim_feed import SimulatedFeed
from execution.executor import LiveExecutor
from execution.order_manager import OrderManager
from execution.roostoo_executor import RoostooExecutor
from execution.sim_executor import SimExecutor
from execution.trade_logger import TradeLogger
from features.extractor import FeatureExtractor
from models.inference import AlphaEngine
from models.model_wrapper import ModelWrapper
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from data.resampler import CandleResampler, MultiResampler
from strategy.factor_engine import FactorEngine
from strategy.monitor import StrategyMonitor

logger = logging.getLogger(__name__)


STRATEGY_PROFILES: dict[str, dict] = {
    "capital_preservation_v1": {
        "alpha": {
            "engine": "rule_based",
            "resample_minutes": 5,
            "multi_timeframes": [15, 60],
        },
        "strategy": {
            "profile": "capital_preservation_v1",
            "use_model_overlay": False,
            "position_size_pct": 0.05,
            "base_size_pct": 0.015,
            "max_size_pct": 0.05,
            "kelly_fraction": 0.25,
            "estimated_win_rate": 0.53,
            "estimated_payoff": 1.25,
            "confirmation_bars": 3,
            "min_entry_score": 0.72,
            "max_blocker_score": 0.22,
            "min_exit_score": 0.40,
            "neutral_entry_size_multiplier": 0.25,
            "risk_off_entry_size_multiplier": 0.00,
            "min_volatility_size_multiplier": 0.20,
            "require_trend_alignment": True,
            "min_supporting_factors": 3,
            "min_supporting_categories": 3,
            "urgent_entry_score": 0.90,
            "signal_horizon_minutes": 360,
            "exit_horizon_minutes": 20,
            "base_stop_loss_pct": 0.008,
            "take_profit_1_rr": 0.8,
            "take_profit_2_rr": 1.4,
            "min_volume_ratio": 1.20,
            "min_order_book_imbalance": 1.04,
            "max_funding_rate": 0.0003,
            "max_taker_ratio": 1.08,
            "max_open_interest_change": 0.015,
            "max_volatility": 0.018,
            "factor_weights": {
                "market_regime": 0.24,
                "trend_alignment": 0.24,
                "momentum_impulse": 0.14,
                "breakout_confirmation": 0.10,
                "volume_confirmation": 0.16,
                "liquidity_balance": 0.08,
                "perp_crowding": 0.18,
                "volatility_regime": 0.12,
            },
        },
        "regime": {
            "enabled": True,
            "benchmark_symbols": ["BTC/USDT", "ETH/USDT"],
            "risk_on_threshold": 0.32,
            "neutral_threshold": 0.12,
            "breadth_min_symbols": 10,
            "volatility_ceiling": 0.015,
        },
        "trend": {
            "breakout_lookback": 24,
            "trend_slope_lookback": 24,
            "volume_zscore_window": 24,
            "breakout_min_distance": 0.18,
            "breakout_overheat_distance": 1.20,
            "min_trend_slope": 0.0007,
            "min_volume_zscore": 0.10,
            "vol_target_floor": 0.010,
        },
        "risk": {
            "max_portfolio_exposure": 0.18,
            "max_single_exposure": 0.05,
            "trailing_stop_pct": 0.018,
            "atr_stop_multiplier": 1.6,
            "daily_drawdown_limit": 0.025,
            "max_orders_per_minute": 8,
            "break_even_trigger_pct": 0.012,
            "break_even_buffer_pct": 0.0015,
        },
    },
    "core_satellite_rotation_v1": {
        "symbols": [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "LINK/USDT",
            "XRP/USDT",
        ],
        "alpha": {
            "engine": "rule_based",
            "resample_minutes": 5,
            "multi_timeframes": [15, 60],
        },
        "strategy": {
            "profile": "core_satellite_rotation_v1",
            "use_model_overlay": False,
            "position_size_pct": 0.055,
            "base_size_pct": 0.018,
            "max_size_pct": 0.06,
            "kelly_fraction": 0.28,
            "estimated_win_rate": 0.54,
            "estimated_payoff": 1.30,
            "confirmation_bars": 3,
            "min_entry_score": 0.74,
            "max_blocker_score": 0.18,
            "min_exit_score": 0.38,
            "min_supporting_factors": 4,
            "min_supporting_categories": 3,
            "require_trend_alignment": True,
            "urgent_entry_score": 0.88,
            "signal_horizon_minutes": 300,
            "exit_horizon_minutes": 20,
            "base_stop_loss_pct": 0.008,
            "take_profit_1_rr": 0.8,
            "take_profit_2_rr": 1.5,
            "model_filter_threshold": 0.05,
            "model_exit_threshold": -0.10,
            "model_size_weight": 0.15,
            "neutral_entry_size_multiplier": 0.10,
            "risk_off_entry_size_multiplier": 0.00,
            "min_volatility_size_multiplier": 0.18,
            "min_volume_ratio": 1.10,
            "min_order_book_imbalance": 1.03,
            "max_funding_rate": 0.00022,
            "max_taker_ratio": 1.04,
            "max_open_interest_change": 0.010,
            "open_interest_lookback_samples": 60,
            "max_volatility": 0.0155,
            "core_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "satellite_symbols": [
                "BNB/USDT",
                "LINK/USDT",
                "XRP/USDT",
            ],
            "allow_satellite_in_neutral": False,
            "top_n_entries_per_cycle": 1,
            "max_active_positions": 2,
            "satellite_max_active_positions": 1,
            "satellite_min_entry_score_bonus": 0.04,
            "core_priority_bonus": 0.05,
            "factor_weights": {
                "market_regime": 0.30,
                "trend_alignment": 0.28,
                "momentum_impulse": 0.10,
                "breakout_confirmation": 0.06,
                "volume_confirmation": 0.14,
                "liquidity_balance": 0.10,
                "perp_crowding": 0.10,
                "volatility_regime": 0.14,
            },
        },
        "regime": {
            "enabled": True,
            "benchmark_symbols": ["BTC/USDT", "ETH/USDT"],
            "risk_on_threshold": 0.28,
            "neutral_threshold": 0.10,
            "breadth_min_symbols": 4,
            "volatility_ceiling": 0.015,
        },
        "trend": {
            "breakout_lookback": 24,
            "trend_slope_lookback": 24,
            "volume_zscore_window": 24,
            "breakout_min_distance": 0.16,
            "breakout_overheat_distance": 1.15,
            "min_trend_slope": 0.0007,
            "min_volume_zscore": 0.08,
            "vol_target_floor": 0.010,
        },
        "risk": {
            "max_portfolio_exposure": 0.18,
            "max_single_exposure": 0.06,
            "trailing_stop_pct": 0.018,
            "atr_stop_multiplier": 1.6,
            "daily_drawdown_limit": 0.025,
            "max_orders_per_minute": 8,
            "break_even_trigger_pct": 0.012,
            "break_even_buffer_pct": 0.0015,
        },
    },
    "regime_trend_v1": {
        "alpha": {
            "engine": "rule_based",
            "resample_minutes": 5,
            "multi_timeframes": [15, 60],
        },
        "strategy": {
            "profile": "regime_trend_v1",
            "use_model_overlay": False,
            "confirmation_bars": 2,
            "min_entry_score": 0.60,
            "max_blocker_score": 0.38,
            "min_exit_score": 0.52,
            "neutral_entry_size_multiplier": 0.50,
            "risk_off_entry_size_multiplier": 0.00,
            "min_volatility_size_multiplier": 0.35,
            "require_trend_alignment": True,
            "min_supporting_factors": 2,
            "min_supporting_categories": 2,
            "factor_weights": {
                "market_regime": 0.20,
                "trend_alignment": 0.24,
                "momentum_impulse": 0.18,
                "breakout_confirmation": 0.12,
                "volume_confirmation": 0.14,
                "liquidity_balance": 0.08,
                "perp_crowding": 0.14,
                "volatility_regime": 0.10,
            },
        },
        "regime": {
            "enabled": True,
            "benchmark_symbols": ["BTC/USDT", "ETH/USDT"],
            "risk_on_threshold": 0.25,
            "neutral_threshold": 0.05,
            "breadth_min_symbols": 5,
            "volatility_ceiling": 0.020,
        },
        "trend": {
            "breakout_lookback": 20,
            "trend_slope_lookback": 20,
            "volume_zscore_window": 24,
            "breakout_min_distance": 0.10,
            "breakout_overheat_distance": 1.75,
            "min_trend_slope": 0.0005,
            "min_volume_zscore": -0.25,
            "vol_target_floor": 0.012,
        },
        "risk": {
            "max_portfolio_exposure": 0.45,
            "max_single_exposure": 0.12,
        },
    },
    "trend_pullback_t_v1": {
        "alpha": {
            "engine": "rule_based",
            "resample_minutes": 5,
            "multi_timeframes": [15, 60],
        },
        "strategy": {
            "profile": "trend_pullback_t_v1",
            "use_model_overlay": False,
            "position_size_pct": 0.012,
            "base_size_pct": 0.003,
            "max_size_pct": 0.008,
            "kelly_fraction": 0.08,
            "estimated_win_rate": 0.56,
            "estimated_payoff": 1.10,
            "confirmation_bars": 3,
            "min_entry_score": 0.70,
            "max_blocker_score": 0.16,
            "min_exit_score": 0.30,
            "neutral_entry_size_multiplier": 0.0,
            "risk_off_entry_size_multiplier": 0.0,
            "min_volatility_size_multiplier": 0.02,
            "require_trend_alignment": True,
            "min_supporting_factors": 3,
            "min_supporting_categories": 3,
            "urgent_entry_score": 0.86,
            "signal_horizon_minutes": 240,
            "exit_horizon_minutes": 15,
            "base_stop_loss_pct": 0.005,
            "take_profit_1_rr": 0.6,
            "take_profit_2_rr": 1.0,
            "enable_pullback_reentry": True,
            "enable_overextension_exit": True,
            "pullback_min_breakout_distance": -0.45,
            "pullback_max_breakout_distance": 0.12,
            "pullback_target_breakout_distance": -0.10,
            "pullback_min_rsi": 45.0,
            "pullback_max_rsi": 58.0,
            "pullback_target_rsi": 51.0,
            "pullback_min_momentum": -0.008,
            "pullback_max_momentum": 0.006,
            "pullback_min_volume_ratio": 0.85,
            "pullback_trend_slack": 0.80,
            "overextension_min_breakout_distance": 0.35,
            "overextension_min_rsi": 65.0,
            "overextension_min_volume_zscore": -0.20,
            "overextension_min_taker_ratio": 1.00,
            "min_volume_ratio": 1.05,
            "min_order_book_imbalance": 1.02,
            "max_funding_rate": 0.00025,
            "max_taker_ratio": 1.05,
            "max_open_interest_change": 0.010,
            "max_volatility": 0.013,
            "factor_weights": {
                "market_regime": 0.26,
                "trend_alignment": 0.26,
                "momentum_impulse": 0.06,
                "breakout_confirmation": 0.04,
                "pullback_reentry": 0.24,
                "overextension_exit": 0.28,
                "volume_confirmation": 0.06,
                "liquidity_balance": 0.06,
                "perp_crowding": 0.14,
                "volatility_regime": 0.18,
            },
        },
        "regime": {
            "enabled": True,
            "benchmark_symbols": ["BTC/USDT", "ETH/USDT"],
            "risk_on_threshold": 0.28,
            "neutral_threshold": 0.10,
            "breadth_min_symbols": 8,
            "volatility_ceiling": 0.014,
        },
        "trend": {
            "breakout_lookback": 24,
            "trend_slope_lookback": 24,
            "volume_zscore_window": 24,
            "breakout_min_distance": 0.22,
            "breakout_overheat_distance": 0.90,
            "min_trend_slope": 0.0007,
            "min_volume_zscore": -0.10,
            "vol_target_floor": 0.007,
        },
        "risk": {
            "max_portfolio_exposure": 0.05,
            "max_single_exposure": 0.012,
            "trailing_stop_pct": 0.010,
            "atr_stop_multiplier": 1.10,
            "daily_drawdown_limit": 0.012,
            "max_orders_per_minute": 5,
            "break_even_trigger_pct": 0.006,
            "break_even_buffer_pct": 0.0008,
        },
    },
}


def setup_logging(mode: str = "paper") -> None:
    """Configure console + rotating file logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"trading_{mode}_{ts}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)

    logger.info("Logging to %s", log_file)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_dicts(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _apply_strategy_profile(config: dict, profile_name: Optional[str]) -> None:
    """Apply a named strategy profile as explicit runtime overrides."""
    if not profile_name:
        return
    if profile_name not in STRATEGY_PROFILES:
        valid = ", ".join(sorted(STRATEGY_PROFILES))
        raise SystemExit(
            f"Unknown strategy profile '{profile_name}'. Valid profiles: {valid}"
        )
    _merge_dicts(config, copy.deepcopy(STRATEGY_PROFILES[profile_name]))
    config.setdefault("strategy", {})["profile"] = profile_name


def _build_feature_config(config: dict) -> dict:
    """Merge feature-extractor settings from features and trend sections."""
    feature_cfg = copy.deepcopy(config.get("features", {}))
    trend_cfg = config.get("trend", {})
    feature_cfg.setdefault(
        "breakout_lookback", trend_cfg.get("breakout_lookback", 20)
    )
    feature_cfg.setdefault(
        "trend_slope_lookback", trend_cfg.get("trend_slope_lookback", 20)
    )
    feature_cfg.setdefault(
        "volume_zscore_window", trend_cfg.get("volume_zscore_window", 24)
    )
    return feature_cfg


def _apply_env_overrides(config: dict) -> None:
    """Override config values from environment variables (never commit secrets).

    Competition keys (ROOSTOO_COMP_*) take priority over testing keys (ROOSTOO_*).
    Key and secret are treated as a pair — both must be set for either tier.
    """
    roostoo_cfg = config.setdefault("roostoo", {})

    comp_key = os.environ.get("ROOSTOO_COMP_API_KEY", "")
    comp_secret = os.environ.get("ROOSTOO_COMP_API_SECRET", "")
    test_key = os.environ.get("ROOSTOO_API_KEY", "")
    test_secret = os.environ.get("ROOSTOO_API_SECRET", "")

    if comp_key and comp_secret:
        roostoo_cfg["api_key"] = comp_key
        roostoo_cfg["api_secret"] = comp_secret
        logger.info("Using competition Roostoo API credentials")
    elif comp_key or comp_secret:
        logger.warning(
            "ROOSTOO_COMP_API_KEY and ROOSTOO_COMP_API_SECRET must both be set; ignoring partial comp credentials"
        )
        if test_key and test_secret:
            roostoo_cfg["api_key"] = test_key
            roostoo_cfg["api_secret"] = test_secret
            logger.info("Falling back to testing Roostoo API credentials")
    elif test_key and test_secret:
        roostoo_cfg["api_key"] = test_key
        roostoo_cfg["api_secret"] = test_secret
        logger.info("Using testing Roostoo API credentials")

    exchange_cfg = config.setdefault("exchange", {})
    if os.environ.get("BINANCE_API_KEY"):
        exchange_cfg["api_key"] = os.environ["BINANCE_API_KEY"]
    if os.environ.get("BINANCE_API_SECRET"):
        exchange_cfg["api_secret"] = os.environ["BINANCE_API_SECRET"]


def _validate_roostoo_config(config: dict) -> None:
    """Validate Roostoo API credentials are present. Raises SystemExit if missing."""
    roostoo_cfg = config.get("roostoo", {})
    api_key = roostoo_cfg.get("api_key", "")
    api_secret = roostoo_cfg.get("api_secret", "")
    if not api_key or not api_secret:
        logger.error(
            "Roostoo API credentials missing. Set ROOSTOO_API_KEY and ROOSTOO_API_SECRET "
            "in .env or environment variables."
        )
        raise SystemExit(1)


def _validate_config(config: dict) -> None:
    """Validate config values are in sane ranges. Raises SystemExit on invalid config."""
    alpha_cfg = config.get("alpha", {})
    strategy_cfg = config.get("strategy", {})
    risk_cfg = config.get("risk", {})

    errors = []

    entry = alpha_cfg.get("entry_threshold", 0.6)
    exit_ = alpha_cfg.get("exit_threshold", -0.2)
    if not (0.0 <= entry <= 1.0):
        errors.append(f"alpha.entry_threshold={entry} must be in [0, 1]")
    if exit_ >= entry:
        errors.append(f"alpha.exit_threshold={exit_} must be less than entry_threshold={entry}")

    dd_limit = risk_cfg.get("daily_drawdown_limit", 0.05)
    if not (0.0 < dd_limit < 1.0):
        errors.append(f"risk.daily_drawdown_limit={dd_limit} must be in (0, 1)")

    max_port = risk_cfg.get("max_portfolio_exposure", 0.5)
    if not (0.0 < max_port <= 1.0):
        errors.append(f"risk.max_portfolio_exposure={max_port} must be in (0, 1]")

    max_single = risk_cfg.get("max_single_exposure", 0.15)
    if not (0.0 < max_single <= 1.0):
        errors.append(f"risk.max_single_exposure={max_single} must be in (0, 1]")

    base_size = strategy_cfg.get("base_size_pct", 0.05)
    max_size = strategy_cfg.get("max_size_pct", 0.15)
    if base_size > max_size:
        errors.append(f"strategy.base_size_pct={base_size} must be <= max_size_pct={max_size}")

    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        raise SystemExit(1)


def _resolve_roostoo_starting_capital(
    default_capital: float, balances: Optional[dict[str, float]]
) -> float:
    """Resolve the tracker cash baseline from a Roostoo balance snapshot.

    If a balance snapshot is available, use free USD directly. This prevents the
    portfolio tracker from booting with paper capital in competition mode.
    When the balance request fails entirely, fall back to config capital.
    """
    if balances:
        return float(balances.get("USD", 0.0))
    return default_capital


def _has_meaningful_position(snapshot, min_notional_usd: float) -> bool:
    """Return True if any restored position is large enough to count as active."""
    for pos in snapshot.positions:
        if pos.quantity <= 0 or pos.current_price <= 0:
            continue
        if pos.quantity * pos.current_price >= min_notional_usd:
            return True
    return False


async def _backfill_resampled_history(
    buffer: LiveBuffer,
    symbols: list[str],
    *,
    resampler: Optional[CandleResampler] = None,
    multi_resampler: Optional[MultiResampler] = None,
) -> None:
    """Replay prefetched 1m candles through the live resampler state.

    Prefetch only seeds raw candles in the buffer. The strategy, however, trades
    on completed higher-timeframe bars. Replaying the prefetched history here
    keeps the resampler state and the stored resampled buffers in sync so the
    strategy does not start from an empty 5m/15m/1h context.
    """
    if resampler is None and multi_resampler is None:
        return

    for symbol in symbols:
        candles = await buffer.get_candles(symbol)
        for candle in candles:
            if multi_resampler is not None:
                completed = multi_resampler.push(candle)
                for period, bar in completed.items():
                    if bar is not None:
                        await buffer.push_resampled(period, bar)
            elif resampler is not None:
                completed = resampler.push(candle)
                if completed is not None:
                    await buffer.push_resampled(resampler.minutes, completed)


async def main(config: dict) -> None:
    _apply_strategy_profile(
        config, config.get("strategy", {}).get("profile")
    )
    mode = config.get("mode", "paper")
    logger.info("=== Trading Competition Framework ===")
    logger.info("Mode: %s", mode)
    logger.info("Symbols: %d pairs", len(config.get("symbols", [])))
    logger.info(
        "Strategy profile: %s",
        config.get("strategy", {}).get("profile", "custom"),
    )

    _validate_config(config)

    # ── Build Components ──

    # 1. Data buffer
    data_cfg = config.get("data", {})
    buffer = LiveBuffer(
        max_candles=data_cfg.get("buffer_size", 500),
        max_ticks=data_cfg.get("tick_buffer_size", 5000),
    )

    # 2. Data source (producer) — always Binance WS for candle data
    if mode == "paper":
        feed = SimulatedFeed(config, buffer)
    else:
        # Both "live" and "roostoo" modes use Binance WS for market data
        feed = WSConnector(config, buffer)

        # Prefetch historical candles to eliminate warmup wait
        from features.extractor import FeatureExtractor as _FE

        _tmp_extractor = _FE(_build_feature_config(config))
        engine_type = config.get("alpha", {}).get("engine", "rule_based")
        seq_len = config.get("alpha", {}).get("seq_len", 30)
        resample_minutes = config.get("alpha", {}).get("resample_minutes", 1)
        use_model_overlay = config.get("strategy", {}).get("use_model_overlay", False)
        strategy_lookback = (_tmp_extractor.min_candles + 10) * max(resample_minutes, 1)
        if use_model_overlay and engine_type in ("lstm", "transformer", "ensemble"):
            model_lookback = (
                seq_len + _tmp_extractor.min_candles + 10
            ) * max(resample_minutes, 1)
            n_prefetch = max(strategy_lookback, model_lookback)
        else:
            n_prefetch = strategy_lookback
        n_prefetch = min(n_prefetch, 1000)  # Binance limit
        logger.info("Prefetching %d candles per symbol from Binance REST...", n_prefetch)
        await prefetch_candles(config.get("symbols", []), buffer, n_candles=n_prefetch)

    # 3. Trade logger
    trade_logger = TradeLogger()

    # 4. Executor
    paper_cfg = config.get("paper", {})
    starting_capital = paper_cfg.get("initial_capital", 1000000.0)
    balances: dict[str, float] = {}
    if mode == "paper":
        executor = SimExecutor(paper_cfg, buffer)
    elif mode == "roostoo":
        _validate_roostoo_config(config)
        roostoo_cfg = config.get("roostoo", {})
        roostoo_exec = RoostooExecutor(roostoo_cfg)
        roostoo_exec.set_trade_logger(trade_logger)
        await roostoo_exec.start()

        # Sync initial balance from Roostoo
        balances = await roostoo_exec.get_balance()
        if balances:
            starting_capital = _resolve_roostoo_starting_capital(
                starting_capital, balances
            )
            logger.info("Roostoo free USD balance: $%.2f", starting_capital)
        else:
            logger.warning(
                "Could not fetch Roostoo balance, using config initial_capital"
            )

        executor = roostoo_exec
    else:
        live_exec = LiveExecutor(config.get("exchange", {}))
        await live_exec.start()
        executor = live_exec

    # 5. Feature extractor
    extractor = FeatureExtractor(_build_feature_config(config))
    factor_engine = FactorEngine(config)

    # 6. Alpha engine (with optional LSTM model)
    model: Optional[ModelWrapper] = None
    alpha_cfg = config.get("alpha", {})
    engine_type = alpha_cfg.get("engine", "rule_based")

    if engine_type in ("lstm", "transformer", "ensemble"):
        model_path = alpha_cfg.get("model_path", "")
        model_type = alpha_cfg.get("model_type", "lstm")
        if model_path and Path(model_path).exists():
            model = ModelWrapper(
                model_path, n_features=extractor.N_FEATURES, model_type=model_type
            )
            model.load()
            logger.info("Loaded %s model from %s", model_type, model_path)
        else:
            logger.warning(
                "Model path '%s' not found, falling back to rule_based", model_path
            )
            config["alpha"]["engine"] = "rule_based"

    # 7. Portfolio tracker
    initial_capital = starting_capital
    fee_bps = paper_cfg.get("fee_bps", 10.0)
    tracker = PortfolioTracker(initial_capital, fee_bps)

    # 8. Risk shield
    risk_shield = RiskShield(config)

    # 9. Order manager
    exec_cfg = config.get("execution", {})
    order_timeout = exec_cfg.get("order_timeout_seconds", 0)
    order_manager = OrderManager(executor, tracker, timeout_seconds=order_timeout)
    if order_timeout > 0:
        logger.info("Order timeout: %ds for pending limit orders", order_timeout)

    # 10. Candle resampler (optional, for N-min alpha gating)
    resample_minutes = alpha_cfg.get("resample_minutes", 1)
    resampler = CandleResampler(resample_minutes) if resample_minutes > 1 else None
    if resampler:
        logger.info("Candle resampler: %d-min bars for alpha scoring", resample_minutes)

    # 10b. Multi-timeframe resampler (for higher-TF filters)
    multi_timeframes = alpha_cfg.get("multi_timeframes", [])
    multi_resampler = None
    if multi_timeframes:
        # Build periods: primary resample + higher TFs
        all_periods = sorted(set([resample_minutes] + multi_timeframes))
        multi_resampler = MultiResampler(all_periods)
        resampler = None  # multi_resampler supersedes single resampler
        logger.info(
            "Multi-timeframe resampler: periods=%s (primary=%d)",
            all_periods,
            multi_resampler.primary_minutes,
        )

    if mode != "paper":
        await _backfill_resampled_history(
            buffer,
            config.get("symbols", []),
            resampler=resampler,
            multi_resampler=multi_resampler,
        )

    # 10c. ICIR tracker (optional, for per-symbol adaptive weights)
    icir_tracker = None
    icir_prior_path = alpha_cfg.get("icir_prior_path", "")
    if alpha_cfg.get("icir_window"):
        from models.icir_tracker import BayesianICIRTracker
        import json

        prior_weights = {}
        if icir_prior_path and Path(icir_prior_path).exists():
            with open(icir_prior_path) as f:
                prior_weights = json.load(f)
            logger.info(
                "Loaded ICIR priors for %d symbols from %s",
                len(prior_weights),
                icir_prior_path,
            )

        icir_tracker = BayesianICIRTracker(
            prior_weights=prior_weights,
            n_factors=4,
            window=alpha_cfg.get("icir_window", 100),
            min_samples=alpha_cfg.get("icir_min_samples", 30),
            min_lambda=alpha_cfg.get("icir_min_lambda", 0.3),
            tau=alpha_cfg.get("icir_tau", 50.0),
        )
        logger.info(
            "ICIR tracker enabled: window=%d, min_samples=%d, min_lambda=%.2f",
            alpha_cfg.get("icir_window", 100),
            alpha_cfg.get("icir_min_samples", 30),
            alpha_cfg.get("icir_min_lambda", 0.3),
        )

    # 10d. Trade tracker for adaptive Kelly
    trade_tracker = None
    if config.get("strategy", {}).get("adaptive_kelly", False):
        from strategy.trade_tracker import TradeTracker

        strategy_cfg = config.get("strategy", {})
        trade_tracker = TradeTracker(
            window=strategy_cfg.get("kelly_window", 50),
            min_trades=strategy_cfg.get("kelly_min_trades", 10),
            prior_win_rate=strategy_cfg.get("estimated_win_rate", 0.55),
            prior_payoff=strategy_cfg.get("estimated_payoff", 1.5),
        )
        logger.info(
            "Adaptive Kelly sizing enabled (window=%d)",
            strategy_cfg.get("kelly_window", 50),
        )

    alpha_engine = AlphaEngine(config, extractor, model, icir_tracker=icir_tracker)

    # 11. Strategy monitor (orchestrator)
    monitor = StrategyMonitor(
        config=config,
        buffer=buffer,
        extractor=extractor,
        factor_engine=factor_engine,
        alpha_engine=alpha_engine,
        risk_shield=risk_shield,
        tracker=tracker,
        order_manager=order_manager,
        resampler=resampler,
        multi_resampler=multi_resampler,
        trade_tracker=trade_tracker,
        icir_tracker=icir_tracker,
        executor=executor,
        trade_logger=trade_logger,
    )

    # 11b. Position recovery on restart (Roostoo mode only)
    if mode == "roostoo" and isinstance(executor, RoostooExecutor):
        min_seed_position_notional = float(
            config.get("roostoo", {}).get("seed_trade_min_position_notional_usd", 1.0)
        )
        for asset, qty in balances.items():
            if asset == "USD" or qty <= 0:
                continue
            symbol = f"{asset}/USDT"
            if symbol not in config.get("symbols", []):
                logger.warning("Skipping unknown asset %s during position recovery", asset)
                continue
            ticker_price = await executor.get_ticker(symbol)
            if ticker_price and ticker_price > 0:
                tracker.restore_position(symbol, qty, entry_price=ticker_price)
                if symbol in monitor.strategies:
                    monitor.strategies[symbol]._state = StrategyState.HOLDING
                    monitor.strategies[symbol]._entry_price = ticker_price
                logger.info("Recovered position: %s qty=%.6f @ $%.2f", symbol, qty, ticker_price)
            else:
                logger.warning("Could not get ticker for %s, skipping position recovery", symbol)

        initial_capital = tracker.rebase_baselines()
        logger.info("Roostoo startup NAV baseline: $%.2f", initial_capital)

        # 11c. Seed trade — buy $2 of BTC to ensure participation record
        has_any_position = _has_meaningful_position(
            tracker.snapshot(), min_seed_position_notional
        )
        if not has_any_position:
            if any(pos.quantity > 0 for pos in tracker.snapshot().positions):
                    logger.info(
                    "Only dust positions detected (< $%.2f), seed trade still allowed",
                    min_seed_position_notional,
                )
            seed_symbol = "BTC/USDT"
            seed_price = await executor.get_ticker(seed_symbol)
            if seed_price and seed_price > 0:
                seed_qty = 2.0 / seed_price  # $2 worth (safely above $1 MiniOrder)
                seed_order = Order(
                    symbol=seed_symbol,
                    side=Side.BUY,
                    order_type=OrderType.MARKET,
                    quantity=seed_qty,
                )
                validated_seed_order = risk_shield.validate(
                    seed_order,
                    tracker,
                    market_price=seed_price,
                )
                if validated_seed_order is None:
                    logger.warning("Seed trade rejected by risk shield")
                else:
                    result = await order_manager.submit(validated_seed_order)
                logger.info(
                    "Seed trade: BUY $2 of %s qty=%.8f — status=%s",
                    seed_symbol,
                    validated_seed_order.quantity,
                    result.status.value,
                )
            else:
                logger.warning("Could not get BTC ticker for seed trade")

    # ── Graceful Shutdown ──
    shutdown_event = asyncio.Event()

    def handle_signal(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Run ──
    logger.info("Starting trading loop...")

    # 12. Supplementary Binance data feed (order book, funding, taker ratio)
    supp_feed = BinanceSupplementaryFeed(config.get("symbols", []), buffer)

    # NAV snapshot interval for risk metrics (every 60 iterations ~= 1 hour at 1m candles)
    NAV_SNAPSHOT_INTERVAL = 60  # noqa: F841

    METRICS_LOG_INTERVAL = 120

    async def run_with_shutdown():
        feed_task = asyncio.create_task(feed.start())
        monitor_task = asyncio.create_task(monitor.run())
        supp_task = asyncio.create_task(supp_feed.start())

        # Periodic NAV snapshot task
        async def nav_snapshot_loop():
            iteration = 0
            last_date = datetime.utcnow().strftime("%Y-%m-%d")
            while not shutdown_event.is_set():
                await asyncio.sleep(60)  # every minute
                iteration += 1
                tracker.record_nav_snapshot()

                # Backup daily reset (in case monitor misses it)
                current_date = datetime.utcnow().strftime("%Y-%m-%d")
                if current_date != last_date:
                    logger.info("NAV loop: day boundary %s → %s, resetting daily state", last_date, current_date)
                    risk_shield.reset_daily()
                    tracker.reset_daily()
                    last_date = current_date

                if iteration % METRICS_LOG_INTERVAL == 0:
                    metrics = tracker.compute_risk_metrics()
                    logger.info(
                        "Risk Metrics | Sharpe=%.3f | Sortino=%.3f | Calmar=%.3f | "
                        "Composite=%.3f | Return=%.2f%% | MaxDD=%.2f%%",
                        metrics.sharpe_ratio,
                        metrics.sortino_ratio,
                        metrics.calmar_ratio,
                        metrics.composite_score,
                        metrics.total_return_pct,
                        metrics.max_drawdown * 100,
                    )

        nav_task = asyncio.create_task(nav_snapshot_loop())

        # Wait for shutdown signal or task completion
        done, pending = await asyncio.wait(
            [
                feed_task,
                monitor_task,
                supp_task,
                nav_task,
                asyncio.create_task(shutdown_event.wait()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cleanup
        logger.info("Shutting down...")
        await order_manager.cancel_all()
        await monitor.stop()
        await feed.stop()
        if supp_feed is not None:
            await supp_feed.stop()

        if mode == "roostoo" and isinstance(executor, RoostooExecutor):
            await executor.stop()

        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Final report with risk metrics
        snap = tracker.snapshot()
        total_return = ((snap.nav - initial_capital) / initial_capital) * 100
        metrics = tracker.compute_risk_metrics()

        logger.info("=== Final Report ===")
        logger.info("NAV: $%.2f (started: $%.2f)", snap.nav, initial_capital)
        logger.info("Return: %.2f%%", total_return)
        logger.info("Max Drawdown: %.2f%%", snap.drawdown * 100)
        logger.info("Sharpe Ratio: %.3f", metrics.sharpe_ratio)
        logger.info("Sortino Ratio: %.3f", metrics.sortino_ratio)
        logger.info("Calmar Ratio: %.3f", metrics.calmar_ratio)
        logger.info("Composite Score (0.4S+0.3Sh+0.3C): %.3f", metrics.composite_score)
        logger.info("Cash: $%.2f", snap.cash)
        n_positions = sum(1 for pos in snap.positions if pos.quantity > 0)
        logger.info("Active Positions: %d", n_positions)
        for pos in snap.positions:
            if pos.quantity > 0:
                logger.info(
                    "  Position: %s qty=%.6f entry=%.2f current=%.2f pnl=%.2f",
                    pos.symbol,
                    pos.quantity,
                    pos.entry_price,
                    pos.current_price,
                    pos.unrealized_pnl,
                )

    await run_with_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Competition Framework")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Config file path"
    )
    parser.add_argument(
        "--mode", default=None, help="Override mode: paper, live, or roostoo"
    )
    parser.add_argument(
        "--engine",
        choices=["rule_based", "lstm", "ensemble"],
        default=None,
        help="Override optional model overlay engine: rule_based, lstm, or ensemble",
    )
    parser.add_argument(
        "--use-model-overlay",
        action="store_true",
        help="Enable the optional model overlay (filter/confidence/size only)",
    )
    parser.add_argument(
        "--strategy-profile",
        choices=sorted(STRATEGY_PROFILES),
        default=None,
        help="Apply a named strategy profile with explicit runtime overrides",
    )
    args = parser.parse_args()

    # Load .env file if present (so credentials work without shell scripts)
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

    config = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    if args.strategy_profile:
        config.setdefault("strategy", {})["profile"] = args.strategy_profile
    _apply_strategy_profile(config, config.get("strategy", {}).get("profile"))
    if args.engine:
        config.setdefault("alpha", {})["engine"] = args.engine
    if args.use_model_overlay:
        config.setdefault("strategy", {})["use_model_overlay"] = True

    _apply_env_overrides(config)
    setup_logging(config.get("mode", "paper"))
    asyncio.run(main(config))
