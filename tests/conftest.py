"""Shared fixtures for all tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pytest

from core.models import (
    OHLCV,
    Order,
    OrderStatus,
    OrderType,
    Position,
    PortfolioSnapshot,
    Side,
    Signal,
    StrategyState,
)


# ---------------------------------------------------------------------------
# OHLCV candle helpers
# ---------------------------------------------------------------------------

def make_candle(
    symbol: str = "BTC/USDT",
    close: float = 100.0,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 10.0,
    ts: datetime | None = None,
    is_closed: bool = True,
) -> OHLCV:
    """Build a single OHLCV candle with sensible defaults."""
    open_ = open_ if open_ is not None else close * 0.999
    high = high if high is not None else close * 1.002
    low = low if low is not None else close * 0.998
    ts = ts or datetime(2025, 1, 1)
    return OHLCV(
        symbol=symbol, open=open_, high=high, low=low,
        close=close, volume=volume, timestamp=ts, is_closed=is_closed,
    )


def make_candle_series(
    n: int = 60,
    start_price: float = 100.0,
    symbol: str = "BTC/USDT",
    trend: float = 0.0,
    noise: float = 0.5,
    seed: int = 42,
) -> List[OHLCV]:
    """Generate a deterministic candle series for testing.

    Args:
        n: number of candles
        start_price: initial close price
        trend: drift per candle (e.g. 0.1 means +0.1 per candle)
        noise: std dev of random walk
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    candles = []
    price = start_price

    base_time = datetime(2025, 1, 1)
    for i in range(n):
        price += trend + rng.normal(0, noise)
        price = max(price, 1.0)  # floor at 1
        o = price * (1 + rng.normal(0, 0.001))
        h = price * (1 + abs(rng.normal(0, 0.003)))
        l = price * (1 - abs(rng.normal(0, 0.003)))
        candles.append(OHLCV(
            symbol=symbol,
            open=round(o, 2),
            high=round(h, 2),
            low=round(l, 2),
            close=round(price, 2),
            volume=round(rng.uniform(5, 50), 2),
            timestamp=base_time + timedelta(minutes=i),
            is_closed=True,
        ))
    return candles


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def candles_60():
    """60 candles, random walk around 100."""
    return make_candle_series(60)


@pytest.fixture
def candles_120():
    """120 candles for sequence extraction."""
    return make_candle_series(120)


@pytest.fixture
def uptrend_candles():
    """60 candles with clear upward trend."""
    return make_candle_series(60, trend=0.5, noise=0.2)


@pytest.fixture
def downtrend_candles():
    """60 candles with clear downward trend."""
    return make_candle_series(60, start_price=200.0, trend=-0.5, noise=0.2)


@pytest.fixture
def default_config() -> dict:
    """Minimal config matching config/default.yaml structure."""
    return {
        "mode": "paper",
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "data": {"buffer_size": 500, "candle_interval": "1m"},
        "features": {
            "rsi_period": 14,
            "ema_fast": 12,
            "ema_slow": 26,
            "atr_period": 14,
            "volatility_window": 20,
            "momentum_window": 10,
        },
        "alpha": {
            "engine": "rule_based",
            "entry_threshold": 0.6,
            "exit_threshold": -0.2,
            "seq_len": 30,
        },
        "strategy": {
            "max_positions_per_symbol": 1,
            "position_size_pct": 0.10,
        },
        "risk": {
            "max_portfolio_exposure": 0.50,
            "max_single_exposure": 0.15,
            "trailing_stop_pct": 0.03,
            "atr_stop_multiplier": 2.0,
            "daily_drawdown_limit": 0.05,
            "max_orders_per_minute": 10,
        },
        "paper": {
            "initial_capital": 100000.0,
            "slippage_bps": 5,
            "fee_bps": 10,
        },
    }


@pytest.fixture
def portfolio_snapshot() -> PortfolioSnapshot:
    """Portfolio snapshot with 100k cash and no positions."""
    return PortfolioSnapshot(
        timestamp=datetime(2025, 1, 1),
        cash=100000.0,
        positions=[],
        nav=100000.0,
        daily_pnl=0.0,
        peak_nav=100000.0,
        drawdown=0.0,
    )


@pytest.fixture
def portfolio_with_position() -> PortfolioSnapshot:
    """Portfolio with a BTC position."""
    pos = Position(
        symbol="BTC/USDT",
        quantity=1.0,
        entry_price=100.0,
        current_price=105.0,
        unrealized_pnl=5.0,
        peak_price=107.0,
        state=StrategyState.HOLDING,
    )
    return PortfolioSnapshot(
        timestamp=datetime(2025, 1, 1),
        cash=90000.0,
        positions=[pos],
        nav=90105.0,
        daily_pnl=105.0,
        peak_nav=90107.0,
        drawdown=0.0,
    )


def make_filled_buy(
    symbol: str = "BTC/USDT",
    qty: float = 1.0,
    price: float = 100.0,
) -> Order:
    return Order(
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=qty,
        status=OrderStatus.FILLED,
        filled_price=price,
        filled_quantity=qty,
        filled_at=datetime.utcnow(),
    )


def make_filled_sell(
    symbol: str = "BTC/USDT",
    qty: float = 1.0,
    price: float = 110.0,
) -> Order:
    return Order(
        symbol=symbol,
        side=Side.SELL,
        order_type=OrderType.MARKET,
        quantity=qty,
        status=OrderStatus.FILLED,
        filled_price=price,
        filled_quantity=qty,
        filled_at=datetime.utcnow(),
    )


def make_signal(
    symbol: str = "BTC/USDT",
    alpha: float = 0.7,
    source: str = "rule_based",
) -> Signal:
    return Signal(
        symbol=symbol,
        alpha_score=alpha,
        confidence=abs(alpha),
        timestamp=datetime(2025, 1, 1),
        source=source,
    )
