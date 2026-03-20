from __future__ import annotations

import os

import pytest

from core.models import Order, OrderStatus, OrderType, Side
from execution.roostoo_executor import RoostooExecutor


def _smoke_enabled() -> bool:
    return os.getenv("RUN_ROOSTOO_SMOKE") == "1"


def _order_smoke_enabled() -> bool:
    return os.getenv("RUN_ROOSTOO_ORDER_SMOKE") == "1"


def _smoke_config() -> dict:
    api_key = os.getenv("ROOSTOO_COMP_API_KEY") or os.getenv("ROOSTOO_API_KEY", "")
    api_secret = os.getenv("ROOSTOO_COMP_API_SECRET") or os.getenv(
        "ROOSTOO_API_SECRET", ""
    )
    if not api_key or not api_secret:
        pytest.skip("Roostoo smoke credentials are not configured")
    return {
        "base_url": os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com"),
        "api_key": api_key,
        "api_secret": api_secret,
    }


@pytest.mark.asyncio
async def test_roostoo_live_balance_and_ticker_smoke():
    if not _smoke_enabled():
        pytest.skip("Set RUN_ROOSTOO_SMOKE=1 to run live Roostoo smoke tests")

    symbol = os.getenv("ROOSTOO_SMOKE_SYMBOL", "BTC/USDT")
    executor = RoostooExecutor(_smoke_config())
    try:
        await executor.start()
        balances = await executor.get_balance()
        ticker = await executor.get_ticker(symbol)
    finally:
        await executor.stop()

    assert "USD" in balances
    assert ticker is not None
    assert ticker > 0


@pytest.mark.asyncio
async def test_roostoo_live_order_smoke():
    if not _smoke_enabled() or not _order_smoke_enabled():
        pytest.skip(
            "Set RUN_ROOSTOO_SMOKE=1 and RUN_ROOSTOO_ORDER_SMOKE=1 to run live order smoke"
        )

    symbol = os.getenv("ROOSTOO_ORDER_SMOKE_SYMBOL", "BTC/USDT")
    quantity = float(os.getenv("ROOSTOO_ORDER_SMOKE_QTY", "0"))
    price = float(os.getenv("ROOSTOO_ORDER_SMOKE_PRICE", "0"))
    if quantity <= 0 or price <= 0:
        pytest.skip("Explicit ROOSTOO_ORDER_SMOKE_QTY and ROOSTOO_ORDER_SMOKE_PRICE are required")

    executor = RoostooExecutor(_smoke_config())
    try:
        await executor.start()
        order = Order(
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
        )
        result = await executor.execute(order)
    finally:
        await executor.stop()

    assert result.status in {
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.REJECTED,
    }
