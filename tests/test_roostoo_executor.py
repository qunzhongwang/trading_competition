from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.models import Order, OrderStatus, OrderType, Side
from execution.roostoo_executor import RoostooExecutor


class TestRoostooSymbolMapping:
    def test_symbol_roundtrip(self):
        assert RoostooExecutor.to_roostoo_symbol("BTC/USDT") == "BTC/USD"
        assert RoostooExecutor.to_internal_symbol("BTC/USD") == "BTC/USDT"


class TestRoostooExecutor:
    @pytest.mark.asyncio
    async def test_execute_maps_filled_response(self):
        executor = RoostooExecutor({"api_key": "key", "api_secret": "secret"})
        executor._pair_info["BTC/USDT"] = {"qty_precision": 4}
        executor._signed_request = AsyncMock(
            return_value={
                "Success": True,
                "OrderDetail": {
                    "OrderID": "abc123",
                    "FilledQuantity": "1.25",
                    "FilledAverPrice": "101.5",
                },
            }
        )

        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.25,
        )
        result = await executor.execute(order)

        assert result.status == OrderStatus.FILLED
        assert result.order_id == "abc123"
        assert result.filled_quantity == pytest.approx(1.25)
        assert result.filled_price == pytest.approx(101.5)

    @pytest.mark.asyncio
    async def test_execute_keeps_partial_fill_active(self):
        executor = RoostooExecutor({"api_key": "key", "api_secret": "secret"})
        executor._pair_info["BTC/USDT"] = {"qty_precision": 4}
        executor._signed_request = AsyncMock(
            return_value={
                "Success": True,
                "OrderDetail": {
                    "OrderID": "partial-1",
                    "FilledQuantity": "0.4",
                    "FilledAverPrice": "100.25",
                },
            }
        )

        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=100.0,
        )
        result = await executor.execute(order)

        assert result.status == OrderStatus.PARTIALLY_FILLED
        assert result.filled_quantity == pytest.approx(0.4)
        assert result.filled_price == pytest.approx(100.25)

    @pytest.mark.asyncio
    async def test_get_balance_parses_spot_wallet(self):
        executor = RoostooExecutor({"api_key": "key", "api_secret": "secret"})
        executor._signed_request = AsyncMock(
            return_value={
                "Success": True,
                "SpotWallet": {
                    "USD": {"Free": "2500.0"},
                    "BTC": {"Free": "0.15"},
                    "ETH": {"Free": "0"},
                },
            }
        )

        balances = await executor.get_balance()

        assert balances == {"USD": 2500.0, "BTC": 0.15}

    @pytest.mark.asyncio
    async def test_get_balance_parses_legacy_wallet_shape(self):
        executor = RoostooExecutor({"api_key": "key", "api_secret": "secret"})
        executor._signed_request = AsyncMock(
            return_value={
                "Success": True,
                "Wallet": {
                    "USD": {"Free": "100.0"},
                    "SOL": {"Free": "3.5"},
                },
            }
        )

        balances = await executor.get_balance()

        assert balances == {"USD": 100.0, "SOL": 3.5}

    @pytest.mark.asyncio
    async def test_get_balance_preserves_zero_usd_snapshot(self):
        executor = RoostooExecutor({"api_key": "key", "api_secret": "secret"})
        executor._signed_request = AsyncMock(
            return_value={
                "Success": True,
                "SpotWallet": {},
            }
        )

        balances = await executor.get_balance()

        assert balances == {"USD": 0.0}

    @pytest.mark.asyncio
    async def test_load_exchange_info_caches_pair_contract(self):
        executor = RoostooExecutor({"api_key": "key", "api_secret": "secret"})
        executor.get_exchange_info = AsyncMock(
            return_value={
                "TradePairs": {
                    "BTC/USD": {"AmountPrecision": 4, "MiniOrder": "10"},
                    "ETH/USD": {"AmountPrecision": 3, "MiniOrder": "5"},
                }
            }
        )

        await executor._load_exchange_info()

        assert executor._pair_info["BTC/USDT"] == {
            "min_qty": pytest.approx(0.0001),
            "qty_precision": 4,
            "min_notional": pytest.approx(10.0),
        }
        assert executor._pair_info["ETH/USDT"] == {
            "min_qty": pytest.approx(0.001),
            "qty_precision": 3,
            "min_notional": pytest.approx(5.0),
        }

    @pytest.mark.asyncio
    async def test_get_ticker_parses_last_price(self):
        executor = RoostooExecutor({"api_key": "key", "api_secret": "secret"})
        executor._signed_request = AsyncMock(
            return_value={
                "Success": True,
                "Data": {
                    "BTC/USD": {"LastPrice": "103245.12"},
                },
            }
        )

        price = await executor.get_ticker("BTC/USDT")

        assert price == pytest.approx(103245.12)

    @pytest.mark.asyncio
    async def test_get_status_maps_exchange_status(self):
        executor = RoostooExecutor({"api_key": "key", "api_secret": "secret"})
        executor._signed_request = AsyncMock(
            return_value={
                "Success": True,
                "OrderMatched": [
                    {
                        "Status": "PARTIALLY_FILLED",
                        "Side": "SELL",
                        "Type": "LIMIT",
                        "Quantity": "2.0",
                        "FilledQuantity": "0.75",
                        "FilledAverPrice": "99.8",
                    }
                ],
            }
        )

        order = await executor.get_status("oid-1", "BTC/USDT")

        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.side == Side.SELL
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == pytest.approx(2.0)
        assert order.filled_quantity == pytest.approx(0.75)
        assert order.filled_price == pytest.approx(99.8)
