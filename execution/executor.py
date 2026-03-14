from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from core.models import Order

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    async def execute(self, order: Order) -> Order:
        """Execute an order. Returns the order with updated status/fill info."""
        ...

    @abstractmethod
    async def cancel(self, order_id: str, symbol: str) -> Order:
        """Cancel an open order."""
        ...

    @abstractmethod
    async def get_status(self, order_id: str, symbol: str) -> Order:
        """Get current status of an order."""
        ...


class LiveExecutor(BaseExecutor):
    """Executes orders against a real exchange via ccxt."""

    def __init__(self, config: dict):
        self._exchange_name: str = config.get("name", "binance")
        self._api_key: str = config.get("api_key", "")
        self._api_secret: str = config.get("api_secret", "")
        self._exchange = None

    async def start(self) -> None:
        import ccxt.async_support as ccxt_async

        exchange_class = getattr(ccxt_async, self._exchange_name)
        self._exchange = exchange_class({
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "enableRateLimit": True,
        })
        logger.info("LiveExecutor connected to %s", self._exchange_name)

    async def stop(self) -> None:
        if self._exchange:
            await self._exchange.close()

    async def execute(self, order: Order) -> Order:
        from core.models import OrderStatus, OrderType
        from datetime import datetime

        try:
            if order.order_type == OrderType.MARKET:
                result = await self._exchange.create_market_order(
                    order.symbol,
                    order.side.value.lower(),
                    order.quantity,
                )
            else:
                result = await self._exchange.create_limit_order(
                    order.symbol,
                    order.side.value.lower(),
                    order.quantity,
                    order.price,
                )

            order.status = OrderStatus.FILLED
            order.filled_price = float(result.get("average", result.get("price", 0)))
            order.filled_quantity = float(result.get("filled", order.quantity))
            order.filled_at = datetime.utcnow()

            logger.info(
                "Live order filled: %s %s %.6f @ %.2f",
                order.side.value, order.symbol, order.filled_quantity, order.filled_price,
            )

        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error("Live order failed: %s", e)

        return order

    async def cancel(self, order_id: str, symbol: str) -> Order:
        try:
            await self._exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled order %s", order_id)
        except Exception as e:
            logger.error("Cancel failed: %s", e)
        return Order(order_id=order_id, symbol=symbol, side="BUY", order_type="MARKET", quantity=0)

    async def get_status(self, order_id: str, symbol: str) -> Order:
        result = await self._exchange.fetch_order(order_id, symbol)
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=result["side"].upper(),
            order_type=result["type"].upper(),
            quantity=float(result["amount"]),
            filled_quantity=float(result.get("filled", 0)),
            filled_price=float(result.get("average", 0)) if result.get("average") else None,
        )
