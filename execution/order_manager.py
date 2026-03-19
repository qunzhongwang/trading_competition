from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Dict, List

from core.models import Order, OrderStatus
from execution.executor import BaseExecutor
from risk.tracker import PortfolioTracker

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order lifecycle: submit, track, cancel, and notify on fills."""

    def __init__(
        self,
        executor: BaseExecutor,
        tracker: PortfolioTracker,
        timeout_seconds: float = 0,
    ):
        self._executor = executor
        self._tracker = tracker
        self._active_orders: Dict[str, Order] = {}
        self._fill_callbacks: List[Callable[[Order], None]] = []
        self._timeout_seconds = timeout_seconds

    async def submit(self, order: Order) -> Order:
        """Submit order to executor and handle the result."""
        logger.info(
            "Submitting: %s %s %s qty=%.6f",
            order.side.value,
            order.symbol,
            order.order_type.value,
            order.quantity,
        )

        order = await self._executor.execute(order)

        if order.status == OrderStatus.FILLED:
            self._on_fill(order)
        elif order.status in (
            OrderStatus.SUBMITTED,
            OrderStatus.PENDING,
            OrderStatus.PARTIALLY_FILLED,
        ):
            self._active_orders[order.order_id] = order
        else:
            logger.warning("Order %s status: %s", order.order_id, order.status.value)

        return order

    async def cancel(self, order_id: str) -> None:
        """Cancel an active order."""
        order = self._active_orders.get(order_id)
        if order is None:
            logger.warning("Order %s not found in active orders", order_id)
            return

        cancelled = await self._executor.cancel(order_id, order.symbol)
        self._active_orders.pop(order_id, None)
        logger.info("Cancelled order %s", order_id)

        # Notify callbacks of cancellation
        for cb in self._fill_callbacks:
            try:
                cb(cancelled)
            except Exception as e:
                logger.error("Callback error on cancel: %s", e)

    async def check_pending(self) -> None:
        """Check and update status of pending orders. Cancel stale orders."""
        now = datetime.utcnow()
        to_remove = []
        to_cancel = []

        # Check for timed-out orders first
        if self._timeout_seconds > 0:
            for order_id, order in self._active_orders.items():
                age = (now - order.created_at).total_seconds()
                if age > self._timeout_seconds:
                    to_cancel.append(order_id)
                    logger.info(
                        "Order %s timed out after %.0fs (limit: %.0fs), cancelling",
                        order_id,
                        age,
                        self._timeout_seconds,
                    )

        for order_id in to_cancel:
            await self.cancel(order_id)

        for order_id, order in self._active_orders.items():
            try:
                updated = await self._executor.get_status(order_id, order.symbol)
                if updated.status == OrderStatus.FILLED:
                    order.filled_price = updated.filled_price
                    order.filled_quantity = updated.filled_quantity
                    order.status = OrderStatus.FILLED
                    self._on_fill(order)
                    to_remove.append(order_id)
                elif updated.status == OrderStatus.CANCELLED:
                    to_remove.append(order_id)
            except Exception as e:
                logger.error("Error checking order %s: %s", order_id, e)

        for order_id in to_remove:
            self._active_orders.pop(order_id, None)

    async def cancel_all(self) -> None:
        """Cancel all active orders (used during shutdown)."""
        if not self._active_orders:
            return
        logger.info("Cancelling %d active orders...", len(self._active_orders))
        for order_id in list(self._active_orders.keys()):
            await self.cancel(order_id)

    def register_fill_callback(self, cb: Callable[[Order], None]) -> None:
        """Register a callback to be called on order fills."""
        self._fill_callbacks.append(cb)

    def _on_fill(self, order: Order) -> None:
        """Handle a filled order."""
        # Update portfolio tracker
        self._tracker.on_fill(order)

        # Notify all registered callbacks
        for cb in self._fill_callbacks:
            try:
                cb(order)
            except Exception as e:
                logger.error("Fill callback error: %s", e)

    @property
    def active_orders(self) -> Dict[str, Order]:
        return dict(self._active_orders)

    @property
    def has_pending(self) -> bool:
        return len(self._active_orders) > 0
