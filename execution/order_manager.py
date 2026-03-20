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
        self._error_counts: Dict[str, int] = {}
        self._max_errors: int = 5

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

        if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            reported_qty = order.filled_quantity
            reported_price = order.filled_price
            order.filled_quantity = 0.0
            order.filled_price = None
            self._apply_fill_update(
                order,
                cumulative_filled_qty=reported_qty,
                cumulative_avg_price=reported_price,
                status=order.status,
            )
            if order.status == OrderStatus.PARTIALLY_FILLED:
                self._active_orders[order.order_id] = order
        elif order.status in (OrderStatus.SUBMITTED, OrderStatus.PENDING):
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
        cancelled.quantity = order.quantity
        cancelled.filled_quantity = order.filled_quantity
        cancelled.filled_price = order.filled_price
        logger.info("Cancelled order %s", order_id)

        # Notify callbacks of cancellation
        self._notify_callbacks(cancelled)

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
                prev_qty = order.filled_quantity or 0.0
                prev_avg = order.filled_price or 0.0
                updated = await self._executor.get_status(order_id, order.symbol)
                if updated.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    self._apply_fill_update(
                        order,
                        cumulative_filled_qty=updated.filled_quantity,
                        cumulative_avg_price=updated.filled_price,
                        status=updated.status,
                        previous_filled_qty=prev_qty,
                        previous_avg_price=prev_avg,
                    )
                    if updated.status == OrderStatus.FILLED:
                        to_remove.append(order_id)
                elif updated.status == OrderStatus.CANCELLED:
                    order.status = OrderStatus.CANCELLED
                    self._notify_callbacks(order)
                    to_remove.append(order_id)
                self._error_counts.pop(order_id, None)
            except Exception as e:
                logger.error("Error checking order %s: %s", order_id, e)
                self._error_counts[order_id] = self._error_counts.get(order_id, 0) + 1
                if self._error_counts[order_id] >= self._max_errors:
                    logger.warning(
                        "Order %s failed status check %d times, removing as CANCELLED",
                        order_id,
                        self._max_errors,
                    )
                    order.status = OrderStatus.CANCELLED
                    to_remove.append(order_id)
                    self._notify_callbacks(order)

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
        self._notify_callbacks(order)

    def _notify_callbacks(self, order: Order) -> None:
        for cb in self._fill_callbacks:
            try:
                cb(order)
            except Exception as e:
                logger.error("Fill callback error: %s", e)

    def _apply_fill_update(
        self,
        order: Order,
        cumulative_filled_qty: float,
        cumulative_avg_price: float | None,
        status: OrderStatus,
        *,
        previous_filled_qty: float | None = None,
        previous_avg_price: float | None = None,
    ) -> None:
        """Apply only the incremental fill delta to the tracker and callbacks.

        Executors and polling endpoints usually report cumulative filled quantity.
        Convert that into a delta so partial fills do not double-count position updates.
        """
        prev_qty = (
            previous_filled_qty
            if previous_filled_qty is not None
            else (order.filled_quantity or 0.0)
        )
        prev_avg = (
            previous_avg_price
            if previous_avg_price is not None
            else (order.filled_price or 0.0)
        )
        new_qty = max(0.0, cumulative_filled_qty or 0.0)
        delta_qty = max(0.0, new_qty - prev_qty)

        order.status = status
        order.filled_quantity = new_qty
        if cumulative_avg_price is not None:
            order.filled_price = cumulative_avg_price

        if delta_qty <= 1e-12:
            return

        delta_price = cumulative_avg_price
        if cumulative_avg_price is not None and prev_qty > 0 and prev_avg > 0:
            delta_value = cumulative_avg_price * new_qty - prev_avg * prev_qty
            if abs(delta_value) > 1e-12:
                delta_price = delta_value / delta_qty

        fill_event = order.model_copy(
            update={
                "quantity": delta_qty,
                "filled_quantity": delta_qty,
                "filled_price": delta_price,
                "status": status,
            }
        )
        self._on_fill(fill_event)

    @property
    def active_orders(self) -> Dict[str, Order]:
        return dict(self._active_orders)

    @property
    def has_pending(self) -> bool:
        return len(self._active_orders) > 0
