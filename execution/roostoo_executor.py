"""Roostoo mock exchange executor — REST API order execution.

Uses Binance WebSocket for market data; Roostoo REST API only for order
execution and balance tracking.

Symbol mapping: internal BTC/USDT -> Roostoo BTC/USD at the boundary.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from core.models import Order, OrderStatus, OrderType, Side
from data.roostoo_auth import RoostooAuth
from execution.executor import BaseExecutor

logger = logging.getLogger(__name__)

# Max retries for transient HTTP errors (429, 5xx)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds


class RoostooExecutor(BaseExecutor):
    """Executes orders against the Roostoo mock exchange via REST API."""

    def __init__(self, config: dict):
        self._base_url: str = config.get("base_url", "https://mock-api.roostoo.com")
        api_key = config.get("api_key", "")
        api_secret = config.get("api_secret", "")
        self._auth = RoostooAuth(api_key, api_secret)
        self._session: Optional[aiohttp.ClientSession] = None
        # Pair info cache: symbol -> {min_qty, qty_precision, min_notional}
        self._pair_info: Dict[str, Dict[str, Any]] = {}
        self._trade_logger = None  # set externally via set_trade_logger

    def set_trade_logger(self, trade_logger) -> None:
        """Inject the TradeLogger for structured event logging."""
        self._trade_logger = trade_logger

    async def start(self) -> None:
        """Initialize HTTP session and cache exchange info."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
        # Validate server time
        ok = await self._auth.validate_server_time(self._base_url)
        if not ok:
            logger.warning("Server time drift is large; proceeding anyway")

        # Cache exchange info
        await self._load_exchange_info()
        logger.info("RoostooExecutor started, %d pairs loaded", len(self._pair_info))

    async def stop(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("RoostooExecutor stopped")

    # ── Symbol Mapping ──

    @staticmethod
    def to_roostoo_symbol(internal_symbol: str) -> str:
        """Convert internal symbol (BTC/USDT) to Roostoo (BTC/USD)."""
        return internal_symbol.replace("/USDT", "/USD")

    @staticmethod
    def to_internal_symbol(roostoo_symbol: str) -> str:
        """Convert Roostoo symbol (BTC/USD) to internal (BTC/USDT)."""
        return roostoo_symbol.replace("/USD", "/USDT")

    # ── Core Methods ──

    async def execute(self, order: Order) -> Order:
        """Place an order on Roostoo. POST /v3/place_order."""
        roostoo_pair = self.to_roostoo_symbol(order.symbol)
        timestamp = self._auth.get_timestamp()

        params = {
            "pair": roostoo_pair,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": str(self._round_quantity(order.symbol, order.quantity)),
            "timestamp": str(timestamp),
        }
        if order.order_type == OrderType.LIMIT and order.price is not None:
            params["price"] = str(order.price)

        start_time = time.monotonic()
        data = await self._signed_request("POST", "/v3/place_order", params)
        latency_ms = (time.monotonic() - start_time) * 1000

        if data and data.get("Success"):
            order_data = data.get("OrderDetail", {})
            executed_qty = float(order_data.get("FilledQuantity", order.quantity))
            order.filled_price = float(order_data.get("FilledAverPrice", 0) or order_data.get("Price", order.price or 0))
            order.filled_quantity = executed_qty
            order.order_id = str(order_data.get("OrderID", order.order_id))
            order.filled_at = datetime.utcnow()
            # Detect partial fills
            if abs(executed_qty - order.quantity) > 1e-10 and executed_qty < order.quantity:
                order.status = OrderStatus.PARTIALLY_FILLED
                logger.info(
                    "Roostoo order partially filled: %s %s %.6f/%.6f @ %.2f (latency %.0fms)",
                    order.side.value,
                    order.symbol,
                    executed_qty,
                    order.quantity,
                    order.filled_price,
                    latency_ms,
                )
            else:
                order.status = OrderStatus.FILLED
                logger.info(
                    "Roostoo order filled: %s %s %.6f @ %.2f (latency %.0fms)",
                    order.side.value,
                    order.symbol,
                    order.filled_quantity,
                    order.filled_price,
                    latency_ms,
                )
        elif data and not data.get("Success"):
            order.status = OrderStatus.REJECTED
            err = data.get("ErrMsg", "Unknown error")
            logger.error("Roostoo order rejected: %s — %s", order.symbol, err)
        else:
            order.status = OrderStatus.REJECTED
            logger.error("Roostoo order failed: no response for %s", order.symbol)

        # Log to trade logger
        if self._trade_logger:
            await self._trade_logger.log_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.filled_price or order.price,
                order_id=order.order_id,
                status=order.status.value,
                roostoo_response=data,
                latency_ms=latency_ms,
            )

        return order

    async def cancel(self, order_id: str, symbol: str) -> Order:
        """Cancel an order. POST /v3/cancel_order."""
        roostoo_pair = self.to_roostoo_symbol(symbol)
        timestamp = self._auth.get_timestamp()

        params = {
            "pair": roostoo_pair,
            "order_id": order_id,
            "timestamp": str(timestamp),
        }
        data = await self._signed_request("POST", "/v3/cancel_order", params)

        status = OrderStatus.CANCELLED
        if data and not data.get("Success"):
            logger.warning(
                "Cancel may have failed for %s: %s", order_id, data.get("ErrMsg")
            )

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
            status=status,
        )

    async def get_status(self, order_id: str, symbol: str) -> Order:
        """Query order status. POST /v3/query_order."""
        roostoo_pair = self.to_roostoo_symbol(symbol)
        timestamp = self._auth.get_timestamp()

        params = {
            "pair": roostoo_pair,
            "order_id": order_id,
            "timestamp": str(timestamp),
        }
        data = await self._signed_request("POST", "/v3/query_order", params)

        if data and data.get("Success"):
            matches = data.get("OrderMatched", [])
            if matches:
                order_data = matches[0]
                status_str = order_data.get("Status", "PENDING")
                status_map = {
                    "FILLED": OrderStatus.FILLED,
                    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                    "CANCELLED": OrderStatus.CANCELLED,
                    "NEW": OrderStatus.SUBMITTED,
                }
                status = status_map.get(status_str, OrderStatus.PENDING)
                side_str = order_data.get("Side", "BUY").upper()
                return Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=Side(side_str),
                    order_type=OrderType(order_data.get("Type", "MARKET")),
                    quantity=float(order_data.get("Quantity", 0)),
                    filled_quantity=float(order_data.get("FilledQuantity", 0)),
                    filled_price=float(order_data.get("FilledAverPrice", 0)) or None,
                    status=status,
                )

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
            status=OrderStatus.PENDING,
        )

    # ── Balance & Exchange Info ──

    async def get_balance(self) -> Dict[str, float]:
        """Get account balances. GET /v3/balance."""
        timestamp = self._auth.get_timestamp()
        params = {"timestamp": str(timestamp)}
        data = await self._signed_request("GET", "/v3/balance", params)
        logger.debug("Roostoo /v3/balance raw response: %s", data)

        balances: Dict[str, float] = {}
        if data and data.get("Success"):
            # API may return "SpotWallet" or "Wallet" depending on version
            wallet = data.get("SpotWallet") or data.get("Wallet") or {}
            for asset, amounts in wallet.items():
                free = float(amounts.get("Free", 0))
                if free > 0 or asset == "USD":
                    balances[asset] = free
        elif data:
            logger.warning("Roostoo balance failed: %s", data.get("ErrMsg", data))
        return balances

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange info (pair precisions, min orders). GET /v3/exchangeInfo."""
        data = await self._unsigned_request("GET", "/v3/exchangeInfo")
        return data or {}

    async def get_ticker(self, symbol: str) -> Optional[float]:
        """Get latest ticker price for a Roostoo pair. GET /v3/ticker."""
        roostoo_pair = self.to_roostoo_symbol(symbol)
        timestamp = self._auth.get_timestamp()
        params = {"pair": roostoo_pair, "timestamp": str(timestamp)}
        data = await self._signed_request("GET", "/v3/ticker", params)
        if data and data.get("Success"):
            ticker_data = data.get("Data", {})
            pair_data = ticker_data.get(roostoo_pair, {})
            price = pair_data.get("LastPrice", 0)
            if price:
                return float(price)
        return None

    # ── Internal Helpers ──

    async def _load_exchange_info(self) -> None:
        """Cache pair precision and min order info from exchange."""
        data = await self.get_exchange_info()
        if not data:
            logger.warning("Failed to load exchange info")
            return
        trade_pairs = data.get("TradePairs", {})
        if not trade_pairs:
            logger.warning("No TradePairs in exchange info response")
            return
        for symbol, info in trade_pairs.items():
            internal = self.to_internal_symbol(symbol)
            qty_precision = int(info.get("AmountPrecision", 8))
            self._pair_info[internal] = {
                "min_qty": 10 ** (-qty_precision),
                "qty_precision": qty_precision,
                "min_notional": float(info.get("MiniOrder", 0)),
            }

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to the pair's precision."""
        info = self._pair_info.get(symbol, {})
        precision = info.get("qty_precision", 8)
        return round(quantity, precision)

    async def _signed_request(
        self, method: str, endpoint: str, params: dict
    ) -> Optional[Dict]:
        """Make a signed request with retry on transient errors."""
        headers, query_string = self._auth.sign(params)
        url = f"{self._base_url}{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                if method == "GET":
                    resp = await self._session.get(
                        f"{url}?{query_string}", headers=headers
                    )
                else:
                    resp = await self._session.post(
                        url,
                        data=query_string,
                        headers={
                            **headers,
                            "Content-Type": "application/x-www-form-urlencoded",
                        },
                    )

                if resp.status in (429, 500, 502, 503, 504):
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Roostoo %s %s returned %d, retrying in %.1fs...",
                        method,
                        endpoint,
                        resp.status,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                data = await resp.json(content_type=None)

                if self._trade_logger:
                    await self._trade_logger.log_api(
                        endpoint=endpoint,
                        params=params,
                        response_code=resp.status,
                        success=resp.status == 200,
                    )
                return data

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Roostoo request error: %s, retrying in %.1fs", e, wait
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "Roostoo request failed after %d retries: %s", MAX_RETRIES, e
                    )
                    if self._trade_logger:
                        await self._trade_logger.log_api(
                            endpoint=endpoint,
                            params=params,
                            success=False,
                            error_msg=str(e),
                        )
        return None

    async def _unsigned_request(
        self, method: str, endpoint: str, params: Optional[dict] = None
    ) -> Optional[Dict]:
        """Make an unsigned request (public endpoints)."""
        url = f"{self._base_url}{endpoint}"
        try:
            if method == "GET":
                resp = await self._session.get(url, params=params)
            else:
                resp = await self._session.post(url, data=params)
            return await resp.json(content_type=None)
        except Exception as e:
            logger.error("Roostoo unsigned request failed: %s", e)
            return None
