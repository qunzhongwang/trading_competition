from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import List

import websockets

from core.models import OHLCV, Tick
from data.buffer import LiveBuffer

logger = logging.getLogger(__name__)


class WSConnector:
    """Connects to exchange WebSocket, parses messages, pushes to buffer.

    Currently supports Binance kline and trade streams.
    Auto-reconnects with exponential backoff.
    """

    def __init__(self, config: dict, buffer: LiveBuffer):
        exchange_cfg = config.get("exchange", {})
        self._ws_url: str = exchange_cfg.get("ws_url", "wss://stream.binance.com:9443/ws")
        self._symbols: List[str] = config.get("symbols", [])
        self._interval: str = config.get("data", {}).get("candle_interval", "1m")
        self._buffer = buffer
        self._reconnect_delay: float = 1.0
        self._max_reconnect_delay: float = 60.0
        self._running = False

    async def start(self) -> None:
        """Launch one WebSocket task per symbol stream."""
        self._running = True
        tasks = [self._listen(sym) for sym in self._symbols]
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        self._running = False

    async def _listen(self, symbol: str) -> None:
        """Main loop: connect, subscribe, parse messages."""
        stream_symbol = symbol.replace("/", "").lower()
        stream_name = f"{stream_symbol}@kline_{self._interval}"
        url = f"{self._ws_url}/{stream_name}"

        delay = self._reconnect_delay

        while self._running:
            try:
                logger.info("Connecting to %s", url)
                async with websockets.connect(url) as ws:
                    delay = self._reconnect_delay  # reset on success
                    logger.info("Connected: %s", stream_name)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw_msg)
                            await self._handle_message(msg, symbol)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON: %s", raw_msg[:100])

            except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                if not self._running:
                    break
                logger.warning("Connection lost for %s: %s. Reconnecting in %.1fs", symbol, e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

            except Exception as e:
                logger.error("Unexpected error for %s: %s", symbol, e)
                if not self._running:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

    async def _handle_message(self, msg: dict, symbol: str) -> None:
        """Parse Binance kline message and push to buffer."""
        if "k" not in msg:
            return

        k = msg["k"]
        candle = OHLCV(
            symbol=symbol,
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            timestamp=datetime.utcfromtimestamp(k["t"] / 1000),
            is_closed=k.get("x", False),
        )
        await self._buffer.push_candle(candle)


class BinanceSupplementaryFeed:
    """Supplementary data feed from free public Binance API.

    Collects: order book depth, funding rate, taker buy/sell ratio.
    All endpoints are free and require no API key.
    """

    SPOT_WS_URL = "wss://stream.binance.com:9443/ws"
    FUTURES_WS_URL = "wss://fstream.binance.com/ws"
    FUTURES_REST_URL = "https://fapi.binance.com"

    def __init__(self, symbols: list, buffer: LiveBuffer):
        self._symbols = symbols
        self._buffer = buffer
        self._running = False

    async def start(self) -> None:
        self._running = True
        tasks = [
            asyncio.create_task(self._listen_depth()),
            asyncio.create_task(self._listen_funding()),
            asyncio.create_task(self._poll_taker_ratio()),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        self._running = False

    async def _listen_depth(self) -> None:
        """Subscribe to order book depth for all symbols."""
        streams = "/".join(
            f"{s.replace('/', '').lower()}@depth20@100ms" for s in self._symbols
        )
        url = f"{self.SPOT_WS_URL}/{streams}"

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Depth WS connected: %d symbols", len(self._symbols))
                    while self._running:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        # Combined stream format: {"stream": "...", "data": {...}}
                        if "data" in data:
                            data = data["data"]
                        # Convert stream name back to symbol
                        symbol = self._stream_to_symbol(data.get("s", ""))
                        if symbol:
                            await self._buffer.push_depth(
                                symbol, data.get("bids", []), data.get("asks", [])
                            )
            except Exception as e:
                logger.warning("Depth WS error: %s, reconnecting...", e)
                await asyncio.sleep(5)

    async def _listen_funding(self) -> None:
        """Subscribe to mark price stream for funding rate."""
        streams = "/".join(
            f"{s.replace('/', '').lower()}@markPrice@1s" for s in self._symbols
        )
        url = f"{self.FUTURES_WS_URL}/{streams}"

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info("Funding WS connected: %d symbols", len(self._symbols))
                    while self._running:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)
                        if "data" in data:
                            data = data["data"]
                        symbol = self._stream_to_symbol(data.get("s", ""))
                        if symbol:
                            rate = float(data.get("r", 0.0))
                            await self._buffer.push_funding(symbol, rate)
            except Exception as e:
                logger.warning("Funding WS error: %s, reconnecting...", e)
                await asyncio.sleep(5)

    async def _poll_taker_ratio(self) -> None:
        """Poll taker long/short ratio every 5 minutes."""
        import aiohttp

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    for symbol in self._symbols:
                        api_symbol = symbol.replace("/", "")
                        url = (
                            f"{self.FUTURES_REST_URL}/futures/data/takerlongshortRatio"
                            f"?symbol={api_symbol}&period=5m&limit=1"
                        )
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data:
                                    ratio = float(data[0].get("buySellRatio", 1.0))
                                    await self._buffer.push_taker_ratio(symbol, ratio)
            except Exception as e:
                logger.warning("Taker ratio poll error: %s", e)
            await asyncio.sleep(300)  # 5 minutes

    def _stream_to_symbol(self, raw_symbol: str) -> str:
        """Convert Binance symbol (BTCUSDT) to our format (BTC/USDT)."""
        for sym in self._symbols:
            if sym.replace("/", "") == raw_symbol:
                return sym
        return ""
