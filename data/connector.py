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
