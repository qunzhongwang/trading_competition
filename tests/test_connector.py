from __future__ import annotations

import asyncio

import pytest

from data.buffer import LiveBuffer
from data.connector import BinanceSupplementaryFeed, WSConnector


class _DummyConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
class TestConnectorShutdown:
    async def test_ws_connector_stop_closes_connections_and_tasks(self):
        connector = WSConnector(
            {
                "symbols": ["BTC/USDT"],
                "exchange": {"ws_url": "wss://example.invalid/ws"},
            },
            LiveBuffer(),
        )
        connection = _DummyConnection()
        task = asyncio.create_task(asyncio.sleep(60))

        connector._running = True
        connector._connections.add(connection)
        connector._tasks = [task]

        await connector.stop()

        assert connector._running is False
        assert connection.closed is True
        assert task.cancelled()

    async def test_supplementary_feed_stop_closes_connections_and_tasks(self):
        feed = BinanceSupplementaryFeed(["BTC/USDT"], LiveBuffer())
        connection = _DummyConnection()
        task = asyncio.create_task(asyncio.sleep(60))

        feed._running = True
        feed._connections.add(connection)
        feed._tasks = [task]

        await feed.stop()

        assert feed._running is False
        assert connection.closed is True
        assert task.cancelled()
