from __future__ import annotations

import copy

import pytest

import main as main_module
from core.models import OrderStatus, Side, StrategyState


def _build_roostoo_config(default_config: dict) -> dict:
    config = copy.deepcopy(default_config)
    config["mode"] = "roostoo"
    config["symbols"] = ["BTC/USDT", "ETH/USDT"]
    config.setdefault("strategy", {})["profile"] = "regime_trend_v1"
    config["exchange"] = {"name": "binance", "ws_url": "wss://example.invalid/ws"}
    config["roostoo"] = {
        "base_url": "https://mock-api.roostoo.com",
        "api_key": "test-key",
        "api_secret": "test-secret",
    }
    config["execution"] = {"order_timeout_seconds": 0, "limit_offset_bps": 5}
    config["paper"]["initial_capital"] = 1_000_000.0
    return config


class _RecordingTracker(main_module.PortfolioTracker):
    created: list["_RecordingTracker"] = []

    def __init__(self, initial_capital: float, fee_bps: float = 10.0):
        super().__init__(initial_capital, fee_bps)
        self.__class__.created.append(self)


class _FakeStrategy:
    def __init__(self) -> None:
        self._state = StrategyState.FLAT
        self._entry_price = 0.0


class _FakeFeed:
    def __init__(self, config, buffer):
        self.config = config
        self.buffer = buffer
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


class _FakeTradeLogger:
    pass


class _FakeOrderManager:
    created: "_FakeOrderManager | None" = None

    def __init__(self, executor, tracker, timeout_seconds: float = 0):
        self.executor = executor
        self.tracker = tracker
        self.timeout_seconds = timeout_seconds
        self.submitted = []
        self.cancelled_all = False
        self.__class__.created = self

    async def submit(self, order):
        self.submitted.append(order)
        order.status = OrderStatus.FILLED
        return order

    async def cancel_all(self) -> None:
        self.cancelled_all = True


class _FakeMonitor:
    created: "_FakeMonitor | None" = None

    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.strategies = {sym: _FakeStrategy() for sym in config.get("symbols", [])}
        self.stopped = False
        self.__class__.created = self

    async def run(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True


class _FakeRoostooExecutor:
    balances: dict[str, float] = {}
    tickers: dict[str, float] = {}
    created: "_FakeRoostooExecutor | None" = None

    def __init__(self, config: dict):
        self.config = config
        self.trade_logger = None
        self.started = False
        self.stopped = False
        self.__class__.created = self

    def set_trade_logger(self, trade_logger) -> None:
        self.trade_logger = trade_logger

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def get_balance(self) -> dict[str, float]:
        return dict(self.__class__.balances)

    async def get_ticker(self, symbol: str) -> float | None:
        return self.__class__.tickers.get(symbol)


def _install_roostoo_mode_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingTracker.created = []
    _FakeOrderManager.created = None
    _FakeMonitor.created = None
    _FakeRoostooExecutor.created = None

    monkeypatch.setattr(main_module, "PortfolioTracker", _RecordingTracker)
    monkeypatch.setattr(main_module, "OrderManager", _FakeOrderManager)
    monkeypatch.setattr(main_module, "StrategyMonitor", _FakeMonitor)
    monkeypatch.setattr(main_module, "RoostooExecutor", _FakeRoostooExecutor)
    monkeypatch.setattr(main_module, "WSConnector", _FakeFeed)
    monkeypatch.setattr(main_module, "BinanceSupplementaryFeed", _FakeFeed)
    monkeypatch.setattr(main_module, "TradeLogger", _FakeTradeLogger)
    monkeypatch.setattr(main_module.signal, "signal", lambda *args, **kwargs: None)

    async def _noop_prefetch(*args, **kwargs):
        return None

    monkeypatch.setattr(main_module, "prefetch_candles", _noop_prefetch)


class TestMainRoostooMode:
    @pytest.mark.asyncio
    async def test_restores_positions_and_strategy_state_from_balances(
        self, default_config, monkeypatch
    ):
        _install_roostoo_mode_fakes(monkeypatch)
        _FakeRoostooExecutor.balances = {"USD": 1250.0, "ETH": 2.5}
        _FakeRoostooExecutor.tickers = {"ETH/USDT": 3200.0}

        await main_module.main(_build_roostoo_config(default_config))

        tracker = _RecordingTracker.created[-1]
        eth_position = tracker.get_position("ETH/USDT")
        assert tracker.snapshot().cash == pytest.approx(1250.0)
        assert tracker.snapshot().nav == pytest.approx(9250.0)
        assert tracker.snapshot().daily_pnl == pytest.approx(0.0)
        assert eth_position.quantity == pytest.approx(2.5)
        assert eth_position.entry_price == pytest.approx(3200.0)
        assert eth_position.state == StrategyState.HOLDING

        monitor = _FakeMonitor.created
        assert monitor is not None
        assert monitor.config["strategy"]["profile"] == "regime_trend_v1"
        assert monitor.strategies["ETH/USDT"]._state == StrategyState.HOLDING
        assert monitor.strategies["ETH/USDT"]._entry_price == pytest.approx(3200.0)
        assert _FakeOrderManager.created is not None
        assert _FakeOrderManager.created.submitted == []

    @pytest.mark.asyncio
    async def test_submits_seed_trade_when_account_has_no_positions(
        self, default_config, monkeypatch
    ):
        _install_roostoo_mode_fakes(monkeypatch)
        _FakeRoostooExecutor.balances = {"USD": 250.0}
        _FakeRoostooExecutor.tickers = {"BTC/USDT": 50000.0}

        await main_module.main(_build_roostoo_config(default_config))

        tracker = _RecordingTracker.created[-1]
        order_manager = _FakeOrderManager.created
        assert tracker.snapshot().cash == pytest.approx(250.0)
        assert order_manager is not None
        assert len(order_manager.submitted) == 1

        seed_order = order_manager.submitted[0]
        assert seed_order.symbol == "BTC/USDT"
        assert seed_order.side == Side.BUY
        assert seed_order.quantity == pytest.approx(2.0 / 50000.0)

    @pytest.mark.asyncio
    async def test_dust_position_does_not_block_seed_trade(
        self, default_config, monkeypatch
    ):
        _install_roostoo_mode_fakes(monkeypatch)
        _FakeRoostooExecutor.balances = {"USD": 250.0, "BTC": 0.00001}
        _FakeRoostooExecutor.tickers = {"BTC/USDT": 50000.0}

        await main_module.main(_build_roostoo_config(default_config))

        order_manager = _FakeOrderManager.created
        assert order_manager is not None
        assert len(order_manager.submitted) == 1
        assert order_manager.submitted[0].symbol == "BTC/USDT"
