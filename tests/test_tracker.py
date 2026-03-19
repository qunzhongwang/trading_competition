"""Tests for risk/tracker.py — PortfolioTracker cash, positions, PnL, NAV."""

import pytest

from core.models import StrategyState
from risk.tracker import PortfolioTracker
from tests.conftest import make_filled_buy, make_filled_sell


@pytest.fixture
def tracker():
    return PortfolioTracker(initial_capital=100_000.0, fee_bps=10.0)


class TestInitialization:
    def test_initial_state(self, tracker):
        snap = tracker.snapshot()
        assert snap.cash == 100_000.0
        assert snap.nav == 100_000.0
        assert snap.drawdown == 0.0
        assert snap.daily_pnl == 0.0
        assert len(snap.positions) == 0


class TestBuyFill:
    def test_cash_decreases(self, tracker):
        order = make_filled_buy(price=100.0, qty=10.0)
        tracker.on_fill(order)
        # cost = 100*10 = 1000, fee = 1000 * 0.001 = 1.0
        assert tracker.snapshot().cash == pytest.approx(100_000.0 - 1000.0 - 1.0)

    def test_position_created(self, tracker):
        order = make_filled_buy(price=50.0, qty=5.0)
        tracker.on_fill(order)
        pos = tracker.get_position("BTC/USDT")
        assert pos.quantity == 5.0
        assert pos.entry_price == 50.0
        assert pos.state == StrategyState.HOLDING

    def test_weighted_average_entry(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        tracker.on_fill(make_filled_buy(price=200.0, qty=1.0))
        pos = tracker.get_position("BTC/USDT")
        assert pos.quantity == 2.0
        assert pos.entry_price == pytest.approx(150.0)


class TestSellFill:
    def test_full_sell_clears_position(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=2.0))
        tracker.on_fill(make_filled_sell(price=110.0, qty=2.0))
        pos = tracker.get_position("BTC/USDT")
        assert pos.quantity == 0.0
        assert pos.state == StrategyState.FLAT

    def test_realized_pnl(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=2.0))
        tracker.on_fill(make_filled_sell(price=110.0, qty=2.0))
        pos = tracker.get_position("BTC/USDT")
        # PnL = (110 - 100) * 2 = 20
        assert pos.realized_pnl == pytest.approx(20.0)

    def test_partial_sell(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=10.0))
        tracker.on_fill(make_filled_sell(price=110.0, qty=5.0))
        pos = tracker.get_position("BTC/USDT")
        assert pos.quantity == 5.0
        assert pos.realized_pnl == pytest.approx(50.0)

    def test_cash_increases_on_sell(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        cash_after_buy = tracker.snapshot().cash
        tracker.on_fill(make_filled_sell(price=110.0, qty=1.0))
        # sell proceeds = 110*1 = 110, fee = 110 * 0.001 = 0.11
        assert tracker.snapshot().cash == pytest.approx(cash_after_buy + 110.0 - 0.11)


class TestNAVAndDrawdown:
    def test_nav_includes_positions(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=10.0))
        tracker.update_prices("BTC/USDT", 100.0)
        nav = tracker.snapshot().nav
        # cash = 100000 - 1000 - 1 = 98999, pos value = 100 * 10 = 1000
        assert nav == pytest.approx(98999.0 + 1000.0)

    def test_drawdown_calculation(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=100.0))
        # Peak NAV after buy should be ~100000
        tracker.update_prices("BTC/USDT", 80.0)  # 20% drop
        snap = tracker.snapshot()
        assert snap.drawdown > 0

    def test_peak_nav_updates(self, tracker):
        initial_peak = tracker.snapshot().peak_nav
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        tracker.update_prices("BTC/USDT", 200.0)
        # Trigger NAV recalc via on_fill
        tracker.on_fill(make_filled_sell(price=200.0, qty=0.5))
        assert tracker.snapshot().peak_nav >= initial_peak


class TestExposure:
    def test_zero_exposure_no_positions(self, tracker):
        assert tracker.get_exposure("BTC/USDT") == 0.0
        assert tracker.get_total_exposure() == 0.0

    def test_single_exposure(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=10.0))
        tracker.update_prices("BTC/USDT", 100.0)
        exp = tracker.get_exposure("BTC/USDT")
        # ~1000 / ~100000 ≈ 1%
        assert 0.009 < exp < 0.012

    def test_total_exposure(self, tracker):
        tracker.on_fill(make_filled_buy(symbol="BTC/USDT", price=100.0, qty=10.0))
        tracker.on_fill(make_filled_buy(symbol="ETH/USDT", price=50.0, qty=20.0))
        tracker.update_prices("BTC/USDT", 100.0)
        tracker.update_prices("ETH/USDT", 50.0)
        total = tracker.get_total_exposure()
        assert total > 0


class TestUpdatePrices:
    def test_updates_unrealized_pnl(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=5.0))
        tracker.update_prices("BTC/USDT", 120.0)
        pos = tracker.get_position("BTC/USDT")
        assert pos.unrealized_pnl == pytest.approx(100.0)  # (120-100)*5

    def test_updates_peak_price(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=1.0))
        tracker.update_prices("BTC/USDT", 150.0)
        pos = tracker.get_position("BTC/USDT")
        assert pos.peak_price == 150.0


class TestResetDaily:
    def test_resets_daily_start_nav(self, tracker):
        tracker.on_fill(make_filled_buy(price=100.0, qty=10.0))
        tracker.reset_daily()
        snap = tracker.snapshot()
        assert snap.daily_pnl == pytest.approx(0.0, abs=0.1)


class TestInvalidFill:
    def test_fill_with_no_price(self, tracker):
        order = make_filled_buy(price=100.0, qty=1.0)
        order.filled_price = None
        tracker.on_fill(order)
        # Should be a no-op
        assert tracker.snapshot().cash == 100_000.0

    def test_fill_with_zero_qty(self, tracker):
        order = make_filled_buy(price=100.0, qty=0.0)
        order.filled_quantity = 0.0
        tracker.on_fill(order)
        assert tracker.snapshot().cash == 100_000.0
