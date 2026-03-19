"""Tests for strategy/trade_tracker.py — adaptive Kelly parameters."""

from __future__ import annotations


from strategy.trade_tracker import TradeTracker


class TestTradeTracker:
    def test_no_trades_returns_priors(self):
        tt = TradeTracker(prior_win_rate=0.55, prior_payoff=1.5)
        wr, payoff = tt.get_kelly_params()
        assert wr == 0.55
        assert payoff == 1.5

    def test_all_wins_shifts_win_rate_up(self):
        tt = TradeTracker(
            window=50, min_trades=5, prior_win_rate=0.55, prior_payoff=1.5
        )
        for _ in range(10):
            tt.record_trade(100.0, 110.0)

        wr, payoff = tt.get_kelly_params()
        assert wr > 0.55  # shifted toward 1.0
        assert wr <= 1.0

    def test_all_losses_shifts_win_rate_down(self):
        tt = TradeTracker(
            window=50, min_trades=5, prior_win_rate=0.55, prior_payoff=1.5
        )
        for _ in range(10):
            tt.record_trade(100.0, 90.0)

        wr, payoff = tt.get_kelly_params()
        assert wr < 0.55  # shifted toward 0.0

    def test_blending_with_few_trades(self):
        tt = TradeTracker(
            window=50, min_trades=10, prior_win_rate=0.55, prior_payoff=1.5
        )
        # Only 2 trades — alpha = 0.2, mostly prior
        tt.record_trade(100.0, 110.0)
        tt.record_trade(100.0, 110.0)

        wr, _ = tt.get_kelly_params()
        # Should be close to prior (alpha=0.2 → 0.2*1.0 + 0.8*0.55 = 0.64)
        assert 0.55 < wr < 0.75

    def test_n_trades_count(self):
        tt = TradeTracker()
        assert tt.n_trades == 0
        tt.record_trade(100.0, 110.0)
        assert tt.n_trades == 1

    def test_zero_entry_price_ignored(self):
        tt = TradeTracker()
        tt.record_trade(0.0, 110.0)
        assert tt.n_trades == 0
