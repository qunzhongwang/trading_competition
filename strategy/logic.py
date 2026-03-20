from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from core.models import (
    FactorBias,
    FactorSnapshot,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Side,
    Signal,
    StrategyIntent,
    StrategyState,
    TradeInstruction,
    UrgencyLevel,
)

logger = logging.getLogger(__name__)


class StrategyLogic:
    """Per-symbol state machine for the long-only strategy.

    State transitions:
        FLAT + alpha > entry_threshold (confirmed N bars) → emit BUY → LONG_PENDING
        LONG_PENDING + partial fill → LONG_PENDING
        LONG_PENDING + fill → HOLDING
        LONG_PENDING + cancel/reject → FLAT or HOLDING (if partial fill exists)
        HOLDING + alpha < exit_threshold → emit SELL → EXIT_PENDING
        EXIT_PENDING + fill → FLAT
        EXIT_PENDING + cancel/reject → HOLDING
        HOLDING + risk stop → emit SELL → FLAT (handled externally)
    """

    def __init__(self, symbol: str, config: dict):
        self._symbol = symbol
        self._state: StrategyState = StrategyState.FLAT

        alpha_cfg = config.get("alpha", {})
        strategy_cfg = config.get("strategy", {})
        self._entry_threshold: float = alpha_cfg.get("entry_threshold", 0.6)
        self._exit_threshold: float = alpha_cfg.get("exit_threshold", -0.2)
        self._position_size_pct: float = strategy_cfg.get("position_size_pct", 0.10)

        # Half-Kelly sizing parameters
        self._base_size_pct: float = strategy_cfg.get("base_size_pct", 0.05)
        self._max_size_pct: float = strategy_cfg.get("max_size_pct", 0.15)
        self._kelly_fraction: float = strategy_cfg.get("kelly_fraction", 0.5)
        self._win_rate: float = strategy_cfg.get("estimated_win_rate", 0.55)
        self._payoff_ratio: float = strategy_cfg.get("estimated_payoff", 1.5)
        self._urgent_alpha_threshold: float = strategy_cfg.get(
            "urgent_alpha_threshold", 0.85
        )
        exec_cfg = config.get("execution", {})
        self._limit_offset_bps: float = exec_cfg.get("limit_offset_bps", 5)

        # Signal confirmation: require N consecutive bars above entry threshold
        self._confirmation_bars: int = strategy_cfg.get("confirmation_bars", 2)
        self._alpha_history: deque = deque(maxlen=max(self._confirmation_bars, 1))
        self._factor_history: deque = deque(maxlen=max(self._confirmation_bars, 1))

        # Graduated exit tiers
        self._exit_tiers: List[Dict] = strategy_cfg.get("exit_tiers", [])
        self._exit_tier_reached: int = 0

        # Track entry price for adaptive Kelly (read before tracker zeroes it on sell)
        self._entry_price: float = 0.0
        self._initial_hold_qty: float = 0.0

        # Optional TradeTracker for adaptive Kelly (injected after init)
        self._trade_tracker = None

        # Alpha decay
        self._decay_half_life_s: float = alpha_cfg.get("decay_half_life_s", 999999)

        # Explicit strategy intent settings
        self._min_entry_score: float = strategy_cfg.get("min_entry_score", 0.62)
        self._max_blocker_score: float = strategy_cfg.get("max_blocker_score", 0.35)
        self._min_exit_score: float = strategy_cfg.get("min_exit_score", 0.55)
        self._signal_horizon_minutes: int = strategy_cfg.get(
            "signal_horizon_minutes", 240
        )
        self._exit_horizon_minutes: int = strategy_cfg.get(
            "exit_horizon_minutes", 30
        )
        self._base_stop_loss_pct: float = strategy_cfg.get("base_stop_loss_pct", 0.012)
        self._take_profit_1_rr: float = strategy_cfg.get("take_profit_1_rr", 1.2)
        self._take_profit_2_rr: float = strategy_cfg.get("take_profit_2_rr", 2.0)
        self._urgent_entry_score: float = strategy_cfg.get("urgent_entry_score", 0.82)
        self._model_filter_threshold: float = strategy_cfg.get(
            "model_filter_threshold", 0.05
        )
        self._model_exit_threshold: float = strategy_cfg.get(
            "model_exit_threshold", -0.10
        )
        self._model_size_weight: float = strategy_cfg.get("model_size_weight", 0.15)
        self._min_supporting_factors: int = strategy_cfg.get(
            "min_supporting_factors", 2
        )
        self._min_supporting_categories: int = strategy_cfg.get(
            "min_supporting_categories", 2
        )
        self._require_trend_alignment: bool = strategy_cfg.get(
            "require_trend_alignment", True
        )
        self._entry_filled_qty: float = 0.0
        self._exit_filled_qty: float = 0.0
        self._exit_fill_notional: float = 0.0

    @property
    def state(self) -> StrategyState:
        return self._state

    @property
    def symbol(self) -> str:
        return self._symbol

    def on_signal(
        self, signal: Signal, portfolio: PortfolioSnapshot, current_price: float = 0.0
    ) -> Optional[Order]:
        """Process alpha signal and decide whether to trade.

        Args:
            signal: Alpha signal
            portfolio: Current portfolio state
            current_price: Latest market price for this symbol

        Returns an Order if action is needed, None otherwise.
        """
        if self._state == StrategyState.FLAT:
            # Use decayed alpha for threshold comparison (disabled if half_life >= 999999)
            if self._decay_half_life_s < 999999:
                effective_alpha = signal.decayed_alpha(
                    datetime.utcnow(), self._decay_half_life_s
                )
            else:
                effective_alpha = signal.alpha_score
            # Always append effective alpha to history for confirmation tracking
            self._alpha_history.append(effective_alpha)

            # Clear streak if alpha drops below entry threshold
            if effective_alpha <= self._entry_threshold and len(self._alpha_history) > 0:
                # Check if the previous values formed a streak that's now broken
                if any(a > self._entry_threshold for a in list(self._alpha_history)[:-1]):
                    self._alpha_history.clear()

            if self._confirmed_entry():
                qty = self._compute_buy_quantity(
                    portfolio, current_price, signal.alpha_score
                )
                if qty <= 0:
                    return None

                # Use MARKET for urgent alpha, LIMIT otherwise to save on fees
                if signal.alpha_score > self._urgent_alpha_threshold:
                    order_type = OrderType.MARKET
                    price = None
                    order_type_label = "MARKET"
                else:
                    order_type = OrderType.LIMIT
                    price = round(
                        current_price * (1 - self._limit_offset_bps / 10000), 8
                    )
                    order_type_label = "LIMIT"

                self._state = StrategyState.LONG_PENDING
                self._alpha_history.clear()
                logger.info(
                    "[%s] FLAT → LONG_PENDING: alpha=%.3f (confirmed %d bars), qty=%.6f, order=%s",
                    self._symbol,
                    signal.alpha_score,
                    self._confirmation_bars,
                    qty,
                    order_type_label,
                )
                order = Order(
                    symbol=self._symbol,
                    side=Side.BUY,
                    order_type=order_type,
                    quantity=qty,
                )
                if price is not None:
                    order.price = price
                return order

        elif self._state == StrategyState.HOLDING:
            # Graduated exits: check tiers first, then fallback to single threshold
            exit_order = self._check_graduated_exit(signal, portfolio)
            if exit_order is not None:
                return exit_order

        elif self._state == StrategyState.LONG_PENDING:
            # Waiting for fill — no action
            pass

        return None

    def on_factors(
        self,
        factors: FactorSnapshot,
        portfolio: PortfolioSnapshot,
        current_price: float = 0.0,
        model_signal: Optional[Signal] = None,
    ) -> Optional[StrategyIntent]:
        """Generate a human-readable strategy intent from explicit factor observations.

        The strategy is now factor-first: explicit observations determine whether we
        should act, while the optional model signal only filters or slightly adjusts
        conviction and sizing.
        """
        effective_entry = factors.entry_score
        effective_blocker = factors.blocker_score
        effective_exit = factors.exit_score
        confidence = factors.confidence
        reasoning = [obs.thesis for obs in factors.observations if obs.bias != FactorBias.NEUTRAL]

        if model_signal is not None:
            if model_signal.alpha_score < self._model_filter_threshold:
                effective_blocker = min(1.0, max(effective_blocker, abs(model_signal.alpha_score)))
                reasoning.append(
                    f"Model overlay weak ({model_signal.alpha_score:.3f}) so entry conviction is filtered"
                )
            elif model_signal.alpha_score > 0:
                boost = self._model_size_weight * model_signal.alpha_score
                effective_entry = min(1.0, effective_entry + boost)
                confidence = min(1.0, 0.7 * confidence + 0.3 * model_signal.confidence)
                reasoning.append(
                    f"Model overlay supportive ({model_signal.alpha_score:.3f}) and boosts conviction"
                )
            if model_signal.alpha_score <= self._model_exit_threshold:
                effective_exit = max(effective_exit, abs(model_signal.alpha_score))
                reasoning.append(
                    f"Model overlay turned risk-off ({model_signal.alpha_score:.3f})"
                )

        if self._state == StrategyState.FLAT:
            structure_ok = self._entry_structure_ok(factors)
            self._factor_history.append(
                effective_entry
                if effective_blocker <= self._max_blocker_score and structure_ok
                else 0.0
            )
            if (
                effective_entry < self._min_entry_score
                or effective_blocker > self._max_blocker_score
                or not structure_ok
            ):
                if any(v >= self._min_entry_score for v in list(self._factor_history)[:-1]):
                    self._factor_history.clear()
                return None

            if not self._confirmed_factor_entry():
                return None

            qty = self._compute_buy_quantity_from_score(
                portfolio,
                current_price=current_price,
                conviction_score=effective_entry,
                threshold=self._min_entry_score,
            )
            if qty <= 0:
                return None

            size_notional = qty * current_price
            size_pct = size_notional / portfolio.nav if portfolio.nav > 0 else 0.0
            urgency = (
                UrgencyLevel.HIGH
                if effective_entry >= self._urgent_entry_score
                else UrgencyLevel.MEDIUM
            )
            entry_type = OrderType.MARKET if urgency == UrgencyLevel.HIGH else OrderType.LIMIT
            entry_price = None
            if entry_type == OrderType.LIMIT:
                entry_price = round(current_price * (1 - self._limit_offset_bps / 10000), 8)

            intent = StrategyIntent(
                signal_time=factors.timestamp,
                symbol=self._symbol,
                direction=Side.BUY,
                thesis=self._compose_thesis(factors.supporting_factors, reasoning),
                entry_type=entry_type,
                entry_price=entry_price,
                size_pct=size_pct,
                size_notional=size_notional,
                quantity=qty,
                signal_horizon=self._format_horizon(self._signal_horizon_minutes),
                expected_move=self._expected_move_text(factors, bullish=True),
                stop_loss=self._stop_loss_text(current_price),
                take_profit=self._take_profit_text(current_price),
                invalidate_condition=self._invalidate_text(factors),
                urgency=urgency,
                confidence=min(1.0, max(0.0, confidence)),
                factor_names=factors.supporting_factors,
                reasoning=reasoning[:4],
            )
            self._state = StrategyState.LONG_PENDING
            self._factor_history.clear()
            logger.info(
                "[%s] factor entry intent: score=%.3f blocker=%.3f qty=%.6f order=%s",
                self._symbol,
                effective_entry,
                effective_blocker,
                qty,
                entry_type.value,
            )
            return intent

        if self._state == StrategyState.HOLDING:
            pos_qty = self._position_quantity(portfolio)
            if pos_qty <= 0:
                self.force_flat()
                return None

            if effective_exit < self._min_exit_score and effective_blocker < self._max_blocker_score:
                return None

            size_notional = pos_qty * current_price
            urgency = (
                UrgencyLevel.HIGH
                if effective_blocker >= 0.65 or effective_exit >= 0.80
                else UrgencyLevel.MEDIUM
            )
            intent = StrategyIntent(
                signal_time=factors.timestamp,
                symbol=self._symbol,
                direction=Side.SELL,
                thesis=self._compose_exit_thesis(factors.blocking_factors, reasoning),
                entry_type=OrderType.MARKET,
                entry_price=None,
                size_pct=size_notional / portfolio.nav if portfolio.nav > 0 else 0.0,
                size_notional=size_notional,
                quantity=pos_qty,
                signal_horizon=self._format_horizon(self._exit_horizon_minutes),
                expected_move="Protect capital before the long thesis fully decays",
                stop_loss="Exit immediately on instruction",
                take_profit="N/A",
                invalidate_condition="Exit is cancelled only if blocking factors fully clear before execution",
                urgency=urgency,
                confidence=min(1.0, max(0.0, max(effective_exit, effective_blocker))),
                factor_names=factors.blocking_factors,
                reasoning=reasoning[:4],
            )
            logger.info(
                "[%s] factor exit intent: exit=%.3f blocker=%.3f qty=%.6f",
                self._symbol,
                effective_exit,
                effective_blocker,
                pos_qty,
            )
            self._reset_exit_fill_tracking()
            self._state = StrategyState.EXIT_PENDING
            return intent

        return None

    def build_instruction(
        self, intent: StrategyIntent, current_price: float
    ) -> TradeInstruction:
        """Convert a strategy intent into a concrete trade instruction."""
        horizon_minutes = (
            self._signal_horizon_minutes
            if intent.direction == Side.BUY
            else self._exit_horizon_minutes
        )
        expires_at = intent.signal_time + timedelta(minutes=horizon_minutes)
        entry_price = intent.entry_price
        if intent.entry_type == OrderType.MARKET and current_price > 0:
            entry_price = current_price

        return TradeInstruction(
            signal_time=intent.signal_time,
            symbol=intent.symbol,
            direction=intent.direction,
            thesis=intent.thesis,
            entry_type=intent.entry_type,
            entry_price=entry_price if intent.entry_type == OrderType.LIMIT else None,
            size_pct=intent.size_pct,
            size_notional=intent.size_notional,
            quantity=intent.quantity,
            signal_horizon=intent.signal_horizon,
            expected_move=intent.expected_move,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            invalidate_condition=intent.invalidate_condition,
            urgency=intent.urgency,
            confidence=intent.confidence,
            factor_names=intent.factor_names,
            reasoning=intent.reasoning,
            expires_at=expires_at,
        )

    def _confirmed_entry(self) -> bool:
        """Check if we have N consecutive bars above entry threshold."""
        if len(self._alpha_history) < self._confirmation_bars:
            return False
        return all(a > self._entry_threshold for a in self._alpha_history)

    def _confirmed_factor_entry(self) -> bool:
        if len(self._factor_history) < self._confirmation_bars:
            return False
        return all(score >= self._min_entry_score for score in self._factor_history)

    def _check_graduated_exit(
        self, signal: Signal, portfolio: PortfolioSnapshot
    ) -> Optional[Order]:
        """Handle exit logic with optional graduated tiers."""
        # Use decayed alpha for exit threshold comparison (disabled if half_life >= 999999)
        if self._decay_half_life_s < 999999:
            effective_alpha = signal.decayed_alpha(
                datetime.utcnow(), self._decay_half_life_s
            )
        else:
            effective_alpha = signal.alpha_score

        pos_qty = 0.0
        for pos in portfolio.positions:
            if pos.symbol == self._symbol:
                pos_qty = pos.quantity
                break

        if pos_qty <= 0:
            self._state = StrategyState.FLAT
            self._exit_tier_reached = 0
            self._initial_hold_qty = 0.0
            return None

        # Graduated exit tiers
        if self._exit_tiers:
            for i, tier in enumerate(self._exit_tiers):
                if i <= self._exit_tier_reached - 1:
                    continue  # already triggered this tier
                if effective_alpha < tier["threshold"]:
                    sell_pct = tier["sell_pct"]
                    sell_qty = pos_qty * sell_pct if sell_pct < 1.0 else pos_qty
                    sell_qty = min(sell_qty, pos_qty)
                    self._exit_tier_reached = i + 1

                    if sell_pct >= 1.0 or sell_qty >= pos_qty - 1e-12:
                        self._state = StrategyState.FLAT
                        self._exit_tier_reached = 0
                        self._initial_hold_qty = 0.0
                        logger.info(
                            "[%s] HOLDING → FLAT: alpha=%.3f < tier %d (%.2f), selling all %.6f",
                            self._symbol,
                            signal.alpha_score,
                            i,
                            tier["threshold"],
                            sell_qty,
                        )
                    else:
                        logger.info(
                            "[%s] HOLDING partial exit: alpha=%.3f < tier %d (%.2f), selling %.1f%% = %.6f",
                            self._symbol,
                            signal.alpha_score,
                            i,
                            tier["threshold"],
                            sell_pct * 100,
                            sell_qty,
                        )

                    return Order(
                        symbol=self._symbol,
                        side=Side.SELL,
                        order_type=OrderType.MARKET,
                        quantity=sell_qty,
                    )
            return None

        # Fallback: single exit threshold
        if effective_alpha < self._exit_threshold:
            logger.info(
                "[%s] HOLDING → FLAT: alpha=%.3f < %.3f, selling %.6f",
                self._symbol,
                signal.alpha_score,
                self._exit_threshold,
                pos_qty,
            )
            self._state = StrategyState.FLAT
            return Order(
                symbol=self._symbol,
                side=Side.SELL,
                order_type=OrderType.MARKET,
                quantity=pos_qty,
            )

        return None

    def on_fill(self, order: Order) -> None:
        """Called when an order is filled."""
        if order.symbol != self._symbol:
            return

        if order.side == Side.BUY and self._state == StrategyState.LONG_PENDING:
            self._record_entry_fill(order)
            self._exit_tier_reached = 0
            if order.status == OrderStatus.PARTIALLY_FILLED:
                logger.info(
                    "[%s] LONG_PENDING partial fill: qty=%.6f @ %.2f",
                    self._symbol,
                    order.filled_quantity,
                    order.filled_price,
                )
            else:
                self._state = StrategyState.HOLDING
                logger.info(
                    "[%s] LONG_PENDING → HOLDING (filled @ %.2f)",
                    self._symbol,
                    order.filled_price,
                )

        elif order.side == Side.SELL and self._state in (
            StrategyState.HOLDING,
            StrategyState.EXIT_PENDING,
        ):
            self._record_exit_fill(order)
            if order.status == OrderStatus.PARTIALLY_FILLED:
                self._state = StrategyState.EXIT_PENDING
                logger.info(
                    "[%s] EXIT_PENDING partial fill: qty=%.6f @ %.2f",
                    self._symbol,
                    order.filled_quantity,
                    order.filled_price,
                )
            else:
                avg_exit_price = self._average_exit_price()
                if (
                    self._trade_tracker is not None
                    and self._entry_price > 0
                    and avg_exit_price > 0
                ):
                    self._trade_tracker.record_trade(self._entry_price, avg_exit_price)
                self._state = StrategyState.FLAT
                self._entry_price = 0.0
                self._entry_filled_qty = 0.0
                self._reset_exit_fill_tracking()
                logger.info(
                    "[%s] EXIT_PENDING → FLAT (sold @ %.2f)",
                    self._symbol,
                    avg_exit_price or (order.filled_price or 0.0),
                )

    def on_cancel(self, order: Order) -> None:
        """Called when an order is cancelled or rejected."""
        if order.symbol != self._symbol:
            return

        if self._state == StrategyState.LONG_PENDING:
            if order.filled_quantity > 0:
                self._state = StrategyState.HOLDING
                logger.info(
                    "[%s] LONG_PENDING → HOLDING (entry order cancelled after partial fill)",
                    self._symbol,
                )
            else:
                self._state = StrategyState.FLAT
                self._entry_price = 0.0
                self._entry_filled_qty = 0.0
                logger.info("[%s] LONG_PENDING → FLAT (order cancelled)", self._symbol)
        elif self._state == StrategyState.EXIT_PENDING:
            self._state = StrategyState.HOLDING
            self._reset_exit_fill_tracking()
            logger.info("[%s] EXIT_PENDING → HOLDING (exit order cancelled)", self._symbol)

    def force_flat(self) -> None:
        """Force state to FLAT (used by circuit breaker)."""
        self._state = StrategyState.FLAT
        self._alpha_history.clear()
        self._factor_history.clear()
        self._exit_tier_reached = 0
        self._entry_price = 0.0
        self._entry_filled_qty = 0.0
        self._reset_exit_fill_tracking()

    def set_trade_tracker(self, tracker) -> None:
        """Inject TradeTracker for adaptive Kelly sizing."""
        self._trade_tracker = tracker

    def _compute_buy_quantity(
        self, portfolio: PortfolioSnapshot, current_price: float, alpha_score: float
    ) -> float:
        """Compute buy quantity using Half-Kelly position sizing."""
        return self._compute_buy_quantity_from_score(
            portfolio,
            current_price=current_price,
            conviction_score=alpha_score,
            threshold=self._entry_threshold,
        )

    def _compute_buy_quantity_from_score(
        self,
        portfolio: PortfolioSnapshot,
        current_price: float,
        conviction_score: float,
        threshold: float,
    ) -> float:
        """Compute buy quantity from a generic conviction score."""
        if portfolio.cash <= 0 or current_price <= 0:
            return 0.0

        # Use adaptive Kelly params if available, else static priors
        if self._trade_tracker is not None:
            win_rate, payoff_ratio = self._trade_tracker.get_kelly_params()
        else:
            win_rate = self._win_rate
            payoff_ratio = self._payoff_ratio

        # Half-Kelly scaling
        raw_kelly = win_rate - (1 - win_rate) / payoff_ratio
        conviction_intensity = (conviction_score - threshold) / (
            1.0 - threshold
        )
        conviction_intensity = max(0.0, min(1.0, conviction_intensity))

        scaled = self._kelly_fraction * raw_kelly * conviction_intensity
        position_pct = self._base_size_pct + scaled * (
            self._max_size_pct - self._base_size_pct
        )
        position_pct = max(self._base_size_pct, min(self._max_size_pct, position_pct))

        logger.info(
            "[%s] Half-Kelly sizing: raw_kelly=%.4f, conviction_intensity=%.4f, position_pct=%.4f",
            self._symbol,
            raw_kelly,
            conviction_intensity,
            position_pct,
        )

        allocation = portfolio.nav * position_pct
        allocation = min(allocation, portfolio.cash * 0.99)
        allocation = max(0.0, allocation)
        return allocation / current_price if allocation > 0 else 0.0

    def _position_quantity(self, portfolio: PortfolioSnapshot) -> float:
        for pos in portfolio.positions:
            if pos.symbol == self._symbol:
                return pos.quantity
        return 0.0

    def _entry_structure_ok(self, factors: FactorSnapshot) -> bool:
        bullish_obs = [obs for obs in factors.observations if obs.bias == FactorBias.BULLISH]
        if len(bullish_obs) < self._min_supporting_factors:
            return False
        categories = {obs.category for obs in bullish_obs}
        if len(categories) < self._min_supporting_categories:
            return False
        if self._require_trend_alignment and "trend_alignment" not in factors.supporting_factors:
            return False
        return True

    def _record_entry_fill(self, order: Order) -> None:
        fill_qty = order.filled_quantity or 0.0
        fill_price = order.filled_price or 0.0
        if fill_qty <= 0 or fill_price <= 0:
            return
        new_total_qty = self._entry_filled_qty + fill_qty
        if new_total_qty <= 0:
            return
        if self._entry_filled_qty > 0 and self._entry_price > 0:
            old_value = self._entry_price * self._entry_filled_qty
            new_value = fill_price * fill_qty
            self._entry_price = (old_value + new_value) / new_total_qty
        else:
            self._entry_price = fill_price
        self._entry_filled_qty = new_total_qty

    def _record_exit_fill(self, order: Order) -> None:
        fill_qty = order.filled_quantity or 0.0
        fill_price = order.filled_price or 0.0
        if fill_qty <= 0 or fill_price <= 0:
            return
        self._exit_filled_qty += fill_qty
        self._exit_fill_notional += fill_qty * fill_price

    def _average_exit_price(self) -> float:
        if self._exit_filled_qty <= 0:
            return 0.0
        return self._exit_fill_notional / self._exit_filled_qty

    def _reset_exit_fill_tracking(self) -> None:
        self._exit_filled_qty = 0.0
        self._exit_fill_notional = 0.0

    @staticmethod
    def _format_horizon(minutes: int) -> str:
        if minutes % 60 == 0:
            return f"{minutes // 60}h"
        return f"{minutes}m"

    @staticmethod
    def _compose_thesis(factor_names: List[str], reasoning: List[str]) -> str:
        if factor_names:
            names = ", ".join(factor_names[:3])
            return f"Enter long on explicit factor alignment: {names}"
        return reasoning[0] if reasoning else "Enter long on factor alignment"

    @staticmethod
    def _compose_exit_thesis(factor_names: List[str], reasoning: List[str]) -> str:
        if factor_names:
            names = ", ".join(factor_names[:3])
            return f"Exit long because blocking factors activated: {names}"
        return reasoning[0] if reasoning else "Exit long because the trade thesis has decayed"

    def _expected_move_text(
        self, factors: FactorSnapshot, bullish: bool = True
    ) -> str:
        moves = [
            obs.expected_move_bps
            for obs in factors.observations
            if obs.bias == (FactorBias.BULLISH if bullish else FactorBias.BEARISH)
        ]
        if not moves:
            return "Move expectation unavailable"
        low = max(20.0, min(moves) * 0.7)
        high = max(low, max(moves) * 1.1)
        sign = "+" if bullish else "-"
        return f"{sign}{low / 100:.2f}% to {sign}{high / 100:.2f}%"

    def _stop_loss_text(self, current_price: float) -> str:
        if current_price <= 0:
            return "Exit if the primary trend breaks"
        stop = current_price * (1 - self._base_stop_loss_pct)
        return (
            f"Exit on a {self._base_stop_loss_pct * 100:.1f}% adverse move "
            f"(reference {stop:.2f}) or on primary trend failure"
        )

    def _take_profit_text(self, current_price: float) -> str:
        if current_price <= 0:
            return "Trim at 1.2R / 2.0R"
        tp1 = current_price * (1 + self._base_stop_loss_pct * self._take_profit_1_rr)
        tp2 = current_price * (1 + self._base_stop_loss_pct * self._take_profit_2_rr)
        return f"Scale out near {tp1:.2f} and {tp2:.2f}"

    @staticmethod
    def _invalidate_text(factors: FactorSnapshot) -> str:
        invalidators = [
            obs.invalidate_condition
            for obs in factors.observations
            if obs.bias == FactorBias.BULLISH
        ]
        if invalidators:
            return invalidators[0]
        return "Invalidate if supporting factors revert to neutral"
