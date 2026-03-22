from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from core.models import OHLCV, Order, OrderStatus, Side, StrategyState
from data.buffer import LiveBuffer
from data.resampler import CandleResampler, MultiResampler
from execution.order_manager import OrderManager
from execution.trade_logger import TradeLogger
from features.extractor import FeatureExtractor
from models.inference import AlphaEngine
from risk.risk_shield import RiskShield
from risk.tracker import PortfolioTracker
from strategy.factor_engine import FactorEngine
from strategy.logic import StrategyLogic

if TYPE_CHECKING:
    from execution.base import BaseExecutor

logger = logging.getLogger(__name__)


class StrategyMonitor:
    """Central event loop / orchestrator.

    Consumes candles from the buffer, runs the feature → alpha → strategy → risk → execution pipeline.
    Event-driven: blocks on buffer.wait_for_update(), triggered by new closed candles.
    """

    def __init__(
        self,
        config: dict,
        buffer: LiveBuffer,
        extractor: FeatureExtractor,
        alpha_engine: AlphaEngine,
        risk_shield: RiskShield,
        tracker: PortfolioTracker,
        order_manager: OrderManager,
        factor_engine: Optional[FactorEngine] = None,
        resampler: Optional[CandleResampler] = None,
        multi_resampler: Optional[MultiResampler] = None,
        trade_tracker=None,
        icir_tracker=None,
        executor: Optional["BaseExecutor"] = None,
        trade_logger: Optional[TradeLogger] = None,
    ):
        self._config = config
        self._buffer = buffer
        self._extractor = extractor
        self._factor_engine = factor_engine or FactorEngine(config)
        self._alpha_engine = alpha_engine
        self._risk_shield = risk_shield
        self._tracker = tracker
        self._order_manager = order_manager
        self._running = False
        self._resampler = resampler
        self._multi_resampler = multi_resampler
        self._trade_tracker = trade_tracker
        self._icir_tracker = icir_tracker
        self._executor = executor
        self._trade_logger = trade_logger

        # Multi-TF timeframes to fetch for alpha filter
        alpha_cfg = config.get("alpha", {})
        self._multi_timeframes = alpha_cfg.get("multi_timeframes", [])
        self._use_model_overlay = config.get("strategy", {}).get(
            "use_model_overlay", False
        )
        strategy_cfg = config.get("strategy", {})
        default_slot_count = max(len(config.get("symbols", [])), 1)
        self._core_symbols: Set[str] = set(strategy_cfg.get("core_symbols", []))
        self._satellite_symbols: Set[str] = set(
            strategy_cfg.get("satellite_symbols", [])
        )
        self._allow_satellite_in_neutral: bool = strategy_cfg.get(
            "allow_satellite_in_neutral", True
        )
        self._min_entry_score: float = float(
            strategy_cfg.get("min_entry_score", 0.0)
        )
        self._top_n_entries_per_cycle: int = max(
            1, int(strategy_cfg.get("top_n_entries_per_cycle", default_slot_count))
        )
        self._max_active_positions: int = max(
            1, int(strategy_cfg.get("max_active_positions", default_slot_count))
        )
        self._satellite_max_active_positions: int = max(
            0,
            int(
                strategy_cfg.get(
                    "satellite_max_active_positions", len(self._satellite_symbols)
                )
            ),
        )
        self._satellite_min_entry_score_bonus: float = float(
            strategy_cfg.get("satellite_min_entry_score_bonus", 0.0)
        )
        self._core_priority_bonus: float = float(
            strategy_cfg.get("core_priority_bonus", 0.0)
        )
        self._primary_minutes = 1
        if multi_resampler is not None:
            self._primary_minutes = multi_resampler.primary_minutes
        elif resampler is not None:
            self._primary_minutes = resampler.minutes

        # Per-symbol strategy state machines
        symbols = config.get("symbols", [])
        self._strategies: Dict[str, StrategyLogic] = {
            sym: StrategyLogic(sym, config) for sym in symbols
        }

        # Inject trade tracker into each strategy for adaptive Kelly
        if trade_tracker is not None:
            for strategy in self._strategies.values():
                strategy.set_trade_tracker(trade_tracker)

        # Register fill callback for strategy logic
        self._order_manager.register_fill_callback(self._on_order_event)

        # Min candles before we start trading (engine-aware)
        engine_type = config.get("alpha", {}).get("engine", "rule_based")
        seq_len = config.get("alpha", {}).get("seq_len", 30)
        if self._use_model_overlay and engine_type in ("lstm", "transformer", "ensemble"):
            self._warmup_candles = (extractor.min_candles + seq_len) * self._primary_minutes
        else:
            self._warmup_candles = extractor.min_candles * self._primary_minutes

        # Day boundary tracking for circuit breaker / tracker daily reset
        self._last_trading_date: Optional[str] = None

        # Time-based periodic logging (wall clock, not iteration count)
        self._last_status_log: float = 0.0
        self._status_log_interval: float = 60.0  # 1 minute

        # ICIR tracking: store previous factors per symbol for online learning
        self._prev_factors: Dict[str, list] = {}
        self._prev_prices: Dict[str, float] = {}
        self._latest_market_context: Optional[dict] = None

    async def run(self) -> None:
        """Main event loop."""
        self._running = True
        logger.info("StrategyMonitor started, warmup=%d candles", self._warmup_candles)

        iteration = 0
        while self._running:
            got_update = await self._buffer.wait_for_update(timeout=5.0)
            if not got_update:
                continue

            iteration += 1
            try:
                await self._process_iteration(iteration)
            except Exception:
                logger.exception("Error in iteration %d, continuing", iteration)

    async def stop(self) -> None:
        self._running = False
        logger.info("StrategyMonitor stopped")

    @property
    def strategies(self) -> Dict[str, StrategyLogic]:
        return self._strategies

    async def _process_iteration(self, iteration: int) -> None:
        """Process one iteration: for each symbol, run the full pipeline.

        Price updates and stop checks run every 1-min candle.
        Alpha scoring and trade decisions only run when the resampler emits
        a completed N-min bar (or every candle if no resampler).
        """
        # Day boundary detection — reset circuit breaker and daily PnL at midnight UTC
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_trading_date is not None and today != self._last_trading_date:
            logger.info("Day boundary crossed: %s → %s, resetting daily state", self._last_trading_date, today)
            self._risk_shield.reset_daily()
            self._tracker.reset_daily()
        self._last_trading_date = today

        # Check pending limit orders for fills
        await self._order_manager.check_pending()

        snapshot = self._tracker.snapshot()
        latest_candles: Dict[str, OHLCV] = {}
        atr_values: Dict[str, float] = {}
        symbol_state: Dict[str, Dict[str, Any]] = {}

        # Track which symbols have a completed resampled bar this iteration
        alpha_ready: Set[str] = set()

        for symbol, strategy in self._strategies.items():
            candles = await self._buffer.get_candles(symbol)
            if not candles:
                continue

            latest_candles[symbol] = candles[-1]

            # Update position prices (every 1-min candle)
            self._tracker.update_prices(symbol, candles[-1].close)

            # Gate alpha on resampled bar completion
            if self._multi_resampler is not None:
                resampled_bars = self._multi_resampler.push(candles[-1])
                # Store higher-TF completed bars in the buffer
                for period, bar in resampled_bars.items():
                    if bar is not None:
                        await self._buffer.push_resampled(period, bar)
                # Alpha gating on primary (smallest) period
                primary = self._multi_resampler.primary_minutes
                if resampled_bars.get(primary) is not None:
                    alpha_ready.add(symbol)
            elif self._resampler is not None:
                resampled = self._resampler.push(candles[-1])
                if resampled is not None:
                    await self._buffer.push_resampled(self._resampler.minutes, resampled)
                    alpha_ready.add(symbol)
            else:
                alpha_ready.add(symbol)

            # Check warmup
            if len(candles) < self._warmup_candles:
                if iteration % 50 == 1:
                    logger.info(
                        "[%s] warming up: %d/%d candles",
                        symbol,
                        len(candles),
                        self._warmup_candles,
                    )
                continue

            # ── Feature Extraction (always, for ATR stops) ──
            supplementary = await self._buffer.get_supplementary(symbol)
            strategy_candles = candles
            if self._primary_minutes > 1:
                resampled_candles = await self._buffer.get_resampled_candles(
                    symbol, self._primary_minutes, n=max(self._extractor.min_candles + 10, 60)
                )
                if resampled_candles:
                    strategy_candles = resampled_candles

            if len(strategy_candles) < self._extractor.min_candles:
                continue

            features = self._extractor.extract(
                strategy_candles, supplementary=supplementary
            )
            atr_values[symbol] = features.atr

            # Fetch higher-TF candles once so the same context is available to
            # both the market-regime builder and the per-symbol factor pipeline.
            candles_15m = None
            candles_1h = None
            if self._multi_timeframes:
                if 15 in self._multi_timeframes:
                    candles_15m = await self._buffer.get_resampled_candles(
                        symbol, 15, n=50
                    )
                if 60 in self._multi_timeframes:
                    candles_1h = await self._buffer.get_resampled_candles(
                        symbol, 60, n=50
                    )

            symbol_state[symbol] = {
                "candles": candles,
                "strategy_candles": strategy_candles,
                "supplementary": supplementary,
                "features": features,
                "candles_15m": candles_15m,
                "candles_1h": candles_1h,
            }

            # ── Strategy Gating (only on completed resampled bars) ──
            if symbol not in alpha_ready:
                continue

        market_context = self._build_market_context(symbol_state)
        self._latest_market_context = market_context
        buy_candidates: list[dict[str, Any]] = []

        for symbol, strategy in self._strategies.items():
            state = symbol_state.get(symbol)
            if state is None or symbol not in alpha_ready:
                continue

            candles = state["candles"]
            strategy_candles = state["strategy_candles"]
            supplementary = state["supplementary"]
            features = state["features"]
            candles_15m = state["candles_15m"]
            candles_1h = state["candles_1h"]

            # ICIR online learning: record previous factors vs realized return
            if self._icir_tracker is not None and symbol in self._prev_factors:
                prev_price = self._prev_prices.get(symbol, 0.0)
                if prev_price > 0:
                    realized_return = (candles[-1].close - prev_price) / prev_price
                    self._icir_tracker.record(
                        symbol, self._prev_factors[symbol], realized_return
                    )

            # Store current factors for ICIR next iteration
            if self._icir_tracker is not None:
                self._prev_factors[symbol] = [
                    (50.0 - features.rsi) / 50.0,
                    max(-1.0, min(1.0, features.momentum * 20.0)),
                    max(
                        -1.0,
                        min(
                            1.0,
                            (features.ema_fast - features.ema_slow)
                            / features.ema_slow
                            * 100.0,
                        ),
                    )
                    if features.ema_slow > 0
                    else 0.0,
                    min(1.0, features.volatility * 50.0),
                ]
                self._prev_prices[symbol] = candles[-1].close

            factor_history_window = getattr(
                self._factor_engine, "supplementary_history_window", 2
            )
            supplementary_history_len = max(
                self._alpha_engine._seq_len,
                factor_history_window,
            )
            if self._use_model_overlay and self._primary_minutes > 1:
                supplementary_history_len *= self._primary_minutes
            supplementary_history = await self._buffer.get_supplementary_history(
                symbol, supplementary_history_len
            )
            model_supplementary_history = self._align_supplementary_history(
                supplementary_history,
                seq_len=self._alpha_engine._seq_len,
            )

            # Fetch higher-TF candles for multi-TF filter
            candles_15m = None
            candles_1h = None
            if self._multi_timeframes:
                if 15 in self._multi_timeframes:
                    candles_15m = await self._buffer.get_resampled_candles(
                        symbol, 15, n=50
                    )
                if 60 in self._multi_timeframes:
                    candles_1h = await self._buffer.get_resampled_candles(
                        symbol, 60, n=50
                    )

            factor_snapshot = self._factor_engine.evaluate(
                features,
                supplementary=supplementary,
                supplementary_history=supplementary_history,
                candles_15m=candles_15m,
                candles_1h=candles_1h,
                market_context=market_context,
            )
            model_signal = None
            if self._use_model_overlay:
                model_signal = self._alpha_engine.score(
                    strategy_candles,
                    supplementary=supplementary,
                    supplementary_history=model_supplementary_history,
                    candles_15m=candles_15m,
                    candles_1h=candles_1h,
                )

            if self._trade_logger is not None:
                await self._trade_logger.log_factor_snapshot(
                    symbol=factor_snapshot.symbol,
                    regime=factor_snapshot.regime,
                    entry_score=factor_snapshot.entry_score,
                    blocker_score=factor_snapshot.blocker_score,
                    confidence=factor_snapshot.confidence,
                    observations=[
                        obs.model_dump(mode="json")
                        for obs in factor_snapshot.observations
                    ],
                    summary=factor_snapshot.summary,
                )

            # ── Strategy Intent → Instruction ──
            snapshot = self._tracker.snapshot()
            intent = strategy.on_factors(
                factor_snapshot,
                snapshot,
                current_price=candles[-1].close,
                model_signal=model_signal,
            )

            if intent is not None:
                if self._trade_logger is not None:
                    await self._trade_logger.log_strategy_intent(
                        intent.model_dump(mode="json")
                    )

                if intent.direction == Side.BUY:
                    buy_candidates.append(
                        {
                            "symbol": symbol,
                            "strategy": strategy,
                            "intent": intent,
                            "current_price": candles[-1].close,
                            "factor_snapshot": factor_snapshot,
                        }
                    )
                    continue

                instruction = strategy.build_instruction(
                    intent, current_price=candles[-1].close
                )
                if self._trade_logger is not None:
                    await self._trade_logger.log_trade_instruction(
                        instruction.model_dump(mode="json")
                    )

                order = instruction.to_order()
                # ── Risk Validation ──
                validated = self._risk_shield.validate(
                    order,
                    self._tracker,
                    market_price=candles[-1].close,
                )
                if validated is not None:
                    result = await self._order_manager.submit(validated)
                else:
                    # Risk rejected the order — reset strategy state
                    strategy.on_cancel(order)

        # ── Post-trade Risk Checks ──
        ranked_buy_candidates = self._rank_buy_candidates(buy_candidates)
        selected_buy_candidates = ranked_buy_candidates[: self._top_n_entries_per_cycle]
        skipped_buy_candidates = ranked_buy_candidates[self._top_n_entries_per_cycle :]
        for candidate in skipped_buy_candidates:
            self._cancel_buy_candidate(
                candidate,
                "not in top entry cohort for this cycle",
            )

        for idx, candidate in enumerate(selected_buy_candidates):
            unsubmitted_symbols = {
                item["symbol"] for item in selected_buy_candidates[idx:]
            }
            if (
                self._active_symbol_count(exclude_symbols=unsubmitted_symbols)
                >= self._max_active_positions
            ):
                self._cancel_buy_candidate(
                    candidate,
                    "max active position cap reached",
                )
                continue

            if (
                candidate["symbol"] in self._satellite_symbols
                and self._active_satellite_count(exclude_symbols=unsubmitted_symbols)
                >= self._satellite_max_active_positions
            ):
                self._cancel_buy_candidate(
                    candidate,
                    "satellite position cap reached",
                )
                continue

            strategy = candidate["strategy"]
            intent = candidate["intent"]
            current_price = candidate["current_price"]
            instruction = strategy.build_instruction(
                intent,
                current_price=current_price,
            )
            if self._trade_logger is not None:
                await self._trade_logger.log_trade_instruction(
                    instruction.model_dump(mode="json")
                )

            order = instruction.to_order()
            validated = self._risk_shield.validate(
                order,
                self._tracker,
                market_price=current_price,
            )
            if validated is not None:
                await self._order_manager.submit(validated)
            else:
                strategy.on_cancel(order)

        # Trailing stops and ATR stops
        stop_orders = self._risk_shield.check_stops(
            self._tracker, latest_candles, atr_values
        )
        for stop_order in stop_orders:
            validated = self._risk_shield.validate(
                stop_order,
                self._tracker,
                market_price=latest_candles.get(stop_order.symbol).close
                if latest_candles.get(stop_order.symbol) is not None
                else 0.0,
                is_stop=True,
            )
            if validated is not None:
                await self._order_manager.submit(validated)
                # Update strategy state
                if stop_order.symbol in self._strategies:
                    self._strategies[stop_order.symbol].force_flat()

        # Circuit breaker check
        if self._risk_shield.check_circuit_breaker(self._tracker):
            await self._liquidate_all()

        # Periodic logging (time-based, not iteration-based)
        now = time.monotonic()
        if now - self._last_status_log >= self._status_log_interval:
            self._last_status_log = now
            snap = self._tracker.snapshot()
            holdings = [
                f"{p.symbol}:{self._format_quantity(p.quantity)}@{p.current_price:.2f}"
                for p in snap.positions
                if p.state == StrategyState.HOLDING
            ]
            logger.info(
                "Iter %d | NAV=%.2f | Cash=%.2f | PnL=%.2f | DD=%.2f%% | %s | Holdings=%s",
                iteration,
                snap.nav,
                snap.cash,
                snap.daily_pnl,
                snap.drawdown * 100,
                self._format_regime_status(self._latest_market_context),
                holdings or "none",
            )

            # Roostoo mode: fetch live balance to verify API connectivity
            if self._executor is not None and hasattr(self._executor, "get_balance"):
                try:
                    roostoo_bal = await self._executor.get_balance()
                    if roostoo_bal:
                        bal_parts = [
                            self._format_balance_item(asset, value)
                            for asset, value in roostoo_bal.items()
                        ]
                        logger.info("Roostoo balance | %s", " | ".join(bal_parts))
                    else:
                        logger.warning("Roostoo balance fetch returned empty")
                except Exception:
                    logger.warning("Roostoo balance fetch failed", exc_info=True)

    def _align_supplementary_history(
        self,
        history: dict,
        seq_len: int,
    ) -> dict:
        """Align raw per-minute supplementary history to the strategy timeframe."""
        if self._primary_minutes <= 1:
            return {
                key: list(values)[-seq_len:]
                for key, values in history.items()
            }

        aligned = {}
        for key, values in history.items():
            series = list(values)
            if not series:
                aligned[key] = []
                continue
            sampled = []
            idx = len(series) - 1
            while idx >= 0 and len(sampled) < seq_len:
                sampled.append(series[idx])
                idx -= self._primary_minutes
            sampled.reverse()
            aligned[key] = sampled
        return aligned

    def _rank_buy_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            reason = self._buy_candidate_rejection_reason(candidate)
            if reason is not None:
                self._cancel_buy_candidate(candidate, reason)
                continue
            ranked.append(candidate)
        ranked.sort(key=self._candidate_priority_score, reverse=True)
        return ranked

    def _buy_candidate_rejection_reason(
        self, candidate: dict[str, Any]
    ) -> Optional[str]:
        symbol = candidate["symbol"]
        snapshot = candidate["factor_snapshot"]
        regime = snapshot.regime
        if regime == "risk_off":
            return "market regime is risk_off"
        if symbol in self._satellite_symbols and regime == "neutral":
            if not self._allow_satellite_in_neutral:
                return "satellite entries disabled outside risk_on"
        if (
            symbol in self._satellite_symbols
            and snapshot.entry_score
            < (self._min_entry_score + self._satellite_min_entry_score_bonus)
        ):
            return "satellite entry score below stricter threshold"
        return None

    def _candidate_priority_score(self, candidate: dict[str, Any]) -> float:
        snapshot = candidate["factor_snapshot"]
        symbol = candidate["symbol"]
        score = snapshot.entry_score - 0.5 * snapshot.blocker_score
        if symbol in self._core_symbols:
            score += self._core_priority_bonus
        return score

    def _cancel_buy_candidate(self, candidate: dict[str, Any], reason: str) -> None:
        symbol = candidate["symbol"]
        strategy = candidate["strategy"]
        current_price = candidate["current_price"]
        intent = candidate["intent"]
        logger.info("[%s] skipped buy candidate: %s", symbol, reason)
        order = strategy.build_instruction(
            intent,
            current_price=current_price,
        ).to_order()
        strategy.on_cancel(order)

    def _active_symbol_count(self, exclude_symbols: Optional[Set[str]] = None) -> int:
        excluded = exclude_symbols or set()
        active_symbols = {
            pos.symbol
            for pos in self._tracker.snapshot().positions
            if pos.quantity > 0 and pos.symbol not in excluded
        }
        active_states = {
            StrategyState.LONG_PENDING,
            StrategyState.HOLDING,
            StrategyState.EXIT_PENDING,
        }
        active_symbols.update(
            symbol
            for symbol, strategy in self._strategies.items()
            if symbol not in excluded and strategy._state in active_states
        )
        return len(active_symbols)

    def _active_satellite_count(
        self, exclude_symbols: Optional[Set[str]] = None
    ) -> int:
        excluded = exclude_symbols or set()
        return len(
            self._satellite_symbols.intersection(
                {
                    pos.symbol
                    for pos in self._tracker.snapshot().positions
                    if pos.quantity > 0 and pos.symbol not in excluded
                }
            ).union(
                {
                    symbol
                    for symbol, strategy in self._strategies.items()
                    if (
                        symbol in self._satellite_symbols
                        and symbol not in excluded
                        and strategy._state
                        in {
                            StrategyState.LONG_PENDING,
                            StrategyState.HOLDING,
                            StrategyState.EXIT_PENDING,
                        }
                    )
                }
            )
        )

    async def _liquidate_all(self) -> None:
        """Emergency liquidation: sell all positions."""
        logger.critical("LIQUIDATING ALL POSITIONS")
        snapshot = self._tracker.snapshot()

        for pos in snapshot.positions:
            if pos.quantity > 0:
                from core.models import OrderType, Side

                order = Order(
                    symbol=pos.symbol,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                )
                await self._order_manager.submit(order)
                if pos.symbol in self._strategies:
                    self._strategies[pos.symbol].force_flat()

    def _on_order_event(self, order: Order) -> None:
        """Callback for order fill/cancel events."""
        if order.symbol in self._strategies:
            if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                self._strategies[order.symbol].on_fill(order)
            elif order.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
                self._strategies[order.symbol].on_cancel(order)

    def _build_market_context(
        self, symbol_state: Dict[str, Dict[str, Any]]
    ) -> Optional[dict]:
        regime_cfg = self._config.get("regime", {})
        if not regime_cfg.get("enabled", False):
            return None
        if not symbol_state:
            return None

        benchmark_symbols = regime_cfg.get(
            "benchmark_symbols", ["BTC/USDT", "ETH/USDT"]
        )
        benchmark_scores: Dict[str, float] = {}
        benchmark_funding: list[float] = []
        benchmark_volatility: list[float] = []

        for symbol in benchmark_symbols:
            state = symbol_state.get(symbol)
            if state is None:
                continue
            features = state["features"]
            benchmark_scores[symbol] = self._benchmark_score(
                features,
                candles_1h=state.get("candles_1h"),
            )
            benchmark_funding.append(
                state["supplementary"].get("funding_rate", features.funding_rate)
            )
            benchmark_volatility.append(features.volatility)

        if not benchmark_scores:
            return None

        feature_list = [state["features"] for state in symbol_state.values()]
        positive_count = sum(
            1
            for feat in feature_list
            if feat.momentum > 0 and feat.ema_fast >= feat.ema_slow
        )
        breadth = positive_count / len(feature_list)
        breadth_score = max(-1.0, min(1.0, 2.0 * breadth - 1.0))
        volume_expansion = sum(
            max(0.0, min(1.0, (feat.volume_ratio - 1.0) / 1.5))
            for feat in feature_list
        ) / max(len(feature_list), 1)
        benchmark_avg = sum(benchmark_scores.values()) / len(benchmark_scores)
        volatility_ceiling = regime_cfg.get("volatility_ceiling", 0.020)
        avg_benchmark_vol = sum(benchmark_volatility) / max(len(benchmark_volatility), 1)
        vol_stress = max(
            0.0,
            min(1.0, avg_benchmark_vol / max(volatility_ceiling, 1e-6) - 1.0),
        )

        score = (
            0.50 * benchmark_avg
            + 0.30 * breadth_score
            + 0.10 * volume_expansion
            - 0.10 * vol_stress
        )
        risk_on_threshold = regime_cfg.get("risk_on_threshold", 0.25)
        neutral_threshold = regime_cfg.get("neutral_threshold", 0.05)
        breadth_min_symbols = max(0, int(regime_cfg.get("breadth_min_symbols", 0)))
        breadth_ok = (
            positive_count >= min(breadth_min_symbols, len(feature_list))
            if breadth_min_symbols > 0
            else True
        )

        if score >= risk_on_threshold and breadth_ok:
            regime = "risk_on"
        elif score >= neutral_threshold:
            regime = "neutral"
        else:
            regime = "risk_off"

        return {
            "regime": regime,
            "score": max(-1.0, min(1.0, score)),
            "breadth": breadth,
            "positive_symbols": positive_count,
            "breadth_ok": breadth_ok,
            "volume_expansion": volume_expansion,
            "avg_funding": sum(benchmark_funding) / max(len(benchmark_funding), 1),
            "avg_benchmark_volatility": avg_benchmark_vol,
            "vol_stress": vol_stress,
            "benchmarks": benchmark_scores,
        }

    @staticmethod
    def _benchmark_score(features: Any, candles_1h: Optional[list[OHLCV]]) -> float:
        ema_spread = 0.0
        if features.ema_slow > 0:
            ema_spread = (features.ema_fast - features.ema_slow) / features.ema_slow

        hourly_momentum = 0.0
        if candles_1h and len(candles_1h) >= 6 and candles_1h[-6].close > 0:
            hourly_momentum = (
                candles_1h[-1].close - candles_1h[-6].close
            ) / candles_1h[-6].close

        score = ema_spread * 120.0 + features.momentum * 8.0 + hourly_momentum * 10.0
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _format_quantity(quantity: float) -> str:
        if quantity >= 100:
            return f"{quantity:.2f}"
        if quantity >= 1:
            return f"{quantity:.4f}"
        if quantity >= 0.01:
            return f"{quantity:.6f}"
        return f"{quantity:.8f}"

    @classmethod
    def _format_balance_item(cls, asset: str, value: float) -> str:
        if asset == "USD":
            return f"{asset}:{value:.2f}"
        return f"{asset}:{cls._format_quantity(value)}"

    @classmethod
    def _format_regime_status(cls, market_context: Optional[dict]) -> str:
        if not market_context:
            return "Regime=n/a"
        regime = market_context.get("regime", "n/a")
        score = float(market_context.get("score", 0.0))
        breadth = float(market_context.get("breadth", 0.0))
        return f"Regime={regime}(score={score:.3f}, breadth={breadth:.2f})"
