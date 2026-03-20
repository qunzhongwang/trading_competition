from __future__ import annotations

import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class StrategyState(str, Enum):
    FLAT = "FLAT"
    LONG_PENDING = "LONG_PENDING"
    HOLDING = "HOLDING"
    EXIT_PENDING = "EXIT_PENDING"


class FactorBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Tick(BaseModel):
    symbol: str
    price: float
    quantity: float
    timestamp: datetime
    is_buyer_maker: bool = False


class OHLCV(BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    is_closed: bool = True


class FeatureVector(BaseModel):
    symbol: str
    timestamp: datetime
    rsi: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    atr: float = 0.0
    momentum: float = 0.0
    volatility: float = 0.0
    order_book_imbalance: float = 0.0  # bid/ask volume ratio
    volume_ratio: float = 0.0  # current vol / rolling avg
    funding_rate: float = 0.0  # perpetual funding rate
    taker_ratio: float = 0.0  # taker buy/sell ratio
    raw: Dict[str, float] = Field(default_factory=dict)


class Signal(BaseModel):
    symbol: str
    alpha_score: float  # [-1.0, 1.0] — positive = bullish
    confidence: float = 1.0  # [0.0, 1.0]
    timestamp: datetime
    source: str = "rule_based"  # "rule_based" | "lstm" | "ensemble"

    def decayed_alpha(self, now: datetime, half_life_s: float = 150.0) -> float:
        """Return alpha score with exponential time decay.

        decay = 2^(-age_s / half_life_s)
        Set half_life_s to a very large value (999999) to effectively disable.
        """
        age_s = (now - self.timestamp).total_seconds()
        if age_s <= 0 or half_life_s <= 0:
            return self.alpha_score
        decay = math.pow(2.0, -age_s / half_life_s)
        return self.alpha_score * decay


class FactorObservation(BaseModel):
    symbol: str
    name: str
    category: str
    timestamp: datetime
    bias: FactorBias = FactorBias.NEUTRAL
    strength: float = 0.0  # [0, 1]
    value: float = 0.0
    threshold: float = 0.0
    horizon_minutes: int = 0
    expected_move_bps: float = 0.0
    thesis: str = ""
    invalidate_condition: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FactorSnapshot(BaseModel):
    symbol: str
    timestamp: datetime
    regime: str = "neutral"
    entry_score: float = 0.0  # [0, 1]
    exit_score: float = 0.0  # [0, 1]
    blocker_score: float = 0.0  # [0, 1]
    confidence: float = 0.0  # [0, 1]
    observations: List[FactorObservation] = Field(default_factory=list)
    supporting_factors: List[str] = Field(default_factory=list)
    blocking_factors: List[str] = Field(default_factory=list)
    summary: str = ""


class StrategyIntent(BaseModel):
    signal_time: datetime
    symbol: str
    direction: Side
    thesis: str
    entry_type: OrderType
    entry_price: Optional[float] = None
    size_pct: float = 0.0
    size_notional: float = 0.0
    quantity: float = 0.0
    signal_horizon: str = ""
    expected_move: str = ""
    stop_loss: str = ""
    take_profit: str = ""
    invalidate_condition: str = ""
    urgency: UrgencyLevel = UrgencyLevel.LOW
    confidence: float = 0.0
    factor_names: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    source: str = "strategy_logic"


class TradeInstruction(BaseModel):
    signal_time: datetime
    symbol: str
    direction: Side
    thesis: str
    entry_type: OrderType
    entry_price: Optional[float] = None
    size_pct: float = 0.0
    size_notional: float = 0.0
    quantity: float = 0.0
    signal_horizon: str = ""
    expected_move: str = ""
    stop_loss: str = ""
    take_profit: str = ""
    invalidate_condition: str = ""
    urgency: UrgencyLevel = UrgencyLevel.LOW
    confidence: float = 0.0
    factor_names: List[str] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None

    def to_order(self) -> "Order":
        order = Order(
            symbol=self.symbol,
            side=self.direction,
            order_type=self.entry_type,
            quantity=self.quantity,
        )
        if self.entry_price is not None and self.entry_type == OrderType.LIMIT:
            order.price = self.entry_price
        return order


class Order(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0


class Position(BaseModel):
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_price: float = 0.0  # for trailing stop
    state: StrategyState = StrategyState.FLAT


class PortfolioSnapshot(BaseModel):
    timestamp: datetime
    cash: float
    positions: List[Position] = Field(default_factory=list)
    nav: float = 0.0  # net asset value
    daily_pnl: float = 0.0
    peak_nav: float = 0.0
    drawdown: float = 0.0  # lifetime drawdown from peak_nav
    daily_drawdown: float = 0.0


class RiskMetrics(BaseModel):
    """Risk-adjusted performance metrics for competition scoring.

    Competition score = 0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    composite_score: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    total_return_pct: float = 0.0
    n_trading_days: int = 0
