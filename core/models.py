from __future__ import annotations

import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

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
    order_book_imbalance: float = 0.0   # bid/ask volume ratio
    volume_ratio: float = 0.0           # current vol / rolling avg
    funding_rate: float = 0.0           # perpetual funding rate
    taker_ratio: float = 0.0            # taker buy/sell ratio
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
    drawdown: float = 0.0


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
