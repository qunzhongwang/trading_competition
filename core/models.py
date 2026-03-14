from __future__ import annotations

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
    raw: Dict[str, float] = Field(default_factory=dict)


class Signal(BaseModel):
    symbol: str
    alpha_score: float  # [-1.0, 1.0] — positive = bullish
    confidence: float = 1.0  # [0.0, 1.0]
    timestamp: datetime
    source: str = "rule_based"  # "rule_based" | "lstm" | "ensemble"


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
