"""Pydantic models for external API responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MarketOutcome(BaseModel):
    """Outcome within a market."""

    token_id: str
    outcome: str
    price: float | None = None


class Market(BaseModel):
    """Market data from Gamma API."""

    condition_id: str
    question: str
    description: str | None = None
    end_date: datetime | None = None
    outcomes: list[MarketOutcome]
    category: str | None = None
    volume: float | None = None
    liquidity: float | None = None
    active: bool = True
    closed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderBookLevel(BaseModel):
    """Single price level in order book."""

    price: float
    size: float


class OrderBook(BaseModel):
    """Order book snapshot."""

    token_id: str
    timestamp: datetime
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    market_id: str | None = None


class Trade(BaseModel):
    """Recent trade data."""

    token_id: str
    timestamp: datetime
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    trade_id: str | None = None

