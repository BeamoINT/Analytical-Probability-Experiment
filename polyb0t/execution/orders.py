"""Order models and management."""

import uuid
from datetime import datetime, timedelta
from enum import Enum

from polyb0t.config import get_settings


class OrderStatus(str, Enum):
    """Order status enumeration."""

    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


class Order:
    """Represents a trading order."""

    def __init__(
        self,
        token_id: str,
        market_id: str,
        side: OrderSide,
        price: float,
        size: float,
        order_type: OrderType = OrderType.LIMIT,
        order_id: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        """Initialize order.

        Args:
            token_id: Token identifier.
            market_id: Market condition ID.
            side: BUY or SELL.
            price: Limit price.
            size: Order size (in dollars).
            order_type: Order type (default LIMIT).
            order_id: Optional order ID (generated if None).
            timeout_seconds: Order timeout (uses config default if None).
        """
        settings = get_settings()

        self.order_id = order_id or str(uuid.uuid4())
        self.token_id = token_id
        self.market_id = market_id
        self.side = side
        self.order_type = order_type
        self.price = price
        self.size = size
        self.filled_size = 0.0
        self.status = OrderStatus.OPEN
        self.created_at = datetime.utcnow()

        timeout = timeout_seconds or settings.order_timeout_seconds
        self.expires_at = self.created_at + timedelta(seconds=timeout)

    @property
    def remaining_size(self) -> float:
        """Remaining unfilled size."""
        return self.size - self.filled_size

    @property
    def is_expired(self) -> bool:
        """Check if order is expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)."""
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED) and not self.is_expired

    def fill(self, fill_size: float, fill_price: float) -> None:
        """Record a fill.

        Args:
            fill_size: Size of fill.
            fill_price: Price of fill.
        """
        self.filled_size += fill_size

        if self.filled_size >= self.size:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Cancel order."""
        if self.is_active:
            self.status = OrderStatus.CANCELLED

    def expire(self) -> None:
        """Mark order as expired."""
        if self.is_active:
            self.status = OrderStatus.EXPIRED

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Order(id={self.order_id[:8]}, {self.side.value} {self.token_id[:8]} "
            f"@ {self.price:.4f}, size={self.size:.2f}, "
            f"filled={self.filled_size:.2f}, status={self.status.value})"
        )


class Fill:
    """Represents an order fill."""

    def __init__(
        self,
        fill_id: str,
        order_id: str,
        token_id: str,
        price: float,
        size: float,
        fee: float,
        filled_at: datetime | None = None,
    ) -> None:
        """Initialize fill.

        Args:
            fill_id: Unique fill identifier.
            order_id: Associated order ID.
            token_id: Token identifier.
            price: Fill price.
            size: Fill size.
            fee: Trading fee.
            filled_at: Fill timestamp (default now).
        """
        self.fill_id = fill_id
        self.order_id = order_id
        self.token_id = token_id
        self.price = price
        self.size = size
        self.fee = fee
        self.filled_at = filled_at or datetime.utcnow()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Fill(id={self.fill_id[:8]}, order={self.order_id[:8]}, "
            f"price={self.price:.4f}, size={self.size:.2f}, fee={self.fee:.4f})"
        )

