"""Portfolio management and position tracking."""

import logging
from datetime import datetime
from typing import Any

from polyb0t.config import get_settings
from polyb0t.execution.orders import Fill, Order, OrderSide

logger = logging.getLogger(__name__)


class Position:
    """Represents a position in a token."""

    def __init__(
        self,
        token_id: str,
        market_id: str,
        side: str,
        quantity: float,
        avg_entry_price: float,
    ) -> None:
        """Initialize position.

        Args:
            token_id: Token identifier.
            market_id: Market condition ID.
            side: LONG or SHORT.
            quantity: Position size.
            avg_entry_price: Average entry price.
        """
        self.token_id = token_id
        self.market_id = market_id
        self.side = side
        self.quantity = quantity
        self.avg_entry_price = avg_entry_price
        self.current_price = avg_entry_price
        self.opened_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    @property
    def exposure(self) -> float:
        """Dollar exposure of position."""
        return self.quantity

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized PnL."""
        if self.side == "LONG":
            return (self.current_price - self.avg_entry_price) * (
                self.quantity / self.avg_entry_price
            )
        else:
            return (self.avg_entry_price - self.current_price) * (
                self.quantity / self.avg_entry_price
            )

    def update_price(self, current_price: float) -> None:
        """Update current market price.

        Args:
            current_price: New market price.
        """
        self.current_price = current_price
        self.updated_at = datetime.utcnow()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Position({self.side} {self.token_id[:8]}, "
            f"qty={self.quantity:.2f}, avg_price={self.avg_entry_price:.4f}, "
            f"current={self.current_price:.4f}, pnl={self.unrealized_pnl:.2f})"
        )


class Portfolio:
    """Manages portfolio state, positions, and PnL tracking."""

    def __init__(self, initial_cash: float | None = None) -> None:
        """Initialize portfolio.

        Args:
            initial_cash: Initial cash balance (uses config default if None).
        """
        settings = get_settings()
        self.initial_cash = initial_cash or settings.paper_bankroll
        self.cash_balance = self.initial_cash
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.total_fees = 0.0

        logger.info(f"Initialized portfolio with ${self.initial_cash:.2f}")

    @property
    def total_exposure(self) -> float:
        """Total dollar exposure across all positions."""
        return sum(pos.exposure for pos in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        """Total portfolio equity (cash + unrealized PnL)."""
        return self.cash_balance + self.unrealized_pnl

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    def process_fill(self, order: Order, fill: Fill) -> None:
        """Process an order fill and update portfolio.

        Args:
            order: The order that was filled.
            fill: The fill details.
        """
        # Deduct cost from cash
        cost = fill.size + fill.fee
        self.cash_balance -= cost
        self.total_fees += fill.fee

        # Update or create position
        if order.token_id in self.positions:
            self._update_position(order, fill)
        else:
            self._create_position(order, fill)

        logger.info(
            f"Processed fill: {fill.size:.2f} @ {fill.price:.4f}, "
            f"fee={fill.fee:.4f}, cash={self.cash_balance:.2f}"
        )

    def _create_position(self, order: Order, fill: Fill) -> None:
        """Create new position from fill.

        Args:
            order: Order that was filled.
            fill: Fill details.
        """
        side = "LONG" if order.side == OrderSide.BUY else "SHORT"
        position = Position(
            token_id=order.token_id,
            market_id=order.market_id,
            side=side,
            quantity=fill.size,
            avg_entry_price=fill.price,
        )
        self.positions[order.token_id] = position
        logger.info(f"Created new position: {position}")

    def _update_position(self, order: Order, fill: Fill) -> None:
        """Update existing position with new fill.

        Args:
            order: Order that was filled.
            fill: Fill details.
        """
        position = self.positions[order.token_id]

        # Check if same side (add to position) or opposite (reduce/close)
        order_side = "LONG" if order.side == OrderSide.BUY else "SHORT"

        if order_side == position.side:
            # Add to position
            total_cost = (
                position.quantity * position.avg_entry_price + fill.size * fill.price
            )
            position.quantity += fill.size
            position.avg_entry_price = total_cost / position.quantity
            position.updated_at = datetime.utcnow()
            logger.info(f"Added to position: {position}")
        else:
            # Reduce or close position
            if fill.size >= position.quantity:
                # Close position
                pnl = position.unrealized_pnl
                self.realized_pnl += pnl
                del self.positions[order.token_id]
                logger.info(
                    f"Closed position in {order.token_id}, realized PnL: {pnl:.2f}"
                )
            else:
                # Partial close
                pnl = (fill.size / position.quantity) * position.unrealized_pnl
                self.realized_pnl += pnl
                position.quantity -= fill.size
                position.updated_at = datetime.utcnow()
                logger.info(
                    f"Reduced position in {order.token_id}, realized PnL: {pnl:.2f}"
                )

    def update_market_prices(self, prices: dict[str, float]) -> None:
        """Update current market prices for all positions.

        Args:
            prices: Dict of token_id -> current price.
        """
        for token_id, price in prices.items():
            if token_id in self.positions:
                self.positions[token_id].update_price(price)

    def get_position_dict(self) -> dict[str, Any]:
        """Get positions as dictionary for risk checks.

        Returns:
            Dict of token_id -> position info.
        """
        return {
            token_id: {
                "token_id": pos.token_id,
                "market_id": pos.market_id,
                "side": pos.side,
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
                "exposure": pos.exposure,
                "unrealized_pnl": pos.unrealized_pnl,
            }
            for token_id, pos in self.positions.items()
        }

    def get_summary(self) -> dict[str, Any]:
        """Get portfolio summary.

        Returns:
            Dictionary with portfolio metrics.
        """
        return {
            "cash_balance": self.cash_balance,
            "total_exposure": self.total_exposure,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_equity": self.total_equity,
            "num_positions": self.num_positions,
            "total_fees": self.total_fees,
            "return_pct": (
                (self.total_equity - self.initial_cash) / self.initial_cash * 100
            ),
        }

