"""Paper trading simulator with realistic fill simulation."""

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from polyb0t.config import get_settings
from polyb0t.data.models import OrderBook, Trade
from polyb0t.data.storage import SimulatedFillDB, SimulatedOrderDB
from polyb0t.execution.orders import Fill, Order, OrderSide, OrderStatus, OrderType
from polyb0t.execution.portfolio import Portfolio
from polyb0t.models.strategy_baseline import TradingSignal

logger = logging.getLogger(__name__)


class PaperTradingSimulator:
    """Simulates order execution with realistic fill logic.

    Fill Logic:
        - Limit orders fill conservatively: only when subsequent trades cross
          the order price or when book depth suggests likely fill.
        - Applies configurable slippage and fees.
        - Tracks open orders, partial fills, and expirations.
    """

    def __init__(self, portfolio: Portfolio, db_session: Session) -> None:
        """Initialize simulator.

        Args:
            portfolio: Portfolio instance to manage.
            db_session: Database session for persistence.
        """
        self.settings = get_settings()
        self.portfolio = portfolio
        self.db_session = db_session
        self.open_orders: dict[str, Order] = {}

    def place_order(
        self,
        signal: TradingSignal,
        size: float,
        cycle_id: str,
    ) -> Order:
        """Place a simulated order.

        Args:
            signal: Trading signal to execute.
            size: Order size in dollars.
            cycle_id: Current cycle identifier for tracking.

        Returns:
            Created Order object.
        """
        # Determine order price from signal
        # For BUY: place at best ask (aggressive) or slightly below
        # For SELL: place at best bid (aggressive) or slightly above
        # For MVP, use mid price with small improvement
        price = signal.p_market

        # Adjust price slightly in our favor
        if signal.side == "BUY":
            # Try to buy slightly below market
            price = signal.p_market * 0.998
        else:
            # Try to sell slightly above market
            price = signal.p_market * 1.002

        # Clamp to valid range
        price = max(0.01, min(0.99, price))

        side = OrderSide.BUY if signal.side == "BUY" else OrderSide.SELL

        order = Order(
            token_id=signal.token_id,
            market_id=signal.market_id,
            side=side,
            price=price,
            size=size,
            order_type=OrderType.LIMIT,
        )

        self.open_orders[order.order_id] = order

        # Persist to database
        self._save_order_to_db(order, cycle_id)

        logger.info(f"Placed order: {order}")
        return order

    def process_market_data(
        self,
        orderbooks: dict[str, OrderBook],
        recent_trades: dict[str, list[Trade]],
        cycle_id: str,
    ) -> None:
        """Process market data to simulate fills.

        Args:
            orderbooks: Current orderbook snapshots.
            recent_trades: Recent trades per token.
            cycle_id: Current cycle ID.
        """
        # Check for expired orders
        self._check_expirations()

        # Attempt to fill orders based on market data
        for order_id, order in list(self.open_orders.items()):
            if not order.is_active:
                continue

            # Get relevant market data
            orderbook = orderbooks.get(order.token_id)
            trades = recent_trades.get(order.token_id, [])

            # Attempt fill
            filled = self._attempt_fill(order, orderbook, trades, cycle_id)

            # Remove from open orders if fully filled or no longer active
            if not order.is_active:
                del self.open_orders[order_id]

    def _attempt_fill(
        self,
        order: Order,
        orderbook: OrderBook | None,
        recent_trades: list[Trade],
        cycle_id: str,
    ) -> bool:
        """Attempt to fill an order based on market conditions.

        Args:
            order: Order to fill.
            orderbook: Current orderbook.
            recent_trades: Recent trades.
            cycle_id: Current cycle ID.

        Returns:
            True if order was filled (fully or partially).
        """
        # Check trades first - conservative approach: only fill if trade crossed our price
        for trade in recent_trades:
            if self._trade_crosses_order(order, trade):
                fill_price = self._calculate_fill_price(order, trade.price)
                fill_size = min(order.remaining_size, trade.size)

                if fill_size > 0:
                    self._execute_fill(order, fill_price, fill_size, cycle_id)
                    return True

        # If no trade crosses, check orderbook depth
        # Conservative: only fill if opposite side has significant depth at our price
        if orderbook:
            if self._orderbook_suggests_fill(order, orderbook):
                # Fill at order price (we got lucky)
                fill_price = order.price
                # Fill partial size based on available depth
                fill_size = self._estimate_fillable_size(order, orderbook)

                if fill_size > 0:
                    self._execute_fill(order, fill_price, fill_size, cycle_id)
                    return True

        return False

    def _trade_crosses_order(self, order: Order, trade: Trade) -> bool:
        """Check if a trade crosses (can fill) an order.

        Args:
            order: The order.
            trade: The trade.

        Returns:
            True if trade can fill order.
        """
        if order.side == OrderSide.BUY:
            # Buy order fills if trade price <= our limit price
            return trade.price <= order.price
        else:
            # Sell order fills if trade price >= our limit price
            return trade.price >= order.price

    def _orderbook_suggests_fill(self, order: Order, orderbook: OrderBook) -> bool:
        """Check if orderbook suggests order might fill.

        Args:
            order: The order.
            orderbook: Current orderbook.

        Returns:
            True if likely to fill.
        """
        if order.side == OrderSide.BUY:
            # Check if there are asks at or below our price
            if not orderbook.asks:
                return False
            best_ask = orderbook.asks[0].price
            return best_ask <= order.price
        else:
            # Check if there are bids at or above our price
            if not orderbook.bids:
                return False
            best_bid = orderbook.bids[0].price
            return best_bid >= order.price

    def _estimate_fillable_size(self, order: Order, orderbook: OrderBook) -> float:
        """Estimate how much of order can be filled based on depth.

        Args:
            order: The order.
            orderbook: Current orderbook.

        Returns:
            Estimated fillable size.
        """
        # Conservative: fill a fraction of available depth
        if order.side == OrderSide.BUY:
            # Sum ask depth at our price or better
            depth = sum(
                level.size for level in orderbook.asks if level.price <= order.price
            )
        else:
            # Sum bid depth at our price or better
            depth = sum(
                level.size for level in orderbook.bids if level.price >= order.price
            )

        # Fill at most 30% of available depth or remaining order size
        fillable = min(depth * 0.3, order.remaining_size)
        return fillable

    def _calculate_fill_price(self, order: Order, market_price: float) -> float:
        """Calculate fill price including slippage.

        Args:
            order: The order.
            market_price: Market price.

        Returns:
            Fill price after slippage.
        """
        slippage_pct = self.settings.slippage_bps / 10000.0

        if order.side == OrderSide.BUY:
            # Slippage increases buy price
            fill_price = market_price * (1 + slippage_pct)
        else:
            # Slippage decreases sell price
            fill_price = market_price * (1 - slippage_pct)

        # Clamp to valid range
        fill_price = max(0.01, min(0.99, fill_price))
        return fill_price

    def _execute_fill(
        self,
        order: Order,
        fill_price: float,
        fill_size: float,
        cycle_id: str,
    ) -> None:
        """Execute a fill.

        Args:
            order: Order to fill.
            fill_price: Fill price.
            fill_size: Fill size.
            cycle_id: Current cycle ID.
        """
        # Calculate fee
        fee = fill_size * (self.settings.fee_bps / 10000.0)

        # Create fill record
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            token_id=order.token_id,
            price=fill_price,
            size=fill_size,
            fee=fee,
        )

        # Update order
        order.fill(fill_size, fill_price)

        # Update portfolio
        self.portfolio.process_fill(order, fill)

        # Persist fill to database
        self._save_fill_to_db(fill)

        # Update order in database
        self._update_order_in_db(order)

        logger.info(f"Executed fill: {fill}")

    def _check_expirations(self) -> None:
        """Check and expire old orders."""
        for order in list(self.open_orders.values()):
            if order.is_expired:
                order.expire()
                self._update_order_in_db(order)
                logger.info(f"Order expired: {order.order_id}")

    def _save_order_to_db(self, order: Order, cycle_id: str) -> None:
        """Save order to database.

        Args:
            order: Order to save.
            cycle_id: Current cycle ID.
        """
        db_order = SimulatedOrderDB(
            order_id=order.order_id,
            cycle_id=cycle_id,
            token_id=order.token_id,
            market_id=order.market_id,
            side=order.side.value,
            order_type=order.order_type.value,
            price=order.price,
            size=order.size,
            status=order.status.value,
            filled_size=order.filled_size,
            created_at=order.created_at,
            expires_at=order.expires_at,
        )
        self.db_session.add(db_order)
        self.db_session.commit()

    def _update_order_in_db(self, order: Order) -> None:
        """Update order status in database.

        Args:
            order: Order to update.
        """
        db_order = (
            self.db_session.query(SimulatedOrderDB)
            .filter_by(order_id=order.order_id)
            .first()
        )
        if db_order:
            db_order.status = order.status.value
            db_order.filled_size = order.filled_size
            self.db_session.commit()

    def _save_fill_to_db(self, fill: Fill) -> None:
        """Save fill to database.

        Args:
            fill: Fill to save.
        """
        db_fill = SimulatedFillDB(
            fill_id=fill.fill_id,
            order_id=fill.order_id,
            token_id=fill.token_id,
            price=fill.price,
            size=fill.size,
            fee=fill.fee,
            filled_at=fill.filled_at,
        )
        self.db_session.add(db_fill)
        self.db_session.commit()

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        for order in list(self.open_orders.values()):
            order.cancel()
            self._update_order_in_db(order)
            logger.info(f"Cancelled order: {order.order_id}")

        self.open_orders.clear()

