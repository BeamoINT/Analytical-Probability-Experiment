"""Trade Lifecycle Service.

Tracks trades from open to close with accurate P&L calculation.
This service ensures win rate and P&L metrics are based on actual
trade profitability, not just execution success.

Usage:
1. When OPEN_POSITION intent executes: call record_trade_opened()
2. When CLOSE_POSITION intent executes: call record_trade_closed()
3. For metrics: call get_performance_stats()
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class TradeLifecycleService:
    """Service to track trade lifecycle from entry to exit with P&L calculation."""

    def __init__(self, db_session: Session):
        """Initialize with a database session.

        Args:
            db_session: SQLAlchemy session for database operations.
        """
        self.db_session = db_session

    def record_trade_opened(
        self,
        intent_id: str,
        token_id: str,
        market_id: Optional[str],
        entry_price: float,
        entry_size_usd: float,
        side: str = "BUY",
        entry_time: Optional[datetime] = None,
    ) -> str:
        """Record when a position is opened.

        Args:
            intent_id: The OPEN_POSITION intent ID.
            token_id: Token being traded.
            market_id: Market ID (optional).
            entry_price: Entry price.
            entry_size_usd: Position size in USD.
            side: Trade side (BUY or SELL).
            entry_time: Time of entry (defaults to now).

        Returns:
            trade_id: Unique ID for this trade.
        """
        from polyb0t.data.storage import ClosedTradeDB

        trade_id = str(uuid.uuid4())
        entry_time = entry_time or datetime.utcnow()

        trade = ClosedTradeDB(
            trade_id=trade_id,
            open_intent_id=intent_id,
            token_id=token_id,
            market_id=market_id,
            side=side,
            entry_price=entry_price,
            entry_size_usd=entry_size_usd,
            entry_time=entry_time,
            status="OPEN",
        )

        self.db_session.add(trade)
        self.db_session.commit()

        logger.info(
            f"Trade opened: {trade_id[:8]} token={token_id[:16]}... "
            f"entry=${entry_price:.4f} size=${entry_size_usd:.2f}"
        )

        return trade_id

    def record_trade_closed(
        self,
        token_id: str,
        exit_price: float,
        exit_size_usd: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        exit_reason: str = "MANUAL",
        close_intent_id: Optional[str] = None,
    ) -> Optional["ClosedTradeDB"]:
        """Record when a position is closed and calculate P&L.

        Args:
            token_id: Token being closed.
            exit_price: Exit price.
            exit_size_usd: Exit size in USD (defaults to entry size).
            exit_time: Time of exit (defaults to now).
            exit_reason: Reason for closing (TAKE_PROFIT, STOP_LOSS, TIME_EXIT, MANUAL, etc.).
            close_intent_id: The CLOSE_POSITION intent ID (optional).

        Returns:
            The updated ClosedTradeDB record with P&L, or None if no open trade found.
        """
        from polyb0t.data.storage import ClosedTradeDB

        # Find the open trade for this token
        trade = (
            self.db_session.query(ClosedTradeDB)
            .filter(ClosedTradeDB.token_id == token_id)
            .filter(ClosedTradeDB.status == "OPEN")
            .order_by(ClosedTradeDB.entry_time.desc())
            .first()
        )

        if not trade:
            logger.warning(f"No open trade found for token {token_id[:16]}...")
            return None

        exit_time = exit_time or datetime.utcnow()

        # Calculate P&L
        # For a BUY (long): P&L = (exit_price - entry_price) / entry_price
        # For a SELL (short): P&L = (entry_price - exit_price) / entry_price
        if trade.side == "SELL":
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price if trade.entry_price > 0 else 0
        else:  # BUY (default)
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price if trade.entry_price > 0 else 0

        pnl_usd = trade.entry_size_usd * pnl_pct

        # Calculate hold time
        hold_time_hours = (exit_time - trade.entry_time).total_seconds() / 3600

        # Update trade record
        trade.exit_price = exit_price
        trade.exit_size_usd = exit_size_usd or trade.entry_size_usd
        trade.exit_time = exit_time
        trade.exit_reason = exit_reason
        trade.close_intent_id = close_intent_id
        trade.realized_pnl_usd = pnl_usd
        trade.realized_pnl_pct = pnl_pct * 100  # Store as percentage (e.g., 5.2 for 5.2%)
        trade.is_winner = pnl_pct > 0
        trade.hold_time_hours = hold_time_hours
        trade.status = "CLOSED"
        trade.updated_at = datetime.utcnow()

        self.db_session.commit()

        logger.info(
            f"Trade closed: {trade.trade_id[:8]} "
            f"entry=${trade.entry_price:.4f} exit=${exit_price:.4f} "
            f"P&L: {pnl_pct*100:+.2f}% (${pnl_usd:+.2f}) "
            f"hold={hold_time_hours:.1f}h {'WIN' if trade.is_winner else 'LOSS'}"
        )

        return trade

    def get_open_trade(self, token_id: str) -> Optional["ClosedTradeDB"]:
        """Get the open trade for a token.

        Args:
            token_id: Token to look up.

        Returns:
            ClosedTradeDB record if found, None otherwise.
        """
        from polyb0t.data.storage import ClosedTradeDB

        return (
            self.db_session.query(ClosedTradeDB)
            .filter(ClosedTradeDB.token_id == token_id)
            .filter(ClosedTradeDB.status == "OPEN")
            .first()
        )

    def get_open_trades(self) -> List["ClosedTradeDB"]:
        """Get all open trades.

        Returns:
            List of open ClosedTradeDB records.
        """
        from polyb0t.data.storage import ClosedTradeDB

        return (
            self.db_session.query(ClosedTradeDB)
            .filter(ClosedTradeDB.status == "OPEN")
            .all()
        )

    def get_closed_trades(
        self, since: Optional[datetime] = None, limit: int = 100
    ) -> List["ClosedTradeDB"]:
        """Get closed trades.

        Args:
            since: Only return trades closed after this time.
            limit: Maximum number of trades to return.

        Returns:
            List of closed ClosedTradeDB records.
        """
        from polyb0t.data.storage import ClosedTradeDB

        query = (
            self.db_session.query(ClosedTradeDB)
            .filter(ClosedTradeDB.status == "CLOSED")
        )

        if since:
            query = query.filter(ClosedTradeDB.exit_time >= since)

        return query.order_by(ClosedTradeDB.exit_time.desc()).limit(limit).all()

    def get_performance_stats(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get performance statistics from closed trades.

        Args:
            since: Only include trades closed after this time.

        Returns:
            Dictionary with performance metrics:
            - closed_trades: Number of closed trades
            - win_rate: Percentage of winning trades (0-1)
            - total_pnl_pct: Sum of all P&L percentages
            - total_pnl_usd: Sum of all P&L in USD
            - avg_pnl_pct: Average P&L per trade
            - best_trade_pct: Best single trade P&L
            - worst_trade_pct: Worst single trade P&L
            - avg_hold_hours: Average hold time in hours
        """
        trades = self.get_closed_trades(since=since, limit=1000)

        if not trades:
            return {
                "closed_trades": 0,
                "win_rate": 0,
                "total_pnl_pct": 0,
                "total_pnl_usd": 0,
                "avg_pnl_pct": 0,
                "best_trade_pct": 0,
                "worst_trade_pct": 0,
                "avg_hold_hours": 0,
            }

        winners = sum(1 for t in trades if t.is_winner)
        total_pnl_pct = sum(t.realized_pnl_pct or 0 for t in trades)
        total_pnl_usd = sum(t.realized_pnl_usd or 0 for t in trades)
        pnl_values = [t.realized_pnl_pct or 0 for t in trades]
        hold_times = [t.hold_time_hours or 0 for t in trades]

        return {
            "closed_trades": len(trades),
            "win_rate": winners / len(trades),
            "total_pnl_pct": total_pnl_pct,
            "total_pnl_usd": total_pnl_usd,
            "avg_pnl_pct": total_pnl_pct / len(trades),
            "best_trade_pct": max(pnl_values),
            "worst_trade_pct": min(pnl_values),
            "avg_hold_hours": sum(hold_times) / len(hold_times) if hold_times else 0,
        }

    def close(self):
        """Close the database session."""
        self.db_session.close()


def get_trade_lifecycle_service() -> TradeLifecycleService:
    """Create a new TradeLifecycleService with a fresh session.

    Returns:
        TradeLifecycleService instance.

    Note: Caller is responsible for calling .close() when done.
    """
    from polyb0t.data.storage import get_session
    return TradeLifecycleService(get_session())
