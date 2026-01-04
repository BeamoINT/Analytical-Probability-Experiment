"""Portfolio reporting and metrics."""

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from polyb0t.data.storage import (
    PnLSnapshotDB,
    PortfolioPositionDB,
    SimulatedFillDB,
    SimulatedOrderDB,
    SignalDB,
    TradeIntentDB,
)
from polyb0t.execution.portfolio import Portfolio

logger = logging.getLogger(__name__)


class Reporter:
    """Generate reports and metrics."""

    def __init__(self, db_session: Session) -> None:
        """Initialize reporter.

        Args:
            db_session: Database session.
        """
        self.db_session = db_session

    def generate_daily_report(self, portfolio: Portfolio) -> dict[str, Any]:
        """Generate daily performance report.

        Args:
            portfolio: Current portfolio state.

        Returns:
            Dictionary with daily report data.
        """
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        report = {
            "date": today.isoformat(),
            "portfolio": portfolio.get_summary(),
            "daily_stats": self._get_daily_stats(today),
            "positions": self._get_position_summary(),
            "recent_fills": self._get_recent_fills(limit=10),
            "top_signals": self._get_top_signals(limit=10),
            "intent_stats": self._get_intent_stats(today),
        }

        return report

    def _get_daily_stats(self, today: datetime) -> dict[str, Any]:
        """Get statistics for today.

        Args:
            today: Start of today.

        Returns:
            Daily statistics dict.
        """
        tomorrow = today + timedelta(days=1)

        # Count orders
        orders_today = (
            self.db_session.query(SimulatedOrderDB)
            .filter(SimulatedOrderDB.created_at >= today)
            .filter(SimulatedOrderDB.created_at < tomorrow)
            .count()
        )

        # Count fills
        fills_today = (
            self.db_session.query(SimulatedFillDB)
            .filter(SimulatedFillDB.filled_at >= today)
            .filter(SimulatedFillDB.filled_at < tomorrow)
            .count()
        )

        # Sum fees
        total_fees = (
            self.db_session.query(func.sum(SimulatedFillDB.fee))
            .filter(SimulatedFillDB.filled_at >= today)
            .filter(SimulatedFillDB.filled_at < tomorrow)
            .scalar()
            or 0.0
        )

        # Count signals
        signals_today = (
            self.db_session.query(SignalDB)
            .filter(SignalDB.timestamp >= today)
            .filter(SignalDB.timestamp < tomorrow)
            .count()
        )

        return {
            "orders_placed": orders_today,
            "fills_executed": fills_today,
            "signals_generated": signals_today,
            "total_fees": round(total_fees, 2),
        }

    def _get_intent_stats(self, today: datetime) -> dict[str, Any]:
        """Get intent lifecycle stats for today (live mode observability)."""
        tomorrow = today + timedelta(days=1)
        rows = (
            self.db_session.query(TradeIntentDB.status, func.count(TradeIntentDB.id))
            .filter(TradeIntentDB.created_at >= today)
            .filter(TradeIntentDB.created_at < tomorrow)
            .group_by(TradeIntentDB.status)
            .all()
        )
        counts = {status: int(n) for status, n in rows}
        return {
            "created": sum(counts.values()),
            "pending": counts.get("PENDING", 0),
            "approved": counts.get("APPROVED", 0),
            "executed": counts.get("EXECUTED", 0),
            "executed_dryrun": counts.get("EXECUTED_DRYRUN", 0),
            "failed": counts.get("FAILED", 0),
            "expired": counts.get("EXPIRED", 0),
            "rejected": counts.get("REJECTED", 0),
            "superseded": counts.get("SUPERSEDED", 0),
            "by_status": counts,
        }

    def _get_position_summary(self) -> list[dict[str, Any]]:
        """Get summary of current positions.

        Returns:
            List of position dicts.
        """
        positions = self.db_session.query(PortfolioPositionDB).all()

        return [
            {
                "token_id": pos.token_id,
                "market_id": pos.market_id,
                "side": pos.side,
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
            }
            for pos in positions
        ]

    def _get_recent_fills(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent fills.

        Args:
            limit: Maximum number of fills to return.

        Returns:
            List of fill dicts.
        """
        fills = (
            self.db_session.query(SimulatedFillDB)
            .order_by(SimulatedFillDB.filled_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "token_id": fill.token_id,
                "price": fill.price,
                "size": fill.size,
                "fee": fill.fee,
                "filled_at": fill.filled_at.isoformat() if fill.filled_at else None,
            }
            for fill in fills
        ]

    def _get_top_signals(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent signals with highest edge.

        Args:
            limit: Maximum number of signals to return.

        Returns:
            List of signal dicts.
        """
        signals = (
            self.db_session.query(SignalDB)
            .filter(SignalDB.timestamp >= datetime.utcnow() - timedelta(days=1))
            .order_by(SignalDB.edge.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "token_id": sig.token_id,
                "market_id": sig.market_id,
                "signal_type": sig.signal_type,
                "edge": sig.edge,
                "p_market": sig.p_market,
                "p_model": sig.p_model,
                "confidence": sig.confidence,
                "timestamp": sig.timestamp.isoformat() if sig.timestamp else None,
            }
            for sig in signals
        ]

    def save_pnl_snapshot(
        self,
        portfolio: Portfolio,
        cycle_id: str,
    ) -> None:
        """Save portfolio PnL snapshot to database.

        Args:
            portfolio: Portfolio instance.
            cycle_id: Current cycle ID.
        """
        summary = portfolio.get_summary()

        # Calculate drawdown
        peak_equity = portfolio.initial_cash
        # Get historical max equity
        max_equity_record = (
            self.db_session.query(func.max(PnLSnapshotDB.total_equity)).scalar()
        )
        if max_equity_record:
            peak_equity = max(peak_equity, max_equity_record)

        drawdown_pct = (
            ((peak_equity - summary["total_equity"]) / peak_equity * 100)
            if peak_equity > 0
            else 0
        )

        snapshot = PnLSnapshotDB(
            cycle_id=cycle_id,
            timestamp=datetime.utcnow(),
            total_equity=summary["total_equity"],
            cash_balance=summary["cash_balance"],
            total_exposure=summary["total_exposure"],
            unrealized_pnl=summary["unrealized_pnl"],
            realized_pnl=summary["realized_pnl"],
            num_positions=summary["num_positions"],
            drawdown_pct=drawdown_pct,
            meta_json=summary,
        )

        self.db_session.add(snapshot)
        self.db_session.commit()

        logger.info(
            f"Saved PnL snapshot: equity=${summary['total_equity']:.2f}, "
            f"drawdown={drawdown_pct:.2f}%"
        )

