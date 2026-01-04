"""FastAPI application for health checks and reporting."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from polyb0t.config import get_settings
from polyb0t.data.storage import get_session
from polyb0t.execution.portfolio import Portfolio
from polyb0t.services.health import get_health_status
from polyb0t.services.reporter import Reporter

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PolyB0T API",
    description="Paper trading bot for Polymarket",
    version="0.1.0",
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"name": "PolyB0T", "version": "0.1.0", "status": "running"}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Health status information.
    """
    health_status = get_health_status()
    status_dict = health_status.to_dict()

    if not health_status.is_healthy:
        return JSONResponse(status_code=503, content=status_dict)

    return status_dict


@app.get("/status")
async def status() -> dict[str, Any]:
    """Get current trading status.

    Returns:
        Current portfolio status and positions.
    """
    try:
        db_session = get_session()
        reporter = Reporter(db_session)

        # Get position summary
        positions = reporter._get_position_summary()

        # Calculate metrics
        total_exposure = sum(pos.get("quantity", 0) for pos in positions)
        unrealized_pnl = sum(pos.get("unrealized_pnl", 0) for pos in positions)

        db_session.close()

        return {
            "num_positions": len(positions),
            "total_exposure": total_exposure,
            "unrealized_pnl": unrealized_pnl,
            "positions": positions,
        }

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report")
async def report() -> dict[str, Any]:
    """Get trading report.

    Returns:
        Daily trading report with PnL and activity.
    """
    try:
        db_session = get_session()
        reporter = Reporter(db_session)

        # Create a temporary portfolio instance to get summary
        # In production, this would reference the actual running portfolio
        settings = get_settings()
        portfolio = Portfolio(settings.paper_bankroll)

        # Generate report
        daily_report = reporter.generate_daily_report(portfolio)

        db_session.close()

        return daily_report

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Get trading metrics.

    Returns:
        Key trading metrics.
    """
    try:
        db_session = get_session()
        reporter = Reporter(db_session)

        # Get recent fills
        recent_fills = reporter._get_recent_fills(limit=5)

        # Get top signals
        top_signals = reporter._get_top_signals(limit=5)

        db_session.close()

        return {
            "recent_fills": recent_fills,
            "top_signals": top_signals,
        }

    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/intents")
async def list_intents(
    status: str | None = None, limit: int = 50
) -> dict[str, Any]:
    """Get trade intents.

    Args:
        status: Filter by status (PENDING, APPROVED, REJECTED, EXPIRED, EXECUTED).
        limit: Maximum number of intents to return.

    Returns:
        List of trade intents.
    """
    try:
        db_session = get_session()
        from polyb0t.execution.intents import IntentManager
        from polyb0t.data.storage import TradeIntentDB

        intent_manager = IntentManager(db_session)

        query = db_session.query(TradeIntentDB).order_by(TradeIntentDB.created_at.desc())

        if status:
            query = query.filter_by(status=status.upper())

        db_intents = query.limit(limit).all()
        intents = [intent_manager._load_intent_from_db(db_intent) for db_intent in db_intents]

        db_session.close()

        return {
            "intents": [intent.to_dict() for intent in intents],
            "count": len(intents),
        }

    except Exception as e:
        logger.error(f"Error listing intents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/intents/{intent_id}")
async def get_intent(intent_id: str) -> dict[str, Any]:
    """Get specific trade intent by ID.

    Args:
        intent_id: Intent identifier.

    Returns:
        Trade intent details.
    """
    try:
        db_session = get_session()
        from polyb0t.execution.intents import IntentManager

        intent_manager = IntentManager(db_session)
        intent = intent_manager.get_intent(intent_id)

        db_session.close()

        if not intent:
            raise HTTPException(status_code=404, detail="Intent not found")

        return intent.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting intent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/intents/{intent_id}/approve")
async def approve_intent(
    intent_id: str, approved_by: str = "api_user"
) -> dict[str, Any]:
    """Approve a trade intent.

    Args:
        intent_id: Intent identifier.
        approved_by: User identifier (optional).

    Returns:
        Approval result.
    """
    try:
        settings = get_settings()

        if settings.mode != "live":
            raise HTTPException(
                status_code=400,
                detail="Intent approval only available in live mode",
            )

        db_session = get_session()
        from polyb0t.execution.intents import IntentManager

        intent_manager = IntentManager(db_session)
        approved_intent = intent_manager.approve_intent(intent_id, approved_by)

        db_session.close()

        if not approved_intent:
            raise HTTPException(
                status_code=400,
                detail="Intent not found, expired, or cannot be approved",
            )

        logger.info(
            f"Intent approved via API: {intent_id[:8]}",
            extra={"intent_id": intent_id, "approved_by": approved_by},
        )

        return {
            "success": True,
            "message": "Intent approved",
            "intent": approved_intent.to_dict(),
            "dry_run": settings.dry_run,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving intent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/intents/{intent_id}/reject")
async def reject_intent(intent_id: str) -> dict[str, Any]:
    """Reject a trade intent.

    Args:
        intent_id: Intent identifier.

    Returns:
        Rejection result.
    """
    try:
        db_session = get_session()
        from polyb0t.execution.intents import IntentManager

        intent_manager = IntentManager(db_session)
        rejected = intent_manager.reject_intent(intent_id)

        db_session.close()

        if not rejected:
            raise HTTPException(
                status_code=400,
                detail="Intent not found or cannot be rejected",
            )

        logger.info(
            f"Intent rejected via API: {intent_id[:8]}",
            extra={"intent_id": intent_id},
        )

        return {
            "success": True,
            "message": "Intent rejected",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rejecting intent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

