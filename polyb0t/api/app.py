"""FastAPI application for health checks, reporting, and web dashboard."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from polyb0t.config import get_settings
from polyb0t.data.storage import get_session
from polyb0t.execution.portfolio import Portfolio
from polyb0t.services.health import get_health_status
from polyb0t.services.reporter import Reporter

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PolyB0T API",
    description="AI-powered trading bot for Polymarket with web dashboard",
    version="0.2.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

ws_manager = ConnectionManager()


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


# ============================================================================
# DASHBOARD API ENDPOINTS
# ============================================================================

@app.get("/api/dashboard")
async def get_dashboard() -> dict[str, Any]:
    """Get comprehensive dashboard data.
    
    Returns all status information for the web dashboard.
    """
    try:
        from polyb0t.services.status_aggregator import get_status_aggregator
        
        aggregator = get_status_aggregator()
        return aggregator.get_full_status(use_cache=True)
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experts")
async def get_experts() -> dict[str, Any]:
    """Get detailed expert information.
    
    Returns all 24 experts with their states, metrics, and versions.
    """
    try:
        from polyb0t.services.status_aggregator import get_status_aggregator
        
        aggregator = get_status_aggregator()
        moe_status = aggregator.get_moe_status()
        
        return {
            "total_experts": moe_status.total_experts,
            "state_counts": moe_status.state_counts,
            "experts": moe_status.top_experts,
            "gating_accuracy": moe_status.gating_accuracy,
        }
        
    except Exception as e:
        logger.error(f"Error getting experts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/meta")
async def get_meta_controller() -> dict[str, Any]:
    """Get Meta-Controller status.
    
    Returns mixture learning status and synergy insights.
    """
    try:
        from polyb0t.services.status_aggregator import get_status_aggregator
        
        aggregator = get_status_aggregator()
        return aggregator.get_meta_controller_status().to_dict()
        
    except Exception as e:
        logger.error(f"Error getting meta-controller status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mixtures")
async def get_mixtures() -> dict[str, Any]:
    """Get mixture performance history.
    
    Returns recent mixture predictions and their outcomes.
    """
    try:
        from polyb0t.ml.moe.meta_controller import get_meta_controller
        
        meta = get_meta_controller()
        status = meta.get_status()
        
        return {
            "total_mixtures": status["total_mixtures"],
            "resolved_mixtures": status["resolved_mixtures"],
            "total_profit_pct": status["total_profit_pct"],
            "win_rate": status["win_rate"],
            "top_mixtures": status["top_mixtures"],
            "synergy_insights": status["synergy_insights"],
            "category_performance": status["category_performance"],
        }
        
    except Exception as e:
        logger.error(f"Error getting mixtures: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades")
async def get_trades(limit: int = 50) -> dict[str, Any]:
    """Get recent trades with expert attribution.
    
    Args:
        limit: Maximum number of trades to return.
    """
    try:
        from polyb0t.services.status_aggregator import get_status_aggregator
        
        aggregator = get_status_aggregator()
        activity = aggregator.get_recent_activity(limit=limit)
        
        return {
            "trades": activity,
            "count": len(activity),
        }
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system")
async def get_system() -> dict[str, Any]:
    """Get system resource status.
    
    Returns CPU, memory, disk usage.
    """
    try:
        from polyb0t.services.status_aggregator import get_status_aggregator
        
        aggregator = get_status_aggregator()
        return aggregator.get_system_status().to_dict()
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ============================================================================

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates.
    
    Sends status updates every 5 seconds.
    """
    await ws_manager.connect(websocket)
    
    try:
        from polyb0t.services.status_aggregator import get_status_aggregator
        aggregator = get_status_aggregator()
        
        while True:
            # Send status update
            status = aggregator.get_full_status(use_cache=False)
            await websocket.send_json({
                "type": "status_update",
                "data": status,
            })
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# ============================================================================
# STATIC FILES FOR WEB DASHBOARD
# ============================================================================

# Check if web build exists
WEB_BUILD_DIR = Path(__file__).parent.parent / "web" / "dist"

if WEB_BUILD_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_BUILD_DIR / "assets")), name="assets")
    
    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        """Serve the web dashboard."""
        index_path = WEB_BUILD_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return HTMLResponse("<h1>Dashboard not built. Run: cd polyb0t/web && npm run build</h1>")
else:
    # Serve a simple HTML page if no build exists
    @app.get("/dashboard", response_class=HTMLResponse)
    async def serve_simple_dashboard():
        """Serve a simple dashboard without build."""
        return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>PolyB0T Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               background: #0a0a0a; color: #e0e0e0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00ff88; }
        .card { background: #1a1a1a; border-radius: 8px; padding: 20px; margin: 10px 0; }
        .status { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
        .metric { background: #252525; padding: 15px; border-radius: 6px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #00ff88; }
        .metric-label { color: #888; font-size: 12px; text-transform: uppercase; }
        .expert { padding: 10px; margin: 5px 0; background: #252525; border-radius: 4px; }
        .active { border-left: 3px solid #00ff88; }
        .suspended { border-left: 3px solid #ffa500; }
        .deprecated { border-left: 3px solid #ff4444; }
        .refresh-btn { background: #00ff88; color: #000; border: none; padding: 10px 20px; 
                       border-radius: 4px; cursor: pointer; font-weight: bold; }
        .refresh-btn:hover { background: #00cc66; }
    </style>
</head>
<body>
    <div class="container">
        <h1>PolyB0T Dashboard</h1>
        <button class="refresh-btn" onclick="loadData()">Refresh</button>
        
        <div class="card">
            <h2>System Status</h2>
            <div class="status" id="system-status">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Trading Status</h2>
            <div class="status" id="trading-status">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Mixture of Experts (24)</h2>
            <div class="status" id="moe-status">Loading...</div>
            <div id="experts-list"></div>
        </div>
        
        <div class="card">
            <h2>Meta-Controller</h2>
            <div class="status" id="meta-status">Loading...</div>
        </div>
    </div>
    
    <script>
        async function loadData() {
            try {
                const res = await fetch('/api/dashboard');
                const data = await res.json();
                
                // System status
                document.getElementById('system-status').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${data.system.cpu_percent.toFixed(1)}%</div>
                        <div class="metric-label">CPU</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.system.memory_percent.toFixed(1)}%</div>
                        <div class="metric-label">Memory</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.system.uptime_hours.toFixed(1)}h</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                `;
                
                // Trading status
                const t = data.trading;
                document.getElementById('trading-status').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${t.is_live ? 'LIVE' : 'DRY RUN'}</div>
                        <div class="metric-label">Mode</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">$${t.balance_usd.toFixed(2)}</div>
                        <div class="metric-label">Balance</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${t.position_count}</div>
                        <div class="metric-label">Positions</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${t.trades_today}</div>
                        <div class="metric-label">Trades Today</div>
                    </div>
                `;
                
                // MoE status
                const m = data.moe;
                document.getElementById('moe-status').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${m.active_experts}</div>
                        <div class="metric-label">Active</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${m.suspended_experts}</div>
                        <div class="metric-label">Suspended</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${m.probation_experts}</div>
                        <div class="metric-label">Probation</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(m.gating_accuracy * 100).toFixed(1)}%</div>
                        <div class="metric-label">Routing Accuracy</div>
                    </div>
                `;
                
                // Expert list
                if (m.top_experts && m.top_experts.length > 0) {
                    document.getElementById('experts-list').innerHTML = m.top_experts.map(e => `
                        <div class="expert ${e.state}">
                            <strong>${e.domain}</strong> [${e.state}]
                            - Profit: ${(e.profit_pct * 100).toFixed(1)}%
                            - Trades: ${e.trades}
                            - Win: ${(e.win_rate * 100).toFixed(0)}%
                        </div>
                    `).join('');
                }
                
                // Meta-controller
                const mc = data.meta_controller;
                document.getElementById('meta-status').innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${mc.total_mixtures}</div>
                        <div class="metric-label">Total Mixtures</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(mc.win_rate * 100).toFixed(1)}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(mc.total_profit_pct * 100).toFixed(1)}%</div>
                        <div class="metric-label">Total Profit</div>
                    </div>
                `;
                
            } catch (e) {
                console.error('Error loading data:', e);
            }
        }
        
        // Load data on page load
        loadData();
        
        // Auto-refresh every 10 seconds
        setInterval(loadData, 10000);
        
        // WebSocket for real-time updates
        try {
            const ws = new WebSocket('ws://' + window.location.host + '/ws/live');
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === 'status_update') {
                    // Update with real-time data
                    loadData();
                }
            };
        } catch (e) {
            console.log('WebSocket not available, using polling');
        }
    </script>
</body>
</html>
        """)

