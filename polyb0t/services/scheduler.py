"""Main trading scheduler and orchestration."""

import asyncio
import logging
import time
import uuid
from typing import Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from polyb0t.config import get_settings
from polyb0t.data import CLOBClient, GammaClient, init_db
from polyb0t.data.models import OrderBook, OrderBookLevel, Trade
from polyb0t.data.storage import (
    MarketDB,
    MarketOutcomeDB,
    OrderBookSnapshotDB,
    SignalDB,
    TradeDB,
    get_session,
)
from polyb0t.execution import PaperTradingSimulator, Portfolio
from polyb0t.execution.intents import IntentManager, IntentStatus, IntentType
from polyb0t.execution.live_executor import LiveExecutor
from polyb0t.models import BaselineStrategy, FeatureEngine, RiskManager
from polyb0t.models.exit_manager import ExitManager
from polyb0t.models.kill_switches import KillSwitchManager
from polyb0t.models.filters import MarketFilter
from polyb0t.models.strategy_baseline import TradingSignal
from polyb0t.services.health import get_health_status
from polyb0t.services.reporter import Reporter
from polyb0t.services.system_monitor import get_system_monitor, start_system_monitor
from polyb0t.services.arbitrage_scanner import get_arbitrage_scanner, ArbitrageOpportunity

logger = logging.getLogger(__name__)


def _check_strategy_mode() -> None:
    """Check AI/MoE readiness. Rules mode has been removed."""
    settings = get_settings()
    
    logger.info("Strategy mode: AI/MoE (rules mode removed)")
    logger.info(f"Placing orders: {settings.placing_orders}")
    
    # AI mode is always on now, but we allow running without a trained model
    # (the bot will just collect data until model is ready)
    try:
        from polyb0t.ml.ai_orchestrator import get_ai_orchestrator
        orchestrator = get_ai_orchestrator()
        if orchestrator.is_ai_ready():
            logger.info("AI model is ready for trading")
        else:
            logger.info("AI model not ready - will collect training data")
            if settings.placing_orders:
                logger.warning("Orders enabled but no trained model - bot will not trade until model is ready")
    except ImportError as e:
        logger.error(f"AI modules not available: {e}")
        if settings.placing_orders:
            logger.error("Cannot start with orders enabled but no AI module")
            raise SystemExit(1)


class TradingScheduler:
    """Main trading loop scheduler."""

    def __init__(self) -> None:
        """Initialize trading scheduler."""
        self.settings = get_settings()
        self.is_running = False
        self.health = get_health_status()
        self.consecutive_errors = 0

        # Check strategy mode before anything else
        _check_strategy_mode()

        # Start system resource monitoring
        self.system_monitor = get_system_monitor()
        start_system_monitor()
        logger.info("System resource monitor started")

        # Initialize database
        init_db()

        # Initialize components
        self.portfolio = Portfolio()
        self.risk_manager = RiskManager()
        self.strategy = BaselineStrategy()
        self.market_filter = MarketFilter()
        
        # AI orchestrator (for AI mode and data collection)
        self.ai_orchestrator: Any | None = None
        if self.settings.strategy_mode == "ai" or not self.settings.placing_orders:
            try:
                from polyb0t.ml.ai_orchestrator import get_ai_orchestrator
                self.ai_orchestrator = get_ai_orchestrator()
                stats = self.ai_orchestrator.get_training_stats()
                logger.info(
                    f"AI Orchestrator initialized: "
                    f"{stats['collector']['total_examples']} examples, "
                    f"model_ready={stats['is_ready']}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize AI orchestrator: {e}")

        # Arbitrage scanner (optional)
        self.arbitrage_scanner = None
        if self.settings.enable_arbitrage_scanner:
            from polyb0t.services.arbitrage_scanner import ArbitrageScanner
            self.arbitrage_scanner = ArbitrageScanner()
            logger.info("Arbitrage scanner enabled")

        # WebSocket client for real-time market data
        self.ws_client = None
        self._ws_connected = False
        try:
            from polyb0t.data.ws_client import get_ws_client
            self.ws_client = get_ws_client()
            logger.info("WebSocket client initialized")
        except Exception as e:
            logger.warning(f"WebSocket client not available: {e}")
        
        # Whale tracker for detecting large market-moving trades
        self.whale_tracker = None
        try:
            from polyb0t.services.whale_tracker import get_whale_tracker
            self.whale_tracker = get_whale_tracker()
            # Connect to WebSocket if available
            if self.ws_client:
                self.whale_tracker.connect_to_websocket(self.ws_client)
            logger.info("Whale tracker initialized")
        except Exception as e:
            logger.warning(f"Whale tracker not available: {e}")
        
        # Discord notifier for alerts and status updates
        self.discord_notifier = None
        try:
            from polyb0t.services.discord_notifier import get_discord_notifier
            self.discord_notifier = get_discord_notifier()
            if self.discord_notifier.enabled:
                logger.info("Discord notifier initialized")
            else:
                logger.info("Discord notifier disabled (no webhook URL configured)")
        except Exception as e:
            logger.warning(f"Discord notifier not available: {e}")
        
        # Track last hourly/daily report times
        self._last_hourly_report = datetime.utcnow()
        self._last_daily_report = datetime.utcnow()
        
        # Track last training time when we showed metrics debrief
        # This ensures we only show the model performance debrief once per training cycle
        self._last_metrics_debrief_training_time: datetime | None = None

        logger.info(
            f"Trading scheduler initialized "
            f"(mode={self.settings.strategy_mode}, placing_orders={self.settings.placing_orders})"
        )

    async def run(self) -> None:
        """Run main trading loop continuously."""
        self.is_running = True
        logger.info("Starting trading loop")
        
        # Start WebSocket connection for real-time market data
        if self.ws_client:
            try:
                connected = await self.ws_client.connect()
                if connected:
                    self._ws_connected = True
                    logger.info("WebSocket connected - using real-time market data")
                else:
                    logger.warning("WebSocket connection failed - falling back to HTTP polling")
            except Exception as e:
                logger.warning(f"WebSocket startup error: {e} - falling back to HTTP polling")
        
        # Send Discord startup notification
        await self._send_discord_startup()
        
        # Backfill missing price data on startup (if enabled)
        if self.settings.ml_enable_backfill:
            try:
                from polyb0t.ml.backfill import HistoricalDataBackfiller
                logger.info("Checking for missing price data to backfill...")
                backfiller = HistoricalDataBackfiller(self.settings.ml_data_db)
                stats = backfiller.get_price_history_stats()
                logger.info(
                    f"Price history: {stats['total_price_points']} points, "
                    f"{stats['unique_tokens']} tokens, "
                    f"{stats['coverage_days']:.1f} days coverage, "
                    f"avg interval: {stats['avg_interval_minutes']:.1f}min "
                    f"(target: {stats['target_interval_minutes']}min)"
                )
            except Exception as e:
                logger.warning(f"Price history backfill check failed: {e}")

        while self.is_running:
            cycle_start = time.time()
            cycle_id = str(uuid.uuid4())

            try:
                await self._run_cycle(cycle_id)

                # Update health
                cycle_duration = time.time() - cycle_start
                self.health.update_cycle(cycle_duration)
                self.health.mark_healthy()
                self.consecutive_errors = 0

                logger.info(
                    f"Cycle {cycle_id[:8]} completed in {cycle_duration:.2f}s",
                    extra={"cycle_id": cycle_id, "duration": cycle_duration},
                )

            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)
                self.health.mark_unhealthy(str(e))
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.settings.kill_switch_on_errors:
                    logger.error(
                        "Kill switch triggered: too many consecutive cycle errors",
                        extra={
                            "consecutive_errors": self.consecutive_errors,
                            "threshold": self.settings.kill_switch_on_errors,
                        },
                    )
                    self.is_running = False
                    break

            # Check for periodic Discord reports
            await self._check_discord_reports()
            
            # Wait before next cycle
            await asyncio.sleep(self.settings.loop_interval_seconds)

    async def _run_cycle(self, cycle_id: str) -> None:
        """Run single trading cycle.

        Args:
            cycle_id: Unique cycle identifier.
        """
        db_session = get_session()

        try:
            intent_manager = IntentManager(db_session)
            kill_switches = KillSwitchManager(db_session)
            execution_summary: dict[str, Any] | None = None
            balance_summary: dict[str, Any] | None = None
            account_state_summary: dict[str, Any] | None = None
            executor: LiveExecutor | None = None

            # Live mode: process already-approved intents (human-in-the-loop).
            # This is the ONLY path to any order submission, and is hard-gated by:
            # - explicit approval
            # - settings.dry_run (no order submission when true)
            if self.settings.mode == "live":
                # Hygiene: expire + backfill + collapse duplicates so dedup is DB-backed and stable.
                # This enforces the invariant: <= 1 pending intent per fingerprint.
                expired_count = intent_manager.expire_old_intents()
                intent_manager.backfill_missing_fingerprints(statuses=[IntentStatus.PENDING, IntentStatus.APPROVED])
                dedup_count = intent_manager.cleanup_duplicate_pending_intents(mode="supersede")
                if expired_count > 0 or dedup_count.get("deduped", 0) > 0:
                    logger.info(
                        f"Intent cleanup: expired={expired_count}, deduped={dedup_count.get('deduped', 0)}"
                    )

                # Balance snapshot (best-effort; safe, read-only). Used for sizing & observability.
                try:
                    from polyb0t.services.balance import BalanceService

                    bal = BalanceService(db_session=db_session)
                    snap = bal.fetch_usdc_balance()
                    bal.persist_snapshot(cycle_id=cycle_id, snap=snap)
                    balance_summary = {
                        "total_usdc": snap.total_usdc,
                        "reserved_usdc": snap.reserved_usdc,
                        "available_usdc": snap.available_usdc,
                    }
                    logger.info(
                        f"Balance snapshot: total={snap.total_usdc:.2f} USDC, "
                        f"reserved={snap.reserved_usdc:.2f}, available={snap.available_usdc:.2f}",
                        extra={
                            "cycle_id": cycle_id,
                            "total_usdc": snap.total_usdc,
                            "reserved_usdc": snap.reserved_usdc,
                            "available_usdc": snap.available_usdc,
                        },
                    )
                    
                    # === GLOBAL STOP-LOSS CHECK ===
                    # Update smart heuristics engine with current equity
                    if self.settings.enable_global_stop_loss:
                        try:
                            from polyb0t.models.smart_heuristics import get_smart_heuristics_engine
                            heuristics = get_smart_heuristics_engine()
                            heuristics.update_equity(snap.total_usdc)
                            
                            if heuristics.is_listen_only():
                                logger.warning(
                                    "🛑 LISTEN-ONLY MODE: Bot is in observation mode due to drawdown. "
                                    "No new positions will be opened.",
                                    extra={"cycle_id": cycle_id, "total_usdc": snap.total_usdc},
                                )
                        except Exception as e:
                            logger.debug(f"Global stop-loss check failed: {e}")
                            
                except Exception as e:
                    balance_summary = {"error": str(e)}
                    logger.warning(f"Balance snapshot unavailable: {e}")

                # Account state snapshot (best-effort; may require auth). Used for open orders/positions.
                held_long_token_ids: set[str] = set()
                markets_with_existing_positions: set[str] = set()  # Blocks new buys for markets we already hold
                try:
                    from polyb0t.data.account_state import AccountStateProvider
                    from polyb0t.data.storage import AccountStateDB

                    async with AccountStateProvider() as provider:
                        state = await provider.fetch_account_state()
                    if state:
                        # Track LONG holdings so we can safely allow SELL intents that reduce existing
                        # positions (including positions opened manually by the user).
                        try:
                            held_long_token_ids = {
                                p.token_id
                                for p in (state.positions or [])
                                if getattr(p, "token_id", None)
                                and float(getattr(p, "quantity", 0) or 0) > 0
                                and str(getattr(p, "side", "LONG")).upper() == "LONG"
                            }
                        except Exception as e:
                            logger.debug(f"Could not extract held long tokens: {e}")
                            held_long_token_ids = set()
                        
                        # CRITICAL: Track markets where we already have positions
                        # This prevents buying back into markets we just sold from
                        # Once we have ANY position in a market, we don't open more until ALL are closed
                        try:
                            markets_with_existing_positions = {
                                str(p.market_id)
                                for p in (state.positions or [])
                                if getattr(p, "market_id", None)
                                and float(getattr(p, "quantity", 0) or 0) > 0
                            }
                            if markets_with_existing_positions:
                                logger.info(
                                    f"Markets with existing positions (blocked for new buys): {len(markets_with_existing_positions)} markets",
                                    extra={"market_ids": list(markets_with_existing_positions)[:10]},  # Log first 10
                                )
                        except Exception as e:
                            logger.debug(f"Could not extract markets with existing positions: {e}")
                            markets_with_existing_positions = set()

                        db_session.add(
                            AccountStateDB(
                                cycle_id=cycle_id,
                                timestamp=datetime.utcnow(),
                                wallet_address=state.wallet_address,
                                cash_balance=state.cash_balance,
                                total_equity=state.total_equity,
                                open_orders_count=len(state.open_orders),
                                positions_count=len(state.positions),
                                positions=[p.to_dict() for p in state.positions],
                                open_orders=[o.to_dict() for o in state.open_orders],
                                meta_json=state.metadata,
                            )
                        )
                        db_session.commit()
                        account_state_summary = {
                            "open_orders_count": len(state.open_orders),
                            "positions_count": len(state.positions),
                        }
                except Exception as e:
                    account_state_summary = {"error": str(e)}

                # Handle stale orders: cancel old unfilled sell orders and resubmit at market price
                # This prevents holding losing positions while waiting for limit orders to fill
                if self.settings.enable_panic_sell and not self.settings.dry_run:
                    try:
                        from polyb0t.services.stale_order_manager import StaleOrderManager
                        
                        stale_manager = StaleOrderManager()
                        stale_result = stale_manager.process_stale_orders()
                        
                        if stale_result.get("resubmitted", 0) > 0:
                            logger.info(
                                f"Stale order refresh: {stale_result['resubmitted']} orders resubmitted at market price",
                                extra=stale_result,
                            )
                        elif stale_result.get("stale_orders", 0) > 0:
                            logger.warning(
                                f"Stale orders detected but resubmit failed",
                                extra=stale_result,
                            )
                    except Exception as e:
                        logger.warning(f"Stale order check failed: {e}")

                if self.settings.dry_run:
                    logger.info(
                        "Dry-run live mode: no positions will be opened; intents are recommendations only",
                        extra={"cycle_id": cycle_id, "dry_run": True},
                    )
                if self.settings.auto_approve_intents:
                    logger.warning(
                        "AUTO-APPROVE ENABLED: intents will be approved automatically (no human gate). "
                        "If dry_run=false and credentials are configured, this will place REAL orders.",
                        extra={"cycle_id": cycle_id, "dry_run": self.settings.dry_run},
                    )
                executor = LiveExecutor(db_session=db_session, intent_manager=intent_manager)
                
                # PLACING_ORDERS check: skip all execution if disabled
                if not self.settings.placing_orders:
                    logger.info("Placing orders DISABLED - skipping execution (data collection mode)")
                    execution_summary = {"processed": 0, "executed": 0, "failed": 0, "dry_run": True, "reason": "placing_orders=false"}
                else:
                    execution_summary = executor.process_approved_intents(cycle_id=cycle_id)

            # Step 1: Fetch markets
            markets, gamma_list_diag = await self._fetch_markets(db_session)
            logger.info(f"Fetched {len(markets)} markets")

            # Step 2: Filter tradable universe (initial pass without orderbooks)
            tradable_markets, initial_rejections = self.market_filter.filter_markets(markets)
            logger.info(
                f"Filtered to {len(tradable_markets)} tradable markets",
                extra={"initial_rejections": initial_rejections},
            )

            if not tradable_markets:
                # Log detailed diagnostics to help debug filter issues
                logger.warning(
                    "No tradable markets found - FILTER ANALYSIS",
                    extra={
                        "markets_fetched": len(markets),
                        "rejection_breakdown": initial_rejections,
                        "top_rejection_reasons": sorted(
                            initial_rejections.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:5] if initial_rejections else [],
                        "settings": {
                            "resolve_min_days": self.settings.resolve_min_days,
                            "resolve_max_days": self.settings.resolve_max_days,
                            "min_liquidity": self.settings.min_liquidity,
                            "max_spread": self.settings.max_spread,
                        },
                        "hint": "Run 'polyb0t diagnose-filters' for detailed analysis",
                    },
                )
                return

            # Live mode: limit universe size to keep 10s cadence and avoid rate limits.
            gamma_enrich_diag: dict[str, Any] | None = None
            markets_for_data = tradable_markets  # Default: use tradable markets
            
            if self.settings.mode == "live":
                # Broad scan -> narrow enrichment/data-fetch to top-N by volume to avoid rate limits.
                enrich_limit = int(getattr(self.settings, "live_enrich_markets_limit", 50))
                clob_limit = int(getattr(self.settings, "live_clob_markets_limit", 50))

                # Enrich markets based on ML settings
                if self.settings.enable_ml and self.settings.ml_data_collection_limit > 0:
                    # ML MODE: Enrich more markets for comprehensive data collection
                    # Sort by volume and take top N
                    markets_sorted = sorted(
                        tradable_markets,
                        key=lambda m: m.volume or 0,
                        reverse=True
                    )
                    markets_to_enrich = markets_sorted[:min(self.settings.ml_data_collection_limit, len(markets_sorted))]
                    tradable_for_trading = markets_sorted[:10]  # Still trade on top 10 only
                    
                    logger.info(
                        f"ML mode: enriching {len(markets_to_enrich)} markets for data collection, "
                        f"trading on {len(tradable_for_trading)}"
                    )
                else:
                    # STANDARD MODE: scan broad, enrich only top-N by volume
                    markets_sorted = sorted(
                        tradable_markets,
                        key=lambda m: m.volume or 0,
                        reverse=True,
                    )
                    markets_to_enrich = markets_sorted[: min(enrich_limit, len(markets_sorted))]
                    tradable_for_trading = markets_to_enrich
                    logger.info(
                        "Live mode scan/enrich limits",
                        extra={
                            "markets_scanned": len(markets),
                            "markets_tradable_initial": len(tradable_markets),
                            "enrich_limit": enrich_limit,
                            "clob_limit": clob_limit,
                            "enriching": len(markets_to_enrich),
                        },
                    )

                # Enrich markets with outcomes (token_ids + Gamma prices)
                markets_for_data, gamma_enrich_diag = await self._enrich_markets_with_outcomes(markets_to_enrich)
                self._persist_market_outcomes(markets_for_data, db_session)
                
                zero_outcomes = sum(1 for m in markets_for_data if not m.outcomes)
                if zero_outcomes:
                    logger.warning(
                        "Some markets missing outcomes after Gamma enrichment",
                        extra={"markets_missing_outcomes": zero_outcomes},
                    )
                
                # Update tradable_markets for trading (subset of enriched markets)
                if self.settings.enable_ml:
                    tradable_markets = [m for m in markets_for_data if m in tradable_for_trading]
                else:
                    tradable_markets = markets_for_data

                # Final cap: only fetch CLOB data for top-N enriched markets (rate-limit safety)
                if clob_limit > 0 and len(markets_for_data) > clob_limit:
                    markets_for_data = sorted(
                        markets_for_data,
                        key=lambda m: m.volume or 0,
                        reverse=True,
                    )[:clob_limit]

            # Step 3: Fetch orderbooks and trades for enriched markets
            orderbooks, trades, clob_diag = await self._fetch_market_data(
                markets_for_data, db_session
            )
            logger.info(
                f"Market data refreshed: orderbooks={len(orderbooks)}, trades={len(trades)}"
            )

            # Per-cycle price visibility diagnostics (non-spammy, once/cycle)
            token_ids = [o.token_id for m in tradable_markets for o in m.outcomes if o.token_id]
            token_ids_unique = sorted(set(token_ids))
            outcomes_total = sum(len(m.outcomes) for m in tradable_markets)
            gamma_prices_available = sum(
                1 for m in tradable_markets for o in m.outcomes if o.price is not None
            )
            logger.info(
                "Price inputs diagnostics",
                extra={
                    "cycle_id": cycle_id,
                    "universe_markets": len(tradable_markets),
                    "outcomes_total": outcomes_total,
                    "token_ids_resolved": len(token_ids),
                    "token_ids_unique": len(token_ids_unique),
                    "token_ids_sample": token_ids_unique[:10],
                    "sources_attempted": {
                        "clob_orderbooks": len(token_ids_unique) > 0,
                        "clob_trades": len(token_ids_unique) > 0,
                        "gamma_outcome_prices": outcomes_total > 0,
                    },
                    "gamma": {
                        "list_markets": gamma_list_diag,
                        "enrich_markets": gamma_enrich_diag,
                        "outcome_prices_available": gamma_prices_available,
                    },
                    "clob": clob_diag,
                },
            )

            # Step 4: Filter markets by spread, depth, and liquidity
            tradable_markets, filter_rejections = self.market_filter.filter_markets(
                tradable_markets, orderbooks
            )
            logger.info(
                f"After all filtering: {len(tradable_markets)} tradable markets",
                extra={"filter_rejections": filter_rejections},
            )

            # Step 4.5: Scan for arbitrage opportunities (risk-free profit)
            arbitrage_intents_created = 0
            if self.arbitrage_scanner and self.settings.mode == "live":
                try:
                    # Convert orderbooks to dict format for arbitrage scanner
                    ob_dicts: dict[str, dict[str, Any]] = {}
                    for token_id, ob in orderbooks.items():
                        if ob and hasattr(ob, 'bids') and hasattr(ob, 'asks'):
                            ob_dicts[token_id] = {
                                "bids": [{"price": l.price, "size": l.size} for l in (ob.bids or [])],
                                "asks": [{"price": l.price, "size": l.size} for l in (ob.asks or [])],
                            }
                    
                    # Scan all markets (not just tradable) for arbitrage
                    arb_opportunities = self.arbitrage_scanner.scan_all(
                        markets=markets_for_data,  # Broad market set
                        orderbooks=ob_dicts,
                    )
                    
                    if arb_opportunities:
                        logger.warning(
                            f"🎯 ARBITRAGE FOUND: {len(arb_opportunities)} opportunities!",
                            extra={
                                "opportunities": [o.to_dict() for o in arb_opportunities[:3]],
                                "best_profit_pct": arb_opportunities[0].profit_pct,
                            },
                        )
                        
                        # Create intents for arbitrage opportunities
                        for opp in arb_opportunities[:3]:  # Max 3 arbs per cycle
                            try:
                                arb_intents = self.arbitrage_scanner.create_arbitrage_intents(
                                    opportunity=opp,
                                    intent_manager=intent_manager,
                                    cycle_id=cycle_id,
                                    max_usd_per_arb=self.settings.arbitrage_max_usd_per_opportunity,
                                )
                                arbitrage_intents_created += len(arb_intents)
                                
                                # Auto-approve if configured (DANGER: use with caution)
                                if self.settings.arbitrage_auto_execute:
                                    for intent in arb_intents:
                                        intent_manager.approve_intent(
                                            intent.intent_id,
                                            note="Auto-approved arbitrage (risk-free)",
                                        )
                                        logger.warning(f"Auto-approved arbitrage intent: {intent.intent_id[:8]}")
                            except Exception as e:
                                logger.error(f"Failed to create arb intent: {e}")
                    else:
                        logger.debug("No arbitrage opportunities found this cycle")
                        
                except Exception as e:
                    logger.warning(f"Arbitrage scan failed: {e}")

            # Step 5: Generate trading signals with balance-aware sizing
            available_for_signals = float(balance_summary.get("available_usdc", 0.0) or 0.0) if balance_summary else 0.0
            reserved_for_signals = float(balance_summary.get("reserved_usdc", 0.0) or 0.0) if balance_summary else 0.0
            
            # === AI DATA COLLECTION ===
            # Collect data for AI training (runs regardless of strategy mode)
            if self.ai_orchestrator:
                self.system_monitor.set_context("collecting")
                try:
                    # Track markets for AI training
                    market_dicts = [
                        {"condition_id": m.condition_id, "outcomes": [{"token_id": o.token_id} for o in m.outcomes]}
                        for m in markets_for_data
                    ]
                    new_tracked = self.ai_orchestrator.track_markets(market_dicts)
                    
                    # Collect comprehensive snapshots from orderbooks
                    snapshots_collected = 0
                    examples_created_this_cycle = False
                    now = datetime.utcnow()
                    
                    for m in tradable_markets:
                        for o in m.outcomes:
                            if not o.token_id:
                                continue
                            ob = orderbooks.get(o.token_id)
                            if not ob or not ob.bids or not ob.asks:
                                continue
                            
                            # === CORE PRICE DATA ===
                            best_bid = ob.bids[0].price
                            best_ask = ob.asks[0].price
                            mid = (best_bid + best_ask) / 2
                            spread = best_ask - best_bid
                            spread_pct = spread / mid if mid > 0 else 0
                            
                            # === ORDERBOOK DEPTH ===
                            bid_depth_5 = sum(l.size for l in ob.bids[:5])
                            ask_depth_5 = sum(l.size for l in ob.asks[:5])
                            bid_depth_10 = sum(l.size for l in ob.bids[:10])
                            ask_depth_10 = sum(l.size for l in ob.asks[:10])
                            total_bid_depth = bid_depth_5 + bid_depth_10
                            total_ask_depth = ask_depth_5 + ask_depth_10
                            imbalance = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth) if (total_bid_depth + total_ask_depth) > 0 else 0
                            
                            best_bid_size = ob.bids[0].size if ob.bids else 0
                            best_ask_size = ob.asks[0].size if ob.asks else 0
                            bid_ask_size_ratio = best_bid_size / best_ask_size if best_ask_size > 0 else 1
                            
                            # === VOLUME & LIQUIDITY ===
                            volume_24h = float(m.volume or 0)
                            liquidity = float(m.liquidity or 0)
                            liquidity_bid = sum(l.size * l.price for l in ob.bids[:10])
                            liquidity_ask = sum(l.size * l.price for l in ob.asks[:10])
                            
                            # === MOMENTUM (from collector's price history) ===
                            momentum_1h = self.ai_orchestrator.collector.compute_momentum(o.token_id, 1)
                            momentum_4h = self.ai_orchestrator.collector.compute_momentum(o.token_id, 4)
                            momentum_24h = self.ai_orchestrator.collector.compute_momentum(o.token_id, 24)
                            momentum_7d = self.ai_orchestrator.collector.compute_momentum(o.token_id, 168)
                            
                            # === VOLATILITY ===
                            volatility_1h = self.ai_orchestrator.collector.compute_volatility(o.token_id, 1)
                            volatility_24h = self.ai_orchestrator.collector.compute_volatility(o.token_id, 24)
                            volatility_7d = self.ai_orchestrator.collector.compute_volatility(o.token_id, 168)
                            
                            # === TIMING FEATURES ===
                            days_to_res = self._days_to_resolution(m.end_date)
                            hours_to_res = days_to_res * 24
                            hour_of_day = now.hour
                            day_of_week = now.weekday()
                            is_weekend = day_of_week >= 5
                            
                            # === MARKET METADATA ===
                            category = getattr(m, 'category', '')
                            market_slug = getattr(m, 'slug', '')
                            question = getattr(m, 'question', '')
                            description = getattr(m, 'description', '')
                            
                            # === DERIVED FEATURES ===
                            price_vs_volume = mid / volume_24h if volume_24h > 0 else 0
                            liq_per_vol = liquidity / volume_24h if volume_24h > 0 else 0
                            spread_adj_edge = max(0, abs(mid - 0.5) - spread_pct)
                            
                            created = self.ai_orchestrator.collect_snapshot(
                                token_id=o.token_id,
                                market_id=m.condition_id,
                                price=mid,
                                bid=best_bid,
                                ask=best_ask,
                                spread=spread,
                                spread_pct=spread_pct,
                                mid_price=mid,
                                orderbook_imbalance=imbalance,
                                volume_24h=volume_24h,
                                liquidity=liquidity,
                                liquidity_bid=liquidity_bid,
                                liquidity_ask=liquidity_ask,
                                bid_depth=total_bid_depth,
                                ask_depth=total_ask_depth,
                                bid_depth_5=bid_depth_5,
                                ask_depth_5=ask_depth_5,
                                bid_depth_10=bid_depth_10,
                                ask_depth_10=ask_depth_10,
                                bid_levels=len(ob.bids),
                                ask_levels=len(ob.asks),
                                best_bid_size=best_bid_size,
                                best_ask_size=best_ask_size,
                                bid_ask_size_ratio=bid_ask_size_ratio,
                                momentum_1h=momentum_1h,
                                momentum_4h=momentum_4h,
                                momentum_24h=momentum_24h,
                                momentum_7d=momentum_7d,
                                volatility_1h=volatility_1h,
                                volatility_24h=volatility_24h,
                                volatility_7d=volatility_7d,
                                days_to_resolution=days_to_res,
                                hours_to_resolution=hours_to_res,
                                hour_of_day=hour_of_day,
                                day_of_week=day_of_week,
                                category=category,
                                market_slug=market_slug,
                                question_length=len(question),
                                description_length=len(description),
                                price_vs_volume_ratio=price_vs_volume,
                                liquidity_per_dollar_volume=liq_per_vol,
                                spread_adjusted_edge=spread_adj_edge,
                            )
                            snapshots_collected += 1
                            if created:
                                examples_created_this_cycle = True
                    
                    self.ai_orchestrator.finish_example_cycle(examples_created=examples_created_this_cycle)
                    
                    # Log AI training progress
                    stats = self.ai_orchestrator.get_training_stats()
                    logger.info(
                        f"AI data collection: {snapshots_collected} snapshots, "
                        f"{stats['collector']['total_examples']} total examples, "
                        f"{stats['collector']['labeled_examples']} labeled, "
                        f"model_ready={stats['is_ready']}",
                        extra={
                            "snapshots_collected": snapshots_collected,
                            "new_markets_tracked": new_tracked,
                            "total_examples": stats['collector']['total_examples'],
                            "labeled_examples": stats['collector']['labeled_examples'],
                            "can_train": stats['can_train'],
                        }
                    )
                    
                    # Run training if due
                    if self.ai_orchestrator.should_train():
                        logger.info("Starting AI training cycle...")
                        self.system_monitor.set_context("training")
                        try:
                            self.ai_orchestrator.run_training()
                        finally:
                            self.system_monitor.set_context("normal")
                    else:
                        self.system_monitor.set_context("normal")  # Done collecting

                except Exception as e:
                    logger.warning(f"AI data collection failed: {e}", exc_info=True)
                    self.system_monitor.set_context("normal")  # Reset context on error
            
            # === SIGNAL GENERATION ===
            # AI/MoE-only signal generation (rules mode has been removed)
            signals: list[TradingSignal] = []
            signal_rejections: dict[str, int] = {}
            
            if self.ai_orchestrator and self.ai_orchestrator.is_ai_ready():
                # AI/MoE signal generation
                signals, signal_rejections = self._generate_ai_signals(
                    tradable_markets, orderbooks, available_for_signals, reserved_for_signals
                )
            else:
                # No model ready - just collect data, no signals
                logger.debug("AI model not ready - collecting data only")
                signal_rejections = {"ai_model_not_ready": len(tradable_markets) * 2}
            
            self._save_signals(signals, cycle_id, db_session)
            logger.info(
                f"Signals computed: {len(signals)} (AI/MoE mode)",
                extra={"signal_rejections": signal_rejections},
            )

            stale_expired = 0
            if self.settings.mode == "live":
                active_fingerprints: set[str] = set()
                for signal in signals:
                    size_usd = signal.sizing_result.size_usd_final if signal.sizing_result else 0.0
                    active_fingerprints.add(
                        intent_manager._fingerprint_intent(
                            intent_type=IntentType.OPEN_POSITION,
                            market_id=signal.market_id,
                            token_id=signal.token_id,
                            side=signal.side,
                            price=signal.p_market,
                            size_usd=size_usd,
                        )
                    )
                stale_expired = intent_manager.expire_stale_open_intents(active_fingerprints)
            
            # Collect ML training data from BROAD MARKET SET (if enabled or data collection phase)
            # This learns from many markets, not just ones we trade
            # Data collection runs even if ML is disabled (for Phase 1: data collection)
            if self.settings.enable_ml or self.settings.ml_data_collection_limit > 0:
                try:
                    # Use markets_for_data (enriched, broader than tradable_markets)
                    # This gives us 5-10x more training data
                    collected = self.strategy.collect_training_data(
                        markets=markets_for_data,  # Enriched markets (30-50 vs 10 tradable)
                        orderbooks=orderbooks,
                        trades=trades,
                        cycle_id=cycle_id,
                    )
                    if collected > 0:
                        logger.info(
                            f"Collected {collected} ML training examples from {len(markets_for_data)} markets "
                            f"(trading on {len(tradable_markets)} markets)",
                            extra={
                                "cycle_id": cycle_id,
                                "examples_collected": collected,
                                "markets_tracked": len(markets_for_data),
                                "markets_traded": len(tradable_markets),
                            }
                        )
                    
                    # Dense price snapshots (every 15 min by default)
                    # Captures higher-resolution price movements for deep learning
                    try:
                        from polyb0t.ml.backfill import HistoricalDataBackfiller
                        backfiller = HistoricalDataBackfiller(self.settings.ml_data_db)
                        snapshots = backfiller.collect_dense_snapshots(
                            markets=markets_for_data,
                            orderbooks=orderbooks,
                        )
                        if snapshots > 0:
                            logger.debug(f"Collected {snapshots} dense price snapshots")
                    except Exception as e:
                        logger.debug(f"Dense snapshot collection skipped: {e}")
                        
                except Exception as e:
                    logger.warning(f"ML data collection failed: {e}")
            if signals:
                sample = []
                for s in signals[:3]:
                    sample.append(
                        {
                            "token_id": s.token_id,
                            "side": s.side,
                            "p_market": s.p_market,
                            "p_model": s.p_model,
                            "edge_net": s.edge_net,
                            "edge_raw": s.edge_raw,
                            "source": s.features.get("p_market_source"),
                            "fill_price": s.fill_estimate.expected_price if s.fill_estimate else None,
                            "size_usd": s.sizing_result.size_usd_final if s.sizing_result else None,
                        }
                    )
                logger.info("Signal sample", extra={"sample": sample})

            # Step 5b: Exit management (CLOSE_POSITION intents) - approval gated.
            exit_created = 0
            exit_skipped = 0
            
            # MASTER SWITCH: Skip all exit management if disabled (BUY ONLY mode)
            if not self.settings.enable_exit_management:
                logger.debug("Exit management disabled (enable_exit_management=false) - BUY ONLY mode")
            elif self.settings.mode == "live":
                try:
                    from polyb0t.data.account_state import AccountStateProvider
                    from polyb0t.execution.portfolio import Position
                    from polyb0t.data.storage import TradeIntentDB

                    async with AccountStateProvider() as provider:
                        # Best-effort: may return empty if endpoints/auth not available.
                        account_state = await provider.fetch_account_state()

                    # Determine which positions to manage for exits.
                    # If manage_all_positions=true, manage ALL positions in the account.
                    # Otherwise, only manage positions that THIS BOT opened (tracked in DB).
                    if self.settings.manage_all_positions:
                        managed_tokens = None  # None means "manage all"
                        logger.info("Exit management: managing ALL positions (manage_all_positions=true)")
                    else:
                        managed_tokens = {
                            r[0]
                            for r in (
                                db_session.query(TradeIntentDB.token_id)
                                .filter(TradeIntentDB.intent_type == "OPEN_POSITION")
                                .filter(TradeIntentDB.status.in_(["EXECUTED", "EXECUTED_DRYRUN"]))
                                .distinct()
                                .all()
                            )
                            if r and r[0]
                        }
                        logger.debug(f"Exit management: tracking {len(managed_tokens)} bot-opened positions")

                    # Convert observed account positions into Position objects (for exit logic only).
                    observed_positions: dict[str, Position] = {}
                    for p in account_state.positions:
                        # If managed_tokens is None, manage all; otherwise filter to bot-opened only
                        if managed_tokens is not None and p.token_id not in managed_tokens:
                            continue
                        mk = str(p.market_id or "unknown")
                        observed_positions[p.token_id] = Position(
                            token_id=p.token_id,
                            market_id=mk,
                            side=p.side if p.side in ("LONG", "SHORT") else "LONG",
                            quantity=float(p.quantity),
                            avg_entry_price=float(p.avg_price),
                        )

                    # Step 5a.5: Diversification cleanup - sell duplicate positions in same market
                    # Group positions by market
                    positions_by_market: dict[str, list] = {}
                    for p in account_state.positions:
                        mk = str(p.market_id or "unknown")
                        if mk not in positions_by_market:
                            positions_by_market[mk] = []
                        positions_by_market[mk].append(p)
                    
                    # For markets with >1 position, keep best (highest value), sell rest
                    diversification_sell_count = 0
                    for market_id, positions in positions_by_market.items():
                        if len(positions) <= self.settings.max_positions_per_market:
                            continue  # Already within limit
                        
                        # Sort by current value (quantity * current_price), highest first
                        positions_sorted = sorted(
                            positions,
                            key=lambda p: float(p.quantity or 0) * float(p.current_price or 0),
                            reverse=True
                        )
                        
                        # Keep the best N positions, sell the rest
                        positions_to_keep = positions_sorted[:self.settings.max_positions_per_market]
                        positions_to_sell = positions_sorted[self.settings.max_positions_per_market:]
                        
                        for p in positions_to_sell:
                            try:
                                # Create exit intent for this position
                                intent = intent_manager.create_exit_intent(
                                    token_id=p.token_id,
                                    market_id=str(p.market_id),
                                    side="SELL",
                                    price=float(p.current_price or 0),
                                    size=float(p.quantity or 0),
                                    reason=f"Diversification cleanup: reducing to {self.settings.max_positions_per_market} position(s) per market",
                                    cycle_id=cycle_id,
                                )
                                if intent:
                                    diversification_sell_count += 1
                                    logger.info(
                                        f"Diversification cleanup: selling position in {market_id}",
                                        extra={"token_id": p.token_id[:20], "quantity": p.quantity, "value": float(p.quantity or 0) * float(p.current_price or 0)}
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to create diversification exit intent: {e}")
                    
                    if diversification_sell_count > 0:
                        logger.info(f"Diversification cleanup: created {diversification_sell_count} sell intents")

                    # Fallback prices from Gamma outcomes (public)
                    fallback_prices: dict[str, float] = {}
                    for m in tradable_markets:
                        for o in m.outcomes:
                            if o.token_id and o.price is not None:
                                fallback_prices[o.token_id] = float(o.price)
                    
                    # CRITICAL: Add position prices from account state (positions may not be in tradable_markets)
                    for p in account_state.positions:
                        if p.token_id and p.current_price and p.current_price > 0:
                            fallback_prices[p.token_id] = float(p.current_price)
                            logger.debug(f"Added position price: {p.token_id[:12]} @ {p.current_price:.3f}")

                    markets_by_id = {m.condition_id: m for m in tradable_markets}
                    exit_mgr = ExitManager()
                    
                    # Log position details for debugging exit logic
                    for tok, pos in observed_positions.items():
                        cur_price = fallback_prices.get(tok)
                        ob = orderbooks.get(tok)
                        if ob and hasattr(ob, 'bids') and ob.bids and ob.asks:
                            cur_price = (ob.bids[0].price + ob.asks[0].price) / 2
                        pnl_pct = 0.0
                        if pos.avg_entry_price > 0 and cur_price:
                            pnl_pct = ((cur_price - pos.avg_entry_price) / pos.avg_entry_price) * 100
                        cur_str = f"${cur_price:.3f}" if cur_price else "N/A"
                        logger.info(
                            f"Position check: {tok[:12]} entry=${pos.avg_entry_price:.3f} "
                            f"current={cur_str} pnl={pnl_pct:+.1f}%"
                        )
                    
                    proposals = exit_mgr.propose_exits(
                        positions=observed_positions,
                        markets=markets_by_id,
                        orderbooks=orderbooks,
                        fallback_prices=fallback_prices,
                    )
                    for prop in proposals:
                        # Get entry price from position for PnL tracking
                        entry_price = None
                        if prop.token_id in observed_positions:
                            entry_price = observed_positions[prop.token_id].avg_entry_price
                        
                        intent = intent_manager.create_exit_intent(
                            token_id=prop.token_id,
                            market_id=prop.market_id,
                            side=prop.side,
                            price=prop.price,
                            size=prop.size,
                            reason=prop.reason,
                            cycle_id=cycle_id,
                            is_emergency=prop.is_emergency,
                            entry_price=entry_price,
                        )
                        if intent is None:
                            exit_skipped += 1
                        else:
                            exit_created += 1
                            if prop.is_emergency:
                                logger.warning(
                                    f"🚨 CRASH EXIT intent created for {prop.token_id[:12]}",
                                    extra={"exit_type": prop.exit_type, "reason": prop.reason}
                                )
                except Exception as e:
                    logger.warning(f"Exit intent proposal skipped due to error: {e}")

            # Step 6: Check drawdown limit
            if self.risk_manager.check_drawdown(self.portfolio.total_equity):
                logger.error("Trading halted due to drawdown limit")
                return

            # Kill switch checks (live safety)
            spreads: dict[str, float] = {}
            # Only evaluate spreads for the *post-filter* tradable universe.
            # This avoids kill-switching on junk books that we already decided not to trade.
            for m in tradable_markets:
                for o in m.outcomes:
                    if not o.token_id:
                        continue
                    ob = orderbooks.get(o.token_id)
                    if not ob or not ob.bids or not ob.asks:
                        continue
                    mid = (ob.bids[0].price + ob.asks[0].price) / 2
                    spreads[o.token_id] = (ob.asks[0].price - ob.bids[0].price) / mid if mid else 0.0
            triggered = kill_switches.check_all_switches(
                current_equity=self.portfolio.total_equity,
                peak_equity=self.risk_manager.peak_equity,
                current_spreads=spreads,
                cycle_id=cycle_id,
            )
            if triggered:
                logger.error(f"Kill switch active; halting cycle. triggered={triggered}")
                return

            # Step 7/8:
            # - Paper mode: simulate fills and place simulated orders
            # - Live mode (dry-run safe): create trade intents only (no execution here)
            if self.settings.mode == "paper":
                simulator = PaperTradingSimulator(self.portfolio, db_session)
                simulator.process_market_data(orderbooks, trades, cycle_id)
                await self._execute_signals(signals, simulator, cycle_id)
            else:
                # Expire old intents first
                expired = intent_manager.expire_old_intents()
                if expired:
                    logger.info(f"Expired intents: {expired}")
                if stale_expired:
                    logger.info(f"Expired stale open intents: {stale_expired}")

                # If auto-trading is enabled, we should not leave old PENDING intents sitting around.
                # Previous versions only auto-approved intents created in the current cycle, which
                # means a restart could strand a backlog of PENDING intents forever (and then dedup
                # prevents new ones). Here we sweep any existing PENDING intents, approve them, and
                # let the executor pick them up.
                if self.settings.mode == "live" and bool(self.settings.auto_approve_intents) and self.settings.placing_orders:
                    try:
                        pending = intent_manager.get_pending_intents()
                        auto_approved_existing = 0
                        for it in pending:
                            try:
                                if intent_manager.approve_intent(it.intent_id, approved_by="auto") is not None:
                                    auto_approved_existing += 1
                            except Exception as e:
                                logger.debug(f"Could not auto-approve intent {it.intent_id}: {e}")
                                continue
                        if auto_approved_existing:
                            logger.info(
                                "Auto-approved existing pending intents",
                                extra={"count": auto_approved_existing, "cycle_id": cycle_id},
                            )
                            # Execute immediately in this cycle as well.
                            if executor is not None and not bool(self.settings.dry_run):
                                execution_summary = executor.process_approved_intents(cycle_id=cycle_id)
                    except Exception as e:
                        logger.warning(f"Error in auto-approve flow: {e}")

                # Create intents from signals (signals are already sized and validated)
                created = 0
                rejected = 0
                skipped_dedup = 0
                created_intent_ids: list[str] = []
                created_intents: list = []  # Full intent objects for market diversification checks
                cycle_allocated_usd = 0.0  # Track cumulative spending within this cycle

                # Daily notional guardrail (live mode).
                # Note: this is a best-effort cap based on recorded intents, not fills.
                # IMPORTANT: Do NOT count EXECUTED_DRYRUN when dry_run=false (no real money was used).
                daily_notional_used = 0.0
                if self.settings.mode == "live" and float(self.settings.max_daily_notional_usd) > 0:
                    try:
                        from datetime import timedelta
                        from polyb0t.data.storage import TradeIntentDB

                        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                        tomorrow = today + timedelta(days=1)
                        statuses = ["EXECUTED"]
                        if bool(self.settings.dry_run):
                            statuses.append("EXECUTED_DRYRUN")
                        # In autonomous mode, also count intents that are pending/approved (expected to execute).
                        if bool(self.settings.auto_approve_intents):
                            statuses.extend(["PENDING", "APPROVED"])
                        rows = (
                            db_session.query(TradeIntentDB)
                            .filter(TradeIntentDB.created_at >= today)
                            .filter(TradeIntentDB.created_at < tomorrow)
                            .filter(TradeIntentDB.intent_type == "OPEN_POSITION")
                            .filter(TradeIntentDB.status.in_(statuses))
                            .all()
                        )
                        for r in rows:
                            v = r.size_usd if getattr(r, "size_usd", None) is not None else r.size
                            if v:
                                daily_notional_used += float(v)
                    except Exception as e:
                        logger.debug(f"Could not calculate daily notional used: {e}")
                        daily_notional_used = 0.0

                # Check if in listen-only mode (global stop-loss triggered)
                is_listen_only = False
                if self.settings.enable_global_stop_loss:
                    try:
                        from polyb0t.models.smart_heuristics import get_smart_heuristics_engine
                        heuristics = get_smart_heuristics_engine()
                        is_listen_only = heuristics.is_listen_only()
                    except Exception as e:
                        logger.debug(f"Could not check listen-only mode: {e}")
                
                if is_listen_only:
                    logger.warning(
                        f"LISTEN-ONLY MODE: Skipping {len(signals)} signals due to global stop-loss",
                        extra={"cycle_id": cycle_id, "signals_count": len(signals)},
                    )
                    # Still log what we would have done
                    for s in signals[:3]:
                        logger.info(
                            f"[LISTEN-ONLY] Would have: {s.side} {s.token_id[:12]} @ {s.p_market:.3f} "
                            f"(edge: {s.edge:+.3f})"
                        )
                else:
                    logger.info(f"Processing {len(signals)} signals. held_long_token_ids={held_long_token_ids}, live_allow_open_sell_intents={self.settings.live_allow_open_sell_intents}")
                
                for signal in sorted(signals, key=lambda s: abs(s.edge), reverse=True):
                    # Skip all new positions in listen-only mode
                    if is_listen_only:
                        rejected += 1
                        continue
                    logger.info(f"Checking signal: token={signal.token_id[:20]}..., side={signal.side}, edge={signal.edge:.4f}")
                    # Safety: SELL can be ambiguous in live trading:
                    # - It can reduce/close an existing LONG position (including manual positions).
                    # - Or it can open/increase a SHORT position (dangerous if unintended).
                    #
                    # Default behavior (live_allow_open_sell_intents=false):
                    # - Allow SELL intents ONLY when the account currently holds the token LONG
                    #   (i.e., this SELL is likely reducing an existing position).
                    # When live_allow_open_sell_intents=true:
                    # - Allow SELL intents even if it would open a SHORT.
                    #
                    # IMPORTANT: Polymarket does not support shorting (selling tokens you don't own).
                    # If live_allow_open_sell_intents=false, skip ALL SELL signals unless we hold the token.
                    if signal.side == "SELL":
                        if not bool(getattr(self.settings, "live_allow_open_sell_intents", False)):
                            if signal.token_id not in held_long_token_ids:
                                rejected += 1
                                logger.info(
                                    "Signal skipped: OPEN_POSITION SELL would open SHORT (live_allow_open_sell_intents=false)",
                                    extra={
                                        "token_id": signal.token_id,
                                        "market_id": signal.market_id,
                                        "edge": signal.edge,
                                    },
                                )
                                continue

                    # Signals already have sizing computed, use it
                    size_usd = signal.sizing_result.size_usd_final if signal.sizing_result else 0.0
                    
                    # Daily notional cap (live)
                    if self.settings.mode == "live":
                        cap = float(self.settings.max_daily_notional_usd)
                        if cap > 0 and (daily_notional_used + float(size_usd)) > cap:
                            rejected += 1
                            logger.info(
                                "Signal rejected: would exceed max_daily_notional_usd",
                                extra={
                                    "daily_used_usd": daily_notional_used,
                                    "size_usd": float(size_usd),
                                    "cap_usd": cap,
                                },
                            )
                            continue
                    
                    # Live mode: do NOT use legacy paper-portfolio risk checks (category exposure, etc.).
                    # Those checks were designed for paper mode bankroll=$10k and will incorrectly reject
                    # real-sized intents. Live mode safety is enforced by:
                    # - PositionSizer caps (5–45% per trade, 90% total cap)
                    # - balance-based reserved exposure
                    # - max_open_orders and kill switches
                    risk_result = None

                    # Exposure guardrail (final check)
                    reserved = float(balance_summary.get("reserved_usdc", 0.0) or 0.0) if balance_summary else 0.0
                    if reserved + size_usd > float(self.settings.max_total_exposure_usd):
                        rejected += 1
                        logger.info(
                            "Signal rejected: would exceed max_total_exposure_usd",
                            extra={"reserved_usdc": reserved, "size_usd": size_usd},
                        )
                        continue

                    # Open orders guardrail
                    oo = int(account_state_summary.get("open_orders_count", 0) or 0) if account_state_summary else 0
                    if oo >= int(self.settings.max_open_orders):
                        rejected += 1
                        logger.info(
                            "Signal rejected: max_open_orders reached",
                            extra={"open_orders_count": oo, "max_open_orders": self.settings.max_open_orders},
                        )
                        continue

                    # STRICT Market diversification: If we already have ANY position in a market,
                    # block all new BUY intents for that market (even if we just sold some)
                    # This prevents the "sell then buy back" loop
                    if signal.side == "BUY" and signal.market_id:
                        market_id_str = str(signal.market_id)
                        if market_id_str in markets_with_existing_positions:
                            rejected += 1
                            logger.info(
                                "Signal rejected: already have position in this market (strict diversification)",
                                extra={
                                    "market_id": signal.market_id,
                                    "token_id": signal.token_id,
                                    "reason": "markets_with_existing_positions block",
                                },
                            )
                            continue
                    
                    # Secondary check: Count existing positions + pending intents (belt-and-suspenders)
                    if signal.side == "BUY" and signal.market_id and account_state:
                        positions_in_market = [
                            p for p in account_state.positions
                            if p.market_id == signal.market_id
                        ]
                        
                        # Also count pending/approved BUY intents for this market from current cycle
                        pending_intents_in_market = [
                            intent for intent in created_intents
                            if intent.market_id == signal.market_id and intent.side == "BUY"
                        ]
                        
                        total_count = len(positions_in_market) + len(pending_intents_in_market)
                        max_positions = int(getattr(self.settings, "max_positions_per_market", 4))
                        
                        # Check if we're already at the limit
                        if total_count >= max_positions:
                            rejected += 1
                            logger.info(
                                "Signal rejected: max_positions_per_market reached",
                                extra={
                                    "market_id": signal.market_id,
                                    "positions_in_market": len(positions_in_market),
                                    "pending_intents_in_market": len(pending_intents_in_market),
                                    "total_count": total_count,
                                    "max_positions_per_market": max_positions,
                                    "token_id": signal.token_id,
                                },
                            )
                            continue

                    # Check remaining available balance for this cycle
                    # Prevents over-spending when multiple orders execute in same cycle
                    initial_available = float(balance_summary.get("available_usdc", 0.0) or 0.0) if balance_summary else 0.0
                    remaining_available = initial_available - cycle_allocated_usd
                    if size_usd > remaining_available:
                        rejected += 1
                        logger.info(
                            "Signal rejected: would exceed remaining available balance for this cycle",
                            extra={
                                "size_usd": size_usd,
                                "remaining_available": remaining_available,
                                "cycle_allocated_usd": cycle_allocated_usd,
                            },
                        )
                        continue

                    # Build enhanced risk checks with fill/sizing info
                    risk_checks = {
                        "approved": True,
                        "max_position_size": getattr(risk_result, "max_position_size", None),
                        "sized_by_kelly": True,
                        "kelly_fraction": signal.sizing_result.kelly_fraction if signal.sizing_result else None,
                        "sizing_reason": signal.sizing_result.sizing_reason if signal.sizing_result else None,
                        "edge_net": signal.edge_net,
                        "edge_raw": signal.edge_raw,
                        "fill_price": signal.fill_estimate.expected_price if signal.fill_estimate else None,
                        "fill_slippage_bps": signal.fill_estimate.slippage_bps if signal.fill_estimate else None,
                    }

                    intent = intent_manager.create_intent_from_signal(
                        signal=signal,
                        size=size_usd,
                        risk_checks=risk_checks,
                        cycle_id=cycle_id,
                    )
                    if intent is None:
                        skipped_dedup += 1
                    else:
                        created += 1
                        created_intent_ids.append(intent.intent_id)
                        created_intents.append(intent)  # Store full intent for market diversification checks
                        cycle_allocated_usd += float(size_usd)  # Track cumulative allocation
                        if self.settings.mode == "live":
                            daily_notional_used += float(size_usd)

                # Autonomous mode: auto-approve and (if not dry-run) auto-execute newly created intents.
                auto_approved = 0
                auto_execution: dict[str, Any] | None = None
                
                # PLACING_ORDERS check: skip all execution if disabled
                if not self.settings.placing_orders:
                    logger.debug(
                        f"Placing orders disabled - skipping execution of {len(created_intent_ids)} intents"
                    )
                elif self.settings.mode == "live" and self.settings.auto_approve_intents:
                    for iid in created_intent_ids:
                        try:
                            if intent_manager.approve_intent(iid, approved_by="auto") is not None:
                                auto_approved += 1
                        except Exception as e:
                            logger.debug(f"Could not auto-approve intent {iid}: {e}")
                            continue

                    if executor is not None and not self.settings.dry_run and auto_approved:
                        auto_execution = executor.process_approved_intents(cycle_id=cycle_id)

                # Comprehensive cycle summary (single structured log)
                markets_scanned_count = len(markets) if 'markets' in locals() else 0
                markets_tradable_count = len(tradable_markets)
                
                # Store for hourly Discord summary
                self._last_markets_scanned = markets_scanned_count
                self._last_markets_tradable = markets_tradable_count
                
                logger.info(
                    "Cycle summary",
                    extra={
                        "cycle_id": cycle_id,
                        "markets_scanned": markets_scanned_count,
                        "markets_filtered": filter_rejections,
                        "markets_tradable": markets_tradable_count,
                        "signals_generated": len(signals),
                        "signal_rejections": signal_rejections,
                        "intents_created": created,
                        "intents_dedup_skipped": skipped_dedup,
                        "intents_risk_rejected": rejected,
                        "intents_auto_approved": auto_approved,
                        "intents_auto_execution": auto_execution,
                        "intents_expired": expired,
                        "intents_stale_expired": stale_expired,
                        "exit_intents_created": exit_created,
                        "exit_intents_dedup_skipped": exit_skipped,
                        "execution": execution_summary or {"processed": 0, "executed": 0, "failed": 0, "dry_run": self.settings.dry_run},
                        "balance": balance_summary or {},
                    },
                )

            # Step 9: Update portfolio prices
            current_prices = self._extract_current_prices(orderbooks)
            self.portfolio.update_market_prices(current_prices)

            # Step 10: Save PnL snapshot (live mode uses real balance, paper mode uses portfolio)
            reporter = Reporter(db_session)
            if self.settings.mode == "live":
                # Live mode should NEVER report paper equity. If balance is unavailable, skip reporting.
                if balance_summary and "total_usdc" in balance_summary:
                    reporter.save_pnl_snapshot_live(
                        cycle_id=cycle_id,
                        total_usdc=balance_summary["total_usdc"],
                        reserved_usdc=balance_summary["reserved_usdc"],
                        available_usdc=balance_summary["available_usdc"],
                    )
                    logger.info(
                        f"Account: balance=${balance_summary['total_usdc']:.2f} USDC, "
                        f"reserved=${balance_summary['reserved_usdc']:.2f}, "
                        f"available=${balance_summary['available_usdc']:.2f}"
                    )
                else:
                    logger.warning(
                        "Live mode: balance unavailable; skipping PnL snapshot to avoid reporting paper equity",
                        extra={"cycle_id": cycle_id, "balance_summary": balance_summary},
                    )
            else:
                reporter.save_pnl_snapshot(self.portfolio, cycle_id)
                logger.info(
                    f"Portfolio: equity=${self.portfolio.total_equity:.2f}, "
                    f"positions={self.portfolio.num_positions}, "
                    f"exposure=${self.portfolio.total_exposure:.2f}"
                )

        finally:
            db_session.close()

    async def _fetch_markets(self, db_session: Session) -> tuple[list[Any], dict[str, Any]]:
        """Fetch markets from Gamma API.

        Args:
            db_session: Database session.

        Returns:
            Tuple of (markets, diagnostics).
        """
        async with GammaClient() as gamma:
            if self.settings.mode == "paper":
                limit = 100
            else:
                # Broad scan in live mode; downstream we cap enrichment/orderbook fetches.
                limit = int(getattr(self.settings, "live_scan_markets_limit", 500))
            markets, diag = await gamma.list_markets_debug(active=True, closed=False, limit=limit)

            # Save to database
            for market in markets:
                db_market = (
                    db_session.query(MarketDB)
                    .filter_by(condition_id=market.condition_id)
                    .first()
                )

                if db_market:
                    # Update existing
                    db_market.volume = market.volume
                    db_market.liquidity = market.liquidity
                    db_market.active = market.active
                    db_market.closed = market.closed
                else:
                    # Create new
                    db_market = MarketDB(
                        condition_id=market.condition_id,
                        question=market.question,
                        description=market.description,
                        end_date=market.end_date,
                        category=market.category,
                        volume=market.volume,
                        liquidity=market.liquidity,
                        active=market.active,
                        closed=market.closed,
                        meta_json=market.metadata,
                    )
                    db_session.add(db_market)

            db_session.commit()

            return markets, diag

    async def _enrich_markets_with_outcomes(
        self, markets: list[Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        """Fetch per-market details from Gamma to populate outcomes/token_ids/prices.

        Gamma list endpoints often omit token/outcome details; we enrich a small live universe.
        """
        semaphore = asyncio.Semaphore(8)
        status_counts: dict[str, int] = {}
        attempted = len(markets)
        success = 0
        async with GammaClient() as gamma:

            async def enrich_one(m: Any) -> tuple[Any, int | None, bool]:
                async with semaphore:
                    detailed, d = await gamma.get_market_debug(m.condition_id)
                    st = d.get("status")
                    ok = detailed is not None
                    return (detailed or m), st, ok

            results = list(await asyncio.gather(*(enrich_one(m) for m in markets)))

        enriched: list[Any] = []
        for m, st, ok in results:
            enriched.append(m)
            status_counts[str(st)] = status_counts.get(str(st), 0) + 1
            if ok:
                success += 1

        diag = {
            "endpoint_template": "/markets/{id}",
            "attempted": attempted,
            "success": success,
            "status_counts": status_counts,
        }
        return enriched, diag

    def _persist_market_outcomes(self, markets: list[Any], db_session: Session) -> None:
        """Upsert outcomes (token_id, outcome, price) into DB for later reuse."""
        for m in markets:
            if not m.outcomes:
                continue
            for o in m.outcomes:
                if not o.token_id:
                    continue
                row = (
                    db_session.query(MarketOutcomeDB)
                    .filter_by(market_condition_id=m.condition_id, token_id=o.token_id)
                    .first()
                )
                if row:
                    row.outcome = o.outcome
                    row.price = o.price
                else:
                    db_session.add(
                        MarketOutcomeDB(
                            market_condition_id=m.condition_id,
                            token_id=o.token_id,
                            outcome=o.outcome,
                            price=o.price,
                        )
                    )
        db_session.commit()

    async def _fetch_market_data(
        self,
        markets: list[Any],
        db_session: Session,
    ) -> tuple[dict[str, OrderBook], dict[str, list[Trade]], dict[str, Any]]:
        """Fetch orderbook and trade data.

        Uses WebSocket data when available, falls back to HTTP polling.

        Args:
            markets: List of markets.
            db_session: Database session.

        Returns:
            Tuple of (orderbooks dict, trades dict, diagnostics).
        """
        orderbooks: dict[str, OrderBook] = {}
        trades: dict[str, list[Trade]] = {}

        token_market_pairs: list[tuple[str, str]] = []
        for market in markets:
            for outcome in market.outcomes:
                if outcome.token_id:
                    token_market_pairs.append((outcome.token_id, market.condition_id))
        
        # === TRY WEBSOCKET FIRST ===
        ws_hits = 0
        ws_misses = []
        
        if self._ws_connected and self.ws_client:
            # Subscribe to any new markets
            all_token_ids = [tid for tid, _ in token_market_pairs]
            await self.ws_client.subscribe(all_token_ids)
            
            # Get data from WebSocket cache
            for token_id, market_id in token_market_pairs:
                ws_book = self.ws_client.get_orderbook(token_id)
                if ws_book and ws_book.best_bid and ws_book.best_ask:
                    # Convert WebSocket orderbook to our format (need OrderBookLevel objects)
                    ob = OrderBook(
                        token_id=token_id,
                        timestamp=ws_book.last_update,
                        bids=[OrderBookLevel(price=level.price, size=level.size) for level in ws_book.bids],
                        asks=[OrderBookLevel(price=level.price, size=level.size) for level in ws_book.asks],
                    )
                    orderbooks[token_id] = ob
                    self._save_orderbook(ob, market_id, db_session)
                    ws_hits += 1
                else:
                    ws_misses.append((token_id, market_id))
            
            # Get recent trades from WebSocket
            for token_id, _ in token_market_pairs:
                ws_trades = self.ws_client.get_recent_trades(asset_id=token_id, limit=50)
                if ws_trades:
                    trades[token_id] = [
                        Trade(
                            token_id=t.asset_id,
                            price=t.price,
                            size=t.size,
                            side=t.side,
                            timestamp=t.timestamp,
                        )
                        for t in ws_trades
                    ]
            
            if ws_hits > 0:
                logger.info(f"WebSocket provided {ws_hits}/{len(token_market_pairs)} orderbooks")
            
            # Only need to HTTP fetch the misses
            token_market_pairs = ws_misses

        markets_no_outcomes = sum(1 for m in markets if not m.outcomes)
        missing_tokens = sum(1 for m in markets for o in m.outcomes if not o.token_id)
        logger.info(
            "CLOB fetch plan",
            extra={
                "universe_markets": len(markets),
                "markets_no_outcomes": markets_no_outcomes,
                "token_ids_resolved": len(token_market_pairs),
                "token_ids_missing": missing_tokens,
                "clob_base_url": self.settings.clob_base_url,
                "orderbook_endpoints": CLOBClient.debug_endpoints()["orderbook"],
                "trades_endpoints": CLOBClient.debug_endpoints()["trades"],
            },
        )

        # Rate-limit safety: throttle live-mode concurrency.
        if self.settings.mode == "live":
            semaphore = asyncio.Semaphore(int(getattr(self.settings, "live_clob_concurrency", 6)))
        else:
            semaphore = asyncio.Semaphore(8)

        async with CLOBClient() as clob:

            ob_attempts = 0
            ob_success = 0
            tr_attempts = 0
            tr_success = 0
            status_counts: dict[str, int] = {}

            async def fetch_one(token_id: str, market_id: str) -> None:
                async with semaphore:
                    # Fetch orderbook
                    nonlocal ob_attempts, ob_success, tr_attempts, tr_success
                    ob_attempts += 1
                    ob, ob_status, ob_ep = await clob.get_orderbook_debug(token_id)
                    status_counts[f"orderbook:{ob_status}"] = status_counts.get(
                        f"orderbook:{ob_status}", 0
                    ) + 1
                    if ob:
                        ob_success += 1
                        orderbooks[token_id] = ob
                        self._save_orderbook(ob, market_id, db_session)

                    # Fetch recent trades (optional; many deployments return 404)
                    fetch_trades = True
                    if self.settings.mode == "live":
                        fetch_trades = bool(getattr(self.settings, "live_fetch_trades", False))
                    if fetch_trades:
                        tr_attempts += 1
                    t, tr_status, tr_ep = await clob.get_trades_debug(token_id, limit=50)
                    status_counts[f"trades:{tr_status}"] = status_counts.get(
                        f"trades:{tr_status}", 0
                    ) + 1
                    if t:
                        tr_success += 1
                        trades[token_id] = t
                        self._save_trades(t, db_session)

            await asyncio.gather(*(fetch_one(tid, mid) for tid, mid in token_market_pairs))

        logger.info(
            "CLOB fetch results",
            extra={
                "orderbook_attempts": ob_attempts,
                "orderbook_success": ob_success,
                "trades_attempts": tr_attempts,
                "trades_success": tr_success,
                "status_counts": status_counts,
            },
        )

        clob_diag = {
            "orderbook_attempts": ob_attempts,
            "orderbook_success": ob_success,
            "trades_attempts": tr_attempts,
            "trades_success": tr_success,
            "status_counts": status_counts,
        }
        return orderbooks, trades, clob_diag

    def _save_orderbook(
        self,
        orderbook: OrderBook,
        market_id: str,
        db_session: Session,
    ) -> None:
        """Save orderbook snapshot to database.

        Args:
            orderbook: OrderBook object.
            market_id: Market condition ID.
            db_session: Database session.
        """
        # Calculate spread and mid price
        spread = None
        mid_price = None

        if orderbook.bids and orderbook.asks:
            best_bid = orderbook.bids[0].price
            best_ask = orderbook.asks[0].price
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2

        db_orderbook = OrderBookSnapshotDB(
            token_id=orderbook.token_id,
            market_id=market_id,
            timestamp=orderbook.timestamp,
            bids=[{"price": b.price, "size": b.size} for b in orderbook.bids],
            asks=[{"price": a.price, "size": a.size} for a in orderbook.asks],
            spread=spread,
            mid_price=mid_price,
        )

        db_session.add(db_orderbook)
        db_session.commit()

    def _save_trades(self, trades: list[Trade], db_session: Session) -> None:
        """Save trades to database.

        Args:
            trades: List of Trade objects.
            db_session: Database session.
        """
        for trade in trades:
            # Check if trade already exists
            if trade.trade_id:
                existing = (
                    db_session.query(TradeDB).filter_by(trade_id=trade.trade_id).first()
                )
                if existing:
                    continue

            db_trade = TradeDB(
                token_id=trade.token_id,
                trade_id=trade.trade_id,
                timestamp=trade.timestamp,
                price=trade.price,
                size=trade.size,
                side=trade.side,
            )
            db_session.add(db_trade)

        db_session.commit()

    def _generate_ai_signals(
        self,
        markets: list,
        orderbooks: dict[str, OrderBook],
        available_usdc: float,
        reserved_usdc: float,
    ) -> tuple[list[TradingSignal], dict[str, int]]:
        """Generate trading signals using AI model only.

        Args:
            markets: List of tradable markets.
            orderbooks: Orderbook data by token_id.
            available_usdc: Available balance.
            reserved_usdc: Reserved balance.

        Returns:
            Tuple of (signals, rejection_counts).
        """
        signals = []
        rejections: dict[str, int] = {}
        
        if not self.ai_orchestrator:
            rejections["ai_orchestrator_not_available"] = 1
            return signals, rejections
            
        from polyb0t.models.position_sizing import PositionSizer
        sizer = PositionSizer()
        
        # === ARBITRAGE SCANNING ===
        # Check for near-resolved markets with guaranteed profit
        arbitrage_scanner = get_arbitrage_scanner()
        arbitrage_scanner.clear_opportunities()
        
        for market in markets:
            for outcome in market.outcomes:
                if not outcome.token_id:
                    continue
                    
                ob = orderbooks.get(outcome.token_id)
                if not ob or not ob.bids or not ob.asks:
                    rejections["no_orderbook"] = rejections.get("no_orderbook", 0) + 1
                    continue
                    
                # Calculate features for AI
                mid = (ob.bids[0].price + ob.asks[0].price) / 2
                spread = (ob.asks[0].price - ob.bids[0].price) / mid if mid > 0 else 0
                bid_depth = sum(l.size for l in ob.bids[:5])
                ask_depth = sum(l.size for l in ob.asks[:5])
                imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
                
                days_to_res = self._days_to_resolution(market.end_date)
                
                features = {
                    "price": mid,
                    "spread": spread,
                    "spread_pct": spread,
                    "volume_24h": float(market.volume or 0),
                    "liquidity": float(market.liquidity or 0),
                    "orderbook_imbalance": imbalance,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "momentum_1h": 0,  # TODO: calculate from price history
                    "momentum_24h": 0,
                    "volatility_24h": 0,
                    "days_to_resolution": days_to_res,
                }
                
                # === CHECK FOR ARBITRAGE FIRST ===
                # Near-resolved markets with NEWS-CONFIRMED profit
                arb_opp = arbitrage_scanner.scan_market(
                    token_id=outcome.token_id,
                    market_id=market.condition_id,
                    market_title=market.question or market.condition_id,
                    price=mid,
                    bid=ob.bids[0].price,
                    ask=ob.asks[0].price,
                    volume_24h=float(market.volume or 0),
                    days_to_resolution=days_to_res,
                    spread_pct=spread,
                    momentum_24h=0,
                    volatility_24h=0,
                    event_end_date=market.end_date,  # For event date checking
                )
                
                if arb_opp:
                    # Create signal for arbitrage opportunity
                    arb_sizing = sizer.compute_size(
                        edge_net=arb_opp.net_profit_pct,
                        confidence=arb_opp.confidence,
                        available_usdc=available_usdc,
                        reserved_usdc=reserved_usdc,
                    )
                    
                    if arb_sizing.size_usd_final >= self.settings.min_order_usd:
                        arb_signal = TradingSignal(
                            token_id=outcome.token_id,
                            market_id=market.condition_id,
                            side="BUY",  # Always buy the near-certain outcome
                            p_market=mid,
                            p_model=arb_opp.expected_value,
                            edge=arb_opp.net_profit_pct,
                            edge_raw=arb_opp.net_profit_pct,
                            edge_net=arb_opp.net_profit_pct,  # Arbitrage edge is already net
                            confidence=arb_opp.confidence,
                            features=features,
                            sizing_result=arb_sizing,
                        )
                        signals.append(arb_signal)
                        logger.info(
                            f"ARBITRAGE: {market.question[:50] if market.question else market.condition_id[:20]} "
                            f"profit={arb_opp.net_profit_pct:.1%} conf={arb_opp.confidence:.0%}"
                        )
                        continue  # Skip regular AI signal for this market
                
                # Get AI signal (with category-based confidence adjustment)
                ai_signal = self.ai_orchestrator.get_ai_signal(
                    token_id=outcome.token_id,
                    market_id=market.condition_id,
                    market_title=market.question or market.condition_id,
                    price=mid,
                    features=features,
                )
                
                if not ai_signal:
                    rejections["ai_no_signal"] = rejections.get("ai_no_signal", 0) + 1
                    continue
                    
                edge = ai_signal["edge"]
                side = ai_signal["side"]
                confidence = ai_signal["confidence"]
                category = ai_signal.get("category", "unknown")
                
                # Check edge threshold
                if abs(edge) < self.settings.edge_threshold:
                    rejections["edge_below_threshold"] = rejections.get("edge_below_threshold", 0) + 1
                    continue
                    
                # Calculate position size
                sizing = sizer.compute_size(
                    edge_net=edge,
                    confidence=confidence,
                    available_usdc=available_usdc,
                    reserved_usdc=reserved_usdc,
                )
                
                if sizing.size_usd_final < self.settings.min_order_usd:
                    rejections["size_too_small"] = rejections.get("size_too_small", 0) + 1
                    continue
                    
                # Create signal
                signal = TradingSignal(
                    token_id=outcome.token_id,
                    market_id=market.condition_id,
                    side=side,
                    p_market=mid,
                    p_model=mid + edge if side == "BUY" else mid - edge,
                    edge=edge,
                    edge_raw=edge,
                    edge_net=edge * 0.98,  # Estimate net edge after fees/slippage
                    confidence=confidence,
                    features=features,
                    sizing_result=sizing,
                )
                
                signals.append(signal)
                logger.info(
                    f"AI signal: {side} {outcome.token_id[:12]} edge={edge:+.3f} "
                    f"confidence={confidence:.0%} size=${sizing.size_usd_final:.2f}"
                )
                
        return signals, rejections

    def _days_to_resolution(self, end_date: datetime | None) -> float:
        """Calculate days until market resolution, handling timezone-aware dates.
        
        Args:
            end_date: Market end date (may be timezone-aware or naive).
            
        Returns:
            Days until resolution, or 30 if unknown.
        """
        if not end_date:
            return 30.0
        
        try:
            now = datetime.utcnow()
            
            # Make both timezone-naive for comparison
            if end_date.tzinfo is not None:
                end_naive = end_date.replace(tzinfo=None)
            else:
                end_naive = end_date
                
            delta = end_naive - now
            return max(0.0, delta.days + delta.seconds / 86400)
        except Exception as e:
            logger.debug(f"Could not calculate days to resolution: {e}")
            return 30.0

    def _save_signals(
        self,
        signals: list[TradingSignal],
        cycle_id: str,
        db_session: Session,
    ) -> None:
        """Save trading signals to database.

        Args:
            signals: List of TradingSignal objects.
            cycle_id: Current cycle ID.
            db_session: Database session.
        """
        for signal in signals:
            db_signal = SignalDB(
                cycle_id=cycle_id,
                token_id=signal.token_id,
                market_id=signal.market_id,
                timestamp=signal.timestamp,
                p_market=signal.p_market,
                p_model=signal.p_model,
                edge=signal.edge,
                features=signal.features,
                signal_type=signal.side,
                confidence=signal.confidence,
            )
            db_session.add(db_signal)

        db_session.commit()

    async def _execute_signals(
        self,
        signals: list[TradingSignal],
        simulator: PaperTradingSimulator,
        cycle_id: str,
    ) -> None:
        """Execute trading signals with risk checks.

        Args:
            signals: List of trading signals.
            simulator: Paper trading simulator.
            cycle_id: Current cycle ID.
        """
        # Sort signals by absolute edge (highest first)
        signals_sorted = sorted(signals, key=lambda s: abs(s.edge), reverse=True)

        for signal in signals_sorted:
            # Get current portfolio state
            current_positions = self.portfolio.get_position_dict()

            # Risk check
            risk_result = self.risk_manager.check_position(
                signal,
                self.portfolio.cash_balance,
                current_positions,
                self.portfolio.total_exposure,
            )

            if not risk_result.approved:
                logger.info(
                    f"Signal rejected by risk check: {signal.token_id[:8]} - "
                    f"{risk_result.reason}"
                )
                continue

            # Place order
            order = simulator.place_order(
                signal,
                risk_result.max_position_size or 0,
                cycle_id,
            )

            logger.info(f"Placed order from signal: {order}")

    def _extract_current_prices(self, orderbooks: dict[str, OrderBook]) -> dict[str, float]:
        """Extract current prices from orderbooks.

        Args:
            orderbooks: Dict of token_id -> OrderBook.

        Returns:
            Dict of token_id -> current price (mid).
        """
        prices = {}

        for token_id, orderbook in orderbooks.items():
            if orderbook.bids and orderbook.asks:
                best_bid = orderbook.bids[0].price
                best_ask = orderbook.asks[0].price
                prices[token_id] = (best_bid + best_ask) / 2

        return prices

    def stop(self) -> None:
        """Stop trading loop."""
        logger.info("Stopping trading loop")
        self.is_running = False
        
        # Disconnect WebSocket
        if self.ws_client and self._ws_connected:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.ws_client.disconnect())
                else:
                    loop.run_until_complete(self.ws_client.disconnect())
            except Exception as e:
                logger.warning(f"WebSocket disconnect error: {e}")
            self._ws_connected = False
    
    async def _send_discord_startup(self) -> None:
        """Send Discord startup notification."""
        if not self.discord_notifier or not self.discord_notifier.enabled:
            return
        
        # In-process flag to prevent duplicate calls
        if getattr(self, '_startup_notification_sent', False):
            return
        
        try:
            # Rate limit: don't send if one was sent in the last hour
            # This prevents spam when the bot restarts frequently
            import os
            startup_marker = "/tmp/polybot_startup_sent"
            if os.path.exists(startup_marker):
                try:
                    mtime = os.path.getmtime(startup_marker)
                    if time.time() - mtime < 3600:  # 1 hour
                        logger.info("Skipping startup notification (sent within last hour)")
                        self._startup_notification_sent = True
                        return
                except Exception as e:
                    logger.debug(f"Could not read startup marker: {e}")
            
            mode = "LIVE" if self.settings.placing_orders else "DATA_COLLECTION"
            
            # Get actual USDC balance
            balance = 0.0
            try:
                from polyb0t.services.balance import BalanceService
                from polyb0t.data.storage import get_session
                db_session = get_session()
                balance_svc = BalanceService(db_session)
                snap = balance_svc.fetch_usdc_balance()
                balance = snap.total_usdc
                balance_svc.close()
                db_session.close()
            except Exception as e:
                logger.debug(f"Could not fetch balance: {e}")
                # Fallback to 0
            
            await self.discord_notifier.send_startup_notification(
                mode=mode,
                balance=balance,
            )
            
            # Mark that we sent a notification
            self._startup_notification_sent = True
            try:
                with open(startup_marker, "w") as f:
                    f.write(str(time.time()))
            except Exception as e:
                logger.debug(f"Could not write startup marker: {e}")
            
            logger.info("Discord startup notification sent")
        except Exception as e:
            logger.warning(f"Failed to send Discord startup notification: {e}")
    
    async def _check_discord_reports(self) -> None:
        """Check if hourly or daily reports should be sent."""
        if not self.discord_notifier or not self.discord_notifier.enabled:
            return
        
        now = datetime.utcnow()
        
        # Hourly summary (every hour)
        if (now - self._last_hourly_report).total_seconds() >= 3600:
            await self._send_hourly_summary()
            self._last_hourly_report = now
        
        # Daily report (every 24 hours, at midnight UTC)
        if (now - self._last_daily_report).total_seconds() >= 86400:
            await self._send_daily_report()
            self._last_daily_report = now
    
    async def _send_hourly_summary(self) -> None:
        """Send hourly Discord summary with training progress and model metrics."""
        if not self.discord_notifier:
            return
        
        try:
            # Get actual USDC balance
            portfolio_value = 0.0
            try:
                from polyb0t.services.balance import BalanceService
                from polyb0t.data.storage import get_session
                db_session = get_session()
                balance_svc = BalanceService(db_session)
                snap = balance_svc.fetch_usdc_balance()
                portfolio_value = snap.total_usdc
                balance_svc.close()
                db_session.close()
            except Exception as e:
                logger.debug(f"Could not fetch balance for hourly: {e}")
            
            # Get AI stats and training info
            ai_status = {}
            active_positions = 0
            training_info = {}
            model_metrics = None
            include_metrics_debrief = False
            
            if self.ai_orchestrator:
                stats = self.ai_orchestrator.get_training_stats()
                
                # Basic AI status
                ai_status = {
                    "active_experts": stats.get("moe", {}).get("active_experts", 0),
                    "training_examples": stats.get("collector", {}).get("total_examples", 0),
                }
                
                # Training timing info
                last_training_str = stats.get("last_training")
                last_training = None
                if last_training_str:
                    try:
                        last_training = datetime.fromisoformat(last_training_str)
                    except (ValueError, TypeError):
                        pass
                
                # Calculate time until next training (2 hour interval)
                training_interval_hours = 2
                if last_training:
                    next_training = last_training + timedelta(hours=training_interval_hours)
                    time_until = next_training - datetime.utcnow()
                    
                    if time_until.total_seconds() > 0:
                        hours = int(time_until.total_seconds() // 3600)
                        minutes = int((time_until.total_seconds() % 3600) // 60)
                        training_info["time_until_training"] = f"{hours}h {minutes}m"
                    else:
                        training_info["time_until_training"] = "Due soon"
                    
                    training_info["last_training"] = last_training.strftime("%Y-%m-%d %H:%M UTC")
                else:
                    training_info["time_until_training"] = "Not yet trained"
                    training_info["last_training"] = "Never"
                
                # Check if we should include model metrics debrief
                # Only show once per training cycle
                if last_training and (
                    self._last_metrics_debrief_training_time is None or
                    last_training > self._last_metrics_debrief_training_time
                ):
                    include_metrics_debrief = True
                    self._last_metrics_debrief_training_time = last_training
                    
                    # Collect model performance metrics for GPT summary
                    moe_stats = stats.get("moe", {})
                    model_metrics = {
                        "total_experts": moe_stats.get("total_experts", 0),
                        "active_experts": moe_stats.get("active_experts", 0),
                        "suspended_experts": moe_stats.get("suspended_experts", 0),
                        "deprecated_experts": moe_stats.get("deprecated_experts", 0),
                        "probation_experts": moe_stats.get("probation_experts", 0),
                        "total_profit_pct": moe_stats.get("total_profit", 0),
                        "total_simulated_trades": moe_stats.get("total_trades", 0),
                        "best_expert": moe_stats.get("best_expert_id", "N/A"),
                        "best_expert_profit": moe_stats.get("best_profit", 0),
                        "training_examples": stats.get("collector", {}).get("total_examples", 0),
                        "labeled_examples": stats.get("collector", {}).get("labeled_examples", 0),
                    }
            
            # Collect performance stats
            performance_stats = {}
            try:
                from polyb0t.data.storage import get_session, TradeIntentDB
                db_session = get_session()
                
                # Get today's executed intents
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                executed_today = (
                    db_session.query(TradeIntentDB)
                    .filter(TradeIntentDB.executed_at >= today_start)
                    .filter(TradeIntentDB.status.in_(["EXECUTED", "EXECUTED_DRYRUN"]))
                    .all()
                )
                trades_today = len(executed_today)
                
                # Calculate win rate from execution results
                wins = sum(1 for t in executed_today if t.execution_result and t.execution_result.get("success"))
                win_rate = wins / trades_today if trades_today > 0 else 0
                
                performance_stats = {
                    "trades_today": trades_today,
                    "win_rate": win_rate,
                    "total_pnl_pct": 0,  # TODO: calculate from closed positions
                    "markets_scanned": getattr(self, '_last_markets_scanned', 0),
                    "markets_tradable": getattr(self, '_last_markets_tradable', 0),
                }
                db_session.close()
            except Exception as e:
                logger.debug(f"Could not get performance stats: {e}")
            
            # Add total experts to ai_status
            if self.ai_orchestrator:
                moe_stats = self.ai_orchestrator.get_training_stats().get("moe", {})
                ai_status["total_experts"] = moe_stats.get("total_experts", 0)
            
            await self.discord_notifier.send_hourly_summary(
                portfolio_value=portfolio_value,
                unrealized_pnl=0.0,
                trades_this_hour=0,
                active_positions=active_positions,
                ai_status=ai_status,
                training_info=training_info,
                model_metrics=model_metrics if include_metrics_debrief else None,
                performance_stats=performance_stats,
            )
            logger.info("Discord hourly summary sent")
        except Exception as e:
            logger.warning(f"Failed to send Discord hourly summary: {e}")
    
    async def _send_daily_report(self) -> None:
        """Send daily Discord performance report."""
        if not self.discord_notifier:
            return
        
        try:
            # Get actual USDC balance
            balance = 0.0
            try:
                from polyb0t.services.balance import BalanceService
                from polyb0t.data.storage import get_session
                db_session = get_session()
                balance_svc = BalanceService(db_session)
                snap = balance_svc.fetch_usdc_balance()
                balance = snap.total_usdc
                balance_svc.close()
                db_session.close()
            except Exception as e:
                logger.debug(f"Could not fetch balance for daily: {e}")
            
            await self.discord_notifier.send_daily_report(
                starting_balance=balance,
                ending_balance=balance,
                total_trades=0,
                win_rate=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                expert_summary={},
            )
            logger.info("Discord daily report sent")
        except Exception as e:
            logger.warning(f"Failed to send Discord daily report: {e}")
