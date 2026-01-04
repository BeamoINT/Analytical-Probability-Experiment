"""Main trading scheduler and orchestration."""

import asyncio
import logging
import time
import uuid
from typing import Any
from datetime import datetime

from sqlalchemy.orm import Session

from polyb0t.config import get_settings
from polyb0t.data import CLOBClient, GammaClient, init_db
from polyb0t.data.models import OrderBook, Trade
from polyb0t.data.storage import (
    MarketDB,
    MarketOutcomeDB,
    OrderBookSnapshotDB,
    SignalDB,
    TradeDB,
    get_session,
)
from polyb0t.execution import PaperTradingSimulator, Portfolio
from polyb0t.execution.intents import IntentManager, IntentStatus
from polyb0t.execution.live_executor import LiveExecutor
from polyb0t.models import BaselineStrategy, FeatureEngine, RiskManager
from polyb0t.models.exit_manager import ExitManager
from polyb0t.models.kill_switches import KillSwitchManager
from polyb0t.models.filters import MarketFilter
from polyb0t.models.strategy_baseline import TradingSignal
from polyb0t.services.health import get_health_status
from polyb0t.services.reporter import Reporter

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Main trading loop scheduler."""

    def __init__(self) -> None:
        """Initialize trading scheduler."""
        self.settings = get_settings()
        self.is_running = False
        self.health = get_health_status()
        self.consecutive_errors = 0

        # Initialize database
        init_db()

        # Initialize components
        self.portfolio = Portfolio()
        self.risk_manager = RiskManager()
        self.strategy = BaselineStrategy()
        self.market_filter = MarketFilter()

        logger.info("Trading scheduler initialized")

    async def run(self) -> None:
        """Run main trading loop continuously."""
        self.is_running = True
        logger.info("Starting trading loop")

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

            # Live mode: process already-approved intents (human-in-the-loop).
            # This is the ONLY path to any order submission, and is hard-gated by:
            # - explicit approval
            # - settings.dry_run (no order submission when true)
            if self.settings.mode == "live":
                # Hygiene: expire + backfill + collapse duplicates so dedup is DB-backed and stable.
                # This enforces the invariant: <= 1 pending intent per fingerprint.
                intent_manager.expire_old_intents()
                intent_manager.backfill_missing_fingerprints(statuses=[IntentStatus.PENDING, IntentStatus.APPROVED])
                intent_manager.cleanup_duplicate_pending_intents(mode="supersede")

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
                except Exception as e:
                    balance_summary = {"error": str(e)}
                    logger.warning(f"Balance snapshot unavailable: {e}")

                # Account state snapshot (best-effort; may require auth). Used for open orders/positions.
                try:
                    from polyb0t.data.account_state import AccountStateProvider
                    from polyb0t.data.storage import AccountStateDB

                    async with AccountStateProvider() as provider:
                        state = await provider.fetch_account_state()
                    if state:
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

                if self.settings.dry_run:
                    logger.info(
                        "Dry-run live mode: no positions will be opened; intents are recommendations only",
                        extra={"cycle_id": cycle_id, "dry_run": True},
                    )
                executor = LiveExecutor(db_session=db_session, intent_manager=intent_manager)
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
                logger.warning("No tradable markets found")
                return

            # Live mode: limit universe size to keep 10s cadence and avoid rate limits.
            gamma_enrich_diag: dict[str, Any] | None = None
            if self.settings.mode == "live":
                tradable_markets = tradable_markets[:10]
                logger.info(f"Live mode universe capped to {len(tradable_markets)} markets")

                # Enrich markets with outcomes (token_ids + Gamma prices) using public Gamma endpoints.
                tradable_markets, gamma_enrich_diag = await self._enrich_markets_with_outcomes(tradable_markets)
                self._persist_market_outcomes(tradable_markets, db_session)
                zero_outcomes = sum(1 for m in tradable_markets if not m.outcomes)
                if zero_outcomes:
                    logger.warning(
                        "Some markets missing outcomes after Gamma enrichment",
                        extra={"markets_missing_outcomes": zero_outcomes},
                    )

            # Step 3: Fetch orderbooks and trades
            orderbooks, trades, clob_diag = await self._fetch_market_data(
                tradable_markets, db_session
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

            # Step 5: Generate trading signals with balance-aware sizing
            available_for_signals = float(balance_summary.get("available_usdc", 0.0) or 0.0) if balance_summary else 0.0
            reserved_for_signals = float(balance_summary.get("reserved_usdc", 0.0) or 0.0) if balance_summary else 0.0
            
            signals, signal_rejections = self.strategy.generate_signals(
                tradable_markets, orderbooks, trades, available_for_signals, reserved_for_signals
            )
            self._save_signals(signals, cycle_id, db_session)
            logger.info(
                f"Signals computed: {len(signals)}",
                extra={"signal_rejections": signal_rejections},
            )
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
            if self.settings.mode == "live":
                try:
                    from polyb0t.data.account_state import AccountStateProvider
                    from polyb0t.execution.portfolio import Position

                    async with AccountStateProvider() as provider:
                        # Best-effort: may return empty if endpoints/auth not available.
                        account_state = await provider.fetch_account_state()

                    # Convert observed account positions into Position objects (for exit logic only).
                    observed_positions: dict[str, Position] = {}
                    for p in account_state.positions:
                        mk = str(p.market_id or "unknown")
                        observed_positions[p.token_id] = Position(
                            token_id=p.token_id,
                            market_id=mk,
                            side=p.side if p.side in ("LONG", "SHORT") else "LONG",
                            quantity=float(p.quantity),
                            avg_entry_price=float(p.avg_price),
                        )

                    # Fallback prices from Gamma outcomes (public)
                    fallback_prices: dict[str, float] = {}
                    for m in tradable_markets:
                        for o in m.outcomes:
                            if o.token_id and o.price is not None:
                                fallback_prices[o.token_id] = float(o.price)

                    markets_by_id = {m.condition_id: m for m in tradable_markets}
                    exit_mgr = ExitManager()
                    proposals = exit_mgr.propose_exits(
                        positions=observed_positions,
                        markets=markets_by_id,
                        orderbooks=orderbooks,
                        fallback_prices=fallback_prices,
                    )
                    for prop in proposals:
                        intent = intent_manager.create_exit_intent(
                            token_id=prop.token_id,
                            market_id=prop.market_id,
                            side=prop.side,
                            price=prop.price,
                            size=prop.size,
                            reason=prop.reason,
                            cycle_id=cycle_id,
                        )
                        if intent is None:
                            exit_skipped += 1
                        else:
                            exit_created += 1
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

                # Create intents from signals (signals are already sized and validated)
                created = 0
                rejected = 0
                skipped_dedup = 0
                for signal in sorted(signals, key=lambda s: abs(s.edge), reverse=True):
                    # Signals already have sizing computed, use it
                    size_usd = signal.sizing_result.size_usd_final if signal.sizing_result else 0.0
                    
                    # Additional risk manager check (legacy compatibility)
                    risk_result = self.risk_manager.check_position(
                        signal,
                        self.portfolio.cash_balance,
                        self.portfolio.get_position_dict(),
                        self.portfolio.total_exposure,
                    )
                    if not risk_result.approved:
                        rejected += 1
                        logger.info(
                            f"Signal rejected by risk manager: {risk_result.reason}",
                            extra={"token_id": signal.token_id, "edge": signal.edge},
                        )
                        continue

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

                    # Build enhanced risk checks with fill/sizing info
                    risk_checks = {
                        "approved": True,
                        "max_position_size": risk_result.max_position_size,
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

                # Comprehensive cycle summary (single structured log)
                logger.info(
                    "Cycle summary",
                    extra={
                        "cycle_id": cycle_id,
                        "markets_scanned": len(markets) if 'markets' in locals() else 0,
                        "markets_filtered": filter_rejections,
                        "markets_tradable": len(tradable_markets),
                        "signals_generated": len(signals),
                        "signal_rejections": signal_rejections,
                        "intents_created": created,
                        "intents_dedup_skipped": skipped_dedup,
                        "intents_risk_rejected": rejected,
                        "intents_expired": expired,
                        "exit_intents_created": exit_created,
                        "exit_intents_dedup_skipped": exit_skipped,
                        "execution": execution_summary or {"processed": 0, "executed": 0, "failed": 0, "dry_run": self.settings.dry_run},
                        "balance": balance_summary or {},
                    },
                )

            # Step 9: Update portfolio prices
            current_prices = self._extract_current_prices(orderbooks)
            self.portfolio.update_market_prices(current_prices)

            # Step 10: Save PnL snapshot
            reporter = Reporter(db_session)
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
            limit = 100 if self.settings.mode == "paper" else 50
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

        Args:
            markets: List of markets.
            db_session: Database session.

        Returns:
            Tuple of (orderbooks dict, trades dict).
        """
        orderbooks: dict[str, OrderBook] = {}
        trades: dict[str, list[Trade]] = {}

        token_market_pairs: list[tuple[str, str]] = []
        for market in markets:
            for outcome in market.outcomes:
                if outcome.token_id:
                    token_market_pairs.append((outcome.token_id, market.condition_id))

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

                    # Fetch recent trades
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

