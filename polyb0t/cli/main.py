"""CLI interface for PolyB0T."""

import asyncio
import json
import logging
import os
import re
import sys
from typing import Any
from datetime import datetime, timedelta

import click
import uvicorn

from polyb0t.config import get_settings, load_env_or_exit
from polyb0t.execution.portfolio import Portfolio
from polyb0t.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """PolyB0T - Autonomous Polymarket Paper Trading Bot.

    Safe-by-default paper trading with risk management.
    """
    # Allow `polyb0t --help` and `polyb0t setup live` to run on a clean machine
    # even before `.env` exists. All other commands require `.env`.
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        return

    ctx = click.get_current_context(silent=True)
    if ctx and ctx.invoked_subcommand == "setup":
        return

    # Load .env and fail fast if missing/incomplete (no silent defaults)
    load_env_or_exit()


@cli.command()
@click.option(
    "--paper",
    is_flag=True,
    default=False,
    help="Run in paper trading mode (simulated)",
)
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Run in live mode with human-in-the-loop approval",
)
def run(paper: bool, live: bool) -> None:
    """Run the trading bot continuously.

    This starts the main trading loop that will:
    1. Fetch markets and filter tradable universe
    2. Generate trading signals
    3. In paper mode: Execute simulated trades
    4. In live mode: Create trade intents requiring approval
    5. Track portfolio and PnL

    The bot runs indefinitely until interrupted (Ctrl+C).
    """
    setup_logging()

    # Validate flags
    if paper and live:
        click.echo("ERROR: Cannot specify both --paper and --live")
        return

    if not paper and not live:
        click.echo("ERROR: Must specify either --paper or --live mode")
        click.echo("  Use: polyb0t run --paper  (for paper trading)")
        click.echo("  Or:  polyb0t run --live   (for live monitoring with approval)")
        return

    settings = get_settings()

    if live:
        logger.info("Starting PolyB0T in LIVE mode (human-in-the-loop)")
        click.echo("\n" + "=" * 60)
        click.echo("⚠️  LIVE MODE - HUMAN-IN-THE-LOOP TRADING")
        click.echo("=" * 60)
        click.echo("• All trading actions require explicit approval")
        click.echo("• Use 'polyb0t intents list' to see pending actions")
        click.echo("• Use 'polyb0t intents approve <id>' to execute")
        if settings.dry_run:
            click.echo("• DRY-RUN MODE: Intents will be logged but NOT executed")
        else:
            click.echo("• LIVE EXECUTION: Approved intents WILL place real orders")
        click.echo("• Press Ctrl+C to stop")
        click.echo("=" * 60 + "\n")
        click.echo("The bot will NEVER ask for your wallet private key.")

        # Startup self-checks (fail fast)
        from polyb0t.services.startup_checks import (
            startup_banner,
            validate_clock_utc,
            validate_db_connectivity,
        )

        click.echo(startup_banner(settings))
        try:
            validate_clock_utc()
            validate_db_connectivity(settings.db_url)
        except Exception as e:
            click.echo(f"ERROR: Startup checks failed: {e}")
            raise SystemExit(2)

        if not settings.dry_run:
            if not settings.auto_approve_intents:
                confirm = click.confirm(
                    "⚠️  You are about to enable LIVE order execution. "
                    "Approved intents will place REAL orders with REAL funds. Continue?",
                    default=False,
                )
                if not confirm:
                    click.echo("Aborted.")
                    return
            else:
                click.echo("⚠️  Autonomous live mode: REAL orders will be placed automatically")
                click.echo("⚠️  No manual approval required - proceed with caution!")
    else:
        logger.info("Starting PolyB0T in PAPER TRADING mode")
        click.echo("\nPaper trading mode - all trades are simulated\n")

    try:
        from polyb0t.services.scheduler import TradingScheduler

        scheduler = TradingScheduler()
        asyncio.run(scheduler.run())
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        click.echo("\nShutdown complete.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        click.echo(f"ERROR: {e}")


@cli.command()
def universe() -> None:
    """Show current tradable universe.

    Fetches and displays markets that pass all filters:
    - Resolution time within configured window
    - Minimum liquidity threshold
    - Active and not closed
    """
    setup_logging()
    logger.info("Fetching tradable universe")

    async def fetch_universe() -> None:
        from polyb0t.data.gamma_client import GammaClient
        from polyb0t.models.filters import MarketFilter

        async with GammaClient() as gamma:
            markets = await gamma.list_markets(active=True, closed=False, limit=100)
            market_filter = MarketFilter()
            tradable = market_filter.filter_markets(markets)

            click.echo(f"\nTotal markets: {len(markets)}")
            click.echo(f"Tradable markets: {len(tradable)}\n")

            for market in tradable[:20]:  # Show first 20
                if market.end_date:
                    end_date = market.end_date
                    now = datetime.now(end_date.tzinfo) if end_date.tzinfo else datetime.utcnow()
                    days_until = (end_date - now).days
                else:
                    days_until = "?"
                click.echo(f"- {market.question[:80]}")
                click.echo(f"  ID: {market.condition_id}")
                click.echo(f"  Resolves in: ~{days_until} days")
                click.echo(f"  Volume: ${market.volume or 0:.0f}\n")

    try:
        asyncio.run(fetch_universe())
    except Exception as e:
        logger.error(f"Error fetching universe: {e}", exc_info=True)
        click.echo(f"ERROR: {e}")


@cli.command()
@click.option("--days", default=14, help="Number of days to backtest")
def backtest(days: int) -> None:
    """Run backtest on recent data.

    Note: Full backtesting not implemented in MVP.
    This command shows how backtest would be structured.
    """
    setup_logging()
    click.echo(f"Backtesting over {days} days...")
    click.echo("\nWARNING: Full backtesting not implemented in MVP.")
    click.echo("To implement:")
    click.echo("1. Fetch historical orderbook/trade data")
    click.echo("2. Replay data through strategy")
    click.echo("3. Simulate execution with historical fills")
    click.echo("4. Calculate performance metrics")


@cli.command()
@click.option("--today", is_flag=True, help="Show today's report")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def report(today: bool, json_output: bool) -> None:
    """Generate trading report.

    Shows portfolio performance, positions, and recent activity.
    """
    setup_logging()
    from polyb0t.data.storage import get_session, init_db

    init_db()

    try:
        db_session = get_session()
        from polyb0t.services.reporter import Reporter

        reporter = Reporter(db_session)
        settings = get_settings()

        # Create portfolio instance
        portfolio = Portfolio(settings.paper_bankroll)

        # Generate report
        daily_report = reporter.generate_daily_report(portfolio)

        # Fetch live account state (read-only) if configured
        account_state = None
        try:
            if settings.user_address:
                async def fetch_state() -> None:
                    nonlocal account_state
                    from polyb0t.data.account_state import AccountStateProvider

                    async with AccountStateProvider() as provider:
                        account_state = await provider.fetch_account_state()

                asyncio.run(fetch_state())
        except Exception as e:
            logger.warning(f"Could not fetch account state: {e}")

        if account_state:
            daily_report["account_state"] = account_state.to_dict()

        if json_output:
            click.echo(json.dumps(daily_report, indent=2))
        else:
            _print_report(daily_report)

        db_session.close()

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        click.echo(f"ERROR: {e}")


@cli.command()
@click.option("--json-output", is_flag=True, help="Output as JSON")
def status(json_output: bool) -> None:
    """Show status summary (intents + last cycle + optional account monitoring).

    This command is read-only and safe in dry-run.
    """
    setup_logging()
    from polyb0t.data.storage import (
        get_session,
        init_db,
        BalanceSnapshotDB,
        PnLSnapshotDB,
        TradeIntentDB,
        SignalDB,
    )
    from sqlalchemy import func

    init_db()
    settings = get_settings()

    db_session = get_session()
    try:
        # Latest cycle snapshot (if any)
        last_snap = (
            db_session.query(PnLSnapshotDB)
            .order_by(PnLSnapshotDB.timestamp.desc())
            .first()
        )

        last_balance = (
            db_session.query(BalanceSnapshotDB)
            .order_by(BalanceSnapshotDB.timestamp.desc())
            .first()
        )

        # Intent counts
        intent_counts = (
            db_session.query(TradeIntentDB.status, func.count(TradeIntentDB.id))
            .group_by(TradeIntentDB.status)
            .all()
        )
        counts = {status: int(n) for status, n in intent_counts}

        # Last price source (best-effort from last signal features)
        last_signal = db_session.query(SignalDB).order_by(SignalDB.timestamp.desc()).first()
        last_source = None
        if last_signal and isinstance(last_signal.features, dict):
            last_source = last_signal.features.get("p_market_source")

        payload = {
            "mode": settings.mode,
            "dry_run": settings.dry_run,
            "last_cycle_time": last_snap.timestamp.isoformat() if last_snap else None,
            "last_cycle_id": last_snap.cycle_id if last_snap else None,
            "exposure_usd": float(last_snap.total_exposure) if last_snap else 0.0,
            "drawdown_pct": float(last_snap.drawdown_pct) if last_snap else 0.0,
            "pending_intents": counts.get("PENDING", 0),
            "approved_intents": counts.get("APPROVED", 0),
            "executed_intents": counts.get("EXECUTED", 0) + counts.get("EXECUTED_DRYRUN", 0),
            "failed_intents": counts.get("FAILED", 0),
            "expired_intents": counts.get("EXPIRED", 0),
            "last_price_source": last_source,
            "intent_counts": counts,
            "total_usdc": float(last_balance.total_usdc) if last_balance and last_balance.total_usdc is not None else None,
            "reserved_usdc": float(last_balance.reserved_usdc) if last_balance and last_balance.reserved_usdc is not None else None,
            "available_usdc": float(last_balance.available_usdc) if last_balance and last_balance.available_usdc is not None else None,
        }

        if json_output:
            click.echo(json.dumps(payload, indent=2))
            return

        click.echo("\nSTATUS\n" + "=" * 60)
        click.echo(f"Mode:                 {payload['mode']}")
        click.echo(f"Dry-run:              {payload['dry_run']}")
        if settings.mode == "live" and settings.dry_run:
            click.echo("Note:                 Dry-run live mode (recommendations only; no orders submitted)")
        click.echo(f"Last cycle:           {payload['last_cycle_time'] or 'N/A'}")
        click.echo(f"Last cycle id:        {payload['last_cycle_id'] or 'N/A'}")
        click.echo(f"Exposure (USD):       {payload['exposure_usd']:.2f}")
        click.echo(f"Drawdown (%):         {payload['drawdown_pct']:.2f}")
        click.echo(f"Last price source:    {payload['last_price_source'] or 'N/A'}")
        if payload["total_usdc"] is not None:
            click.echo(f"USDC total:           {payload['total_usdc']:.2f}")
            click.echo(f"USDC reserved:        {payload['reserved_usdc']:.2f}")
            click.echo(f"USDC available:       {payload['available_usdc']:.2f}")
        click.echo("")
        click.echo("Intents:")
        click.echo(f"  Pending:            {payload['pending_intents']}")
        click.echo(f"  Approved:           {payload['approved_intents']}")
        click.echo(f"  Executed (incl DR): {payload['executed_intents']}")
        click.echo(f"  Failed:             {payload['failed_intents']}")
        click.echo(f"  Expired:            {payload['expired_intents']}")
        click.echo("")

        # On-chain balance (primary source of truth for cash)
        if settings.polygon_rpc_url:
            try:
                from polyb0t.services.balance import BalanceService

                bal_service = BalanceService(db_session=db_session)
                balance_snap = bal_service.fetch_usdc_balance()
                click.echo("On-Chain Balance (Polygon)")
                click.echo("-" * 60)
                click.echo(f"Wallet:            {settings.user_address}")
                click.echo(f"Cash Balance:      ${balance_snap.available_usdc:.2f} USDC")
                click.echo(f"Total Equity:      ${balance_snap.available_usdc:.2f} USDC")
                click.echo(f"  Total USDC:      ${balance_snap.total_usdc:.2f}")
                click.echo(f"  Reserved:        ${balance_snap.reserved_usdc:.2f}")
                click.echo(f"  Available:       ${balance_snap.available_usdc:.2f}")
                click.echo("")
            except Exception as e:
                click.echo(f"On-chain balance: error fetching ({e})\n")

        # Best-effort account monitoring (read-only; may require authenticated endpoints).
        try:
            if settings.user_address:
                async def fetch_state() -> Any:
                    from polyb0t.data.account_state import AccountStateProvider

                    async with AccountStateProvider() as provider:
                        return await provider.fetch_account_state()

                account_state = asyncio.run(fetch_state())
                if account_state:
                    acct = account_state.to_dict()
                    click.echo("Account State (CLOB API - read-only)")
                    click.echo("-" * 60)
                    click.echo(f"Open Orders:   {acct.get('open_orders_count')}")
                    click.echo(f"Positions:     {acct.get('positions_count')}")
                    if acct.get("positions"):
                        click.echo("Positions:")
                        for pos in acct["positions"][:5]:
                            click.echo(
                                f"  - {pos.get('token_id', '')[:12]}..."
                                f" side={pos.get('side')} qty={pos.get('quantity')} "
                                f"avg={pos.get('avg_price')} cur={pos.get('current_price')}"
                            )
                    click.echo("")
        except Exception as e:
            click.echo(f"Account State: error fetching ({e})\n")
    finally:
        db_session.close()


def _print_report(report: dict) -> None:
    """Print report in human-readable format.

    Args:
        report: Report dictionary.
    """
    click.echo("\n" + "=" * 60)
    click.echo("POLYB0T TRADING REPORT")
    click.echo("=" * 60)

    # Portfolio summary
    portfolio = report.get("portfolio", {})
    click.echo("\nPORTFOLIO:")
    click.echo(f"  Cash Balance:     ${portfolio.get('cash_balance', 0):.2f}")
    click.echo(f"  Total Exposure:   ${portfolio.get('total_exposure', 0):.2f}")
    click.echo(f"  Unrealized PnL:   ${portfolio.get('unrealized_pnl', 0):.2f}")
    click.echo(f"  Realized PnL:     ${portfolio.get('realized_pnl', 0):.2f}")
    click.echo(f"  Total Equity:     ${portfolio.get('total_equity', 0):.2f}")
    click.echo(f"  Return:           {portfolio.get('return_pct', 0):.2f}%")
    click.echo(f"  Positions:        {portfolio.get('num_positions', 0)}")
    click.echo(f"  Total Fees:       ${portfolio.get('total_fees', 0):.2f}")

    # Daily stats
    daily = report.get("daily_stats", {})
    click.echo("\nTODAY'S ACTIVITY:")
    click.echo(f"  Signals Generated:  {daily.get('signals_generated', 0)}")
    click.echo(f"  Orders Placed:      {daily.get('orders_placed', 0)}")
    click.echo(f"  Fills Executed:     {daily.get('fills_executed', 0)}")
    click.echo(f"  Fees Paid:          ${daily.get('total_fees', 0):.2f}")

    # Positions
    positions = report.get("positions", [])
    if positions:
        click.echo(f"\nOPEN POSITIONS ({len(positions)}):")
        for pos in positions[:10]:
            click.echo(f"  Token: {pos['token_id'][:12]}...")
            click.echo(f"    Side: {pos['side']}, Qty: ${pos['quantity']:.2f}")
            click.echo(f"    Entry: {pos['avg_entry_price']:.4f}, Current: {pos['current_price']:.4f}")
            click.echo(f"    PnL: ${pos['unrealized_pnl']:.2f}\n")

    # Top signals
    signals = report.get("top_signals", [])
    if signals:
        click.echo(f"\nTOP SIGNALS ({len(signals)}):")
        for sig in signals[:5]:
            click.echo(f"  {sig['signal_type']} {sig['token_id'][:12]}...")
            click.echo(f"    Edge: {sig['edge']:.4f}, Confidence: {sig['confidence']:.2f}")

    click.echo("\n" + "=" * 60 + "\n")


@cli.command(name="db")
@click.argument("action", type=click.Choice(["init", "reset", "prune-intents"]))
@click.option("--older-than-days", default=7, type=int, help="Delete intents older than N days (for prune-intents)")
@click.option("--yes", "assume_yes", is_flag=True, help="Skip confirmation prompt")
def db_command(action: str, older_than_days: int, assume_yes: bool) -> None:
    """Database management commands.

    Actions:
        init          - Initialize database tables
        reset         - Drop and recreate all tables (WARNING: deletes all data)
        prune-intents - Delete old terminal intents (EXPIRED, EXECUTED, FAILED, REJECTED)
    """
    setup_logging()
    from polyb0t.data.storage import Base, _engine, get_session, init_db, TradeIntentDB
    from datetime import datetime, timedelta

    if action == "init":
        click.echo("Initializing database...")
        try:
            init_db()
            click.echo("✓ Database initialized successfully")
        except Exception as e:
            click.echo(f"ERROR: {e}")

    elif action == "reset":
        if not click.confirm(
            "WARNING: This will delete ALL data. Continue?", abort=True
        ):
            return

        click.echo("Resetting database...")
        try:
            if _engine is None:
                init_db()

            if _engine:
                Base.metadata.drop_all(bind=_engine)
                Base.metadata.create_all(bind=_engine)
                click.echo("✓ Database reset successfully")
            else:
                click.echo("ERROR: Database engine not initialized")

        except Exception as e:
            click.echo(f"ERROR: {e}")

    elif action == "prune-intents":
        init_db()
        db_session = get_session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            
            # Count what will be deleted
            terminal_statuses = ["EXPIRED", "EXECUTED", "EXECUTED_DRYRUN", "FAILED", "REJECTED", "SUPERSEDED"]
            count_query = (
                db_session.query(TradeIntentDB)
                .filter(TradeIntentDB.status.in_(terminal_statuses))
                .filter(TradeIntentDB.created_at < cutoff_date)
            )
            count = count_query.count()
            
            if count == 0:
                click.echo(f"No terminal intents older than {older_than_days} days found.")
                return
            
            if not assume_yes:
                click.echo(f"Found {count} terminal intents older than {older_than_days} days.")
                click.echo(f"Cutoff date: {cutoff_date.isoformat()}")
                click.echo(f"Statuses: {', '.join(terminal_statuses)}")
                if not click.confirm("\nDelete these intents?", default=False):
                    click.echo("Cancelled.")
                    return
            
            # Delete
            deleted = count_query.delete(synchronize_session=False)
            db_session.commit()
            
            click.echo(f"✓ Deleted {deleted} old terminal intents")
            
        except Exception as e:
            db_session.rollback()
            click.echo(f"ERROR: {e}")
        finally:
            db_session.close()


@cli.command()
@click.option("--host", default=None, help="API host")
@click.option("--port", default=None, type=int, help="API port")
def api(host: str | None, port: int | None) -> None:
    """Start the FastAPI server.

    Provides HTTP endpoints for:
    - /health - Health check
    - /status - Current status and positions
    - /report - Trading report
    - /metrics - Key metrics
    """
    setup_logging()
    settings = get_settings()

    api_host = host or settings.api_host
    api_port = port or settings.api_port

    click.echo(f"Starting API server on {api_host}:{api_port}")
    click.echo("Endpoints:")
    click.echo(f"  http://{api_host}:{api_port}/health")
    click.echo(f"  http://{api_host}:{api_port}/status")
    click.echo(f"  http://{api_host}:{api_port}/report")
    click.echo(f"  http://{api_host}:{api_port}/metrics\n")

    uvicorn.run(
        "polyb0t.api.app:app",
        host=api_host,
        port=api_port,
        log_level=settings.log_level.lower(),
    )


@cli.group()
def setup() -> None:
    """Setup helpers."""
    pass


@setup.command(name="live")
def setup_live() -> None:
    """Guide user to configure live (read-only) Polymarket credentials safely."""
    click.echo("\nPolyB0T Live Setup (read-only & approval-based)")
    click.echo("=" * 60)
    click.echo("This will configure Polymarket CLOB API credentials for read-only access.")
    click.echo("The bot will NEVER ask for your wallet private key or seed phrase.")
    click.echo("\nSteps:")
    click.echo("1) Open Polymarket and navigate to the CLOB API key management page.")
    click.echo("2) Generate an API Key, Secret, and Passphrase using official UI (requires wallet signing).")
    click.echo("3) Paste the values here. They will be saved to your local .env (gitignored).")
    click.echo("4) We will perform a single authenticated READ-ONLY check to verify.")
    click.echo("\nNOTE: Keep these credentials secret. Do not share or commit them.")
    click.echo("=" * 60 + "\n")

    # Prompt for credentials (hidden input)
    api_key = click.prompt("API Key", hide_input=True, confirmation_prompt=False)
    api_secret = click.prompt("API Secret", hide_input=True, confirmation_prompt=False)
    api_passphrase = click.prompt("API Passphrase", hide_input=True, confirmation_prompt=False)

    # Basic validation
    def _validate_non_empty(value: str, name: str, min_len: int = 8) -> None:
        if not value or len(value.strip()) < min_len:
            raise click.ClickException(f"{name} appears invalid. Please re-run and paste the full value.")
        if re.search(r"\\s", value):
            raise click.ClickException(f"{name} must not contain spaces.")

    _validate_non_empty(api_key, "API Key", min_len=8)
    _validate_non_empty(api_secret, "API Secret", min_len=12)
    _validate_non_empty(api_passphrase, "API Passphrase", min_len=6)

    # Write to .env safely (no echo of secrets)
    env_path = ".env"
    _upsert_env(env_path, {
        "POLYBOT_CLOB_API_KEY": api_key.strip(),
        "POLYBOT_CLOB_API_SECRET": api_secret.strip(),
        "POLYBOT_CLOB_API_PASSPHRASE": api_passphrase.strip(),
        "POLYBOT_MODE": "live",
        "POLYBOT_DRY_RUN": "true",
        "POLYBOT_LOOP_INTERVAL_SECONDS": "10",
        "POLYBOT_USER_ADDRESS": "0x5cbb1a163f426097578eb4de9e3ecd987fc1c0d4",
    })

    # Now that `.env` exists, load it and configure logging.
    load_env_or_exit()
    setup_logging()

    click.echo("\n✅ Credentials saved to .env (gitignored).")
    click.echo("Performing read-only verification...")

    # Verify via authenticated read-only call
    try:
        settings = get_settings()

        async def verify() -> bool:
            from polyb0t.data.account_state import AccountStateProvider

            async with AccountStateProvider() as provider:
                state = await provider.fetch_account_state()
                # consider success if no error metadata
                return not state.metadata or "error" not in state.metadata

        success = asyncio.run(verify())
        if success:
            click.echo("✅ Polymarket connection verified (read-only)")
        else:
            click.echo("⚠️ Verification did not return success. Check credentials and retry.")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg:
            click.echo("❌ Unauthorized (401/403). Please check API Key/Secret/Passphrase.")
        else:
            click.echo(f"⚠️ Could not verify connection: {e}")


def _upsert_env(env_path: str, updates: dict[str, str]) -> None:
    """Update or insert environment variables in a .env file."""
    lines: list[str] = []
    existing = {}

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v

    # Apply updates
    for k, v in updates.items():
        existing[k] = v

    # Rebuild file preserving comments
    new_lines = []
    seen = set()
    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            k, _ = line.split("=", 1)
            k = k.strip()
            if k in existing:
                new_lines.append(f"{k}={existing[k]}")
                seen.add(k)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Append missing keys
    for k, v in updates.items():
        if k not in seen:
            new_lines.append(f"{k}={v}")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


@cli.group()
def intents() -> None:
    """Manage trade intents (live mode only).

    Trade intents are proposed actions awaiting user approval.
    All trading actions in live mode require explicit approval.
    """
    pass


@cli.group()
def auth() -> None:
    """Authentication utilities (safe, read-only checks)."""
    pass


@auth.command(name="check")
def auth_check() -> None:
    """Validate authenticated CLOB access (read-only).

    This command never submits orders and never prints secrets.
    """
    setup_logging()
    settings = get_settings()

    missing = []
    if not settings.clob_api_key:
        missing.append("POLYBOT_CLOB_API_KEY")
    if not settings.clob_api_secret:
        missing.append("POLYBOT_CLOB_API_SECRET")
    if not settings.clob_passphrase:
        missing.append("POLYBOT_CLOB_API_PASSPHRASE")

    if missing:
        click.echo("Auth check FAILED: missing required CLOB credentials:")
        for m in missing:
            click.echo(f"  - {m}")
        raise SystemExit(2)

    # Best-effort authenticated request (read-only):
    # attempt to fetch account state (open orders / positions), which should require auth on most deployments.
    try:
        async def fetch_state() -> Any:
            from polyb0t.data.account_state import AccountStateProvider

            async with AccountStateProvider() as provider:
                return await provider.fetch_account_state()

        state = asyncio.run(fetch_state())
        if state and state.metadata and state.metadata.get("error"):
            click.echo(f"Auth check FAILED: {state.metadata.get('error')}")
            raise SystemExit(2)

        click.echo("Auth OK (read-only).")
        if state:
            click.echo(f"Open orders: {len(state.open_orders)}, positions: {len(state.positions)}")
    except Exception as e:
        click.echo(f"Auth check FAILED: {e}")
        raise SystemExit(2)


@cli.command()
def doctor() -> None:
    """Smoke test connectivity and wiring (no trading)."""
    setup_logging()
    settings = get_settings()

    results: list[tuple[str, bool, str]] = []

    # 1) Gamma reachable
    try:
        async def gamma_ping() -> None:
            from polyb0t.data.gamma_client import GammaClient

            async with GammaClient() as g:
                await g.list_markets(active=True, closed=False, limit=1)

        asyncio.run(gamma_ping())
        results.append(("Gamma API", True, "ok"))
    except Exception as e:
        results.append(("Gamma API", False, str(e)))

    # 2) CLOB public orderbook reachable (best-effort: use token from Gamma if possible)
    try:
        async def clob_ping() -> None:
            from polyb0t.data.gamma_client import GammaClient
            from polyb0t.data.clob_client import CLOBClient

            token_id = None
            async with GammaClient() as g:
                ms = await g.list_markets(active=True, closed=False, limit=5)
                if ms:
                    # Try to get a detailed market to find token ids
                    d = await g.get_market(ms[0].condition_id)
                    if d and d.outcomes:
                        token_id = d.outcomes[0].token_id
            if not token_id:
                raise RuntimeError("Could not resolve token_id for orderbook ping")
            async with CLOBClient() as c:
                ob, status, _ = await c.get_orderbook_debug(token_id)
                if not ob:
                    raise RuntimeError(f"Orderbook fetch failed (status={status})")

        asyncio.run(clob_ping())
        results.append(("CLOB public orderbook", True, "ok"))
    except Exception as e:
        results.append(("CLOB public orderbook", False, str(e)))

    # 3) RPC USDC balance (only if RPC URL provided)
    if settings.polygon_rpc_url:
        try:
            from polyb0t.data.storage import get_session, init_db
            from polyb0t.services.balance import BalanceService

            init_db()
            s = get_session()
            try:
                bs = BalanceService(db_session=s)
                snap = bs.fetch_usdc_balance()
                results.append(("Polygon RPC USDC balance", True, f"total_usdc={snap.total_usdc:.2f}"))
            finally:
                s.close()
        except Exception as e:
            results.append(("Polygon RPC USDC balance", False, str(e)))
    else:
        results.append(("Polygon RPC USDC balance", False, "POLYBOT_POLYGON_RPC_URL not set"))

    # 4) Auth check (only if creds present)
    if settings.clob_api_key and settings.clob_api_secret and settings.clob_passphrase:
        try:
            # reuse same logic without printing secrets
            async def fetch_state() -> Any:
                from polyb0t.data.account_state import AccountStateProvider

                async with AccountStateProvider() as provider:
                    return await provider.fetch_account_state()

            state = asyncio.run(fetch_state())
            if state and state.metadata and state.metadata.get("error"):
                raise RuntimeError(state.metadata.get("error"))
            results.append(("CLOB auth (read-only)", True, "ok"))
        except Exception as e:
            results.append(("CLOB auth (read-only)", False, str(e)))
    else:
        results.append(("CLOB auth (read-only)", False, "credentials not set"))

    # Print concise summary
    click.echo("\nDOCTOR\n" + "=" * 60)
    ok_all = True
    for name, ok, msg in results:
        ok_all = ok_all and ok
        click.echo(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")
    click.echo("=" * 60)
    raise SystemExit(0 if ok_all else 2)


@intents.command(name="cleanup")
@click.option(
    "--mode",
    type=click.Choice(["supersede", "expire"]),
    default="supersede",
    show_default=True,
    help="How to handle duplicate pending intents per fingerprint.",
)
@click.option("--yes", "assume_yes", is_flag=True, help="Run without confirmation")
def cleanup_intents(mode: str, assume_yes: bool) -> None:
    """Cleanup duplicate pending intents so there is <= 1 PENDING intent per fingerprint."""
    setup_logging()
    from polyb0t.data.storage import get_session, init_db

    init_db()
    db_session = get_session()
    try:
        from polyb0t.execution.intents import IntentManager

        intent_manager = IntentManager(db_session)
        # Sweep expiry + backfill legacy NULL fingerprints first.
        expired = intent_manager.expire_old_intents()
        backfilled = intent_manager.backfill_missing_fingerprints()

        if not assume_yes:
            click.echo(
                f"This will cleanup duplicate PENDING intents (mode={mode}).\n"
                f"- expired: {expired}\n"
                f"- backfilled fingerprints: {backfilled}\n"
            )
            click.confirm("Continue?", default=False, abort=True)

        summary = intent_manager.cleanup_duplicate_pending_intents(mode=mode)
        click.echo(
            "✓ Cleanup complete\n"
            f"  scanned_pending={summary.get('scanned', 0)}\n"
            f"  kept={summary.get('kept', 0)}\n"
            f"  deduped={summary.get('deduped', 0)}\n"
            f"  expired_null_fp={summary.get('expired_null_fp', 0)}\n"
        )
    finally:
        db_session.close()


@cli.group()
def orders() -> None:
    """Order utilities (read-only listing; cancel is approval-gated)."""
    pass


@orders.command(name="list")
def orders_list() -> None:
    """List open orders (best-effort, read-only)."""
    setup_logging()
    settings = get_settings()
    if settings.mode != "live":
        click.echo("orders list is intended for live mode.")
        return

    try:
        async def fetch_state() -> Any:
            from polyb0t.data.account_state import AccountStateProvider

            async with AccountStateProvider() as provider:
                return await provider.fetch_account_state()

        state = asyncio.run(fetch_state())
        if not state:
            click.echo("No account state available.")
            return
        if state.metadata and state.metadata.get("error"):
            click.echo(f"Account state error: {state.metadata.get('error')}")
            return
        if not state.open_orders:
            click.echo("No open orders.")
            return

        click.echo(f"\nOpen orders ({len(state.open_orders)}):")
        for o in state.open_orders[:20]:
            click.echo(
                f"- order_id={o.order_id} token={o.token_id[:12]}... side={o.side} "
                f"price={o.price} size={o.size} filled={o.filled_size}"
            )
        click.echo("")
    except Exception as e:
        click.echo(f"ERROR: {e}")


@orders.command(name="cancel")
@click.argument("order_id")
@click.option("--token-id", required=True, help="Token id for the order (required for intent tracking)")
@click.option("--market-id", required=False, help="Market id (optional)")
@click.option("--yes", "assume_yes", is_flag=True, help="Create cancel intent without interactive confirmation")
def orders_cancel(order_id: str, token_id: str, market_id: str | None, assume_yes: bool) -> None:
    """Request cancellation of an order via an approval-gated CANCEL_ORDER intent."""
    setup_logging()
    from polyb0t.data.storage import get_session, init_db

    init_db()
    if not assume_yes:
        click.confirm(
            f"Create CANCEL_ORDER intent for order_id={order_id} ? (requires later approval)",
            default=False,
            abort=True,
        )

    db_session = get_session()
    try:
        from polyb0t.execution.intents import IntentManager

        im = IntentManager(db_session)
        intent = im.create_cancel_order_intent(
            token_id=token_id,
            market_id=market_id,
            order_id=order_id,
            reason=f"User requested cancel for order_id={order_id}",
            cycle_id="manual",
        )
        click.echo(f"✓ Created CANCEL_ORDER intent {intent.intent_id[:8]}... (use `polyb0t intents approve <id>`)")
    finally:
        db_session.close()


@intents.command(name="list")
@click.option("--all", "show_all", is_flag=True, help="Show all intents (not just pending)")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def list_intents(show_all: bool, json_output: bool) -> None:
    """List pending trade intents awaiting approval."""
    setup_logging()
    from polyb0t.data.storage import get_session, init_db

    init_db()

    try:
        db_session = get_session()
        from polyb0t.execution.intents import IntentManager

        intent_manager = IntentManager(db_session)
        # Always sweep expiry + backfill fingerprints + collapse duplicate pending intents
        # before presenting data to humans.
        intent_manager.expire_old_intents()
        intent_manager.backfill_missing_fingerprints()
        intent_manager.cleanup_duplicate_pending_intents(mode="supersede")

        if show_all:
            # Get from database
            from polyb0t.data.storage import TradeIntentDB

            db_intents = (
                db_session.query(TradeIntentDB)
                .order_by(TradeIntentDB.created_at.desc())
                .limit(50)
                .all()
            )
            intents = [intent_manager._load_intent_from_db(db_intent) for db_intent in db_intents]
        else:
            # Only pending
            intents = intent_manager.get_pending_intents()

        if json_output:
            output = [intent.to_dict() for intent in intents]
            click.echo(json.dumps(output, indent=2))
        else:
            _print_intents(intents)

        db_session.close()

    except Exception as e:
        logger.error(f"Error listing intents: {e}", exc_info=True)
        click.echo(f"ERROR: {e}")


@intents.command(name="approve")
@click.argument("intent_id")
@click.option("--yes", "assume_yes", is_flag=True, help="Approve without interactive confirmation")
def approve_intent(intent_id: str, assume_yes: bool) -> None:
    """Approve a trade intent for execution.

    INTENT_ID: The intent identifier to approve
    """
    setup_logging()
    from polyb0t.data.storage import get_session, init_db

    init_db()

    try:
        db_session = get_session()
        from polyb0t.execution.intents import IntentManager

        intent_manager = IntentManager(db_session)

        # Get intent details before approving
        intent = intent_manager.get_intent(intent_id)
        if not intent:
            click.echo(f"ERROR: Intent {intent_id} not found")
            db_session.close()
            return

        # Show details and confirm
        click.echo("\n" + "=" * 60)
        click.echo("INTENT APPROVAL")
        click.echo("=" * 60)
        click.echo(f"Intent ID:    {intent.intent_id}")
        click.echo(f"Type:         {intent.intent_type.value}")
        click.echo(f"Token:        {intent.token_id}")
        click.echo(f"Market:       {intent.market_id}")
        click.echo(f"Side:         {intent.side}")
        click.echo(f"Price:        {intent.price:.4f}" if intent.price else "Price:        N/A")
        click.echo(f"Size USD:     ${intent.size_usd:.2f}" if intent.size_usd else "Size USD:     N/A")
        click.echo(f"Reason:       {intent.reason}")
        click.echo(f"Expires:      {intent.expires_at.isoformat()}")
        click.echo(f"Time left:    {intent.to_dict()['seconds_until_expiry']:.0f}s")
        click.echo("=" * 60)

        if intent.is_expired():
            click.echo("\n❌ Intent has expired and cannot be approved")
            db_session.close()
            return

        settings = get_settings()
        if settings.mode != "live":
            click.echo("\n⚠️  Warning: Not in live mode. Approval will have no effect.")

        confirm = True if assume_yes else click.confirm("\n✓ Approve this intent?", default=False)

        if not confirm:
            click.echo("Cancelled.")
            db_session.close()
            return

        approved = intent_manager.approve_intent(intent_id, approved_by="cli_user")

        if approved:
            click.echo(f"\n✓ Intent {intent_id[:8]}... approved successfully")
            if settings.mode == "live":
                if settings.dry_run:
                    click.echo("  (DRY-RUN mode: marked EXECUTED_DRYRUN; no order will be submitted)")
                else:
                    click.echo("  (Will be executed in next cycle after approval)")
        else:
            click.echo(f"\n❌ Failed to approve intent {intent_id}")

        db_session.close()

    except Exception as e:
        logger.error(f"Error approving intent: {e}", exc_info=True)
        click.echo(f"ERROR: {e}")


@intents.command(name="reject")
@click.argument("intent_id")
@click.option("--yes", "assume_yes", is_flag=True, help="Reject without interactive confirmation")
def reject_intent(intent_id: str, assume_yes: bool) -> None:
    """Reject a trade intent.

    INTENT_ID: The intent identifier to reject
    """
    setup_logging()
    from polyb0t.data.storage import get_session, init_db

    init_db()

    try:
        db_session = get_session()
        from polyb0t.execution.intents import IntentManager

        intent_manager = IntentManager(db_session)

        if not assume_yes:
            confirm = click.confirm(f"Reject intent {intent_id[:8]}... ?", default=False)
            if not confirm:
                click.echo("Cancelled.")
                db_session.close()
                return
        rejected = intent_manager.reject_intent(intent_id)

        if rejected:
            click.echo(f"✓ Intent {intent_id[:8]}... rejected")
        else:
            click.echo(f"❌ Failed to reject intent {intent_id}")

        db_session.close()

    except Exception as e:
        logger.error(f"Error rejecting intent: {e}", exc_info=True)
        click.echo(f"ERROR: {e}")


@intents.command(name="expire")
def expire_intents() -> None:
    """Manually expire old pending intents."""
    setup_logging()
    from polyb0t.data.storage import get_session, init_db

    init_db()

    try:
        db_session = get_session()
        from polyb0t.execution.intents import IntentManager

        intent_manager = IntentManager(db_session)
        count = intent_manager.expire_old_intents()

        click.echo(f"✓ Expired {count} old intents")

        db_session.close()

    except Exception as e:
        logger.error(f"Error expiring intents: {e}", exc_info=True)
        click.echo(f"ERROR: {e}")


def _print_intents(intents: list) -> None:
    """Print intents in human-readable format.

    Args:
        intents: List of TradeIntent objects.
    """
    if not intents:
        click.echo("\nNo intents found.\n")
        return

    click.echo(f"\n{len(intents)} Intent(s):\n")
    # Compact table view for human-in-the-loop triage
    header = (
        f"{'ID':8}  {'STATUS':14}  {'TYPE':14}  {'SIDE':4}  {'PRICE':7}  {'USD':8}  {'EDGE':7}  {'EXP(s)':6}  MARKET/TOKEN"
    )
    click.echo(header)
    click.echo("-" * len(header))

    now = datetime.utcnow()
    for intent in intents:
        short_id = (intent.intent_id or "")[:8]
        status = (intent.status.value if getattr(intent, "status", None) else "")[:14]
        itype = (intent.intent_type.value if getattr(intent, "intent_type", None) else "")[:14]
        side = (intent.side or "")[:4]
        price = f"{intent.price:.3f}" if intent.price is not None else "-"
        usd = f"{intent.size_usd:.2f}" if getattr(intent, "size_usd", None) is not None else "-"
        edge = f"{intent.edge:+.3f}" if getattr(intent, "edge", None) is not None else "-"
        exp_s = int(max(0, (intent.expires_at - now).total_seconds())) if getattr(intent, "expires_at", None) else 0
        mk = (intent.market_id or "")[:10]
        tok = (intent.token_id or "")[:12]
        click.echo(
            f"{short_id:8}  {status:14}  {itype:14}  {side:4}  {price:>7}  {usd:>8}  {edge:>7}  {exp_s:6d}  {mk}/{tok}"
        )
    click.echo("")


if __name__ == "__main__":
    cli()

