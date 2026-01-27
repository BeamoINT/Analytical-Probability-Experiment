"""Database schema and storage using SQLAlchemy."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from polyb0t.config import get_settings


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class MarketDB(Base):
    """Markets table."""

    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    condition_id = Column(String(255), unique=True, nullable=False, index=True)
    question = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    end_date = Column(DateTime, nullable=True, index=True)
    category = Column(String(255), nullable=True, index=True)
    volume = Column(Float, nullable=True)
    liquidity = Column(Float, nullable=True)
    active = Column(Boolean, default=True, index=True)
    closed = Column(Boolean, default=False, index=True)
    # NOTE: `metadata` is reserved on SQLAlchemy declarative models.
    # Keep DB column name stable ("metadata") while renaming Python attribute.
    meta_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MarketOutcomeDB(Base):
    """Per-market outcome mapping (token ids + prices)."""

    __tablename__ = "market_outcomes"
    __table_args__ = (UniqueConstraint("market_condition_id", "token_id", name="uq_market_token"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    market_condition_id = Column(
        String(255), ForeignKey("markets.condition_id"), nullable=False, index=True
    )
    token_id = Column(String(255), nullable=False, index=True)
    outcome = Column(String(255), nullable=True)
    price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class OrderBookSnapshotDB(Base):
    """Order book snapshots."""

    __tablename__ = "orderbook_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String(255), nullable=False, index=True)
    market_id = Column(String(255), nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    bids = Column(JSON, nullable=False)  # List of {price, size}
    asks = Column(JSON, nullable=False)
    spread = Column(Float, nullable=True)
    mid_price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TradeDB(Base):
    """Recent trades."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String(255), nullable=False, index=True)
    trade_id = Column(String(255), nullable=True, unique=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    side = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalDB(Base):
    """Trading signals and features."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(String(255), nullable=False, index=True)
    token_id = Column(String(255), nullable=False, index=True)
    market_id = Column(String(255), nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    p_market = Column(Float, nullable=True)
    p_model = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)
    features = Column(JSON, nullable=True)
    signal_type = Column(String(50), nullable=True)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SimulatedOrderDB(Base):
    """Simulated orders."""

    __tablename__ = "simulated_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(255), unique=True, nullable=False, index=True)
    cycle_id = Column(String(255), nullable=False, index=True)
    token_id = Column(String(255), nullable=False, index=True)
    market_id = Column(String(255), nullable=True, index=True)
    side = Column(String(10), nullable=False)  # BUY or SELL
    order_type = Column(String(20), default="LIMIT")
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    status = Column(String(20), default="OPEN", index=True)  # OPEN, FILLED, CANCELLED, EXPIRED
    filled_size = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, index=True)


class SimulatedFillDB(Base):
    """Simulated fills."""

    __tablename__ = "simulated_fills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fill_id = Column(String(255), unique=True, nullable=False)
    order_id = Column(String(255), ForeignKey("simulated_orders.order_id"), nullable=False)
    token_id = Column(String(255), nullable=False, index=True)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    filled_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PortfolioPositionDB(Base):
    """Current portfolio positions."""

    __tablename__ = "portfolio_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_id = Column(String(255), unique=True, nullable=False, index=True)
    market_id = Column(String(255), nullable=True, index=True)
    side = Column(String(10), nullable=False)  # LONG or SHORT
    quantity = Column(Float, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PnLSnapshotDB(Base):
    """Portfolio PnL snapshots."""

    __tablename__ = "pnl_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_equity = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    total_exposure = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    num_positions = Column(Integer, default=0)
    drawdown_pct = Column(Float, default=0.0)
    meta_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TradeIntentDB(Base):
    """Trade intents awaiting user approval."""

    __tablename__ = "trade_intents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # NOTE: `intent_id` is already a UUID string used throughout the app/CLI/API.
    # `intent_uuid` is an explicit UUID field for schema clarity and future-proofing.
    intent_uuid = Column(String(36), nullable=True, index=True)
    intent_id = Column(String(255), unique=True, nullable=False, index=True)
    cycle_id = Column(String(255), nullable=False, index=True)
    intent_type = Column(
        String(50), nullable=False, index=True
    )  # OPEN_POSITION, CLOSE_POSITION, CLAIM_SETTLEMENT
    fingerprint = Column(String(255), nullable=True, index=True)
    token_id = Column(String(255), nullable=False, index=True)
    market_id = Column(String(255), nullable=True, index=True)
    side = Column(String(10), nullable=True)  # BUY or SELL
    price = Column(Float, nullable=True)
    # Historically this was treated as USD notional. Keep `size` for backward compatibility,
    # but store explicit `size_usd` going forward.
    size = Column(Float, nullable=True)
    size_usd = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)
    p_market = Column(Float, nullable=True)
    p_model = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)  # Human-readable reason
    risk_checks = Column(JSON, nullable=True)  # Risk check results
    signal_data = Column(JSON, nullable=True)  # Full signal details
    status = Column(
        String(20), default="PENDING", index=True
    )  # PENDING, APPROVED, REJECTED, EXPIRED, EXECUTED, EXECUTED_DRYRUN, FAILED, SUPERSEDED
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(String(255), nullable=True)  # User identifier
    executed_at = Column(DateTime, nullable=True)
    execution_result = Column(JSON, nullable=True)
    submitted_order_id = Column(String(255), nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    superseded_by_intent_id = Column(String(255), nullable=True, index=True)
    superseded_at = Column(DateTime, nullable=True)


class AccountStateDB(Base):
    """Account state snapshots (live mode only)."""

    __tablename__ = "account_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    wallet_address = Column(String(255), nullable=True, index=True)
    cash_balance = Column(Float, nullable=True)
    total_equity = Column(Float, nullable=True)
    open_orders_count = Column(Integer, default=0)
    positions_count = Column(Integer, default=0)
    positions = Column(JSON, nullable=True)  # List of position details
    open_orders = Column(JSON, nullable=True)  # List of open order details
    meta_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BalanceSnapshotDB(Base):
    """On-chain collateral snapshot (live mode)."""

    __tablename__ = "balance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    wallet_address = Column(String(255), nullable=True, index=True)
    token_address = Column(String(255), nullable=True, index=True)
    chain_id = Column(Integer, nullable=True, index=True)
    total_usdc = Column(Float, nullable=True)
    reserved_usdc = Column(Float, nullable=True)
    available_usdc = Column(Float, nullable=True)
    meta_json = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class KillSwitchEventDB(Base):
    """Kill switch activation events."""

    __tablename__ = "kill_switch_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(255), unique=True, nullable=False, index=True)
    cycle_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    switch_type = Column(
        String(50), nullable=False
    )  # DRAWDOWN, API_ERROR_RATE, STALE_DATA, SPREAD_ANOMALY, DAILY_LOSS
    trigger_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    cleared_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ClosedTradeDB(Base):
    """Closed trades with realized P&L for accurate metrics tracking."""

    __tablename__ = "closed_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(255), unique=True, nullable=False, index=True)

    # Link to intents
    open_intent_id = Column(String(255), nullable=False, index=True)
    close_intent_id = Column(String(255), nullable=True, index=True)

    # Trade identification
    token_id = Column(String(255), nullable=False, index=True)
    market_id = Column(String(255), nullable=True, index=True)
    side = Column(String(10), nullable=True)  # BUY or SELL (entry side)

    # Entry details
    entry_price = Column(Float, nullable=False)
    entry_size_usd = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)

    # Exit details
    exit_price = Column(Float, nullable=True)
    exit_size_usd = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True, index=True)
    exit_reason = Column(String(100), nullable=True)  # TAKE_PROFIT, STOP_LOSS, TIME_EXIT, MANUAL

    # Realized P&L
    realized_pnl_usd = Column(Float, nullable=True)
    realized_pnl_pct = Column(Float, nullable=True)
    is_winner = Column(Boolean, nullable=True)

    # Hold time
    hold_time_hours = Column(Float, nullable=True)

    # Status: OPEN (entry recorded, awaiting exit) or CLOSED (exit recorded, P&L calculated)
    status = Column(String(20), default="OPEN", index=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Database engine and session management
_engine = None
_SessionLocal = None


def init_db(db_url: str | None = None) -> None:
    """Initialize database connection and create tables."""
    global _engine, _SessionLocal

    if db_url is None:
        settings = get_settings()
        db_url = settings.db_url

    _engine = create_engine(
        db_url,
        echo=False,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
    )

    Base.metadata.create_all(bind=_engine)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Lightweight SQLite schema upgrade for dev usage (no migrations framework).
    # This is intentionally best-effort and additive-only: it will never drop columns.
    if "sqlite" in (db_url or ""):
        try:
            _ensure_sqlite_trade_intents_columns(_engine)
            _ensure_sqlite_balance_snapshots_table(_engine)
            _ensure_sqlite_closed_trades_table(_engine)
        except Exception:
            # Never prevent startup due to a best-effort schema upgrade.
            pass


def _ensure_sqlite_trade_intents_columns(engine: Any) -> None:
    """Ensure new TradeIntent columns exist on SQLite (additive-only)."""
    # Late import to keep module import graph simple.
    import sqlite3

    # Engine URL for sqlite: `sqlite:///./polybot.db` -> filesystem path
    url = str(engine.url)
    if not url.startswith("sqlite:///"):
        return
    path = url.replace("sqlite:///", "", 1)

    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(trade_intents)")
        existing = {row[1] for row in cur.fetchall()}  # name is index 1

        # Column name -> sqlite type
        desired: dict[str, str] = {
            "intent_uuid": "TEXT",
            "fingerprint": "TEXT",
            "size_usd": "REAL",
            "p_market": "REAL",
            "p_model": "REAL",
            "submitted_order_id": "TEXT",
            "error_message": "TEXT",
            "superseded_by_intent_id": "TEXT",
            "superseded_at": "DATETIME",
        }

        for col, col_type in desired.items():
            if col in existing:
                continue
            cur.execute(f"ALTER TABLE trade_intents ADD COLUMN {col} {col_type}")

        con.commit()
    finally:
        con.close()


def _ensure_sqlite_balance_snapshots_table(engine: Any) -> None:
    """Ensure balance_snapshots table exists on SQLite (best-effort)."""
    import sqlite3

    url = str(engine.url)
    if not url.startswith("sqlite:///"):
        return
    path = url.replace("sqlite:///", "", 1)

    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='balance_snapshots'"
        )
        if cur.fetchone():
            return
        cur.execute(
            """CREATE TABLE balance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id VARCHAR(255) NOT NULL,
                timestamp DATETIME NOT NULL,
                wallet_address VARCHAR(255),
                token_address VARCHAR(255),
                chain_id INTEGER,
                total_usdc REAL,
                reserved_usdc REAL,
                available_usdc REAL,
                metadata JSON,
                created_at DATETIME
            )"""
        )
        cur.execute("CREATE INDEX IF NOT EXISTS ix_balance_snapshots_cycle_id ON balance_snapshots (cycle_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_balance_snapshots_timestamp ON balance_snapshots (timestamp)")
        con.commit()
    finally:
        con.close()


def _ensure_sqlite_closed_trades_table(engine: Any) -> None:
    """Ensure closed_trades table exists on SQLite (best-effort)."""
    import sqlite3

    url = str(engine.url)
    if not url.startswith("sqlite:///"):
        return
    path = url.replace("sqlite:///", "", 1)

    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='closed_trades'"
        )
        if cur.fetchone():
            return
        cur.execute(
            """CREATE TABLE closed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id VARCHAR(255) NOT NULL UNIQUE,
                open_intent_id VARCHAR(255) NOT NULL,
                close_intent_id VARCHAR(255),
                token_id VARCHAR(255) NOT NULL,
                market_id VARCHAR(255),
                side VARCHAR(10),
                entry_price REAL NOT NULL,
                entry_size_usd REAL NOT NULL,
                entry_time DATETIME NOT NULL,
                exit_price REAL,
                exit_size_usd REAL,
                exit_time DATETIME,
                exit_reason VARCHAR(100),
                realized_pnl_usd REAL,
                realized_pnl_pct REAL,
                is_winner BOOLEAN,
                hold_time_hours REAL,
                status VARCHAR(20) DEFAULT 'OPEN',
                created_at DATETIME,
                updated_at DATETIME
            )"""
        )
        cur.execute("CREATE INDEX IF NOT EXISTS ix_closed_trades_trade_id ON closed_trades (trade_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_closed_trades_token_id ON closed_trades (token_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_closed_trades_status ON closed_trades (status)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_closed_trades_exit_time ON closed_trades (exit_time)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_closed_trades_open_intent_id ON closed_trades (open_intent_id)")
        con.commit()
    finally:
        con.close()


def get_session() -> Session:
    """Get a database session."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()

