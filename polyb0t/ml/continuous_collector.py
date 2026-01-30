"""Continuous AI Data Collector for training data collection.

This module tracks markets over time and creates training examples
at regular intervals, not just on resolution. It handles:
- Tracking many markets simultaneously
- Creating multiple examples per position over time
- Persisting data across restarts
- Catching up on missed data after shutdown
- Schema versioning for backwards compatibility
- Storage limits with automatic cleanup
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Any, Optional
import threading

logger = logging.getLogger(__name__)

# Schema version - increment when adding new features
SCHEMA_VERSION = 3

# Maximum storage size (120GB)
MAX_STORAGE_BYTES = 120 * 1024 * 1024 * 1024


@dataclass
class MarketSnapshot:
    """A comprehensive snapshot of market state at a point in time.
    
    Version 2: Added many more features for richer training data.
    """
    # === IDENTIFIERS ===
    token_id: str
    market_id: str
    timestamp: datetime
    schema_version: int = SCHEMA_VERSION
    
    # === PRICE DATA ===
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0  # Spread as percentage of price
    mid_price: float = 0.0
    
    # === VOLUME & LIQUIDITY ===
    volume_24h: float = 0.0
    volume_1h: float = 0.0
    volume_6h: float = 0.0
    liquidity: float = 0.0
    liquidity_bid: float = 0.0  # Liquidity on bid side
    liquidity_ask: float = 0.0  # Liquidity on ask side
    
    # === ORDERBOOK FEATURES ===
    orderbook_imbalance: float = 0.0  # -1 to +1
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    bid_depth_5: float = 0.0  # Top 5 levels
    ask_depth_5: float = 0.0
    bid_depth_10: float = 0.0  # Top 10 levels
    ask_depth_10: float = 0.0
    bid_levels: int = 0  # Number of bid levels
    ask_levels: int = 0  # Number of ask levels
    best_bid_size: float = 0.0
    best_ask_size: float = 0.0
    bid_ask_size_ratio: float = 0.0
    
    # === MOMENTUM / PRICE CHANGES ===
    momentum_1h: float = 0.0
    momentum_4h: float = 0.0
    momentum_24h: float = 0.0
    momentum_7d: float = 0.0
    price_change_1h: float = 0.0
    price_change_4h: float = 0.0
    price_change_24h: float = 0.0
    price_high_24h: float = 0.0
    price_low_24h: float = 0.0
    price_range_24h: float = 0.0  # High - Low
    
    # === VOLATILITY ===
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    volatility_7d: float = 0.0
    atr_24h: float = 0.0  # Average True Range
    
    # === TRADE FLOW ===
    trade_count_1h: int = 0
    trade_count_24h: int = 0
    avg_trade_size_1h: float = 0.0
    avg_trade_size_24h: float = 0.0
    buy_volume_1h: float = 0.0
    sell_volume_1h: float = 0.0
    buy_sell_ratio_1h: float = 0.0
    large_trade_count_24h: int = 0  # Trades > $100
    
    # === MARKET METADATA ===
    category: str = ""
    subcategory: str = ""
    market_slug: str = ""
    question_length: int = 0
    description_length: int = 0
    has_icon: bool = False
    
    # === TIMING FEATURES ===
    days_to_resolution: float = 30.0
    hours_to_resolution: float = 720.0
    market_age_days: float = 0.0  # Days since market created
    hour_of_day: int = 0  # 0-23
    day_of_week: int = 0  # 0-6 (Monday=0)
    is_weekend: bool = False
    
    # === MARKET STATE ===
    is_active: bool = True
    is_closed: bool = False
    total_yes_shares: float = 0.0
    total_no_shares: float = 0.0
    open_interest: float = 0.0
    
    # === RELATED MARKETS / CORRELATION ===
    num_related_markets: int = 0
    avg_related_price: float = 0.0
    correlated_market_count: int = 0  # Markets with significant price correlation
    correlated_avg_price: float = 0.0  # Average price of correlated markets
    correlated_momentum: float = 0.0  # Momentum agreement with correlated markets (-1 to +1)
    avg_correlation_strength: float = 0.0  # Average correlation coefficient
    
    # === WHALE TRACKING FEATURES ===
    whale_activity_1h: int = 0  # Number of whale trades in last hour
    whale_net_direction_1h: float = 0.0  # Net whale direction (-1 to +1)
    whale_activity_24h: int = 0  # Number of whale trades in last 24 hours
    whale_net_direction_24h: float = 0.0  # Net whale direction 24h
    largest_trade_24h: float = 0.0  # Largest single trade in USD
    
    # === SOCIAL/ENGAGEMENT (if available) ===
    comment_count: int = 0
    view_count: int = 0
    unique_traders: int = 0
    
    # === DERIVED FEATURES ===
    price_vs_volume_ratio: float = 0.0  # Price relative to volume
    liquidity_per_dollar_volume: float = 0.0
    spread_adjusted_edge: float = 0.0

    # === MICROSTRUCTURE FEATURES (V3) ===
    vpin: float = 0.0  # Volume-sync probability of informed trading
    order_flow_toxicity: float = 0.0  # Kyle's lambda normalized
    trade_impact_10usd: float = 0.0  # Price impact for $10 trade
    trade_impact_100usd: float = 0.0  # Price impact for $100 trade
    amihud_illiquidity: float = 0.0  # Amihud illiquidity ratio

    # === NEWS/SENTIMENT FEATURES (V3) ===
    news_article_count: int = 0  # Number of relevant articles found
    news_recency_hours: float = 999.0  # Hours since most recent article
    news_sentiment_score: float = 0.0  # Aggregate sentiment (-1 to +1)
    news_sentiment_confidence: float = 0.0  # Confidence in sentiment
    keyword_positive_count: int = 0  # Headlines with positive keywords
    keyword_negative_count: int = 0  # Headlines with negative keywords
    headline_confirmation: float = 0.0  # Headline-based outcome signal (-1, 0, +1)
    headline_conf_confidence: float = 0.0  # Confidence in headline confirmation
    intelligent_confirmation: float = 0.0  # GPT-based outcome signal (-1, 0, +1)
    intelligent_conf_confidence: float = 0.0  # Confidence in intelligent confirmation

    # === INSIDER TRACKING FEATURES (V3) ===
    smart_wallet_buy_count_1h: int = 0  # Smart wallet buys in last hour
    smart_wallet_sell_count_1h: int = 0  # Smart wallet sells in last hour
    smart_wallet_net_direction_1h: float = 0.0  # Net direction (-1 to +1)
    smart_wallet_volume_1h: float = 0.0  # Total smart wallet volume (USD)
    avg_buyer_reputation: float = 0.5  # Average reputation of buyers
    avg_seller_reputation: float = 0.5  # Average reputation of sellers
    smart_wallet_buy_count_24h: int = 0  # Smart wallet buys in last 24h
    smart_wallet_sell_count_24h: int = 0  # Smart wallet sells in last 24h
    smart_wallet_net_direction_24h: float = 0.0  # Net direction 24h
    unusual_activity_score: float = 0.0  # Unusual activity detection (0 to 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def get_feature_columns(cls) -> list[str]:
        """Get list of numeric feature columns for training."""
        return [
            # Core price features
            "price", "spread", "spread_pct", "mid_price",
            # Volume & liquidity
            "volume_24h", "volume_1h", "volume_6h", "liquidity",
            "liquidity_bid", "liquidity_ask",
            # Orderbook
            "orderbook_imbalance", "bid_depth", "ask_depth",
            "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10",
            "bid_levels", "ask_levels", "best_bid_size", "best_ask_size",
            "bid_ask_size_ratio",
            # Momentum (backward-looking only - how price changed in the PAST)
            "momentum_1h", "momentum_4h", "momentum_24h", "momentum_7d",
            # NOTE: price_change_1h/4h/24h are LABELS, not features - excluded to prevent data leakage
            "price_high_24h", "price_low_24h", "price_range_24h",
            # Volatility
            "volatility_1h", "volatility_24h", "volatility_7d", "atr_24h",
            # Trade flow
            "trade_count_1h", "trade_count_24h",
            "avg_trade_size_1h", "avg_trade_size_24h",
            "buy_volume_1h", "sell_volume_1h", "buy_sell_ratio_1h",
            "large_trade_count_24h",
            # Timing
            "days_to_resolution", "hours_to_resolution", "market_age_days",
            "hour_of_day", "day_of_week",
            # Market state
            "total_yes_shares", "total_no_shares", "open_interest",
            # Related/social
            "num_related_markets", "avg_related_price",
            "comment_count", "view_count", "unique_traders",
            # Derived
            "price_vs_volume_ratio", "liquidity_per_dollar_volume",
            "spread_adjusted_edge",
            # Microstructure (V3)
            "vpin", "order_flow_toxicity", "trade_impact_10usd",
            "trade_impact_100usd", "amihud_illiquidity",
            # News/Sentiment (V3)
            "news_article_count", "news_recency_hours", "news_sentiment_score",
            "news_sentiment_confidence", "keyword_positive_count",
            "keyword_negative_count", "headline_confirmation",
            "headline_conf_confidence", "intelligent_confirmation",
            "intelligent_conf_confidence",
            # Insider Tracking (V3)
            "smart_wallet_buy_count_1h", "smart_wallet_sell_count_1h",
            "smart_wallet_net_direction_1h", "smart_wallet_volume_1h",
            "avg_buyer_reputation", "avg_seller_reputation",
            "smart_wallet_buy_count_24h", "smart_wallet_sell_count_24h",
            "smart_wallet_net_direction_24h", "unusual_activity_score",
        ]


@dataclass 
class TrainingExample:
    """A labeled training example for AI."""
    example_id: str
    token_id: str
    market_id: str
    created_at: datetime
    schema_version: int = SCHEMA_VERSION
    
    # Features at time of snapshot
    features: dict = field(default_factory=dict)
    
    # Label: what happened next (multiple timeframes)
    price_change_15m: Optional[float] = None
    price_change_1h: Optional[float] = None
    price_change_4h: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_7d: Optional[float] = None
    price_change_to_resolution: Optional[float] = None
    
    # Direction labels (for classification)
    direction_1h: Optional[int] = None  # 1=up, 0=flat, -1=down
    direction_24h: Optional[int] = None
    
    # Final resolution
    resolved_outcome: Optional[int] = None  # 1 = Yes won, 0 = No won
    
    # Metadata
    labeled_at: Optional[datetime] = None
    is_fully_labeled: bool = False
    available_features: list = field(default_factory=list)  # Which features were available


class ContinuousDataCollector:
    """Collects training data continuously from Polymarket."""

    def __init__(self, db_path: str = "data/ai_training.db", max_storage_bytes: int = MAX_STORAGE_BYTES):
        """Initialize the collector.

        Args:
            db_path: Path to SQLite database for persistence.
            max_storage_bytes: Maximum storage size in bytes (default 140GB).
        """
        self.db_path = db_path
        self.max_storage_bytes = max_storage_bytes
        self._ensure_db()
        self._lock = threading.Lock()
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None

        # Track when we last collected data
        self._last_collection_time: Optional[datetime] = None
        self._load_state()

        # Check storage on startup
        self._check_storage()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with WAL mode and busy timeout.

        Returns:
            SQLite connection with proper settings for concurrent access.
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn
        
    def _ensure_db(self) -> None:
        """Create database tables if they don't exist, and migrate old schemas."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Market snapshots table (expanded schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                schema_version INTEGER DEFAULT 1,
                data JSON NOT NULL,
                UNIQUE(token_id, timestamp)
            )
        """)
        
        # === MIGRATE OLD SCHEMA ===
        # Add schema_version column if it doesn't exist (for old databases)
        try:
            cursor.execute("SELECT schema_version FROM market_snapshots LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Migrating database: adding schema_version column to market_snapshots")
            cursor.execute("ALTER TABLE market_snapshots ADD COLUMN schema_version INTEGER DEFAULT 1")
            cursor.execute("ALTER TABLE market_snapshots ADD COLUMN data JSON")
            conn.commit()
        
        # Training examples table (expanded schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_examples (
                example_id TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                schema_version INTEGER DEFAULT 1,
                features TEXT NOT NULL,
                available_features TEXT,
                price_change_15m REAL,
                price_change_1h REAL,
                price_change_4h REAL,
                price_change_24h REAL,
                price_change_7d REAL,
                price_change_to_resolution REAL,
                direction_1h INTEGER,
                direction_24h INTEGER,
                resolved_outcome INTEGER,
                labeled_at TEXT,
                is_fully_labeled INTEGER DEFAULT 0,
                -- Prediction simulation columns
                predicted_change REAL,
                category TEXT,
                market_title TEXT,
                prediction_evaluated INTEGER DEFAULT 0
            )
        """)
        
        # Migrate training_examples table if needed
        try:
            cursor.execute("SELECT schema_version FROM training_examples LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Migrating database: adding new columns to training_examples")
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN schema_version INTEGER DEFAULT 1")
            except sqlite3.OperationalError:
                pass  # Column might already exist
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN available_features TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN price_change_15m REAL")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN price_change_7d REAL")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN direction_1h INTEGER")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN direction_24h INTEGER")
            except sqlite3.OperationalError:
                pass
            conn.commit()
        
        # Migrate for prediction simulation columns
        try:
            cursor.execute("SELECT predicted_change FROM training_examples LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Migrating database: adding prediction simulation columns")
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN predicted_change REAL")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN category TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN market_title TEXT")
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("ALTER TABLE training_examples ADD COLUMN prediction_evaluated INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            conn.commit()
        
        # Tracked markets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracked_markets (
                token_id TEXT PRIMARY KEY,
                market_id TEXT NOT NULL,
                started_tracking TEXT NOT NULL,
                last_snapshot TEXT,
                is_resolved INTEGER DEFAULT 0,
                resolution_outcome INTEGER,
                market_metadata TEXT
            )
        """)
        
        # Migrate tracked_markets table if needed
        try:
            cursor.execute("SELECT market_metadata FROM tracked_markets LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("Migrating database: adding market_metadata column to tracked_markets")
            try:
                cursor.execute("ALTER TABLE tracked_markets ADD COLUMN market_metadata TEXT")
            except sqlite3.OperationalError:
                pass
            conn.commit()
        
        # Collector state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collector_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Price history table (for computing momentum/volatility)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL DEFAULT 0,
                UNIQUE(token_id, timestamp)
            )
        """)
        
        # Storage stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS storage_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_snapshots INTEGER DEFAULT 0,
                total_examples INTEGER DEFAULT 0,
                last_cleanup TEXT,
                db_size_bytes INTEGER DEFAULT 0
            )
        """)
        
        # Initialize storage stats if not exists
        cursor.execute("""
            INSERT OR IGNORE INTO storage_stats (id, total_snapshots, total_examples)
            VALUES (1, 0, 0)
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_token ON market_snapshots(token_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_version ON market_snapshots(schema_version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_examples_labeled ON training_examples(is_fully_labeled)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_examples_version ON training_examples(schema_version)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_token ON price_history(token_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_history_time ON price_history(timestamp)")
        
        conn.commit()
        conn.close()
        
    def _check_storage(self) -> None:
        """Check storage usage and cleanup if needed."""
        try:
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            if db_size > self.max_storage_bytes:
                logger.warning(
                    f"Storage limit exceeded: {db_size / 1e9:.1f}GB > {self.max_storage_bytes / 1e9:.1f}GB. "
                    "Running cleanup..."
                )
                self._cleanup_old_data()
            else:
                logger.info(f"Storage usage: {db_size / 1e9:.2f}GB / {self.max_storage_bytes / 1e9:.0f}GB")
                
        except Exception as e:
            logger.warning(f"Storage check failed: {e}")
            
    def _cleanup_old_data(self) -> None:
        """Remove oldest data to stay under storage limit."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Delete oldest 10% of snapshots
            cursor.execute("""
                DELETE FROM market_snapshots 
                WHERE id IN (
                    SELECT id FROM market_snapshots 
                    ORDER BY timestamp ASC 
                    LIMIT (SELECT COUNT(*) / 10 FROM market_snapshots)
                )
            """)
            deleted_snapshots = cursor.rowcount
            
            # Delete oldest 10% of price history
            cursor.execute("""
                DELETE FROM price_history 
                WHERE id IN (
                    SELECT id FROM price_history 
                    ORDER BY timestamp ASC 
                    LIMIT (SELECT COUNT(*) / 10 FROM price_history)
                )
            """)
            deleted_prices = cursor.rowcount
            
            # Update last cleanup time
            cursor.execute("""
                UPDATE storage_stats SET last_cleanup = ? WHERE id = 1
            """, (datetime.utcnow().isoformat(),))
            
            conn.commit()
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
            
            logger.info(f"Cleanup complete: deleted {deleted_snapshots} snapshots, {deleted_prices} price points")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        finally:
            conn.close()
        
    def _load_state(self) -> None:
        """Load collector state from database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM collector_state WHERE key = 'last_collection_time'")
        row = cursor.fetchone()
        if row:
            self._last_collection_time = datetime.fromisoformat(row[0])
        
        conn.close()
        
    def _save_state(self) -> None:
        """Save collector state to database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self._last_collection_time:
            cursor.execute(
                "INSERT OR REPLACE INTO collector_state (key, value) VALUES (?, ?)",
                ("last_collection_time", self._last_collection_time.isoformat())
            )
        
        conn.commit()
        conn.close()
        
    def get_tracked_market_count(self) -> int:
        """Get number of markets being tracked."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracked_markets WHERE is_resolved = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_total_examples(self) -> int:
        """Get total number of training examples."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_examples")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_labeled_examples(self) -> int:
        """Get number of examples with 24h labels (usable for training)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        # Count examples with 24h price change labels (main training target)
        cursor.execute("SELECT COUNT(*) FROM training_examples WHERE price_change_24h IS NOT NULL")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_resolved_examples(self) -> int:
        """Get number of examples from resolved markets (final ground truth)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_examples WHERE is_fully_labeled = 1")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_unlabeled_examples(self) -> int:
        """Get number of examples waiting for labels."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_examples WHERE is_fully_labeled = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        try:
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM market_snapshots")
            snapshot_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM price_history")
            price_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM training_examples")
            example_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "db_size_gb": db_size / 1e9,
                "max_size_gb": self.max_storage_bytes / 1e9,
                "usage_pct": (db_size / self.max_storage_bytes) * 100,
                "snapshots": snapshot_count,
                "price_points": price_count,
                "examples": example_count,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def add_market_to_track(self, token_id: str, market_id: str, metadata: dict = None) -> bool:
        """Add a market to track for training data.
        
        Args:
            token_id: Token ID to track.
            market_id: Market condition ID.
            metadata: Optional market metadata to store.
            
        Returns:
            True if added, False if already tracking.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO tracked_markets (token_id, market_id, started_tracking, market_metadata) VALUES (?, ?, ?, ?)",
                (token_id, market_id, datetime.utcnow().isoformat(), json.dumps(metadata) if metadata else None)
            )
            conn.commit()
            logger.debug(f"Started tracking market {token_id[:12]} for AI training")
            return True
        except sqlite3.IntegrityError:
            return False  # Already tracking
        finally:
            conn.close()
            
    def record_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Record a market snapshot.
        
        Args:
            snapshot: Market snapshot to record.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Store full snapshot as JSON
            cursor.execute("""
                INSERT OR IGNORE INTO market_snapshots 
                (token_id, market_id, timestamp, schema_version, data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                snapshot.token_id, snapshot.market_id, snapshot.timestamp.isoformat(),
                snapshot.schema_version, json.dumps(snapshot.to_dict())
            ))
            
            # Also store in price history for momentum calculations
            cursor.execute("""
                INSERT OR IGNORE INTO price_history (token_id, timestamp, price, volume)
                VALUES (?, ?, ?, ?)
            """, (
                snapshot.token_id, snapshot.timestamp.isoformat(),
                snapshot.price, snapshot.volume_24h
            ))
            
            # Update last snapshot time
            cursor.execute(
                "UPDATE tracked_markets SET last_snapshot = ? WHERE token_id = ?",
                (snapshot.timestamp.isoformat(), snapshot.token_id)
            )
            
            conn.commit()
        finally:
            conn.close()
            
    def create_training_example(
        self,
        snapshot: MarketSnapshot,
        predicted_change: Optional[float] = None,
        category: Optional[str] = None,
        market_title: Optional[str] = None,
    ) -> str:
        """Create a training example from a snapshot.

        Args:
            snapshot: Market snapshot to create example from.
            predicted_change: What the model predicted (for simulation).
            category: Market category (for category learning).
            market_title: Market title (for categorization).

        Returns:
            Example ID.
        """
        import uuid

        example_id = str(uuid.uuid4())
        features = snapshot.to_dict()

        # Track which features are available (for backwards compat)
        available_features = [k for k, v in features.items() if v is not None and v != 0 and v != ""]

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO training_examples
            (example_id, token_id, market_id, created_at, schema_version, features, available_features, 
             is_fully_labeled, predicted_change, category, market_title, prediction_evaluated)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 0)
        """, (
            example_id, snapshot.token_id, snapshot.market_id,
            datetime.utcnow().isoformat(), snapshot.schema_version,
            json.dumps(features), json.dumps(available_features),
            predicted_change, category, market_title
        ))

        conn.commit()
        conn.close()

        return example_id
    
    def get_price_history(self, token_id: str, hours: int = 24) -> list[tuple[datetime, float]]:
        """Get price history for a token.
        
        Args:
            token_id: Token ID.
            hours: Hours of history to fetch.
            
        Returns:
            List of (timestamp, price) tuples.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT timestamp, price FROM price_history
            WHERE token_id = ? AND timestamp > ?
            ORDER BY timestamp ASC
        """, (token_id, cutoff))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [(datetime.fromisoformat(ts), price) for ts, price in rows]
    
    def compute_momentum(self, token_id: str, hours: int) -> float:
        """Compute price momentum over given hours.
        
        Args:
            token_id: Token ID.
            hours: Hours to look back.
            
        Returns:
            Momentum (price change ratio).
        """
        history = self.get_price_history(token_id, hours)
        if len(history) < 2:
            return 0.0
        
        first_price = history[0][1]
        last_price = history[-1][1]
        
        if first_price <= 0:
            return 0.0
            
        return (last_price - first_price) / first_price
    
    def compute_volatility(self, token_id: str, hours: int) -> float:
        """Compute price volatility over given hours.
        
        Args:
            token_id: Token ID.
            hours: Hours to look back.
            
        Returns:
            Volatility (standard deviation of returns).
        """
        history = self.get_price_history(token_id, hours)
        if len(history) < 3:
            return 0.0
        
        prices = [p for _, p in history]
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if len(returns) < 2:
            return 0.0
        
        import statistics
        try:
            return statistics.stdev(returns)
        except:
            return 0.0
    
    def label_examples(self) -> int:
        """Label unlabeled examples with future price data.
        
        Returns:
            Number of examples labeled.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get unlabeled examples
        cursor.execute("""
            SELECT example_id, token_id, created_at, features
            FROM training_examples 
            WHERE is_fully_labeled = 0
        """)
        
        examples = cursor.fetchall()
        labeled_count = 0
        partial_labeled = 0
        now = datetime.utcnow()
        
        if not examples:
            return 0
        
        for example_id, token_id, created_at_str, features_json in examples:
            created_at = datetime.fromisoformat(created_at_str)
            features = json.loads(features_json)
            initial_price = features.get("price", 0)
            
            if initial_price <= 0:
                continue
                
            # Get price history after this example was created
            # Use price_history table (always populated) instead of market_snapshots
            cursor.execute("""
                SELECT timestamp, price 
                FROM price_history
                WHERE token_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (token_id, created_at_str))
            
            future_snapshots = cursor.fetchall()
            
            price_change_15m = None
            price_change_1h = None
            price_change_4h = None
            price_change_24h = None
            price_change_7d = None
            
            for snap_time_str, snap_price in future_snapshots:
                if snap_price is None:
                    continue
                snap_time = datetime.fromisoformat(snap_time_str)
                hours_later = (snap_time - created_at).total_seconds() / 3600
                
                if snap_price and snap_price > 0:
                    price_change = (snap_price - initial_price) / initial_price
                    
                    if hours_later >= 0.25 and price_change_15m is None:
                        price_change_15m = price_change
                    if hours_later >= 1 and price_change_1h is None:
                        price_change_1h = price_change
                    if hours_later >= 4 and price_change_4h is None:
                        price_change_4h = price_change
                    if hours_later >= 24 and price_change_24h is None:
                        price_change_24h = price_change
                    if hours_later >= 168 and price_change_7d is None:  # 7 days
                        price_change_7d = price_change
                        
            # Compute direction labels
            direction_1h = None
            direction_24h = None
            if price_change_1h is not None:
                if price_change_1h > 0.01:
                    direction_1h = 1
                elif price_change_1h < -0.01:
                    direction_1h = -1
                else:
                    direction_1h = 0
            if price_change_24h is not None:
                if price_change_24h > 0.02:
                    direction_24h = 1
                elif price_change_24h < -0.02:
                    direction_24h = -1
                else:
                    direction_24h = 0
                    
            # Check if market resolved
            cursor.execute(
                "SELECT is_resolved, resolution_outcome FROM tracked_markets WHERE token_id = ?",
                (token_id,)
            )
            market_row = cursor.fetchone()
            is_resolved = market_row[0] if market_row else 0
            resolution_outcome = market_row[1] if market_row else None
            
            # === LABELING LOGIC ===
            # "Fully labeled" = market has resolved (final truth)
            # "Has 24h label" = can be used for training (even if not resolved)
            # We continue tracking until resolution for the best training data
            is_fully_labeled = bool(is_resolved)
            has_24h_label = price_change_24h is not None
            
            # Track partial labels
            has_any_label = any([price_change_15m, price_change_1h, price_change_4h, price_change_24h])
            if has_any_label:
                partial_labeled += 1
            
            # Determine labeled_at timestamp
            # Set when we first get a 24h label (usable for training)
            labeled_at = None
            if has_24h_label or is_fully_labeled:
                labeled_at = now.isoformat()
            
            # Update the example
            cursor.execute("""
                UPDATE training_examples
                SET price_change_15m = ?, price_change_1h = ?, price_change_4h = ?, 
                    price_change_24h = ?, price_change_7d = ?,
                    direction_1h = ?, direction_24h = ?,
                    resolved_outcome = ?, labeled_at = ?, is_fully_labeled = ?
                WHERE example_id = ?
            """, (
                price_change_15m, price_change_1h, price_change_4h, 
                price_change_24h, price_change_7d,
                direction_1h, direction_24h,
                resolution_outcome, labeled_at,
                1 if is_fully_labeled else 0, example_id
            ))
            
            # Count as "labeled" if it has 24h data (usable for training)
            if has_24h_label:
                labeled_count += 1
                
        conn.commit()
        conn.close()
        
        # Log progress
        if labeled_count > 0 or partial_labeled > 0:
            logger.info(
                f"Labeling progress: {labeled_count} fully labeled, "
                f"{partial_labeled} with partial labels, "
                f"{len(examples)} total unlabeled checked"
            )
            
        return labeled_count
    
    def mark_market_resolved(self, token_id: str, outcome: int) -> None:
        """Mark a market as resolved.
        
        Args:
            token_id: Token ID of resolved market.
            outcome: Resolution outcome (1 = Yes, 0 = No).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE tracked_markets SET is_resolved = 1, resolution_outcome = ? WHERE token_id = ?",
            (outcome, token_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Marked market {token_id[:12]} as resolved with outcome {outcome}")
    
    def get_training_data(self, only_labeled: bool = True, min_schema_version: int = 1) -> list[dict]:
        """Get training data for model training.
        
        Handles backwards compatibility by filling missing features with defaults.
        
        Args:
            only_labeled: If True, only return examples with 24h labels (usable for training).
                         Note: "fully labeled" now means resolved, but we can train on
                         any example with a 24h price change label.
            min_schema_version: Minimum schema version to include.
            
        Returns:
            List of training examples as dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT features, schema_version, available_features,
                   price_change_15m, price_change_1h, price_change_4h, 
                   price_change_24h, price_change_7d,
                   direction_1h, direction_24h, resolved_outcome,
                   created_at, category, market_title
            FROM training_examples
            WHERE schema_version >= ?
        """
        
        if only_labeled:
            # Include examples with 24h labels (not just fully resolved)
            # This gives us more training data while still having reliable labels
            query += " AND price_change_24h IS NOT NULL"
            
        cursor.execute(query, (min_schema_version,))
        rows = cursor.fetchall()
        conn.close()
        
        # Get all possible feature columns
        all_feature_cols = MarketSnapshot.get_feature_columns()
        
        data = []
        for (features_json, schema_ver, avail_features_json,
             pc_15m, pc_1h, pc_4h, pc_24h, pc_7d,
             dir_1h, dir_24h, outcome, created_at, category, market_title) in rows:
            
            features = json.loads(features_json)
            available = json.loads(avail_features_json) if avail_features_json else []
            
            # Fill missing features with 0 for backwards compatibility
            for col in all_feature_cols:
                if col not in features:
                    features[col] = 0.0
                    
            # Add labels
            features["label_price_change_15m"] = pc_15m
            features["label_price_change_1h"] = pc_1h
            features["label_price_change_4h"] = pc_4h
            features["label_price_change_24h"] = pc_24h
            features["label_price_change_7d"] = pc_7d
            features["label_direction_1h"] = dir_1h
            features["label_direction_24h"] = dir_24h
            features["label_resolved_outcome"] = outcome
            features["_schema_version"] = schema_ver
            features["_available_features"] = available
            features["created_at"] = created_at  # For time-based sorting
            
            # Add category/market metadata (may override features if stored separately)
            if category:
                features["category"] = category
            if market_title:
                features["market_title"] = market_title
            
            data.append(features)
            
        return data
    
    def get_time_since_last_collection(self) -> Optional[timedelta]:
        """Get time since last data collection.
        
        Returns:
            Timedelta since last collection, or None if never collected.
        """
        if self._last_collection_time is None:
            return None
        return datetime.utcnow() - self._last_collection_time
    
    def update_collection_time(self) -> None:
        """Update the last collection timestamp."""
        self._last_collection_time = datetime.utcnow()
        self._save_state()
        
    def get_stats(self) -> dict:
        """Get collector statistics.
        
        Returns:
            Dictionary of stats.
        """
        storage = self.get_storage_stats()
        return {
            "tracked_markets": self.get_tracked_market_count(),
            "total_examples": self.get_total_examples(),
            "labeled_examples": self.get_labeled_examples(),
            "unlabeled_examples": self.get_unlabeled_examples(),
            "last_collection": self._last_collection_time.isoformat() if self._last_collection_time else None,
            "storage": storage,
            "schema_version": SCHEMA_VERSION,
        }


# Singleton instance
_collector_instance: Optional[ContinuousDataCollector] = None


def get_data_collector(db_path: str = "data/ai_training.db") -> ContinuousDataCollector:
    """Get or create the singleton data collector.
    
    Args:
        db_path: Path to database.
        
    Returns:
        ContinuousDataCollector instance.
    """
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = ContinuousDataCollector(db_path)
    return _collector_instance
