"""Continuous AI Data Collector for training data collection.

This module tracks markets over time and creates training examples
at regular intervals, not just on resolution. It handles:
- Tracking many markets simultaneously
- Creating multiple examples per position over time
- Persisting data across restarts
- Catching up on missed data after shutdown
"""

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Optional
import threading

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """A snapshot of market state at a point in time."""
    token_id: str
    market_id: str
    timestamp: datetime
    price: float
    bid: float
    ask: float
    spread: float
    volume_24h: float
    liquidity: float
    orderbook_imbalance: float
    bid_depth: float
    ask_depth: float
    momentum_1h: float
    momentum_24h: float
    trade_count_1h: int
    category: str
    days_to_resolution: float
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass 
class TrainingExample:
    """A labeled training example for AI."""
    example_id: str
    token_id: str
    market_id: str
    created_at: datetime
    
    # Features at time of snapshot
    features: dict
    
    # Label: what happened next
    price_change_1h: Optional[float] = None
    price_change_4h: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_to_resolution: Optional[float] = None
    resolved_outcome: Optional[int] = None  # 1 = Yes won, 0 = No won
    
    # Metadata
    labeled_at: Optional[datetime] = None
    is_fully_labeled: bool = False


class ContinuousDataCollector:
    """Collects training data continuously from Polymarket."""
    
    def __init__(self, db_path: str = "data/ai_training.db"):
        """Initialize the collector.
        
        Args:
            db_path: Path to SQLite database for persistence.
        """
        self.db_path = db_path
        self._ensure_db()
        self._lock = threading.Lock()
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None
        
        # Track when we last collected data
        self._last_collection_time: Optional[datetime] = None
        self._load_state()
        
    def _ensure_db(self) -> None:
        """Create database tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL,
                bid REAL,
                ask REAL,
                spread REAL,
                volume_24h REAL,
                liquidity REAL,
                orderbook_imbalance REAL,
                bid_depth REAL,
                ask_depth REAL,
                momentum_1h REAL,
                momentum_24h REAL,
                trade_count_1h INTEGER,
                category TEXT,
                days_to_resolution REAL,
                UNIQUE(token_id, timestamp)
            )
        """)
        
        # Training examples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_examples (
                example_id TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                features TEXT NOT NULL,
                price_change_1h REAL,
                price_change_4h REAL,
                price_change_24h REAL,
                price_change_to_resolution REAL,
                resolved_outcome INTEGER,
                labeled_at TEXT,
                is_fully_labeled INTEGER DEFAULT 0
            )
        """)
        
        # Tracked markets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracked_markets (
                token_id TEXT PRIMARY KEY,
                market_id TEXT NOT NULL,
                started_tracking TEXT NOT NULL,
                last_snapshot TEXT,
                is_resolved INTEGER DEFAULT 0,
                resolution_outcome INTEGER
            )
        """)
        
        # Collector state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collector_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_token ON market_snapshots(token_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_examples_labeled ON training_examples(is_fully_labeled)")
        
        conn.commit()
        conn.close()
        
    def _load_state(self) -> None:
        """Load collector state from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM collector_state WHERE key = 'last_collection_time'")
        row = cursor.fetchone()
        if row:
            self._last_collection_time = datetime.fromisoformat(row[0])
        
        conn.close()
        
    def _save_state(self) -> None:
        """Save collector state to database."""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracked_markets WHERE is_resolved = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_total_examples(self) -> int:
        """Get total number of training examples."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_examples")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_labeled_examples(self) -> int:
        """Get number of fully labeled examples."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_examples WHERE is_fully_labeled = 1")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_unlabeled_examples(self) -> int:
        """Get number of examples waiting for labels."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM training_examples WHERE is_fully_labeled = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def add_market_to_track(self, token_id: str, market_id: str) -> bool:
        """Add a market to track for training data.
        
        Args:
            token_id: Token ID to track.
            market_id: Market condition ID.
            
        Returns:
            True if added, False if already tracking.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO tracked_markets (token_id, market_id, started_tracking) VALUES (?, ?, ?)",
                (token_id, market_id, datetime.utcnow().isoformat())
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO market_snapshots 
                (token_id, market_id, timestamp, price, bid, ask, spread, volume_24h,
                 liquidity, orderbook_imbalance, bid_depth, ask_depth, momentum_1h,
                 momentum_24h, trade_count_1h, category, days_to_resolution)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.token_id, snapshot.market_id, snapshot.timestamp.isoformat(),
                snapshot.price, snapshot.bid, snapshot.ask, snapshot.spread,
                snapshot.volume_24h, snapshot.liquidity, snapshot.orderbook_imbalance,
                snapshot.bid_depth, snapshot.ask_depth, snapshot.momentum_1h,
                snapshot.momentum_24h, snapshot.trade_count_1h, snapshot.category,
                snapshot.days_to_resolution
            ))
            
            # Update last snapshot time
            cursor.execute(
                "UPDATE tracked_markets SET last_snapshot = ? WHERE token_id = ?",
                (snapshot.timestamp.isoformat(), snapshot.token_id)
            )
            
            conn.commit()
        finally:
            conn.close()
            
    def create_training_example(self, snapshot: MarketSnapshot) -> str:
        """Create a training example from a snapshot.
        
        Args:
            snapshot: Market snapshot to create example from.
            
        Returns:
            Example ID.
        """
        import uuid
        
        example_id = str(uuid.uuid4())
        features = snapshot.to_dict()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO training_examples 
            (example_id, token_id, market_id, created_at, features, is_fully_labeled)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (
            example_id, snapshot.token_id, snapshot.market_id,
            datetime.utcnow().isoformat(), json.dumps(features)
        ))
        
        conn.commit()
        conn.close()
        
        return example_id
    
    def label_examples(self) -> int:
        """Label unlabeled examples with future price data.
        
        Returns:
            Number of examples labeled.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unlabeled examples
        cursor.execute("""
            SELECT example_id, token_id, created_at, features
            FROM training_examples 
            WHERE is_fully_labeled = 0
        """)
        
        examples = cursor.fetchall()
        labeled_count = 0
        now = datetime.utcnow()
        
        for example_id, token_id, created_at_str, features_json in examples:
            created_at = datetime.fromisoformat(created_at_str)
            features = json.loads(features_json)
            initial_price = features.get("price", 0)
            
            if initial_price <= 0:
                continue
                
            # Get snapshots after this example was created
            cursor.execute("""
                SELECT timestamp, price FROM market_snapshots
                WHERE token_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
            """, (token_id, created_at_str))
            
            future_snapshots = cursor.fetchall()
            
            price_change_1h = None
            price_change_4h = None
            price_change_24h = None
            
            for snap_time_str, snap_price in future_snapshots:
                snap_time = datetime.fromisoformat(snap_time_str)
                hours_later = (snap_time - created_at).total_seconds() / 3600
                
                if snap_price and snap_price > 0:
                    price_change = (snap_price - initial_price) / initial_price
                    
                    if hours_later >= 1 and price_change_1h is None:
                        price_change_1h = price_change
                    if hours_later >= 4 and price_change_4h is None:
                        price_change_4h = price_change
                    if hours_later >= 24 and price_change_24h is None:
                        price_change_24h = price_change
                        
            # Check if we can fully label this example
            hours_since_created = (now - created_at).total_seconds() / 3600
            
            # Check if market resolved
            cursor.execute(
                "SELECT is_resolved, resolution_outcome FROM tracked_markets WHERE token_id = ?",
                (token_id,)
            )
            market_row = cursor.fetchone()
            is_resolved = market_row[0] if market_row else 0
            resolution_outcome = market_row[1] if market_row else None
            
            # Mark as fully labeled if 24h has passed or market resolved
            is_fully_labeled = (hours_since_created >= 24 and price_change_24h is not None) or is_resolved
            
            # Update the example
            cursor.execute("""
                UPDATE training_examples
                SET price_change_1h = ?, price_change_4h = ?, price_change_24h = ?,
                    resolved_outcome = ?, labeled_at = ?, is_fully_labeled = ?
                WHERE example_id = ?
            """, (
                price_change_1h, price_change_4h, price_change_24h,
                resolution_outcome, now.isoformat() if is_fully_labeled else None,
                1 if is_fully_labeled else 0, example_id
            ))
            
            if is_fully_labeled:
                labeled_count += 1
                
        conn.commit()
        conn.close()
        
        if labeled_count > 0:
            logger.info(f"Labeled {labeled_count} training examples")
            
        return labeled_count
    
    def mark_market_resolved(self, token_id: str, outcome: int) -> None:
        """Mark a market as resolved.
        
        Args:
            token_id: Token ID of resolved market.
            outcome: Resolution outcome (1 = Yes, 0 = No).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE tracked_markets SET is_resolved = 1, resolution_outcome = ? WHERE token_id = ?",
            (outcome, token_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Marked market {token_id[:12]} as resolved with outcome {outcome}")
    
    def get_training_data(self, only_labeled: bool = True) -> list[dict]:
        """Get training data for model training.
        
        Args:
            only_labeled: If True, only return fully labeled examples.
            
        Returns:
            List of training examples as dictionaries.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if only_labeled:
            cursor.execute("""
                SELECT features, price_change_1h, price_change_4h, price_change_24h,
                       resolved_outcome
                FROM training_examples
                WHERE is_fully_labeled = 1
            """)
        else:
            cursor.execute("""
                SELECT features, price_change_1h, price_change_4h, price_change_24h,
                       resolved_outcome
                FROM training_examples
            """)
            
        rows = cursor.fetchall()
        conn.close()
        
        data = []
        for features_json, pc_1h, pc_4h, pc_24h, outcome in rows:
            features = json.loads(features_json)
            features["label_price_change_1h"] = pc_1h
            features["label_price_change_4h"] = pc_4h
            features["label_price_change_24h"] = pc_24h
            features["label_resolved_outcome"] = outcome
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
        return {
            "tracked_markets": self.get_tracked_market_count(),
            "total_examples": self.get_total_examples(),
            "labeled_examples": self.get_labeled_examples(),
            "unlabeled_examples": self.get_unlabeled_examples(),
            "last_collection": self._last_collection_time.isoformat() if self._last_collection_time else None,
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
