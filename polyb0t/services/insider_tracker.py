"""Insider transaction tracking for detecting predictive wallets.

This service tracks wallets that historically:
1. Trade before resolution in the correct direction
2. Have high win rates on specific market types
3. Show unusual activity patterns

The goal is to detect "smart money" and use their activity as a predictive signal.
"""

import json
import logging
import os
import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WalletProfile:
    """Profile of a tracked wallet."""
    wallet_address: str
    total_trades: int = 0
    correct_predictions: int = 0
    win_rate: float = 0.0
    avg_trade_size_usd: float = 0.0
    total_volume_usd: float = 0.0
    categories_traded: Dict[str, int] = field(default_factory=dict)
    first_seen: Optional[datetime] = None
    last_trade: Optional[datetime] = None
    reputation_score: float = 0.5  # 0-1, higher = more predictive

    # Timeliness metrics
    avg_hours_before_resolution: float = 0.0
    trades_before_resolution: int = 0


@dataclass
class WalletTrade:
    """A trade by a tracked wallet."""
    wallet_address: str
    market_id: str
    token_id: str
    side: str  # "BUY" or "SELL"
    size: float
    price: float
    value_usd: float
    timestamp: datetime
    resolution_outcome: Optional[int] = None  # None until resolved
    was_correct: Optional[bool] = None  # None until resolved
    hours_before_resolution: Optional[float] = None


@dataclass
class InsiderSignal:
    """Signal from insider activity."""
    market_id: str
    token_id: str
    signal_type: str  # "smart_accumulation", "smart_distribution", "unusual_activity"
    strength: float  # 0-1
    wallet_count: int
    total_volume_usd: float
    avg_wallet_reputation: float
    timestamp: datetime


@dataclass
class InsiderFeatures:
    """Features for ML training from insider activity."""
    smart_wallet_buy_count_1h: int = 0
    smart_wallet_sell_count_1h: int = 0
    smart_wallet_net_direction_1h: float = 0.0
    smart_wallet_volume_1h: float = 0.0
    avg_buyer_reputation: float = 0.5
    avg_seller_reputation: float = 0.5
    smart_wallet_buy_count_24h: int = 0
    smart_wallet_sell_count_24h: int = 0
    smart_wallet_net_direction_24h: float = 0.0
    unusual_activity_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for ML features."""
        return {
            "smart_wallet_buy_count_1h": float(self.smart_wallet_buy_count_1h),
            "smart_wallet_sell_count_1h": float(self.smart_wallet_sell_count_1h),
            "smart_wallet_net_direction_1h": self.smart_wallet_net_direction_1h,
            "smart_wallet_volume_1h": self.smart_wallet_volume_1h,
            "avg_buyer_reputation": self.avg_buyer_reputation,
            "avg_seller_reputation": self.avg_seller_reputation,
            "smart_wallet_buy_count_24h": float(self.smart_wallet_buy_count_24h),
            "smart_wallet_sell_count_24h": float(self.smart_wallet_sell_count_24h),
            "smart_wallet_net_direction_24h": self.smart_wallet_net_direction_24h,
            "unusual_activity_score": self.unusual_activity_score,
        }


class InsiderTracker:
    """Service to track and analyze insider trading patterns.

    Maintains a database of wallet profiles and their trading history.
    Uses this to identify "smart money" and generate predictive features.
    """

    def __init__(
        self,
        db_path: str = "data/insider_tracking.db",
        min_trades_for_reputation: int = 10,
        smart_wallet_threshold: float = 0.65,
        large_trade_usd: float = 1000.0
    ):
        """Initialize the insider tracker.

        Args:
            db_path: Path to SQLite database
            min_trades_for_reputation: Minimum trades before wallet gets score
            smart_wallet_threshold: Win rate to be considered "smart"
            large_trade_usd: Threshold for large trade detection
        """
        self.db_path = db_path
        self.min_trades_for_reputation = min_trades_for_reputation
        self.smart_wallet_threshold = smart_wallet_threshold
        self.large_trade_usd = large_trade_usd

        # Thread safety
        self._lock = threading.Lock()

        # In-memory cache of wallet profiles
        self._wallet_profiles: Dict[str, WalletProfile] = {}
        self._smart_wallets: Set[str] = set()

        # Recent trades cache for real-time analysis
        self._recent_trades: Dict[str, List[WalletTrade]] = defaultdict(list)
        self._recent_trades_max = 1000  # Per token

        # Initialize database
        self._ensure_db()
        self._load_wallet_profiles()

    def _ensure_db(self):
        """Create database tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wallet_profiles (
                    wallet_address TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    avg_trade_size_usd REAL DEFAULT 0.0,
                    total_volume_usd REAL DEFAULT 0.0,
                    reputation_score REAL DEFAULT 0.5,
                    avg_hours_before_resolution REAL DEFAULT 0.0,
                    trades_before_resolution INTEGER DEFAULT 0,
                    categories_traded TEXT,
                    first_seen TEXT,
                    last_trade TEXT,
                    updated_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS wallet_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet_address TEXT,
                    market_id TEXT,
                    token_id TEXT,
                    side TEXT,
                    size REAL,
                    price REAL,
                    value_usd REAL,
                    timestamp TEXT,
                    resolution_outcome INTEGER,
                    was_correct INTEGER,
                    hours_before_resolution REAL,
                    FOREIGN KEY (wallet_address) REFERENCES wallet_profiles(wallet_address)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_wallet_trades_wallet
                ON wallet_trades(wallet_address)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_wallet_trades_market
                ON wallet_trades(market_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_wallet_trades_token
                ON wallet_trades(token_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_wallet_trades_timestamp
                ON wallet_trades(timestamp)
            """)

            conn.commit()

    def _load_wallet_profiles(self):
        """Load wallet profiles from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM wallet_profiles WHERE reputation_score >= ?",
                    (self.smart_wallet_threshold,)
                )

                for row in cursor:
                    profile = WalletProfile(
                        wallet_address=row['wallet_address'],
                        total_trades=row['total_trades'],
                        correct_predictions=row['correct_predictions'],
                        win_rate=row['win_rate'],
                        avg_trade_size_usd=row['avg_trade_size_usd'],
                        total_volume_usd=row['total_volume_usd'],
                        reputation_score=row['reputation_score'],
                        avg_hours_before_resolution=row['avg_hours_before_resolution'],
                        trades_before_resolution=row['trades_before_resolution'],
                        categories_traded=json.loads(row['categories_traded'] or '{}'),
                        first_seen=datetime.fromisoformat(row['first_seen']) if row['first_seen'] else None,
                        last_trade=datetime.fromisoformat(row['last_trade']) if row['last_trade'] else None,
                    )
                    self._wallet_profiles[profile.wallet_address] = profile

                    if profile.reputation_score >= self.smart_wallet_threshold:
                        self._smart_wallets.add(profile.wallet_address)

                logger.info(f"Loaded {len(self._wallet_profiles)} wallet profiles, "
                           f"{len(self._smart_wallets)} smart wallets")

        except Exception as e:
            logger.warning(f"Error loading wallet profiles: {e}")

    def process_trade(
        self,
        wallet_address: str,
        market_id: str,
        token_id: str,
        side: str,
        size: float,
        price: float,
        timestamp: Optional[datetime] = None,
        category: str = ""
    ) -> Optional[InsiderSignal]:
        """Process a trade and check for insider patterns.

        Args:
            wallet_address: Wallet that made the trade
            market_id: Market identifier
            token_id: Token identifier
            side: "BUY" or "SELL"
            size: Trade size
            price: Trade price
            timestamp: Trade timestamp
            category: Market category

        Returns:
            InsiderSignal if smart wallet activity detected, else None
        """
        if not wallet_address:
            return None

        timestamp = timestamp or datetime.utcnow()
        value_usd = size * price

        # Create trade record
        trade = WalletTrade(
            wallet_address=wallet_address,
            market_id=market_id,
            token_id=token_id,
            side=side.upper(),
            size=size,
            price=price,
            value_usd=value_usd,
            timestamp=timestamp
        )

        with self._lock:
            # Update wallet profile
            self._update_wallet_profile(wallet_address, trade, category)

            # Store trade
            self._store_trade(trade)

            # Add to recent trades cache
            self._recent_trades[token_id].append(trade)
            if len(self._recent_trades[token_id]) > self._recent_trades_max:
                self._recent_trades[token_id] = self._recent_trades[token_id][-self._recent_trades_max:]

            # Check if this is a smart wallet trade
            if wallet_address in self._smart_wallets:
                return self._generate_signal(trade)

        return None

    def _update_wallet_profile(
        self,
        wallet_address: str,
        trade: WalletTrade,
        category: str
    ):
        """Update wallet profile with new trade."""
        if wallet_address not in self._wallet_profiles:
            self._wallet_profiles[wallet_address] = WalletProfile(
                wallet_address=wallet_address,
                first_seen=trade.timestamp
            )

        profile = self._wallet_profiles[wallet_address]
        profile.total_trades += 1
        profile.total_volume_usd += trade.value_usd
        profile.avg_trade_size_usd = profile.total_volume_usd / profile.total_trades
        profile.last_trade = trade.timestamp

        if category:
            profile.categories_traded[category] = profile.categories_traded.get(category, 0) + 1

    def _store_trade(self, trade: WalletTrade):
        """Store trade in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO wallet_trades (
                        wallet_address, market_id, token_id, side, size, price,
                        value_usd, timestamp, resolution_outcome, was_correct,
                        hours_before_resolution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.wallet_address,
                    trade.market_id,
                    trade.token_id,
                    trade.side,
                    trade.size,
                    trade.price,
                    trade.value_usd,
                    trade.timestamp.isoformat(),
                    trade.resolution_outcome,
                    1 if trade.was_correct else (0 if trade.was_correct is False else None),
                    trade.hours_before_resolution
                ))
                conn.commit()
        except Exception as e:
            logger.debug(f"Error storing trade: {e}")

    def _generate_signal(self, trade: WalletTrade) -> InsiderSignal:
        """Generate an insider signal from a smart wallet trade."""
        profile = self._wallet_profiles.get(trade.wallet_address)
        reputation = profile.reputation_score if profile else 0.5

        signal_type = "smart_accumulation" if trade.side == "BUY" else "smart_distribution"

        return InsiderSignal(
            market_id=trade.market_id,
            token_id=trade.token_id,
            signal_type=signal_type,
            strength=reputation,
            wallet_count=1,
            total_volume_usd=trade.value_usd,
            avg_wallet_reputation=reputation,
            timestamp=trade.timestamp
        )

    def record_resolution(
        self,
        market_id: str,
        token_id: str,
        winning_outcome: int,
        resolution_time: Optional[datetime] = None
    ):
        """Record market resolution and update wallet reputation scores.

        Args:
            market_id: Market that resolved
            token_id: Token that resolved
            winning_outcome: 1 for Yes, 0 for No
            resolution_time: Time of resolution
        """
        resolution_time = resolution_time or datetime.utcnow()

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all trades for this market
                cursor = conn.execute("""
                    SELECT id, wallet_address, side, timestamp
                    FROM wallet_trades
                    WHERE (market_id = ? OR token_id = ?)
                    AND resolution_outcome IS NULL
                """, (market_id, token_id))

                trades_to_update = list(cursor)

                for trade_id, wallet_address, side, timestamp_str in trades_to_update:
                    trade_time = datetime.fromisoformat(timestamp_str)
                    hours_before = (resolution_time - trade_time).total_seconds() / 3600

                    # Determine if prediction was correct
                    # BUY Yes token = predicting Yes wins
                    # SELL Yes token = predicting No wins
                    predicted_yes = side == "BUY"
                    was_correct = (predicted_yes and winning_outcome == 1) or \
                                 (not predicted_yes and winning_outcome == 0)

                    # Update trade record
                    conn.execute("""
                        UPDATE wallet_trades
                        SET resolution_outcome = ?,
                            was_correct = ?,
                            hours_before_resolution = ?
                        WHERE id = ?
                    """, (winning_outcome, 1 if was_correct else 0, hours_before, trade_id))

                    # Update wallet profile
                    if wallet_address in self._wallet_profiles:
                        profile = self._wallet_profiles[wallet_address]
                        if was_correct:
                            profile.correct_predictions += 1
                        profile.trades_before_resolution += 1

                        # Update timeliness
                        n = profile.trades_before_resolution
                        profile.avg_hours_before_resolution = (
                            (profile.avg_hours_before_resolution * (n - 1) + hours_before) / n
                        )

                        # Recalculate win rate and reputation
                        if profile.trades_before_resolution >= self.min_trades_for_reputation:
                            profile.win_rate = profile.correct_predictions / profile.trades_before_resolution
                            profile.reputation_score = self._calculate_reputation(profile)

                            # Update smart wallet set
                            if profile.reputation_score >= self.smart_wallet_threshold:
                                self._smart_wallets.add(wallet_address)
                            else:
                                self._smart_wallets.discard(wallet_address)

                conn.commit()

                # Persist wallet profiles
                self._save_wallet_profiles()

        except Exception as e:
            logger.warning(f"Error recording resolution: {e}")

    def _calculate_reputation(self, profile: WalletProfile) -> float:
        """Calculate wallet reputation score.

        Based on:
        - Win rate (primary factor)
        - Timeliness (bonus for early predictions)
        - Trade size (bonus for conviction)
        """
        if profile.trades_before_resolution < self.min_trades_for_reputation:
            return 0.5  # Neutral until proven

        # Base: win rate
        base_score = profile.win_rate

        # Bonus for timeliness (trading well before resolution)
        # More hours before = better foresight
        timeliness_bonus = min(0.1, profile.avg_hours_before_resolution / 1000)

        # Bonus for trade size (larger = more conviction)
        size_bonus = min(0.05, profile.avg_trade_size_usd / 20000)

        # Penalty for low sample size
        sample_penalty = 0.0
        if profile.trades_before_resolution < 20:
            sample_penalty = 0.1 * (1 - profile.trades_before_resolution / 20)

        reputation = base_score + timeliness_bonus + size_bonus - sample_penalty

        return min(1.0, max(0.0, reputation))

    def _save_wallet_profiles(self):
        """Save wallet profiles to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for profile in self._wallet_profiles.values():
                    conn.execute("""
                        INSERT OR REPLACE INTO wallet_profiles (
                            wallet_address, total_trades, correct_predictions, win_rate,
                            avg_trade_size_usd, total_volume_usd, reputation_score,
                            avg_hours_before_resolution, trades_before_resolution,
                            categories_traded, first_seen, last_trade, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        profile.wallet_address,
                        profile.total_trades,
                        profile.correct_predictions,
                        profile.win_rate,
                        profile.avg_trade_size_usd,
                        profile.total_volume_usd,
                        profile.reputation_score,
                        profile.avg_hours_before_resolution,
                        profile.trades_before_resolution,
                        json.dumps(profile.categories_traded),
                        profile.first_seen.isoformat() if profile.first_seen else None,
                        profile.last_trade.isoformat() if profile.last_trade else None,
                        datetime.utcnow().isoformat()
                    ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Error saving wallet profiles: {e}")

    def get_insider_features(self, token_id: str) -> InsiderFeatures:
        """Get insider-based features for ML training.

        Args:
            token_id: Token to get features for

        Returns:
            InsiderFeatures with all computed values
        """
        features = InsiderFeatures()

        with self._lock:
            recent_trades = self._recent_trades.get(token_id, [])

        if not recent_trades:
            return features

        now = datetime.utcnow()
        cutoff_1h = now - timedelta(hours=1)
        cutoff_24h = now - timedelta(hours=24)

        # Analyze trades by smart wallets
        smart_buys_1h = []
        smart_sells_1h = []
        smart_buys_24h = []
        smart_sells_24h = []

        for trade in recent_trades:
            if trade.wallet_address not in self._smart_wallets:
                continue

            profile = self._wallet_profiles.get(trade.wallet_address)
            reputation = profile.reputation_score if profile else 0.5

            if trade.timestamp >= cutoff_24h:
                if trade.side == "BUY":
                    smart_buys_24h.append((trade, reputation))
                else:
                    smart_sells_24h.append((trade, reputation))

            if trade.timestamp >= cutoff_1h:
                if trade.side == "BUY":
                    smart_buys_1h.append((trade, reputation))
                else:
                    smart_sells_1h.append((trade, reputation))

        # 1-hour features
        features.smart_wallet_buy_count_1h = len(smart_buys_1h)
        features.smart_wallet_sell_count_1h = len(smart_sells_1h)

        buy_vol_1h = sum(t.value_usd for t, _ in smart_buys_1h)
        sell_vol_1h = sum(t.value_usd for t, _ in smart_sells_1h)
        total_vol_1h = buy_vol_1h + sell_vol_1h

        features.smart_wallet_volume_1h = total_vol_1h

        if total_vol_1h > 0:
            features.smart_wallet_net_direction_1h = (buy_vol_1h - sell_vol_1h) / total_vol_1h

        if smart_buys_1h:
            features.avg_buyer_reputation = sum(r for _, r in smart_buys_1h) / len(smart_buys_1h)
        if smart_sells_1h:
            features.avg_seller_reputation = sum(r for _, r in smart_sells_1h) / len(smart_sells_1h)

        # 24-hour features
        features.smart_wallet_buy_count_24h = len(smart_buys_24h)
        features.smart_wallet_sell_count_24h = len(smart_sells_24h)

        buy_vol_24h = sum(t.value_usd for t, _ in smart_buys_24h)
        sell_vol_24h = sum(t.value_usd for t, _ in smart_sells_24h)
        total_vol_24h = buy_vol_24h + sell_vol_24h

        if total_vol_24h > 0:
            features.smart_wallet_net_direction_24h = (buy_vol_24h - sell_vol_24h) / total_vol_24h

        # Unusual activity score
        # High if there's sudden increase in smart wallet activity
        all_trades_1h = [t for t in recent_trades if t.timestamp >= cutoff_1h]
        all_trades_24h = [t for t in recent_trades if t.timestamp >= cutoff_24h]

        if all_trades_24h:
            hourly_rate = len(all_trades_24h) / 24
            if hourly_rate > 0:
                recent_rate = len(all_trades_1h)
                deviation = recent_rate / hourly_rate
                # Score based on how much recent activity exceeds average
                features.unusual_activity_score = min(1.0, max(0.0, (deviation - 1) / 5))

        return features

    def get_features_dict(self, token_id: str) -> Dict[str, float]:
        """Get insider features as a dictionary for ML training.

        Args:
            token_id: Token to get features for

        Returns:
            Dictionary of feature name -> value
        """
        return self.get_insider_features(token_id).to_dict()

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "total_wallets": len(self._wallet_profiles),
            "smart_wallets": len(self._smart_wallets),
            "cached_tokens": len(self._recent_trades),
            "smart_wallet_threshold": self.smart_wallet_threshold,
            "min_trades_for_reputation": self.min_trades_for_reputation,
        }


# Singleton instance
_insider_tracker: Optional[InsiderTracker] = None


def get_insider_tracker() -> InsiderTracker:
    """Get the singleton InsiderTracker instance."""
    global _insider_tracker
    if _insider_tracker is None:
        try:
            from polyb0t.config.settings import get_settings
            settings = get_settings()
            db_path = getattr(settings, 'insider_tracking_db', 'data/insider_tracking.db')
            min_trades = getattr(settings, 'smart_wallet_min_trades', 10)
            threshold = getattr(settings, 'smart_wallet_threshold', 0.65)
        except:
            db_path = 'data/insider_tracking.db'
            min_trades = 10
            threshold = 0.65

        _insider_tracker = InsiderTracker(
            db_path=db_path,
            min_trades_for_reputation=min_trades,
            smart_wallet_threshold=threshold
        )
    return _insider_tracker


def get_insider_features_for_token(token_id: str) -> Dict[str, float]:
    """Convenience function to get insider features.

    Args:
        token_id: Token identifier

    Returns:
        Dictionary of insider features
    """
    tracker = get_insider_tracker()
    return tracker.get_features_dict(token_id)
