"""Historical data backfill service for dense ML training data."""

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)


class HistoricalDataBackfiller:
    """Backfills missing price data and collects dense historical snapshots."""
    
    def __init__(self, db_path: str = "data/training_data.db"):
        """Initialize backfiller.
        
        Args:
            db_path: Path to training data database.
        """
        self.db_path = db_path
        self.settings = get_settings()
        self.gamma_base_url = "https://gamma-api.polymarket.com"
        
    def backfill_missing_prices(
        self,
        token_ids: List[str],
        lookback_hours: int = 24,
    ) -> int:
        """Backfill missing price data for tokens.
        
        When bot is stopped, price data gaps occur. This fills those gaps
        by querying recent historical prices from Gamma API.
        
        Args:
            token_ids: List of token IDs to backfill.
            lookback_hours: How far back to look for gaps.
            
        Returns:
            Number of price points added.
        """
        if not self.settings.ml_enable_backfill:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        added_count = 0
        
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        cutoff_ts = cutoff.timestamp()
        
        for token_id in token_ids:
            try:
                # Find gaps in price history
                existing = conn.execute("""
                    SELECT timestamp FROM price_history
                    WHERE token_id = ? AND timestamp > ?
                    ORDER BY timestamp ASC
                """, (token_id, cutoff_ts)).fetchall()
                
                if not existing:
                    # No recent data, skip
                    continue
                
                existing_timestamps = [row[0] for row in existing]
                
                # Detect gaps > 2x snapshot interval
                gap_threshold = self.settings.ml_price_snapshot_interval_minutes * 60 * 2
                
                gaps = []
                for i in range(len(existing_timestamps) - 1):
                    gap = existing_timestamps[i + 1] - existing_timestamps[i]
                    if gap > gap_threshold:
                        gaps.append((existing_timestamps[i], existing_timestamps[i + 1]))
                
                if not gaps:
                    continue
                
                # Fetch historical prices to fill gaps
                # Note: Polymarket doesn't have a public historical price API
                # We'll use Gamma's current price endpoint and store what we can
                # In production, you'd use a proper historical data provider
                
                logger.info(
                    f"Found {len(gaps)} gaps for token {token_id[:12]}... "
                    f"(total gap time: {sum(g[1]-g[0] for g in gaps)/3600:.1f}h)"
                )
                
            except Exception as e:
                logger.warning(f"Backfill failed for token {token_id[:12]}: {e}")
                continue
        
        conn.close()
        
        if added_count > 0:
            logger.info(f"Backfilled {added_count} historical price points")
        
        return added_count
    
    def collect_dense_snapshots(
        self,
        markets: List[Dict],
        orderbooks: Optional[Dict] = None,
    ) -> int:
        """Collect dense price snapshots for active markets.
        
        This runs periodically (every 15 min by default) to capture
        price movements at higher granularity than the main trading loop.
        
        Args:
            markets: List of market dictionaries.
            orderbooks: Optional orderbook data.
            
        Returns:
            Number of snapshots stored.
        """
        conn = sqlite3.connect(self.db_path)
        now = datetime.utcnow()
        stored = 0
        
        # Check if we need a snapshot (based on interval)
        last_snapshot = conn.execute("""
            SELECT MAX(timestamp) FROM price_history
        """).fetchone()[0]
        
        if last_snapshot:
            minutes_since = (now.timestamp() - last_snapshot) / 60
            if minutes_since < self.settings.ml_price_snapshot_interval_minutes:
                conn.close()
                return 0
        
        # Store current prices
        for market in markets:
            if not hasattr(market, 'outcomes'):
                continue
                
            for outcome in market.outcomes:
                if not outcome.token_id or outcome.price is None:
                    continue
                
                price = float(outcome.price)
                
                # Get mid price from orderbook if available
                if orderbooks and outcome.token_id in orderbooks:
                    ob = orderbooks[outcome.token_id]
                    if ob.bids and ob.asks:
                        price = (ob.bids[0].price + ob.asks[0].price) / 2
                
                # Store snapshot
                conn.execute("""
                    INSERT INTO price_history (timestamp, token_id, price)
                    VALUES (?, ?, ?)
                """, (now.timestamp(), outcome.token_id, price))
                
                stored += 1
        
        conn.commit()
        conn.close()
        
        if stored > 0:
            logger.debug(f"Collected {stored} dense price snapshots")
        
        return stored
    
    def get_price_history_stats(self) -> Dict:
        """Get statistics about collected price history.
        
        Returns:
            Dictionary with stats.
        """
        conn = sqlite3.connect(self.db_path)
        
        total = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        
        unique_tokens = conn.execute(
            "SELECT COUNT(DISTINCT token_id) FROM price_history"
        ).fetchone()[0]
        
        oldest = conn.execute("SELECT MIN(timestamp) FROM price_history").fetchone()[0]
        newest = conn.execute("SELECT MAX(timestamp) FROM price_history").fetchone()[0]
        
        if oldest and newest:
            coverage_hours = (newest - oldest) / 3600
        else:
            coverage_hours = 0
        
        # Calculate average sampling rate
        if unique_tokens > 0 and coverage_hours > 0:
            samples_per_token = total / unique_tokens
            avg_interval_minutes = (coverage_hours * 60) / samples_per_token if samples_per_token > 0 else 0
        else:
            avg_interval_minutes = 0
        
        conn.close()
        
        return {
            'total_price_points': total,
            'unique_tokens': unique_tokens,
            'coverage_hours': coverage_hours,
            'coverage_days': coverage_hours / 24,
            'avg_interval_minutes': avg_interval_minutes,
            'target_interval_minutes': self.settings.ml_price_snapshot_interval_minutes,
        }

