"""Historical Price Data Collector for comprehensive ML training data.

This module fetches historical price timeseries data from the Polymarket CLOB API
and stores it for use in training the MoE prediction model. It provides the
critical price momentum, volatility, and trend features that the model needs
to make meaningful predictions.

Key features:
- Fetches maximum available price history for all active and recently resolved markets
- Respects Polymarket API rate limits (1500 req/10s for price history endpoint)
- Computes momentum, volatility, and other time-series features from historical data
- Stores data efficiently for fast retrieval during training
- Supports incremental updates to avoid re-fetching unchanged data
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from polyb0t.config import get_settings
from polyb0t.data.clob_client import CLOBClient, PriceHistory, PricePoint
from polyb0t.data.gamma_client import GammaClient
from polyb0t.data.models import Market

logger = logging.getLogger(__name__)


# Rate limits for CLOB API (prices-history endpoint)
# Documented: 1500 req/10s = 150 req/s
# Safe: Use 100 req/s to leave headroom
PRICE_HISTORY_RATE_LIMIT_PER_SEC = 100
PRICE_HISTORY_BATCH_DELAY_MS = 15  # 15ms between requests = ~66 req/s (very safe)


@dataclass
class MarketPriceData:
    """Aggregated price data for a market token."""
    token_id: str
    market_id: str

    # Raw price history
    price_points: list[tuple[datetime, float]]

    # Computed features
    current_price: float
    price_1h_ago: Optional[float]
    price_4h_ago: Optional[float]
    price_24h_ago: Optional[float]
    price_7d_ago: Optional[float]

    # Momentum (price change %)
    momentum_1h: float
    momentum_4h: float
    momentum_24h: float
    momentum_7d: float

    # Volatility (std dev of returns)
    volatility_1h: float
    volatility_4h: float
    volatility_24h: float
    volatility_7d: float

    # Price range
    high_24h: float
    low_24h: float
    range_24h: float

    # Trend indicators
    sma_1h: float
    sma_24h: float
    price_vs_sma_24h: float

    # Data quality
    data_points_available: int
    oldest_data_timestamp: Optional[datetime]
    newest_data_timestamp: Optional[datetime]

    def to_features_dict(self) -> dict[str, Any]:
        """Convert to dictionary of ML features."""
        return {
            "historical_price": self.current_price,
            "historical_price_1h_ago": self.price_1h_ago or self.current_price,
            "historical_price_4h_ago": self.price_4h_ago or self.current_price,
            "historical_price_24h_ago": self.price_24h_ago or self.current_price,
            "historical_price_7d_ago": self.price_7d_ago or self.current_price,
            "historical_momentum_1h": self.momentum_1h,
            "historical_momentum_4h": self.momentum_4h,
            "historical_momentum_24h": self.momentum_24h,
            "historical_momentum_7d": self.momentum_7d,
            "historical_volatility_1h": self.volatility_1h,
            "historical_volatility_4h": self.volatility_4h,
            "historical_volatility_24h": self.volatility_24h,
            "historical_volatility_7d": self.volatility_7d,
            "historical_high_24h": self.high_24h,
            "historical_low_24h": self.low_24h,
            "historical_range_24h": self.range_24h,
            "historical_sma_1h": self.sma_1h,
            "historical_sma_24h": self.sma_24h,
            "historical_price_vs_sma_24h": self.price_vs_sma_24h,
            "historical_data_points": self.data_points_available,
        }


class HistoricalPriceCollector:
    """Collects and processes historical price data from Polymarket CLOB API.

    This class is responsible for:
    1. Fetching price history for markets using the CLOB /prices-history endpoint
    2. Computing momentum, volatility, and trend features from the raw data
    3. Storing processed data for efficient retrieval during ML training
    4. Rate limiting to respect Polymarket API constraints
    """

    def __init__(
        self,
        db_path: str = "data/historical_prices.db",
        requests_per_second: float = 50.0,  # Conservative rate limit
    ):
        """Initialize the historical price collector.

        Args:
            db_path: Path to SQLite database for storing price history.
            requests_per_second: Maximum API requests per second (default: 50).
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.requests_per_second = requests_per_second
        self._request_delay = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Raw price history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history_raw (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                market_id TEXT,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                fetched_at TEXT NOT NULL,
                UNIQUE(token_id, timestamp)
            )
        """)

        # Computed features cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS computed_features (
                token_id TEXT PRIMARY KEY,
                market_id TEXT,
                features TEXT NOT NULL,
                computed_at TEXT NOT NULL,
                data_points INTEGER,
                oldest_timestamp TEXT,
                newest_timestamp TEXT
            )
        """)

        # Collection metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS collection_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_raw_token ON price_history_raw(token_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_raw_time ON price_history_raw(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_raw_token_time ON price_history_raw(token_id, timestamp)")

        conn.commit()
        conn.close()

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._request_delay:
            await asyncio.sleep(self._request_delay - elapsed)
        self._last_request_time = time.time()

    async def fetch_price_history_for_token(
        self,
        clob: CLOBClient,
        token_id: str,
        market_id: str | None = None,
        fidelity: int = 60,  # 1-hour resolution by default
    ) -> Optional[PriceHistory]:
        """Fetch price history for a single token.

        Args:
            clob: CLOB client instance.
            token_id: Token ID to fetch history for.
            market_id: Optional market ID for logging.
            fidelity: Data resolution in minutes (default: 60).

        Returns:
            PriceHistory object or None on error.
        """
        await self._rate_limit()

        try:
            history = await clob.get_price_history_max(token_id, fidelity=fidelity)
            if history and history.points:
                logger.debug(
                    f"Fetched {len(history.points)} price points for token {token_id[:12]}..."
                )
            return history
        except Exception as e:
            logger.warning(f"Failed to fetch price history for {token_id[:12]}: {e}")
            return None

    def store_price_history(
        self,
        token_id: str,
        market_id: str | None,
        history: PriceHistory,
    ) -> int:
        """Store price history points to database.

        Args:
            token_id: Token ID.
            market_id: Market condition ID.
            history: Price history to store.

        Returns:
            Number of new points stored.
        """
        if not history.points:
            return 0

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        stored = 0

        for point in history.points:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO price_history_raw
                    (token_id, market_id, timestamp, price, fetched_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    token_id,
                    market_id,
                    point.timestamp.isoformat(),
                    point.price,
                    now,
                ))
                if cursor.rowcount > 0:
                    stored += 1
            except Exception as e:
                logger.debug(f"Failed to store price point: {e}")

        conn.commit()
        conn.close()

        return stored

    def get_stored_price_history(
        self,
        token_id: str,
        hours: int | None = None,
    ) -> list[tuple[datetime, float]]:
        """Get stored price history for a token.

        Args:
            token_id: Token ID.
            hours: Optional limit to last N hours.

        Returns:
            List of (timestamp, price) tuples.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = """
            SELECT timestamp, price FROM price_history_raw
            WHERE token_id = ?
        """
        params: list[Any] = [token_id]

        if hours:
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            query += " AND timestamp > ?"
            params.append(cutoff)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [(datetime.fromisoformat(ts), price) for ts, price in rows]

    def compute_features_from_history(
        self,
        token_id: str,
        market_id: str | None,
        price_history: list[tuple[datetime, float]],
    ) -> Optional[MarketPriceData]:
        """Compute ML features from price history.

        Args:
            token_id: Token ID.
            market_id: Market condition ID.
            price_history: List of (timestamp, price) tuples, sorted by time.

        Returns:
            MarketPriceData with computed features, or None if insufficient data.
        """
        if len(price_history) < 2:
            return None

        # Get current price and timestamp
        now = datetime.utcnow()
        current_price = price_history[-1][1]
        oldest_ts = price_history[0][0]
        newest_ts = price_history[-1][0]

        # Find prices at various lookback periods
        def get_price_at_lookback(hours: int) -> Optional[float]:
            target_time = now - timedelta(hours=hours)
            # Find closest price point at or before target time
            closest_price = None
            for ts, price in reversed(price_history):
                if ts <= target_time:
                    closest_price = price
                    break
            return closest_price

        price_1h_ago = get_price_at_lookback(1)
        price_4h_ago = get_price_at_lookback(4)
        price_24h_ago = get_price_at_lookback(24)
        price_7d_ago = get_price_at_lookback(168)

        # Compute momentum (percentage price change)
        def compute_momentum(old_price: Optional[float]) -> float:
            if old_price is None or old_price <= 0:
                return 0.0
            return (current_price - old_price) / old_price

        momentum_1h = compute_momentum(price_1h_ago)
        momentum_4h = compute_momentum(price_4h_ago)
        momentum_24h = compute_momentum(price_24h_ago)
        momentum_7d = compute_momentum(price_7d_ago)

        # Compute volatility (standard deviation of returns)
        def compute_volatility(lookback_hours: int) -> float:
            cutoff = now - timedelta(hours=lookback_hours)
            relevant_prices = [p for ts, p in price_history if ts >= cutoff]
            if len(relevant_prices) < 3:
                return 0.0

            returns = []
            for i in range(1, len(relevant_prices)):
                if relevant_prices[i-1] > 0:
                    ret = (relevant_prices[i] - relevant_prices[i-1]) / relevant_prices[i-1]
                    returns.append(ret)

            if len(returns) < 2:
                return 0.0

            import statistics
            try:
                return statistics.stdev(returns)
            except:
                return 0.0

        volatility_1h = compute_volatility(1)
        volatility_4h = compute_volatility(4)
        volatility_24h = compute_volatility(24)
        volatility_7d = compute_volatility(168)

        # Compute 24h high/low/range
        cutoff_24h = now - timedelta(hours=24)
        prices_24h = [p for ts, p in price_history if ts >= cutoff_24h]
        if prices_24h:
            high_24h = max(prices_24h)
            low_24h = min(prices_24h)
            range_24h = high_24h - low_24h
        else:
            high_24h = current_price
            low_24h = current_price
            range_24h = 0.0

        # Compute simple moving averages
        def compute_sma(lookback_hours: int) -> float:
            cutoff = now - timedelta(hours=lookback_hours)
            relevant_prices = [p for ts, p in price_history if ts >= cutoff]
            if not relevant_prices:
                return current_price
            return sum(relevant_prices) / len(relevant_prices)

        sma_1h = compute_sma(1)
        sma_24h = compute_sma(24)
        price_vs_sma_24h = (current_price - sma_24h) / sma_24h if sma_24h > 0 else 0.0

        return MarketPriceData(
            token_id=token_id,
            market_id=market_id or "",
            price_points=price_history,
            current_price=current_price,
            price_1h_ago=price_1h_ago,
            price_4h_ago=price_4h_ago,
            price_24h_ago=price_24h_ago,
            price_7d_ago=price_7d_ago,
            momentum_1h=momentum_1h,
            momentum_4h=momentum_4h,
            momentum_24h=momentum_24h,
            momentum_7d=momentum_7d,
            volatility_1h=volatility_1h,
            volatility_4h=volatility_4h,
            volatility_24h=volatility_24h,
            volatility_7d=volatility_7d,
            high_24h=high_24h,
            low_24h=low_24h,
            range_24h=range_24h,
            sma_1h=sma_1h,
            sma_24h=sma_24h,
            price_vs_sma_24h=price_vs_sma_24h,
            data_points_available=len(price_history),
            oldest_data_timestamp=oldest_ts,
            newest_data_timestamp=newest_ts,
        )

    def cache_computed_features(self, data: MarketPriceData) -> None:
        """Cache computed features to database for fast retrieval.

        Args:
            data: Computed market price data.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO computed_features
            (token_id, market_id, features, computed_at, data_points, oldest_timestamp, newest_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            data.token_id,
            data.market_id,
            json.dumps(data.to_features_dict()),
            datetime.utcnow().isoformat(),
            data.data_points_available,
            data.oldest_data_timestamp.isoformat() if data.oldest_data_timestamp else None,
            data.newest_data_timestamp.isoformat() if data.newest_data_timestamp else None,
        ))

        conn.commit()
        conn.close()

    def get_cached_features(self, token_id: str) -> Optional[dict[str, Any]]:
        """Get cached computed features for a token.

        Args:
            token_id: Token ID.

        Returns:
            Dictionary of features or None if not cached.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT features FROM computed_features WHERE token_id = ?",
            (token_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    async def collect_all_markets(
        self,
        max_markets: int = 5000,
        fidelity: int = 60,
        include_resolved: bool = True,
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        """Collect historical price data for all available markets.

        This is the main entry point for bulk data collection.

        Args:
            max_markets: Maximum number of markets to process.
            fidelity: Data resolution in minutes (default: 60).
            include_resolved: Whether to include resolved markets.
            progress_callback: Optional callback for progress updates.

        Returns:
            Statistics dictionary about the collection.
        """
        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "markets_fetched": 0,
            "tokens_processed": 0,
            "price_points_stored": 0,
            "features_computed": 0,
            "errors": [],
        }

        logger.info(f"Starting historical price collection for up to {max_markets} markets")

        # Fetch markets from Gamma API
        async with GammaClient() as gamma:
            # Fetch both active and closed markets
            markets, diagnostics = await gamma.list_all_markets(
                active=None if include_resolved else True,
                closed=include_resolved,
                batch_size=500,
                max_markets=max_markets,
            )
            stats["markets_fetched"] = len(markets)
            logger.info(f"Fetched {len(markets)} markets from Gamma API")

        if progress_callback:
            progress_callback(f"Fetched {len(markets)} markets, collecting price history...")

        # Collect price history for each token
        async with CLOBClient() as clob:
            total_tokens = sum(len(m.outcomes) for m in markets)
            processed = 0

            for market in markets:
                for outcome in market.outcomes:
                    if not outcome.token_id:
                        continue

                    try:
                        # Fetch price history
                        history = await self.fetch_price_history_for_token(
                            clob,
                            outcome.token_id,
                            market.condition_id,
                            fidelity=fidelity,
                        )

                        if history and history.points:
                            # Store raw price history
                            stored = self.store_price_history(
                                outcome.token_id,
                                market.condition_id,
                                history,
                            )
                            stats["price_points_stored"] += stored

                            # Convert to internal format and compute features
                            price_history = [
                                (p.timestamp, p.price) for p in history.points
                            ]
                            features = self.compute_features_from_history(
                                outcome.token_id,
                                market.condition_id,
                                price_history,
                            )

                            if features:
                                self.cache_computed_features(features)
                                stats["features_computed"] += 1

                        stats["tokens_processed"] += 1
                        processed += 1

                        if progress_callback and processed % 100 == 0:
                            progress_callback(
                                f"Processed {processed}/{total_tokens} tokens, "
                                f"{stats['price_points_stored']} points stored"
                            )

                    except Exception as e:
                        stats["errors"].append(f"{outcome.token_id[:12]}: {str(e)}")
                        logger.warning(f"Error processing token {outcome.token_id[:12]}: {e}")

        stats["completed_at"] = datetime.utcnow().isoformat()

        logger.info(
            f"Historical price collection complete: "
            f"{stats['tokens_processed']} tokens, "
            f"{stats['price_points_stored']} points, "
            f"{stats['features_computed']} features computed"
        )

        return stats

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about stored price data.

        Returns:
            Dictionary of statistics.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM price_history_raw")
        total_points = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT token_id) FROM price_history_raw")
        unique_tokens = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM computed_features")
        cached_features = cursor.fetchone()[0]

        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM price_history_raw
        """)
        row = cursor.fetchone()
        oldest = row[0] if row else None
        newest = row[1] if row else None

        conn.close()

        # Calculate date range
        days_covered = 0
        if oldest and newest:
            try:
                oldest_dt = datetime.fromisoformat(oldest)
                newest_dt = datetime.fromisoformat(newest)
                days_covered = (newest_dt - oldest_dt).days
            except:
                pass

        return {
            "total_price_points": total_points,
            "unique_tokens": unique_tokens,
            "cached_features": cached_features,
            "oldest_data": oldest,
            "newest_data": newest,
            "days_covered": days_covered,
            "avg_points_per_token": total_points / unique_tokens if unique_tokens > 0 else 0,
        }


# Convenience function for one-off collection
async def collect_historical_prices(
    max_markets: int = 5000,
    output_path: str = "data/historical_prices.db",
    fidelity: int = 60,
) -> dict[str, Any]:
    """Convenience function to collect historical price data.

    Args:
        max_markets: Maximum markets to fetch.
        output_path: Database path.
        fidelity: Data resolution in minutes.

    Returns:
        Statistics dictionary.
    """
    collector = HistoricalPriceCollector(db_path=output_path)
    return await collector.collect_all_markets(
        max_markets=max_markets,
        fidelity=fidelity,
    )
