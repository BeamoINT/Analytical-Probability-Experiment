"""Comprehensive Data Collection Scheduler for ML Training.

This module orchestrates all data collection activities to maximize
training data quality while respecting Polymarket API rate limits.

Key responsibilities:
1. On startup: Run bulk historical data collection
2. Continuously: Collect real-time price snapshots
3. Periodically: Update historical data for active markets
4. Combine: Merge historical and continuous data for training

Rate limits (documented):
- Gamma API: 4000 req/10s general, 300 req/10s for /markets
- CLOB API: 1500 req/10s for market data endpoints
- Data API: 200 req/10s for /trades
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from polyb0t.config import get_settings
from polyb0t.data.clob_client import CLOBClient
from polyb0t.data.gamma_client import GammaClient
from polyb0t.data.models import Market
from polyb0t.ml.continuous_collector import ContinuousDataCollector, get_data_collector
from polyb0t.ml.historical_fetcher import HistoricalDataFetcher
from polyb0t.ml.historical_price_collector import HistoricalPriceCollector

logger = logging.getLogger(__name__)


# Rate limit configuration (conservative to avoid throttling)
GAMMA_MARKETS_REQ_PER_SEC = 25  # 300/10s documented, use 25/s
CLOB_PRICE_HISTORY_REQ_PER_SEC = 100  # 1500/10s documented, use 100/s
CLOB_ORDERBOOK_REQ_PER_SEC = 100  # 1500/10s documented, use 100/s


class ComprehensiveDataCollector:
    """Orchestrates all data collection for ML training.

    This class manages:
    1. Historical data collection (one-time bulk fetch on startup)
    2. Continuous data collection (real-time price snapshots)
    3. Price history updates (periodic refresh)
    4. Data combination for training
    """

    def __init__(
        self,
        historical_db_path: str = "data/historical_training.db",
        continuous_db_path: str = "data/ai_training.db",
        price_history_db_path: str = "data/historical_prices.db",
    ):
        """Initialize the comprehensive data collector.

        Args:
            historical_db_path: Path for historical training data.
            continuous_db_path: Path for continuous training data.
            price_history_db_path: Path for price timeseries data.
        """
        self.settings = get_settings()

        # Initialize component collectors
        self.historical_fetcher = HistoricalDataFetcher(
            output_path=historical_db_path,
            price_db_path=price_history_db_path,
            fetch_price_history=True,
        )
        self.continuous_collector = get_data_collector(continuous_db_path)
        self.price_collector = HistoricalPriceCollector(
            db_path=price_history_db_path,
            requests_per_second=CLOB_PRICE_HISTORY_REQ_PER_SEC,
        )

        # State tracking
        self._historical_collection_done = False
        self._last_price_refresh: Optional[datetime] = None
        self._running = False

    async def run_initial_collection(
        self,
        days: int = 365,
        max_markets: int = 50000,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        """Run initial historical data collection.

        This should be called once on startup to fetch bulk historical data.
        Maximizes data collection while respecting API rate limits.

        Args:
            days: Number of days of historical data to fetch.
            max_markets: Maximum markets to process.
            progress_callback: Optional callback for progress updates.

        Returns:
            Statistics about the collection.
        """
        stats = {
            "phase": "initial_collection",
            "started_at": datetime.utcnow().isoformat(),
            "historical_stats": {},
            "price_history_stats": {},
            "errors": [],
        }

        logger.info(
            f"Starting initial comprehensive data collection: "
            f"{days} days, {max_markets} max markets"
        )

        if progress_callback:
            progress_callback("Starting historical data collection...")

        try:
            # Phase 1: Fetch historical resolved markets with price history
            historical_stats = await self.historical_fetcher.fetch_and_save(
                days=days,
                max_markets=max_markets,
                progress_callback=progress_callback,
                fidelity=60,  # Hourly resolution for historical data
            )
            stats["historical_stats"] = historical_stats

            # Phase 2: Collect price history for active markets (for continuous training)
            if progress_callback:
                progress_callback("Collecting price history for active markets...")

            price_stats = await self._collect_active_market_prices(
                max_markets=min(5000, max_markets),  # Cap active markets
                fidelity=15,  # 15-minute resolution for active markets
                progress_callback=progress_callback,
            )
            stats["price_history_stats"] = price_stats

            self._historical_collection_done = True
            self._last_price_refresh = datetime.utcnow()

            stats["completed_at"] = datetime.utcnow().isoformat()

            logger.info(
                f"Initial collection complete: "
                f"{historical_stats.get('examples_saved', 0)} historical examples, "
                f"{price_stats.get('tokens_processed', 0)} active market tokens"
            )

        except Exception as e:
            logger.error(f"Initial collection failed: {e}")
            stats["errors"].append(str(e))
            stats["failed_at"] = datetime.utcnow().isoformat()

        return stats

    async def _collect_active_market_prices(
        self,
        max_markets: int = 5000,
        fidelity: int = 15,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        """Collect price history for currently active markets.

        Args:
            max_markets: Maximum active markets to process.
            fidelity: Data resolution in minutes.
            progress_callback: Optional progress callback.

        Returns:
            Statistics about collection.
        """
        stats = {
            "markets_fetched": 0,
            "tokens_processed": 0,
            "price_points_stored": 0,
            "features_computed": 0,
        }

        # Fetch active markets
        async with GammaClient() as gamma:
            markets, _ = await gamma.list_all_markets(
                active=True,
                closed=False,
                batch_size=500,
                max_markets=max_markets,
            )
            stats["markets_fetched"] = len(markets)

        if progress_callback:
            progress_callback(f"Fetched {len(markets)} active markets, collecting prices...")

        # Collect price history for each token
        async with CLOBClient() as clob:
            total_tokens = sum(len(m.outcomes) for m in markets)
            processed = 0

            for market in markets:
                for outcome in market.outcomes:
                    if not outcome.token_id:
                        continue

                    try:
                        history = await self.price_collector.fetch_price_history_for_token(
                            clob,
                            outcome.token_id,
                            market.condition_id,
                            fidelity=fidelity,
                        )

                        if history and history.points:
                            stored = self.price_collector.store_price_history(
                                outcome.token_id,
                                market.condition_id,
                                history,
                            )
                            stats["price_points_stored"] += stored

                            # Compute and cache features
                            price_history = [(p.timestamp, p.price) for p in history.points]
                            features = self.price_collector.compute_features_from_history(
                                outcome.token_id,
                                market.condition_id,
                                price_history,
                            )
                            if features:
                                self.price_collector.cache_computed_features(features)
                                stats["features_computed"] += 1

                        stats["tokens_processed"] += 1
                        processed += 1

                        if progress_callback and processed % 100 == 0:
                            progress_callback(
                                f"Active markets: {processed}/{total_tokens} tokens, "
                                f"{stats['price_points_stored']} points"
                            )

                    except Exception as e:
                        logger.debug(f"Error collecting prices for {outcome.token_id[:12]}: {e}")
                        processed += 1

        return stats

    async def refresh_price_history(
        self,
        markets: list[Market],
        fidelity: int = 15,
    ) -> dict[str, Any]:
        """Refresh price history for tracked markets.

        Should be called periodically (e.g., every hour) to keep
        price history up to date for active markets.

        Args:
            markets: List of markets to refresh.
            fidelity: Data resolution in minutes.

        Returns:
            Statistics about the refresh.
        """
        stats = {
            "tokens_refreshed": 0,
            "new_points_stored": 0,
        }

        async with CLOBClient() as clob:
            for market in markets:
                for outcome in market.outcomes:
                    if not outcome.token_id:
                        continue

                    try:
                        # Fetch just the last hour of data for incremental update
                        now = datetime.utcnow()
                        start_ts = int((now - timedelta(hours=2)).timestamp())
                        end_ts = int(now.timestamp())

                        history = await self.price_collector.fetch_price_history_for_token(
                            clob,
                            outcome.token_id,
                            market.condition_id,
                            fidelity=fidelity,
                        )

                        if history and history.points:
                            stored = self.price_collector.store_price_history(
                                outcome.token_id,
                                market.condition_id,
                                history,
                            )
                            stats["new_points_stored"] += stored

                            # Update cached features
                            all_history = self.price_collector.get_stored_price_history(
                                outcome.token_id, hours=168  # Last week
                            )
                            if all_history:
                                features = self.price_collector.compute_features_from_history(
                                    outcome.token_id,
                                    market.condition_id,
                                    all_history,
                                )
                                if features:
                                    self.price_collector.cache_computed_features(features)

                        stats["tokens_refreshed"] += 1

                    except Exception as e:
                        logger.debug(f"Error refreshing {outcome.token_id[:12]}: {e}")

        self._last_price_refresh = datetime.utcnow()
        return stats

    def get_combined_training_data(
        self,
        min_data_points: int = 10,
        recent_weight: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Get combined training data from all sources.

        Merges historical and continuous training data, applying
        sample weighting for better model training.

        Args:
            min_data_points: Minimum price history points required.
            recent_weight: Weight multiplier for recent examples.

        Returns:
            Combined list of training examples.
        """
        all_data = []

        # Get continuous training data (labeled examples)
        continuous_data = self.continuous_collector.get_training_data(only_labeled=True)
        logger.info(f"Loaded {len(continuous_data)} continuous training examples")

        # Enhance continuous data with cached price features
        for example in continuous_data:
            token_id = example.get("token_id")
            if token_id:
                cached_features = self.price_collector.get_cached_features(token_id)
                if cached_features:
                    # Add historical price features to example
                    for key, value in cached_features.items():
                        if key not in example:
                            example[key] = value

            # Add sample weight (recent data weighted higher)
            created_at = example.get("created_at")
            if created_at:
                try:
                    created = datetime.fromisoformat(created_at)
                    days_ago = (datetime.utcnow() - created).days
                    # Recent examples (< 7 days) get higher weight
                    weight = recent_weight if days_ago < 7 else 1.0
                    example["_sample_weight"] = weight
                except:
                    example["_sample_weight"] = 1.0
            else:
                example["_sample_weight"] = 1.0

            all_data.append(example)

        # Get historical training data
        historical_stats = self.historical_fetcher.get_statistics()
        logger.info(
            f"Historical data stats: {historical_stats.get('total_examples', 0)} examples, "
            f"{historical_stats.get('unique_markets', 0)} markets"
        )

        # Note: Historical data is already stored in the historical database
        # The MoE trainer will load it separately to avoid memory issues

        logger.info(f"Combined training data: {len(all_data)} examples ready")
        return all_data

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about collected data.

        Returns:
            Statistics dictionary.
        """
        continuous_stats = self.continuous_collector.get_stats()
        price_stats = self.price_collector.get_statistics()
        historical_stats = self.historical_fetcher.get_statistics()

        return {
            "continuous": continuous_stats,
            "price_history": price_stats,
            "historical": historical_stats,
            "initial_collection_done": self._historical_collection_done,
            "last_price_refresh": self._last_price_refresh.isoformat() if self._last_price_refresh else None,
        }


# Singleton instance
_collector_instance: Optional[ComprehensiveDataCollector] = None


def get_comprehensive_collector(
    historical_db_path: str = "data/historical_training.db",
    continuous_db_path: str = "data/ai_training.db",
    price_history_db_path: str = "data/historical_prices.db",
) -> ComprehensiveDataCollector:
    """Get or create the singleton comprehensive data collector.

    Args:
        historical_db_path: Path for historical training data.
        continuous_db_path: Path for continuous training data.
        price_history_db_path: Path for price timeseries data.

    Returns:
        ComprehensiveDataCollector instance.
    """
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = ComprehensiveDataCollector(
            historical_db_path=historical_db_path,
            continuous_db_path=continuous_db_path,
            price_history_db_path=price_history_db_path,
        )
    return _collector_instance
