"""Historical Data Fetcher for Polymarket markets.

Fetches resolved markets from Polymarket's Gamma API and processes them
into training examples for the MoE system. Now enhanced with historical
price data collection for real momentum/volatility features.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from polyb0t.data.clob_client import CLOBClient
from polyb0t.data.gamma_client import GammaClient
from polyb0t.data.models import Market, MarketOutcome
from polyb0t.ml.category_tracker import MarketCategoryTracker
from polyb0t.ml.historical_price_collector import HistoricalPriceCollector, MarketPriceData

logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """Fetches and processes historical Polymarket data for training.

    This class fetches resolved markets from the Gamma API and converts them
    into training examples with labels based on actual market outcomes.

    Enhanced in v2 to also fetch historical price timeseries data from the
    CLOB API, enabling real momentum, volatility, and trend features.
    """

    # Features available from historical resolved markets
    # V2: Now includes historical price-derived features (~35+ features)
    HISTORICAL_FEATURES = [
        # Price features
        "outcome_price",           # Final price of this outcome (0.0 or 1.0 for resolved)
        "initial_price",           # Approximate initial price (from metadata if available)

        # Volume/Liquidity features
        "total_volume",            # Total market volume
        "liquidity",               # Market liquidity
        "volume_per_day",          # Volume / market age

        # Time features
        "market_age_days",         # How long the market existed
        "days_to_resolution",      # Days from creation to resolution
        "hour_of_day",             # Hour when market was created (0-23)
        "day_of_week",             # Day of week (0=Monday, 6=Sunday)
        "is_weekend",              # 1 if weekend, 0 otherwise

        # Market structure features
        "num_outcomes",            # Number of outcomes (usually 2)
        "has_description",         # 1 if market has description, 0 otherwise
        "question_length",         # Length of market question

        # Category (one-hot encoded during training)
        "category",                # Market category string

        # === NEW V2: Historical price-derived features ===
        # These are computed from CLOB /prices-history endpoint
        "historical_price",        # Current/last known price
        "historical_price_1h_ago",
        "historical_price_4h_ago",
        "historical_price_24h_ago",
        "historical_price_7d_ago",
        "historical_momentum_1h",
        "historical_momentum_4h",
        "historical_momentum_24h",
        "historical_momentum_7d",
        "historical_volatility_1h",
        "historical_volatility_4h",
        "historical_volatility_24h",
        "historical_volatility_7d",
        "historical_high_24h",
        "historical_low_24h",
        "historical_range_24h",
        "historical_sma_1h",
        "historical_sma_24h",
        "historical_price_vs_sma_24h",
        "historical_data_points",
    ]

    def __init__(
        self,
        output_path: str = "data/historical_training.db",
        price_db_path: str = "data/historical_prices.db",
        fetch_price_history: bool = True,
    ):
        """Initialize the historical data fetcher.

        Args:
            output_path: Path to save the training database.
            price_db_path: Path for historical price data database.
            fetch_price_history: Whether to fetch historical price timeseries.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.category_tracker = MarketCategoryTracker()
        self.fetch_price_history = fetch_price_history
        self.price_collector = HistoricalPriceCollector(db_path=price_db_path)
        self._price_features_cache: dict[str, dict[str, Any]] = {}

    async def fetch_resolved_markets(
        self,
        days: int = 365,
        max_markets: int = 250000,
        batch_size: int = 500,
    ) -> list[Market]:
        """Fetch all resolved markets from the last N days.

        Args:
            days: Number of days of history to fetch.
            max_markets: Maximum number of markets to fetch.
            batch_size: Pagination batch size.

        Returns:
            List of resolved Market objects.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        logger.info(f"Fetching resolved markets from last {days} days (cutoff: {cutoff})")

        async with GammaClient() as gamma:
            # Fetch all closed markets with pagination
            all_markets, diagnostics = await gamma.list_all_markets(
                active=None,  # Include both active and inactive
                closed=True,  # Only closed/resolved markets
                batch_size=batch_size,
                max_markets=max_markets,
            )

            logger.info(f"Fetched {len(all_markets)} total closed markets")
            logger.debug(f"Diagnostics: {diagnostics}")

        # Filter to markets resolved within the date range
        resolved_markets = []
        skipped_no_date = 0
        skipped_too_old = 0

        for market in all_markets:
            if market.end_date is None:
                skipped_no_date += 1
                continue

            end_date = market.end_date
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

            if end_date < cutoff:
                skipped_too_old += 1
                continue

            resolved_markets.append(market)

        logger.info(
            f"Filtered to {len(resolved_markets)} markets resolved in last {days} days "
            f"(skipped: {skipped_no_date} no date, {skipped_too_old} too old)"
        )

        return resolved_markets

    def determine_winner(self, market: Market) -> Optional[str]:
        """Determine which outcome won for a resolved market.

        For resolved markets, the winning outcome has price ≈ 1.0,
        losing outcomes have price ≈ 0.0.

        Args:
            market: Resolved market.

        Returns:
            Token ID of the winning outcome, or None if cannot determine.
        """
        if not market.outcomes:
            return None

        winning_outcome = None
        winning_price = 0.0

        for outcome in market.outcomes:
            price = outcome.price or 0.0
            # Winner has price close to 1.0
            if price > 0.9:
                winning_outcome = outcome.token_id
                winning_price = price
            elif price > winning_price and winning_outcome is None:
                # Fallback: highest price wins
                winning_outcome = outcome.token_id
                winning_price = price

        return winning_outcome

    def compute_features(
        self,
        market: Market,
        outcome: MarketOutcome,
        winning_token_id: Optional[str],
    ) -> dict[str, Any]:
        """Compute feature vector for a market outcome.

        V2: Now includes historical price-derived features when available.

        Args:
            market: The market.
            outcome: The specific outcome to compute features for.
            winning_token_id: Token ID of the winning outcome.

        Returns:
            Dictionary of feature name -> value.
        """
        features = {}

        # Price features
        # IMPORTANT: For resolved markets, outcome.price is the POST-resolution price
        # (0.0 for losers, 1.0 for winners), which would leak the label.
        # We use 0.5 (neutral) since we don't have the actual pre-resolution prices.
        # This forces the model to learn from other features rather than the answer.
        features["outcome_price"] = 0.5  # Neutral - actual pre-resolution price unknown
        features["initial_price"] = 0.5  # Markets typically start near 50/50

        # Volume/Liquidity features
        features["total_volume"] = market.volume or 0.0
        features["liquidity"] = market.liquidity or 0.0

        # Calculate market age and volume per day
        market_age_days = 1.0  # Default
        if market.end_date:
            # Try to estimate market age from metadata or use 30 days default
            metadata = market.metadata or {}
            created_at = metadata.get("created_at") or metadata.get("createdAt")

            if created_at:
                try:
                    if isinstance(created_at, str):
                        # Try parsing ISO format
                        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        created = created_at

                    end = market.end_date
                    if end.tzinfo is None:
                        end = end.replace(tzinfo=timezone.utc)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)

                    market_age_days = max(1.0, (end - created).total_seconds() / 86400)
                except Exception:
                    market_age_days = 30.0  # Default assumption
            else:
                market_age_days = 30.0  # Default assumption

        features["market_age_days"] = market_age_days
        features["days_to_resolution"] = market_age_days  # Same as age for resolved markets
        features["volume_per_day"] = (market.volume or 0.0) / max(1.0, market_age_days)

        # Time features (from resolution date)
        if market.end_date:
            end = market.end_date
            features["hour_of_day"] = end.hour
            features["day_of_week"] = end.weekday()
            features["is_weekend"] = 1 if end.weekday() >= 5 else 0
        else:
            features["hour_of_day"] = 12
            features["day_of_week"] = 2
            features["is_weekend"] = 0

        # Market structure features
        features["num_outcomes"] = len(market.outcomes)
        features["has_description"] = 1 if market.description else 0
        features["question_length"] = len(market.question) if market.question else 0

        # Category - categorize_market returns (category, confidence) tuple
        category_result = self.category_tracker.categorize_market(
            market.question or "",
            market.description or "",
        )
        # Extract just the category string, not the confidence
        if isinstance(category_result, tuple):
            features["category"] = category_result[0]
        else:
            features["category"] = category_result

        # === V2: Historical price-derived features ===
        # These provide the crucial momentum/volatility/trend signals
        token_id = outcome.token_id
        if token_id and token_id in self._price_features_cache:
            price_features = self._price_features_cache[token_id]
            features.update(price_features)
        else:
            # Add default values for historical price features
            features.update(self._default_historical_price_features())

        return features

    def _default_historical_price_features(self) -> dict[str, Any]:
        """Return default values for historical price features when not available."""
        return {
            "historical_price": 0.5,
            "historical_price_1h_ago": 0.5,
            "historical_price_4h_ago": 0.5,
            "historical_price_24h_ago": 0.5,
            "historical_price_7d_ago": 0.5,
            "historical_momentum_1h": 0.0,
            "historical_momentum_4h": 0.0,
            "historical_momentum_24h": 0.0,
            "historical_momentum_7d": 0.0,
            "historical_volatility_1h": 0.0,
            "historical_volatility_4h": 0.0,
            "historical_volatility_24h": 0.0,
            "historical_volatility_7d": 0.0,
            "historical_high_24h": 0.5,
            "historical_low_24h": 0.5,
            "historical_range_24h": 0.0,
            "historical_sma_1h": 0.5,
            "historical_sma_24h": 0.5,
            "historical_price_vs_sma_24h": 0.0,
            "historical_data_points": 0,
        }

    def extract_training_examples(
        self,
        market: Market,
    ) -> list[dict[str, Any]]:
        """Convert a resolved market to training examples with labels.

        Creates one example per outcome, with label indicating if that
        outcome was the winner (profitable to hold).

        Args:
            market: Resolved market.

        Returns:
            List of training example dictionaries.
        """
        winning_token_id = self.determine_winner(market)

        if winning_token_id is None:
            logger.debug(f"Could not determine winner for market {market.condition_id}")
            return []

        examples = []

        for outcome in market.outcomes:
            features = self.compute_features(market, outcome, winning_token_id)

            # Label: 1 if this outcome won (profitable), 0 otherwise
            label = 1 if outcome.token_id == winning_token_id else 0

            example = {
                "token_id": outcome.token_id,
                "market_id": market.condition_id,
                "market_title": market.question or "",
                "outcome_name": outcome.outcome,
                "category": features["category"],
                "features": features,
                "label": label,
                "winning_token_id": winning_token_id,
                "resolution_date": market.end_date.isoformat() if market.end_date else None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            examples.append(example)

        return examples

    def _ensure_db(self, db_path: Path) -> None:
        """Ensure the database schema exists.

        Args:
            db_path: Path to the database.
        """
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_title TEXT,
                outcome_name TEXT,
                category TEXT,
                features TEXT NOT NULL,
                label INTEGER NOT NULL,
                winning_token_id TEXT,
                resolution_date TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(token_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_category
            ON historical_examples(category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_label
            ON historical_examples(label)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_market
            ON historical_examples(market_id)
        """)

        conn.commit()
        conn.close()

    def save_to_db(
        self,
        examples: list[dict[str, Any]],
        db_path: Optional[Path] = None,
    ) -> int:
        """Save training examples to SQLite database.

        Args:
            examples: List of training example dictionaries.
            db_path: Path to database (uses self.output_path if not provided).

        Returns:
            Number of examples saved.
        """
        if db_path is None:
            db_path = self.output_path

        self._ensure_db(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        saved = 0
        skipped = 0

        for example in examples:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO historical_examples
                    (token_id, market_id, market_title, outcome_name, category,
                     features, label, winning_token_id, resolution_date, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    example["token_id"],
                    example["market_id"],
                    example["market_title"],
                    example["outcome_name"],
                    example["category"],
                    json.dumps(example["features"]),
                    example["label"],
                    example["winning_token_id"],
                    example["resolution_date"],
                    example["created_at"],
                ))

                if cursor.rowcount > 0:
                    saved += 1
                else:
                    skipped += 1

            except Exception as e:
                logger.warning(f"Failed to save example {example.get('token_id')}: {e}")
                skipped += 1

        conn.commit()
        conn.close()

        logger.info(f"Saved {saved} examples to {db_path} (skipped {skipped} duplicates)")

        return saved

    async def fetch_and_save(
        self,
        days: int = 365,
        max_markets: int = 50000,
        progress_callback: Optional[Callable[[str], None]] = None,
        fidelity: int = 60,
    ) -> dict[str, Any]:
        """Fetch historical markets and save as training examples.

        This is the main entry point for historical data collection.
        V2: Now also fetches historical price timeseries data from CLOB API
        to provide real momentum/volatility features.

        Args:
            days: Number of days of history to fetch.
            max_markets: Maximum number of markets to fetch.
            progress_callback: Optional callback for progress updates.
            fidelity: Price history resolution in minutes (default: 60 = hourly).

        Returns:
            Dictionary with statistics about the fetch operation.
        """
        stats = {
            "days": days,
            "max_markets": max_markets,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "markets_fetched": 0,
            "examples_created": 0,
            "examples_saved": 0,
            "price_history_tokens": 0,
            "price_history_points": 0,
            "categories": {},
            "errors": [],
        }

        try:
            # Fetch resolved markets
            logger.info(f"Starting historical data fetch: {days} days, max {max_markets} markets")
            markets = await self.fetch_resolved_markets(
                days=days,
                max_markets=max_markets,
            )
            stats["markets_fetched"] = len(markets)

            if progress_callback:
                progress_callback(f"Fetched {len(markets)} markets, collecting price history...")

            # === V2: Fetch historical price timeseries data ===
            if self.fetch_price_history:
                await self._collect_price_history_for_markets(
                    markets,
                    fidelity=fidelity,
                    progress_callback=progress_callback,
                    stats=stats,
                )

            if progress_callback:
                progress_callback(f"Processing {len(markets)} markets into training examples...")

            # Convert to training examples
            all_examples = []
            for i, market in enumerate(markets):
                try:
                    examples = self.extract_training_examples(market)
                    all_examples.extend(examples)

                    # Track categories
                    for ex in examples:
                        cat = ex["category"]
                        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

                except Exception as e:
                    logger.warning(f"Error processing market {market.condition_id}: {e}")
                    stats["errors"].append(str(e))

                if progress_callback and (i + 1) % 1000 == 0:
                    progress_callback(f"Processed {i + 1}/{len(markets)} markets...")

            stats["examples_created"] = len(all_examples)

            # Save to database
            if all_examples:
                saved = self.save_to_db(all_examples)
                stats["examples_saved"] = saved

            stats["completed_at"] = datetime.now(timezone.utc).isoformat()

            logger.info(
                f"Historical fetch complete: {stats['markets_fetched']} markets, "
                f"{stats['examples_created']} examples, {stats['examples_saved']} saved, "
                f"{stats['price_history_tokens']} tokens with price history"
            )

        except Exception as e:
            logger.error(f"Historical fetch failed: {e}")
            stats["errors"].append(str(e))
            stats["failed_at"] = datetime.now(timezone.utc).isoformat()

        return stats

    async def _collect_price_history_for_markets(
        self,
        markets: list[Market],
        fidelity: int,
        progress_callback: Optional[Callable[[str], None]],
        stats: dict[str, Any],
    ) -> None:
        """Collect historical price timeseries data for market tokens.

        Args:
            markets: List of markets to collect price history for.
            fidelity: Data resolution in minutes.
            progress_callback: Optional progress callback.
            stats: Stats dictionary to update.
        """
        total_tokens = sum(len(m.outcomes) for m in markets)
        processed = 0

        logger.info(f"Collecting price history for {total_tokens} tokens from {len(markets)} markets")

        async with CLOBClient() as clob:
            for market in markets:
                for outcome in market.outcomes:
                    if not outcome.token_id:
                        continue

                    try:
                        # Fetch price history from CLOB API
                        history = await self.price_collector.fetch_price_history_for_token(
                            clob,
                            outcome.token_id,
                            market.condition_id,
                            fidelity=fidelity,
                        )

                        if history and history.points:
                            # Store raw history
                            stored = self.price_collector.store_price_history(
                                outcome.token_id,
                                market.condition_id,
                                history,
                            )
                            stats["price_history_points"] += stored

                            # Compute features and cache them
                            price_history = [(p.timestamp, p.price) for p in history.points]
                            features_data = self.price_collector.compute_features_from_history(
                                outcome.token_id,
                                market.condition_id,
                                price_history,
                            )

                            if features_data:
                                # Cache in memory for this run
                                self._price_features_cache[outcome.token_id] = features_data.to_features_dict()
                                # Also persist to disk
                                self.price_collector.cache_computed_features(features_data)
                                stats["price_history_tokens"] += 1

                        processed += 1

                        if progress_callback and processed % 200 == 0:
                            progress_callback(
                                f"Collected price history: {processed}/{total_tokens} tokens, "
                                f"{stats['price_history_points']} points"
                            )

                    except Exception as e:
                        logger.debug(f"Error collecting price history for {outcome.token_id[:12]}: {e}")
                        processed += 1

        logger.info(
            f"Price history collection complete: {stats['price_history_tokens']} tokens, "
            f"{stats['price_history_points']} points"
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the stored historical data.

        Returns:
            Dictionary with database statistics.
        """
        if not self.output_path.exists():
            return {"error": "Database does not exist"}

        conn = sqlite3.connect(str(self.output_path))
        cursor = conn.cursor()

        stats = {}

        # Total examples
        cursor.execute("SELECT COUNT(*) FROM historical_examples")
        stats["total_examples"] = cursor.fetchone()[0]

        # Label distribution
        cursor.execute("""
            SELECT label, COUNT(*)
            FROM historical_examples
            GROUP BY label
        """)
        stats["label_distribution"] = {
            str(row[0]): row[1] for row in cursor.fetchall()
        }

        # Category distribution
        cursor.execute("""
            SELECT category, COUNT(*)
            FROM historical_examples
            GROUP BY category
            ORDER BY COUNT(*) DESC
        """)
        stats["category_distribution"] = {
            row[0]: row[1] for row in cursor.fetchall()
        }

        # Unique markets
        cursor.execute("SELECT COUNT(DISTINCT market_id) FROM historical_examples")
        stats["unique_markets"] = cursor.fetchone()[0]

        # Date range
        cursor.execute("""
            SELECT MIN(resolution_date), MAX(resolution_date)
            FROM historical_examples
            WHERE resolution_date IS NOT NULL
        """)
        row = cursor.fetchone()
        stats["date_range"] = {
            "earliest": row[0],
            "latest": row[1],
        }

        conn.close()

        return stats


async def fetch_historical_data(
    days: int = 365,
    output_path: str = "data/historical_training.db",
    max_markets: int = 250000,
) -> dict[str, Any]:
    """Convenience function to fetch historical data.

    Args:
        days: Number of days of history.
        output_path: Path to save database.
        max_markets: Maximum markets to fetch.

    Returns:
        Statistics dictionary.
    """
    fetcher = HistoricalDataFetcher(output_path=output_path)
    return await fetcher.fetch_and_save(
        days=days,
        max_markets=max_markets,
    )
