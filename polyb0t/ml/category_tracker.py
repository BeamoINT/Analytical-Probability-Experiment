"""Market Category Tracker for learning which market types the AI is good/bad at.

This module:
- Categorizes markets by type (politics, sports, crypto, etc.)
- Tracks prediction accuracy per category
- Provides confidence adjustments based on category performance
- Continues collecting data from avoided categories to learn
- Periodically re-evaluates if avoided categories have improved
"""

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# === MARKET CATEGORIES ===
# These are inferred from market titles and descriptions

CATEGORY_KEYWORDS = {
    "politics_us": [
        "trump", "biden", "congress", "senate", "house", "republican", "democrat",
        "gop", "dnc", "president", "election", "vote", "poll", "primary",
        "governor", "mayor", "electoral", "swing state", "red state", "blue state",
        "impeach", "scotus", "supreme court", "cabinet", "white house",
    ],
    "politics_intl": [
        "ukraine", "russia", "putin", "zelensky", "nato", "china", "xi jinping",
        "european union", "brexit", "parliament", "prime minister", "chancellor",
        "president of", "minister", "diplomat", "sanction", "treaty",
    ],
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "token", "blockchain",
        "defi", "nft", "solana", "sol", "cardano", "ada", "dogecoin", "doge",
        "binance", "coinbase", "ftx", "altcoin", "stablecoin", "usdc", "usdt",
    ],
    "sports": [
        "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball",
        "hockey", "tennis", "golf", "ufc", "mma", "boxing", "f1", "formula 1",
        "world cup", "olympics", "super bowl", "championship", "playoffs",
        "mvp", "score", "win", "match", "game", "team",
    ],
    "entertainment": [
        "movie", "film", "oscar", "emmy", "grammy", "album", "song", "artist",
        "celebrity", "tv show", "netflix", "streaming", "box office", "concert",
        "award", "nomination", "actor", "actress", "director",
    ],
    "economics": [
        "fed", "federal reserve", "interest rate", "inflation", "gdp", "recession",
        "stock", "s&p", "nasdaq", "dow jones", "treasury", "bond", "yield",
        "unemployment", "jobs report", "cpi", "ppi", "earnings", "ipo",
    ],
    "tech": [
        "apple", "google", "microsoft", "amazon", "meta", "facebook", "twitter",
        "elon musk", "tesla", "spacex", "openai", "chatgpt", "ai ", "artificial intelligence",
        "iphone", "android", "startup", "silicon valley", "tech company",
    ],
    "weather": [
        "hurricane", "tornado", "earthquake", "flood", "temperature", "weather",
        "climate", "storm", "wildfire", "drought", "snow", "heat wave",
    ],
    "science": [
        "nasa", "space", "mars", "moon", "rocket", "satellite", "asteroid",
        "vaccine", "fda", "drug", "clinical trial", "disease", "virus", "pandemic",
        "discovery", "research", "study",
    ],
    "legal": [
        "court", "judge", "trial", "verdict", "lawsuit", "settlement", "indictment",
        "conviction", "sentence", "appeal", "ruling", "legal", "attorney",
    ],
}

# Minimum samples before we trust category stats
MIN_SAMPLES_FOR_CONFIDENCE = 20

# Minimum accuracy to consider a category "good"
MIN_ACCEPTABLE_ACCURACY = 0.45  # 45% profitable accuracy

# Confidence penalty for poor-performing categories
POOR_CATEGORY_CONFIDENCE_MULTIPLIER = 0.3  # Reduce confidence by 70%

# How often to re-evaluate avoided categories (hours)
REEVALUATION_INTERVAL_HOURS = 24


@dataclass
class CategoryStats:
    """Statistics for a market category."""
    category: str
    total_predictions: int = 0
    correct_predictions: int = 0
    profitable_predictions: int = 0
    total_pnl: float = 0.0  # Simulated P&L
    last_updated: str = ""
    is_avoided: bool = False
    avoided_since: Optional[str] = None
    last_reevaluation: Optional[str] = None
    
    @property
    def accuracy(self) -> float:
        """Directional accuracy."""
        if self.total_predictions == 0:
            return 0.5  # Unknown = assume random
        return self.correct_predictions / self.total_predictions
    
    @property
    def profitable_accuracy(self) -> float:
        """Profitable accuracy (after spread)."""
        if self.total_predictions == 0:
            return 0.5
        return self.profitable_predictions / self.total_predictions
    
    @property
    def avg_pnl(self) -> float:
        """Average P&L per prediction."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_pnl / self.total_predictions
    
    @property
    def confidence_multiplier(self) -> float:
        """Confidence multiplier based on performance."""
        if self.total_predictions < MIN_SAMPLES_FOR_CONFIDENCE:
            return 0.7  # Unknown category = reduce confidence
        
        if self.is_avoided:
            return 0.0  # Avoided = no trades
        
        if self.profitable_accuracy < MIN_ACCEPTABLE_ACCURACY:
            return POOR_CATEGORY_CONFIDENCE_MULTIPLIER
        
        # Scale confidence with accuracy (0.45 -> 0.7, 0.55 -> 1.0, 0.65 -> 1.3)
        return 0.7 + (self.profitable_accuracy - 0.45) * 3
    
    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "profitable_predictions": self.profitable_predictions,
            "accuracy": round(self.accuracy, 3),
            "profitable_accuracy": round(self.profitable_accuracy, 3),
            "avg_pnl": round(self.avg_pnl, 4),
            "confidence_multiplier": round(self.confidence_multiplier, 2),
            "is_avoided": self.is_avoided,
            "avoided_since": self.avoided_since,
            "last_reevaluation": self.last_reevaluation,
            "last_updated": self.last_updated,
        }


class MarketCategoryTracker:
    """Tracks and learns which market categories the AI performs well on."""
    
    def __init__(self, db_path: str = "data/category_stats.db"):
        """Initialize the category tracker.
        
        Args:
            db_path: Path to SQLite database for persistence.
        """
        self.db_path = db_path
        self._ensure_db()
        self._category_cache: Dict[str, CategoryStats] = {}
        self._load_stats()
    
    def _ensure_db(self) -> None:
        """Create database tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Category statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS category_stats (
                category TEXT PRIMARY KEY,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                profitable_predictions INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                is_avoided INTEGER DEFAULT 0,
                avoided_since TEXT,
                last_reevaluation TEXT,
                last_updated TEXT
            )
        """)
        
        # Individual prediction tracking (for detailed analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                token_id TEXT,
                category TEXT,
                predicted_direction INTEGER,  -- 1 = up, -1 = down, 0 = flat
                actual_direction INTEGER,
                predicted_change REAL,
                actual_change REAL,
                was_correct INTEGER,
                was_profitable INTEGER,
                pnl REAL,
                timestamp TEXT,
                UNIQUE(market_id, token_id, timestamp)
            )
        """)
        
        # Market category mappings (cache)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_categories (
                market_id TEXT PRIMARY KEY,
                category TEXT,
                confidence REAL,  -- How confident we are in the categorization
                title TEXT,
                created_at TEXT
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_category ON predictions(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        
        conn.commit()
        conn.close()
    
    def _load_stats(self) -> None:
        """Load category stats from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT category, total_predictions, correct_predictions, profitable_predictions,
                   total_pnl, is_avoided, avoided_since, last_reevaluation, last_updated
            FROM category_stats
        """)
        
        for row in cursor.fetchall():
            stats = CategoryStats(
                category=row[0],
                total_predictions=row[1],
                correct_predictions=row[2],
                profitable_predictions=row[3],
                total_pnl=row[4],
                is_avoided=bool(row[5]),
                avoided_since=row[6],
                last_reevaluation=row[7],
                last_updated=row[8],
            )
            self._category_cache[row[0]] = stats
        
        conn.close()
        logger.info(f"Loaded {len(self._category_cache)} category stats")
    
    def _save_stats(self, category: str) -> None:
        """Save category stats to database."""
        if category not in self._category_cache:
            return
        
        stats = self._category_cache[category]
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO category_stats
            (category, total_predictions, correct_predictions, profitable_predictions,
             total_pnl, is_avoided, avoided_since, last_reevaluation, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stats.category,
            stats.total_predictions,
            stats.correct_predictions,
            stats.profitable_predictions,
            stats.total_pnl,
            1 if stats.is_avoided else 0,
            stats.avoided_since,
            stats.last_reevaluation,
            datetime.utcnow().isoformat(),
        ))
        
        conn.commit()
        conn.close()
    
    def categorize_market(
        self,
        market_id: str,
        title: str,
        description: str = "",
        tags: List[str] = None,
    ) -> Tuple[str, float]:
        """Categorize a market based on its title and description.
        
        Args:
            market_id: Unique market identifier.
            title: Market title.
            description: Market description.
            tags: Optional list of tags.
            
        Returns:
            Tuple of (category, confidence).
        """
        # Check cache first
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT category, confidence FROM market_categories WHERE market_id = ?", (market_id,))
        row = cursor.fetchone()
        if row:
            conn.close()
            return row[0], row[1]
        
        # Combine text for matching
        text = f"{title} {description} {' '.join(tags or [])}".lower()
        
        # Count keyword matches per category
        category_scores: Dict[str, int] = defaultdict(int)
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    category_scores[category] += 1
        
        # Find best category
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            category = best_category[0]
            # Confidence based on number of matching keywords
            confidence = min(1.0, best_category[1] / 3.0)  # 3+ keywords = 100% confidence
        else:
            category = "other"
            confidence = 0.5
        
        # Cache the categorization
        cursor.execute("""
            INSERT OR REPLACE INTO market_categories (market_id, category, confidence, title, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (market_id, category, confidence, title[:500], datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        
        return category, confidence
    
    def record_prediction(
        self,
        market_id: str,
        token_id: str,
        category: str,
        predicted_change: float,
        actual_change: float,
        spread_cost: float = 0.02,
    ) -> None:
        """Record a prediction outcome for learning.
        
        Args:
            market_id: Market identifier.
            token_id: Token identifier.
            category: Market category.
            predicted_change: Predicted price change.
            actual_change: Actual price change.
            spread_cost: Spread cost (for profitability calculation).
        """
        # Determine directions
        predicted_dir = 1 if predicted_change > 0.01 else (-1 if predicted_change < -0.01 else 0)
        actual_dir = 1 if actual_change > 0.01 else (-1 if actual_change < -0.01 else 0)
        
        # Was the prediction correct?
        was_correct = (predicted_dir == actual_dir) if predicted_dir != 0 else False
        
        # Would it have been profitable (after spread)?
        if predicted_dir != 0:
            net_gain = abs(actual_change) - spread_cost if predicted_dir == actual_dir else -(abs(actual_change) + spread_cost)
            was_profitable = net_gain > 0
            pnl = net_gain
        else:
            was_profitable = False
            pnl = 0.0
        
        # Record to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO predictions
                (market_id, token_id, category, predicted_direction, actual_direction,
                 predicted_change, actual_change, was_correct, was_profitable, pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market_id, token_id, category, predicted_dir, actual_dir,
                predicted_change, actual_change,
                1 if was_correct else 0,
                1 if was_profitable else 0,
                pnl,
                datetime.utcnow().isoformat(),
            ))
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record prediction: {e}")
        finally:
            conn.close()
        
        # Update category stats
        if category not in self._category_cache:
            self._category_cache[category] = CategoryStats(category=category)
        
        stats = self._category_cache[category]
        stats.total_predictions += 1
        if was_correct:
            stats.correct_predictions += 1
        if was_profitable:
            stats.profitable_predictions += 1
        stats.total_pnl += pnl
        stats.last_updated = datetime.utcnow().isoformat()
        
        # Check if category should be avoided
        self._check_avoid_status(category)
        
        # Save stats
        self._save_stats(category)
    
    def _check_avoid_status(self, category: str) -> None:
        """Check if a category should be avoided or un-avoided."""
        if category not in self._category_cache:
            return
        
        stats = self._category_cache[category]
        
        # Need minimum samples
        if stats.total_predictions < MIN_SAMPLES_FOR_CONFIDENCE:
            return
        
        # Check if should avoid
        if stats.profitable_accuracy < MIN_ACCEPTABLE_ACCURACY - 0.05:  # Below 40%
            if not stats.is_avoided:
                stats.is_avoided = True
                stats.avoided_since = datetime.utcnow().isoformat()
                logger.warning(
                    f"Category '{category}' marked as AVOIDED "
                    f"(profitable_acc={stats.profitable_accuracy:.1%})"
                )
        
        # Check if should un-avoid (after re-evaluation period)
        elif stats.is_avoided:
            if stats.last_reevaluation:
                last_eval = datetime.fromisoformat(stats.last_reevaluation)
                if (datetime.utcnow() - last_eval).total_seconds() > REEVALUATION_INTERVAL_HOURS * 3600:
                    # Time for re-evaluation
                    if stats.profitable_accuracy >= MIN_ACCEPTABLE_ACCURACY:
                        stats.is_avoided = False
                        stats.avoided_since = None
                        logger.info(
                            f"Category '{category}' UN-AVOIDED after improvement "
                            f"(profitable_acc={stats.profitable_accuracy:.1%})"
                        )
                    stats.last_reevaluation = datetime.utcnow().isoformat()
    
    def get_confidence_multiplier(self, category: str) -> float:
        """Get the confidence multiplier for a category.
        
        Args:
            category: Market category.
            
        Returns:
            Multiplier between 0.0 (avoid) and 1.3+ (high confidence).
        """
        if category not in self._category_cache:
            return 0.8  # Unknown category = slight reduction
        
        return self._category_cache[category].confidence_multiplier
    
    def should_avoid_category(self, category: str) -> bool:
        """Check if a category should be avoided.
        
        Args:
            category: Market category.
            
        Returns:
            True if the category should be avoided.
        """
        if category not in self._category_cache:
            return False
        
        return self._category_cache[category].is_avoided
    
    def get_category_stats(self, category: str) -> Optional[dict]:
        """Get stats for a specific category.
        
        Args:
            category: Market category.
            
        Returns:
            Stats dict or None.
        """
        if category not in self._category_cache:
            return None
        
        return self._category_cache[category].to_dict()
    
    def get_all_stats(self) -> Dict[str, dict]:
        """Get stats for all categories.
        
        Returns:
            Dict mapping category to stats.
        """
        return {cat: stats.to_dict() for cat, stats in self._category_cache.items()}
    
    def get_performance_summary(self) -> dict:
        """Get a summary of category performance.
        
        Returns:
            Summary dict with rankings and recommendations.
        """
        if not self._category_cache:
            return {"categories": [], "avoided": [], "best": [], "worst": []}
        
        # Get all categories with enough data
        evaluated = [
            (cat, stats) for cat, stats in self._category_cache.items()
            if stats.total_predictions >= MIN_SAMPLES_FOR_CONFIDENCE
        ]
        
        if not evaluated:
            return {
                "categories": list(self._category_cache.keys()),
                "message": "Not enough data yet for category analysis",
                "total_predictions": sum(s.total_predictions for s in self._category_cache.values()),
            }
        
        # Sort by profitable accuracy
        sorted_cats = sorted(evaluated, key=lambda x: x[1].profitable_accuracy, reverse=True)
        
        best = [(cat, stats.profitable_accuracy) for cat, stats in sorted_cats[:3]]
        worst = [(cat, stats.profitable_accuracy) for cat, stats in sorted_cats[-3:]]
        avoided = [cat for cat, stats in evaluated if stats.is_avoided]
        
        return {
            "total_categories": len(self._category_cache),
            "evaluated_categories": len(evaluated),
            "total_predictions": sum(s.total_predictions for s in self._category_cache.values()),
            "best_categories": [{"category": c, "profitable_acc": f"{a:.1%}"} for c, a in best],
            "worst_categories": [{"category": c, "profitable_acc": f"{a:.1%}"} for c, a in worst],
            "avoided_categories": avoided,
            "avg_profitable_accuracy": sum(s.profitable_accuracy for _, s in evaluated) / len(evaluated),
        }
    
    def trigger_reevaluation(self, category: str) -> dict:
        """Manually trigger re-evaluation of an avoided category.
        
        Args:
            category: Category to re-evaluate.
            
        Returns:
            Result of re-evaluation.
        """
        if category not in self._category_cache:
            return {"error": f"Category '{category}' not found"}
        
        stats = self._category_cache[category]
        stats.last_reevaluation = datetime.utcnow().isoformat()
        
        if stats.profitable_accuracy >= MIN_ACCEPTABLE_ACCURACY:
            stats.is_avoided = False
            stats.avoided_since = None
            self._save_stats(category)
            return {
                "category": category,
                "status": "un-avoided",
                "profitable_accuracy": stats.profitable_accuracy,
            }
        else:
            self._save_stats(category)
            return {
                "category": category,
                "status": "still-avoided",
                "profitable_accuracy": stats.profitable_accuracy,
                "needs_accuracy": MIN_ACCEPTABLE_ACCURACY,
            }


# Singleton instance
_tracker_instance: Optional[MarketCategoryTracker] = None


def get_category_tracker() -> MarketCategoryTracker:
    """Get or create the singleton category tracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = MarketCategoryTracker()
    return _tracker_instance
