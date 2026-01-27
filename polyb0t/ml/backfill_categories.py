"""Backfill categories for training examples with NULL category.

This script populates the `category` and `market_title` fields for training
examples that were created before category tracking was implemented.
"""

import json
import sqlite3
import logging
from typing import Optional

from polyb0t.ml.category_tracker import get_category_tracker
from polyb0t.config.settings import get_settings

logger = logging.getLogger(__name__)


def backfill_categories(
    batch_size: int = 1000,
    max_examples: Optional[int] = None,
) -> dict:
    """Backfill NULL categories in training_examples table.

    Looks up market titles from:
    1. tracked_markets.market_metadata (best source)
    2. Main database markets table (fallback)

    Then uses category_tracker to categorize based on title keywords.

    Args:
        batch_size: How often to commit (for progress tracking)
        max_examples: Maximum examples to process (None = all)

    Returns:
        Dict with stats: total, categorized, failed, by_category
    """
    settings = get_settings()
    tracker = get_category_tracker()

    ai_db = sqlite3.connect(settings.ai_training_db, timeout=30)
    ai_cursor = ai_db.cursor()

    # Get examples needing categorization
    query = """
        SELECT example_id, market_id, token_id, market_title
        FROM training_examples
        WHERE category IS NULL OR category = ''
    """
    if max_examples:
        query += f" LIMIT {max_examples}"

    ai_cursor.execute(query)
    examples = ai_cursor.fetchall()

    logger.info(f"Found {len(examples)} examples to backfill")

    stats = {
        "total": len(examples),
        "categorized": 0,
        "failed": 0,
        "by_category": {},
    }

    for example_id, market_id, token_id, existing_title in examples:
        title = existing_title

        # Try to get title from tracked_markets if not available
        if not title:
            title = _get_title_from_tracked_markets(ai_cursor, token_id, market_id)

        # Try main database if still no title
        if not title:
            title = _get_title_from_main_db(market_id)

        if not title:
            stats["failed"] += 1
            continue

        # Categorize using existing tracker
        category, confidence = tracker.categorize_market(
            market_id=market_id,
            title=title,
        )

        # Update training example
        ai_cursor.execute("""
            UPDATE training_examples
            SET category = ?, market_title = ?
            WHERE example_id = ?
        """, (category, title, example_id))

        stats["categorized"] += 1
        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

        # Commit in batches and log progress
        if stats["categorized"] % batch_size == 0:
            ai_db.commit()
            logger.info(f"Progress: {stats['categorized']}/{stats['total']} categorized")

    ai_db.commit()
    ai_db.close()

    logger.info(f"Backfill complete: {stats['categorized']} categorized, {stats['failed']} failed")
    logger.info(f"Category distribution: {stats['by_category']}")
    return stats


def _get_title_from_tracked_markets(cursor, token_id: str, market_id: str) -> Optional[str]:
    """Get market title from tracked_markets metadata."""
    cursor.execute(
        "SELECT market_metadata FROM tracked_markets WHERE token_id = ? OR market_id = ?",
        (token_id, market_id)
    )
    row = cursor.fetchone()
    if row and row[0]:
        try:
            metadata = json.loads(row[0])
            return metadata.get("question") or metadata.get("title")
        except json.JSONDecodeError:
            pass
    return None


def _get_title_from_main_db(market_id: str) -> Optional[str]:
    """Get market title from main polybot database."""
    try:
        # Use direct SQLite query - simpler and more reliable
        import os
        db_path = "polybot.db"  # Main database in project root
        if not os.path.exists(db_path):
            return None

        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        cursor.execute("SELECT question FROM markets WHERE condition_id = ?", (market_id,))
        row = cursor.fetchone()
        conn.close()

        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backfill_categories()
