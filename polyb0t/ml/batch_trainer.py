"""Batch Training System for Historical Data.

One-time batch training on historical Polymarket data, replacing the
continuous online learning approach.

IMPORTANT LIMITATIONS:
- Historical data only contains POST-resolution market state
- Pre-resolution prices are unavailable, so price features are set to 0.5 (neutral)
- This means the model learns from metadata (volume, category, timing) not price action
- For proper price-movement prediction, use continuous_collector with live markets
- Models trained on historical data predict "market outcome" not "short-term price change"

The model trained here is useful for:
- Learning which categories/volumes/timing correlate with winning outcomes
- Providing a baseline signal for the MoE ensemble
- Bootstrap training before live data accumulates

For real trading decisions, combine with:
- Live price snapshots from continuous_collector
- Actual momentum/volatility from recent price history
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from polyb0t.config import get_settings
from polyb0t.ml.moe.expert_pool import ExpertPool, get_expert_pool
from polyb0t.ml.moe.trainer import MoETrainer

logger = logging.getLogger(__name__)


class BatchTrainer:
    """One-time batch training on historical data.

    This trainer loads historical training data, converts it to the MoE trainer's
    expected format, trains all experts on the appropriate data subsets, and
    validates on a held-out test set.
    """

    def __init__(
        self,
        data_path: str = "data/historical_training.db",
        test_split: float = 0.2,
        metrics_output: str = "data/training_metrics.json",
    ):
        """Initialize the batch trainer.

        Args:
            data_path: Path to historical training database.
            test_split: Fraction of data to hold out for testing.
            metrics_output: Path to save training metrics JSON.
        """
        self.data_path = Path(data_path)
        self.test_split = test_split
        self.metrics_output = Path(metrics_output)
        self.metrics_output.parent.mkdir(parents=True, exist_ok=True)

        # Path for converted training data
        self.converted_db_path = Path("data/historical_training_converted.db")

        self.settings = get_settings()
        self.expert_pool: Optional[ExpertPool] = None
        self.moe_trainer: Optional[MoETrainer] = None

    def load_data(self) -> tuple[list[dict], list[dict]]:
        """Load and split historical training data.

        Returns:
            Tuple of (train_data, test_data) as lists of example dicts.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training database not found: {self.data_path}")

        conn = sqlite3.connect(str(self.data_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT token_id, market_id, market_title, outcome_name, category,
                   features, label, winning_token_id, resolution_date
            FROM historical_examples
            ORDER BY resolution_date ASC
        """)

        all_examples = []
        for row in cursor.fetchall():
            features = json.loads(row[5]) if row[5] else {}
            example = {
                "token_id": row[0],
                "market_id": row[1],
                "market_title": row[2],
                "outcome_name": row[3],
                "category": row[4] or "other",
                "features": features,
                "label": row[6],
                "target": row[6],  # Alias for compatibility with MoE trainer
                "winning_token_id": row[7],
                "resolution_date": row[8],
            }
            all_examples.append(example)

        conn.close()

        logger.info(f"Loaded {len(all_examples)} examples from {self.data_path}")

        # Time-based split: earlier data for training, later for testing
        split_idx = int(len(all_examples) * (1 - self.test_split))
        train_data = all_examples[:split_idx]
        test_data = all_examples[split_idx:]

        logger.info(
            f"Split data: {len(train_data)} train, {len(test_data)} test "
            f"({self.test_split:.0%} test split)"
        )

        return train_data, test_data

    def _convert_to_moe_db(self, examples: list[dict]) -> None:
        """Convert historical examples to MoE trainer database format.

        Creates a SQLite database with the schema expected by MoETrainer._load_training_data().

        Args:
            examples: List of historical examples.
        """
        # Remove old converted database
        if self.converted_db_path.exists():
            self.converted_db_path.unlink()

        conn = sqlite3.connect(str(self.converted_db_path))
        cursor = conn.cursor()

        # Create table with schema expected by MoE trainer
        cursor.execute("""
            CREATE TABLE training_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id TEXT NOT NULL,
                market_id TEXT,
                category TEXT,
                features TEXT,
                price_change_1h REAL,
                price_change_4h REAL,
                price_change_24h REAL,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX idx_training_category ON training_examples(category)
        """)
        cursor.execute("""
            CREATE INDEX idx_training_price_change ON training_examples(price_change_24h)
        """)

        # Convert and insert examples
        inserted = 0
        for ex in examples:
            features = ex["features"]
            label = ex["label"]  # 0 or 1

            # Convert binary label to simulated price change
            # The MoE trainer uses abs(price_change) > 0.025 to determine is_profitable
            # We need winners to be above threshold and losers below:
            # - Winners (label=1): +0.10 (10%) -> abs(0.10) > 0.025 -> is_profitable=1
            # - Losers (label=0): -0.01 (-1%) -> abs(-0.01) < 0.025 -> is_profitable=0
            # This ensures proper binary classification AND realistic profit simulation
            if label == 1:
                # Winner: moderate positive price change (above threshold)
                price_change = 0.10  # 10% gain
            else:
                # Loser: small negative change (below threshold to be labeled non-profitable)
                price_change = -0.01  # 1% loss

            # Build features dict with all available features
            # NOTE: For historical data, price features are neutral (0.5) since we don't
            # have pre-resolution prices. The model must learn from metadata features.
            moe_features = {
                # Price features (neutral for historical - see historical_fetcher.py)
                "price": features.get("outcome_price", 0.5),  # Always 0.5 for historical
                "initial_price": features.get("initial_price", 0.5),

                # Volume/Liquidity
                "volume_24h": features.get("total_volume", 0.0),
                "liquidity": features.get("liquidity", 0.0),
                "volume_per_day": features.get("volume_per_day", 0.0),

                # Time features
                "days_to_resolution": features.get("days_to_resolution", 30.0),
                "market_age_days": features.get("market_age_days", 30.0),
                "hour_of_day": features.get("hour_of_day", 12),
                "day_of_week": features.get("day_of_week", 2),
                "is_weekend": features.get("is_weekend", 0),

                # Market structure
                "num_outcomes": features.get("num_outcomes", 2),
                "has_description": features.get("has_description", 0),
                "question_length": features.get("question_length", 50),

                # Simulated orderbook features (neutral values)
                "spread": 0.02,
                "bid_depth": features.get("liquidity", 0.0) / 2,
                "ask_depth": features.get("liquidity", 0.0) / 2,
                "best_bid_size": features.get("liquidity", 0.0) / 4,
                "best_ask_size": features.get("liquidity", 0.0) / 4,

                # Momentum (neutral for historical)
                "momentum_1h": 0.0,
                "momentum_4h": 0.0,
                "momentum_24h": 0.0,
                "momentum_7d": 0.0,

                # Volatility (moderate assumption)
                "volatility_1h": 0.02,
                "volatility_24h": 0.05,
                "volatility_7d": 0.1,

                # Category
                "category": ex["category"],
            }

            try:
                cursor.execute("""
                    INSERT INTO training_examples
                    (token_id, market_id, category, features,
                     price_change_1h, price_change_4h, price_change_24h, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ex["token_id"],
                    ex["market_id"],
                    ex["category"],
                    json.dumps(moe_features),
                    price_change,  # Use same value for all horizons
                    price_change,
                    price_change,
                    ex.get("resolution_date") or datetime.now(timezone.utc).isoformat(),
                ))
                inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert example: {e}")

        conn.commit()
        conn.close()

        logger.info(f"Converted {inserted} examples to MoE format at {self.converted_db_path}")

    def train(self) -> dict[str, Any]:
        """Run full MoE training pipeline on historical data.

        Returns:
            Dictionary with training results and metrics.
        """
        logger.info("=" * 60)
        logger.info("STARTING BATCH TRAINING ON HISTORICAL DATA")
        logger.info("=" * 60)

        results = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "data_path": str(self.data_path),
            "test_split": self.test_split,
        }

        try:
            # Load data
            train_data, test_data = self.load_data()
            results["data_stats"] = {
                "total_examples": len(train_data) + len(test_data),
                "train_examples": len(train_data),
                "test_examples": len(test_data),
            }

            # Count categories
            categories = {}
            for ex in train_data + test_data:
                cat = ex.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1
            results["data_stats"]["categories"] = categories

            logger.info(f"Category distribution: {categories}")

            # Convert training data to MoE format database
            logger.info("Converting historical data to MoE format...")
            self._convert_to_moe_db(train_data)

            # Initialize expert pool and trainer with converted database
            logger.info("Initializing MoE expert pool...")
            self.expert_pool = get_expert_pool()

            # Create trainer pointing to converted database
            self.moe_trainer = MoETrainer(
                pool=self.expert_pool,
                db_path=str(self.converted_db_path),
            )

            # Train MoE system
            logger.info(f"Training MoE on {len(train_data)} examples...")
            train_result = self.moe_trainer.train()

            if train_result:
                results["training_result"] = {
                    "experts_trained": train_result.get("experts_trained", 0),
                    "training_time_seconds": train_result.get("training_time", 0),
                    "active_experts": train_result.get("active", 0),
                    "suspended_experts": train_result.get("suspended", 0),
                    "deprecated_experts": train_result.get("deprecated", 0),
                }
            else:
                results["training_result"] = {"error": "Training returned None"}

            # Validate on test set
            logger.info(f"Validating on {len(test_data)} test examples...")
            validation_results = self.validate(test_data)
            results["validation"] = validation_results

            # Get expert metrics
            results["expert_metrics"] = self._collect_expert_metrics()

            # Get gating metrics
            results["gating_metrics"] = self._collect_gating_metrics()

            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            results["status"] = "success"

            logger.info("=" * 60)
            logger.info("BATCH TRAINING COMPLETE")
            logger.info(f"  Experts trained: {results['training_result'].get('experts_trained', 0)}")
            logger.info(f"  Active experts: {results['training_result'].get('active_experts', 0)}")
            logger.info(f"  Test accuracy: {validation_results.get('accuracy', 0):.2%}")
            logger.info(f"  Test profit: {validation_results.get('simulated_profit_pct', 0):.2%}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Batch training failed: {e}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)
            results["failed_at"] = datetime.now(timezone.utc).isoformat()

        # Save metrics
        self.save_metrics(results)

        return results

    def validate(self, test_data: list[dict]) -> dict[str, Any]:
        """Comprehensive validation on held-out test set.

        Args:
            test_data: List of test examples in MoE format.

        Returns:
            Dictionary with validation metrics.
        """
        if not self.expert_pool:
            return {"error": "Expert pool not initialized"}

        metrics = {
            "n_examples": len(test_data),
            "predictions": 0,
            "correct": 0,
            "profitable_predictions": 0,
            "total_profit_pct": 0.0,
            "category_metrics": {},
        }

        # Track per-category metrics
        category_stats = {}

        for example in test_data:
            try:
                # Build features dict for prediction
                features = example["features"]
                pred_features = {
                    "price": features.get("outcome_price", 0.5),
                    "volume_24h": features.get("total_volume", 0.0),
                    "liquidity": features.get("liquidity", 0.0),
                    "days_to_resolution": features.get("days_to_resolution", 30.0),
                    "market_age_days": features.get("market_age_days", 30.0),
                    "hour_of_day": features.get("hour_of_day", 12),
                    "day_of_week": features.get("day_of_week", 2),
                    "is_weekend": features.get("is_weekend", 0),
                    "num_outcomes": features.get("num_outcomes", 2),
                    "spread": 0.02,
                    "category": example.get("category", "other"),
                }

                # Get prediction from expert pool
                prediction = self.expert_pool.predict(pred_features)

                if prediction is None:
                    continue

                metrics["predictions"] += 1

                # Check if prediction was correct
                actual = example.get("label", 0)
                predicted = 1 if prediction > 0.5 else 0

                if predicted == actual:
                    metrics["correct"] += 1

                # Simulate profit/loss
                profit = 0.0
                if prediction > 0.5:  # We would have traded
                    if actual == 1:
                        profit = 0.05  # 5% profit on correct trade
                        metrics["profitable_predictions"] += 1
                    else:
                        profit = -0.05  # 5% loss on wrong trade
                    metrics["total_profit_pct"] += profit

                # Track by category
                cat = example.get("category", "other")
                if cat not in category_stats:
                    category_stats[cat] = {
                        "n": 0,
                        "correct": 0,
                        "profit": 0.0,
                        "trades": 0,
                    }
                category_stats[cat]["n"] += 1
                if predicted == actual:
                    category_stats[cat]["correct"] += 1
                if prediction > 0.5:
                    category_stats[cat]["trades"] += 1
                    category_stats[cat]["profit"] += profit

            except Exception as e:
                logger.debug(f"Validation error for example: {e}")
                continue

        # Calculate summary metrics
        if metrics["predictions"] > 0:
            metrics["accuracy"] = metrics["correct"] / metrics["predictions"]
            trades = sum(1 for _ in test_data if True)  # placeholder
            metrics["win_rate"] = metrics["profitable_predictions"] / max(1, metrics["predictions"])
        else:
            metrics["accuracy"] = 0.0
            metrics["win_rate"] = 0.0

        metrics["simulated_profit_pct"] = metrics["total_profit_pct"]

        # Category breakdown
        for cat, stats in category_stats.items():
            metrics["category_metrics"][cat] = {
                "n_examples": stats["n"],
                "accuracy": stats["correct"] / max(1, stats["n"]),
                "trades": stats["trades"],
                "profit_pct": stats["profit"],
            }

        return metrics

    def _collect_expert_metrics(self) -> dict[str, Any]:
        """Collect metrics from all trained experts.

        Returns:
            Dictionary of expert_id -> metrics.
        """
        if not self.expert_pool:
            return {}

        expert_metrics = {}

        for expert_id, expert in self.expert_pool.experts.items():
            if expert.metrics:
                m = expert.metrics
                expert_metrics[expert_id] = {
                    "state": expert.state.value,
                    "domain": expert.domain,
                    "expert_type": expert.expert_type,
                    "simulated_profit_pct": m.simulated_profit_pct,
                    "simulated_win_rate": m.simulated_win_rate,
                    "simulated_num_trades": m.simulated_num_trades,
                    "simulated_sharpe": m.simulated_sharpe,
                    "simulated_max_drawdown": m.simulated_max_drawdown,
                    "simulated_profit_factor": m.simulated_profit_factor,
                    "n_training_examples": m.n_training_examples,
                    "last_trained": m.last_trained.isoformat() if m.last_trained else None,
                }
            else:
                expert_metrics[expert_id] = {
                    "state": expert.state.value,
                    "domain": expert.domain,
                    "expert_type": expert.expert_type,
                    "error": "No metrics available",
                }

        return expert_metrics

    def _collect_gating_metrics(self) -> dict[str, Any]:
        """Collect metrics from the gating network.

        Returns:
            Dictionary with gating metrics.
        """
        if not self.expert_pool or not self.expert_pool.gating:
            return {"error": "Gating network not available"}

        gating = self.expert_pool.gating

        metrics = {
            "n_experts": len(self.expert_pool.experts),
            "active_experts": sum(
                1 for e in self.expert_pool.experts.values()
                if e.state.value == "active"
            ),
        }

        if hasattr(gating, "metrics") and gating.metrics:
            gm = gating.metrics
            metrics["routing_accuracy"] = gm.routing_accuracy
            metrics["profit_vs_random"] = gm.profit_vs_random
            metrics["n_samples_trained"] = gm.n_samples_trained

        return metrics

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Save training metrics to JSON file.

        Args:
            metrics: Dictionary of metrics to save.
        """
        try:
            # Pretty print with indentation
            with open(self.metrics_output, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            logger.info(f"Saved training metrics to {self.metrics_output}")

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


def run_batch_training(
    data_path: str = "data/historical_training.db",
    test_split: float = 0.2,
    metrics_output: str = "data/training_metrics.json",
) -> dict[str, Any]:
    """Convenience function to run batch training.

    Args:
        data_path: Path to historical training database.
        test_split: Fraction of data for testing.
        metrics_output: Path to save metrics.

    Returns:
        Training results dictionary.
    """
    trainer = BatchTrainer(
        data_path=data_path,
        test_split=test_split,
        metrics_output=metrics_output,
    )
    return trainer.train()
