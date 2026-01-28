"""Batch Training System for Historical Data.

One-time batch training on historical Polymarket data, replacing the
continuous online learning approach.
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
from polyb0t.ml.moe.trainer import MoETrainer, get_moe_trainer

logger = logging.getLogger(__name__)


class BatchTrainer:
    """One-time batch training on historical data.

    This trainer loads historical training data, performs a comprehensive
    training pass on all MoE experts, validates on a held-out test set,
    and saves detailed metrics.
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
            example = {
                "token_id": row[0],
                "market_id": row[1],
                "market_title": row[2],
                "outcome_name": row[3],
                "category": row[4],
                "features": json.loads(row[5]),
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

    def _prepare_moe_format(self, examples: list[dict]) -> list[dict]:
        """Convert historical examples to MoE trainer format.

        The MoE trainer expects a specific format with certain keys.

        Args:
            examples: Historical example dicts.

        Returns:
            Examples formatted for MoE trainer.
        """
        moe_examples = []

        for ex in examples:
            features = ex["features"]

            # Create MoE-compatible example
            moe_ex = {
                "token_id": ex["token_id"],
                "market_id": ex["market_id"],
                "market_title": ex["market_title"],
                "category": ex["category"],
                "target": ex["label"],

                # Flatten features into top-level keys
                "price": features.get("outcome_price", 0.5),
                "volume_24h": features.get("total_volume", 0.0),
                "liquidity": features.get("liquidity", 0.0),
                "days_to_resolution": features.get("days_to_resolution", 30.0),

                # Time features
                "hour_of_day": features.get("hour_of_day", 12),
                "day_of_week": features.get("day_of_week", 2),
                "is_weekend": features.get("is_weekend", 0),

                # Market features
                "market_age_days": features.get("market_age_days", 30.0),
                "num_outcomes": features.get("num_outcomes", 2),
                "question_length": features.get("question_length", 50),

                # Volume derived
                "volume_per_day": features.get("volume_per_day", 0.0),

                # Placeholder features (not available for historical data)
                # Set to neutral values
                "spread": 0.02,  # Assume 2% spread
                "bid_depth": features.get("liquidity", 0.0) / 2,
                "ask_depth": features.get("liquidity", 0.0) / 2,
                "momentum_1h": 0.0,
                "momentum_24h": 0.0,
                "volatility_24h": 0.05,  # Assume 5% volatility

                # Store original features for reference
                "_historical_features": features,
            }

            moe_examples.append(moe_ex)

        return moe_examples

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

            # Prepare data for MoE trainer
            moe_train = self._prepare_moe_format(train_data)
            moe_test = self._prepare_moe_format(test_data)

            # Initialize expert pool and trainer
            logger.info("Initializing MoE expert pool...")
            self.expert_pool = get_expert_pool()
            self.moe_trainer = get_moe_trainer()

            # Override trainer's training data
            self.moe_trainer.training_data = moe_train

            # Train MoE system
            logger.info(f"Training MoE on {len(moe_train)} examples...")
            train_result = self.moe_trainer.train()

            results["training_result"] = {
                "experts_trained": train_result.get("experts_trained", 0),
                "training_time_seconds": train_result.get("training_time", 0),
            }

            # Validate on test set
            logger.info(f"Validating on {len(moe_test)} test examples...")
            validation_results = self.validate(moe_test)
            results["validation"] = validation_results

            # Get expert metrics
            results["expert_metrics"] = self._collect_expert_metrics()

            # Get gating metrics
            results["gating_metrics"] = self._collect_gating_metrics()

            results["completed_at"] = datetime.now(timezone.utc).isoformat()
            results["status"] = "success"

            logger.info("=" * 60)
            logger.info("BATCH TRAINING COMPLETE")
            logger.info(f"  Experts trained: {results['training_result']['experts_trained']}")
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
                # Get prediction from expert pool
                prediction = self.expert_pool.predict(example)

                if prediction is None:
                    continue

                metrics["predictions"] += 1

                # Check if prediction was correct
                actual = example.get("target", 0)
                predicted = 1 if prediction > 0.5 else 0

                if predicted == actual:
                    metrics["correct"] += 1

                # Simulate profit/loss
                # If we predicted profitable (>0.5) and it was profitable (actual=1)
                # we made money. Otherwise we lost.
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
                    }
                category_stats[cat]["n"] += 1
                if predicted == actual:
                    category_stats[cat]["correct"] += 1
                if prediction > 0.5:
                    category_stats[cat]["profit"] += profit if actual == 1 else -0.05

            except Exception as e:
                logger.debug(f"Validation error for example: {e}")
                continue

        # Calculate summary metrics
        if metrics["predictions"] > 0:
            metrics["accuracy"] = metrics["correct"] / metrics["predictions"]
            metrics["win_rate"] = metrics["profitable_predictions"] / max(
                1, sum(1 for ex in test_data if self.expert_pool.predict(ex) or 0 > 0.5)
            )
        else:
            metrics["accuracy"] = 0.0
            metrics["win_rate"] = 0.0

        metrics["simulated_profit_pct"] = metrics["total_profit_pct"]

        # Category breakdown
        for cat, stats in category_stats.items():
            metrics["category_metrics"][cat] = {
                "n_examples": stats["n"],
                "accuracy": stats["correct"] / max(1, stats["n"]),
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
                    "error": "No metrics available",
                }

        return expert_metrics

    def _collect_gating_metrics(self) -> dict[str, Any]:
        """Collect metrics from the gating network.

        Returns:
            Dictionary with gating metrics.
        """
        if not self.expert_pool or not self.expert_pool.gating_network:
            return {"error": "Gating network not available"}

        gating = self.expert_pool.gating_network

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
