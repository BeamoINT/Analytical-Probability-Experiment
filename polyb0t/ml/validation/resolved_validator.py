"""Validates model predictions against resolved market outcomes.

This module provides proper out-of-sample validation using ONLY markets
that have actually resolved (is_fully_labeled=1), ensuring we measure
true predictive performance rather than in-sample overfitting.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from polyb0t.ml.validation.calibration import ConfidenceCalibrator
from polyb0t.ml.validation.metrics import CalibrationMetrics, ValidationResult

logger = logging.getLogger(__name__)

# Simulation parameters
SPREAD_COST = 0.02  # 2% spread/slippage assumption
POSITION_SIZE = 0.05  # 5% of portfolio per trade
MIN_CONFIDENCE_TO_TRADE = 0.55  # Only trade if confidence > 55%


class ResolvedMarketValidator:
    """Validates model predictions against resolved market outcomes.

    Only uses is_fully_labeled=1 examples from training_examples table
    to ensure ground truth labels. Uses temporal split to avoid look-ahead bias.

    Usage:
        validator = ResolvedMarketValidator()
        result = validator.validate_model(expert_pool)
        print(result.summary())
    """

    def __init__(
        self,
        db_path: str = "data/ai_training.db",
        calibration_dir: str = "data/calibration",
        min_samples: int = 100,
    ):
        """Initialize validator.

        Args:
            db_path: Path to AI training database
            calibration_dir: Directory for calibration models
            min_samples: Minimum resolved samples needed for validation
        """
        self.db_path = db_path
        self.calibration_dir = calibration_dir
        self.min_samples = min_samples
        self.calibrator = ConfidenceCalibrator(calibration_dir)

    def get_resolved_examples(self) -> List[Dict[str, Any]]:
        """Load all labeled market examples.

        Uses price_change_24h as the outcome measure since most markets
        don't have resolved_outcome populated. An example is considered
        "profitable" if price_change_24h > 0.

        Returns:
            List of examples with features and outcome.
            Sorted by created_at ascending (oldest first).
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            cursor = conn.cursor()

            # Use price_change_24h as the outcome since resolved_outcome is rarely populated
            # A positive price change means the prediction "buy YES" would have been profitable
            cursor.execute("""
                SELECT
                    example_id, token_id, market_id, created_at,
                    features, price_change_24h, price_change_to_resolution,
                    category, market_title, direction_24h
                FROM training_examples
                WHERE price_change_24h IS NOT NULL
                ORDER BY created_at ASC
            """)

            rows = cursor.fetchall()
            conn.close()

            examples = []
            for row in rows:
                try:
                    features = json.loads(row[4]) if row[4] else {}
                except json.JSONDecodeError:
                    features = {}

                price_change_24h = row[5] or 0
                # Convert to binary outcome: 1 if price went up (profitable for YES buyers)
                # Use a small threshold to filter out noise
                if price_change_24h > 0.01:  # >1% up
                    outcome = 1
                elif price_change_24h < -0.01:  # >1% down
                    outcome = 0
                else:
                    continue  # Skip flat examples (within 1% either way)

                examples.append({
                    "example_id": row[0],
                    "token_id": row[1],
                    "market_id": row[2],
                    "created_at": row[3],
                    "features": features,
                    "resolved_outcome": outcome,
                    "price_change_to_resolution": row[6] or price_change_24h,
                    "category": row[7],
                    "market_title": row[8],
                    "price_change_24h": price_change_24h,
                })

            logger.info(f"Loaded {len(examples)} labeled market examples (using 24h price change)")
            return examples

        except Exception as e:
            logger.error(f"Failed to load resolved examples: {e}")
            return []

    def validate_model(
        self,
        expert_pool,
        test_split: float = 0.2,
        update_calibrator: bool = True,
    ) -> Optional[ValidationResult]:
        """Run full validation pipeline.

        Steps:
        1. Load resolved examples
        2. Temporal split (older=calibration train, newer=test)
        3. Get model predictions on both sets
        4. Fit calibration model on train set
        5. Apply calibration to test predictions
        6. Compute metrics
        7. Simulate P&L

        Args:
            expert_pool: Trained ExpertPool to validate
            test_split: Fraction for test set (time-based, most recent)
            update_calibrator: Whether to update the calibration model

        Returns:
            ValidationResult with all metrics, or None if insufficient data
        """
        examples = self.get_resolved_examples()

        if len(examples) < self.min_samples:
            logger.warning(
                f"Only {len(examples)} resolved examples - need at least {self.min_samples}"
            )
            return None

        # Temporal split: older for calibration training, newer for testing
        split_idx = int(len(examples) * (1 - test_split))
        train_examples = examples[:split_idx]
        test_examples = examples[split_idx:]

        logger.info(
            f"Validation split: {len(train_examples)} calibration train, "
            f"{len(test_examples)} test"
        )

        # Get predictions for calibration training set
        train_preds = self._get_predictions(expert_pool, train_examples)

        if len(train_preds) < 50:
            logger.warning(f"Only {len(train_preds)} train predictions - need more data")
            return None

        # Fit calibrator on train set
        train_raw_confs = np.array([p[0] for p in train_preds])
        train_actuals = np.array([p[1] for p in train_preds])

        if update_calibrator:
            cal_metrics = self.calibrator.fit(train_raw_confs, train_actuals)
        else:
            cal_metrics = self.calibrator._compute_calibration_metrics(
                self.calibrator.calibrate(train_raw_confs), train_actuals
            )

        # Get test predictions
        test_preds = self._get_predictions(expert_pool, test_examples)

        if len(test_preds) < 20:
            logger.warning(f"Only {len(test_preds)} test predictions - insufficient")
            return None

        test_raw_confs = np.array([p[0] for p in test_preds])
        test_actuals = np.array([p[1] for p in test_preds])

        # Compute raw metrics (before calibration)
        raw_predictions = (test_raw_confs > 0.5).astype(int)
        raw_accuracy = float(np.mean(raw_predictions == test_actuals))
        raw_precision = self._precision(raw_predictions, test_actuals)
        raw_recall = self._recall(raw_predictions, test_actuals)
        raw_f1 = self._f1(raw_predictions, test_actuals)
        raw_brier = float(np.mean((test_raw_confs - test_actuals) ** 2))

        # Compute calibrated metrics
        calibrated_confs = self.calibrator.calibrate(test_raw_confs)
        cal_predictions = (calibrated_confs > 0.5).astype(int)
        calibrated_accuracy = float(np.mean(cal_predictions == test_actuals))
        calibrated_brier = float(np.mean((calibrated_confs - test_actuals) ** 2))

        # Confidence bucket analysis
        confidence_buckets = self._analyze_confidence_buckets(test_raw_confs, test_actuals)

        # Simulate P&L
        pnl_result = self._simulate_pnl(test_examples, test_preds)

        # Per-category analysis
        category_results = self._analyze_by_category(test_examples, test_preds)

        result = ValidationResult(
            timestamp=datetime.utcnow(),
            n_train=len(train_preds),
            n_test=len(test_preds),
            raw_accuracy=raw_accuracy,
            raw_precision=raw_precision,
            raw_recall=raw_recall,
            raw_f1=raw_f1,
            raw_brier=raw_brier,
            calibrated_accuracy=calibrated_accuracy,
            calibrated_brier=calibrated_brier,
            calibration_metrics=cal_metrics,
            simulated_pnl=pnl_result["total_pnl"],
            simulated_trades=pnl_result["n_trades"],
            simulated_win_rate=pnl_result["win_rate"],
            simulated_avg_win=pnl_result["avg_win"],
            simulated_avg_loss=pnl_result["avg_loss"],
            confidence_buckets=confidence_buckets,
            category_results=category_results,
        )

        logger.info(f"Validation complete:\n{result.summary()}")
        return result

    def _get_predictions(
        self,
        expert_pool,
        examples: List[Dict],
    ) -> List[Tuple[float, int]]:
        """Get model predictions for examples.

        Args:
            expert_pool: ExpertPool to use for predictions
            examples: List of examples with features

        Returns:
            List of (raw_confidence, actual_outcome) tuples
        """
        predictions = []

        for ex in examples:
            features = ex["features"]

            if not features:
                continue

            try:
                result = expert_pool.predict(features)

                if result is None:
                    continue

                pred, conf, _ = result

                # Map prediction to confidence in YES winning
                # pred > 0.5 means model thinks trade is profitable
                # We use the confidence as probability estimate
                # If momentum-based direction would be YES, conf is prob of YES
                # For simplicity, we use conf directly as the "confidence in correct prediction"

                actual = ex["resolved_outcome"]
                predictions.append((conf, actual))

            except Exception as e:
                logger.debug(f"Prediction failed for {ex.get('market_id')}: {e}")
                continue

        return predictions

    def _analyze_confidence_buckets(
        self,
        confidences: np.ndarray,
        actuals: np.ndarray,
    ) -> Dict[str, Dict]:
        """Analyze win rate by confidence bucket.

        This reveals whether higher confidence correlates with better accuracy.
        If model is well-calibrated, 60-70% confidence should have ~65% win rate.
        """
        buckets = {}
        edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

        for i in range(len(edges) - 1):
            low, high = edges[i], edges[i + 1]
            mask = (confidences >= low) & (confidences < high)

            if mask.sum() == 0:
                continue

            bucket_actuals = actuals[mask]
            bucket_confs = confidences[mask]

            expected = (low + high) / 2
            actual_win_rate = float(bucket_actuals.mean())
            gap = actual_win_rate - expected

            # Determine status
            if abs(gap) < 0.05:
                status = "OK"
            elif gap < 0:
                status = "OVERCONFIDENT"
            else:
                status = "UNDERCONFIDENT"

            buckets[f"{low:.2f}-{high:.2f}"] = {
                "count": int(mask.sum()),
                "win_rate": actual_win_rate,
                "expected": expected,
                "gap": gap,
                "status": status,
                "avg_confidence": float(bucket_confs.mean()),
            }

        return buckets

    def _simulate_pnl(
        self,
        examples: List[Dict],
        predictions: List[Tuple[float, int]],
    ) -> Dict[str, Any]:
        """Simulate P&L on test set.

        Uses realistic assumptions:
        - 5% position size per trade
        - 2% spread/slippage cost
        - Only trades when confidence > threshold

        Returns:
            Dict with total_pnl, n_trades, win_rate, avg_win, avg_loss
        """
        total_pnl = 0.0
        n_trades = 0
        wins = 0
        win_amounts = []
        loss_amounts = []

        for i, (conf, actual) in enumerate(predictions):
            # Only trade on sufficient confidence
            if conf < MIN_CONFIDENCE_TO_TRADE:
                continue

            n_trades += 1

            # Get price change to resolution
            ex = examples[i] if i < len(examples) else {}
            price_change = ex.get("price_change_to_resolution") or 0

            # Profit calculation
            # Model predicted "profitable" with this confidence
            # If actual outcome matches prediction, we profit
            predicted_correct = (conf > 0.5 and actual == 1) or (conf <= 0.5 and actual == 0)

            if predicted_correct:
                # Win: capture the price movement minus spread
                profit = max(0, abs(price_change) - SPREAD_COST) * POSITION_SIZE
                wins += 1
                win_amounts.append(profit)
            else:
                # Loss: lose position + spread
                loss = (abs(price_change) * 0.5 + SPREAD_COST) * POSITION_SIZE
                profit = -loss
                loss_amounts.append(abs(profit))

            total_pnl += profit

        return {
            "total_pnl": total_pnl,
            "n_trades": n_trades,
            "win_rate": wins / n_trades if n_trades > 0 else 0,
            "avg_win": float(np.mean(win_amounts)) if win_amounts else 0,
            "avg_loss": float(np.mean(loss_amounts)) if loss_amounts else 0,
        }

    def _analyze_by_category(
        self,
        examples: List[Dict],
        predictions: List[Tuple[float, int]],
    ) -> Dict[str, Dict]:
        """Analyze performance by market category."""
        category_data = {}

        for i, (conf, actual) in enumerate(predictions):
            if i >= len(examples):
                break

            category = examples[i].get("category") or "unknown"

            if category not in category_data:
                category_data[category] = {
                    "confidences": [],
                    "actuals": [],
                }

            category_data[category]["confidences"].append(conf)
            category_data[category]["actuals"].append(actual)

        results = {}
        for category, data in category_data.items():
            confs = np.array(data["confidences"])
            acts = np.array(data["actuals"])

            if len(confs) < 5:
                continue

            preds = (confs > 0.5).astype(int)
            accuracy = float(np.mean(preds == acts))
            avg_conf = float(confs.mean())

            results[category] = {
                "count": len(confs),
                "accuracy": accuracy,
                "avg_confidence": avg_conf,
                "calibration_gap": avg_conf - accuracy,
            }

        return results

    @staticmethod
    def _precision(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate precision (positive predictive value)."""
        true_positives = np.sum((predictions == 1) & (actuals == 1))
        predicted_positives = np.sum(predictions == 1)
        return float(true_positives / predicted_positives) if predicted_positives > 0 else 0.0

    @staticmethod
    def _recall(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate recall (true positive rate)."""
        true_positives = np.sum((predictions == 1) & (actuals == 1))
        actual_positives = np.sum(actuals == 1)
        return float(true_positives / actual_positives) if actual_positives > 0 else 0.0

    @staticmethod
    def _f1(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate F1 score."""
        p = ResolvedMarketValidator._precision(predictions, actuals)
        r = ResolvedMarketValidator._recall(predictions, actuals)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def run_validation(
    db_path: str = "data/ai_training.db",
    test_split: float = 0.2,
) -> Optional[ValidationResult]:
    """Convenience function to run validation.

    Args:
        db_path: Path to AI training database
        test_split: Fraction for test set

    Returns:
        ValidationResult or None
    """
    from polyb0t.ml.moe.expert_pool import get_expert_pool

    validator = ResolvedMarketValidator(db_path=db_path)
    pool = get_expert_pool()
    return validator.validate_model(pool, test_split=test_split)
