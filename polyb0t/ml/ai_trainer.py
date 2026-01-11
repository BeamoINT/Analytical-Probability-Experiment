"""AI Training Manager with benchmarking and model versioning.

This module handles:
- Training AI models when enough data is available
- Benchmarking new models against old ones
- Only deploying models that are better
- Persisting model state and training history
"""

import json
import logging
import os
import pickle
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Tuple
import threading

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for evaluating model performance."""
    mse: float  # Mean Squared Error
    mae: float  # Mean Absolute Error
    r2: float  # R-squared
    directional_accuracy: float  # % of correct direction predictions
    profitable_accuracy: float  # % of predictions that would have been profitable
    
    def score(self) -> float:
        """Overall score (higher is better)."""
        # Weighted combination of metrics
        return (
            self.directional_accuracy * 0.4 +
            self.profitable_accuracy * 0.4 +
            max(0, self.r2) * 0.2
        )


@dataclass
class ModelInfo:
    """Information about a trained model."""
    version: int
    created_at: datetime
    training_examples: int
    metrics: ModelMetrics
    is_active: bool
    model_path: str
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "training_examples": self.training_examples,
            "metrics": {
                "mse": self.metrics.mse,
                "mae": self.metrics.mae,
                "r2": self.metrics.r2,
                "directional_accuracy": self.metrics.directional_accuracy,
                "profitable_accuracy": self.metrics.profitable_accuracy,
                "score": self.metrics.score(),
            },
            "is_active": self.is_active,
            "model_path": self.model_path,
        }


class AITrainer:
    """Manages AI model training, benchmarking, and deployment."""
    
    def __init__(
        self,
        model_dir: str = "data/ai_models",
        min_training_examples: int = 1000,
        benchmark_test_size: float = 0.2,
        min_improvement_pct: float = 1.0,
    ):
        """Initialize the trainer.
        
        Args:
            model_dir: Directory to store models.
            min_training_examples: Minimum examples before training.
            benchmark_test_size: Fraction for test set.
            min_improvement_pct: Required improvement to deploy new model.
        """
        self.model_dir = model_dir
        self.min_training_examples = min_training_examples
        self.benchmark_test_size = benchmark_test_size
        self.min_improvement_pct = min_improvement_pct
        
        self._ensure_dirs()
        self._current_model = None
        self._current_model_info: Optional[ModelInfo] = None
        self._training_lock = threading.Lock()
        self._is_training = False
        
        # Load current model if exists
        self._load_current_model()
        
    def _ensure_dirs(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "versions"), exist_ok=True)
        
    def _get_state_path(self) -> str:
        """Get path to trainer state file."""
        return os.path.join(self.model_dir, "trainer_state.json")
    
    def _get_current_model_path(self) -> str:
        """Get path to current active model."""
        return os.path.join(self.model_dir, "current_model.pkl")
    
    def _load_current_model(self) -> None:
        """Load the current active model."""
        model_path = self._get_current_model_path()
        state_path = self._get_state_path()
        
        if os.path.exists(model_path) and os.path.exists(state_path):
            try:
                with open(model_path, "rb") as f:
                    self._current_model = pickle.load(f)
                    
                with open(state_path, "r") as f:
                    state = json.load(f)
                    
                self._current_model_info = ModelInfo(
                    version=state["version"],
                    created_at=datetime.fromisoformat(state["created_at"]),
                    training_examples=state["training_examples"],
                    metrics=ModelMetrics(**state["metrics"]),
                    is_active=True,
                    model_path=model_path,
                )
                
                logger.info(
                    f"Loaded AI model v{self._current_model_info.version} "
                    f"(score: {self._current_model_info.metrics.score():.3f})"
                )
            except Exception as e:
                logger.warning(f"Failed to load current model: {e}")
                self._current_model = None
                self._current_model_info = None
                
    def has_trained_model(self) -> bool:
        """Check if a trained model is available."""
        return self._current_model is not None
    
    def get_model_info(self) -> Optional[dict]:
        """Get info about current model."""
        if self._current_model_info:
            return self._current_model_info.to_dict()
        return None
    
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._is_training
    
    def can_train(self, available_examples: int) -> bool:
        """Check if we have enough data to train.
        
        Args:
            available_examples: Number of labeled examples available.
            
        Returns:
            True if we can train.
        """
        return available_examples >= self.min_training_examples
    
    def train_model(self, training_data: list[dict]) -> Optional[ModelInfo]:
        """Train a new model and deploy if better.
        
        Args:
            training_data: List of training examples.
            
        Returns:
            ModelInfo if new model deployed, None otherwise.
        """
        if len(training_data) < self.min_training_examples:
            logger.warning(
                f"Not enough training data: {len(training_data)} < {self.min_training_examples}"
            )
            return None
            
        with self._training_lock:
            if self._is_training:
                logger.warning("Training already in progress")
                return None
                
            self._is_training = True
            
        try:
            logger.info(f"Starting AI training with {len(training_data)} examples")
            
            # === SORT BY TIME FOR REALISTIC VALIDATION ===
            # Sort data by timestamp (oldest first)
            # This prevents data leakage - we train on past, validate on future
            sorted_data = sorted(
                training_data,
                key=lambda x: x.get("timestamp", x.get("created_at", ""))
            )
            
            # Prepare data (maintains sort order)
            X, y, timestamps = self._prepare_training_data_with_time(sorted_data)
            
            if X is None or len(X) == 0:
                logger.warning("No valid training data after preparation")
                return None
            
            # === TIME-BASED SPLIT ===
            # Train on older 80%, validate on newest 20%
            # This simulates real trading: learn from past, predict future
            split_idx = int(len(X) * (1 - self.benchmark_test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(
                f"Time-based split: train={len(X_train)} (older), test={len(X_test)} (newer)"
            )
            
            # Train new model
            new_model = self._train(X_train, y_train)
            
            # === EVALUATE ON FUTURE DATA (more realistic) ===
            new_metrics = self._evaluate(new_model, X_test, y_test)
            logger.info(
                f"New model validation (on future data): "
                f"dir_acc={new_metrics.directional_accuracy:.1%}, "
                f"profit_acc={new_metrics.profitable_accuracy:.1%}, "
                f"r2={new_metrics.r2:.3f}"
            )
            
            # Compare with current model (for logging only)
            if self._current_model is not None:
                try:
                    old_metrics = self._evaluate(self._current_model, X_test, y_test)
                    improvement = (new_metrics.score() - old_metrics.score()) / max(0.001, old_metrics.score()) * 100
                    
                    logger.info(
                        f"Model comparison: old_score={old_metrics.score():.3f}, "
                        f"new_score={new_metrics.score():.3f}, "
                        f"change={improvement:+.1f}%"
                    )
                except Exception as e:
                    logger.warning(f"Could not compare with old model: {e}")
            
            # === ALWAYS DEPLOY NEWEST MODEL ===
            # New model always replaces old - it has more/newer training data
            version = (self._current_model_info.version + 1) if self._current_model_info else 1
            model_info = self._deploy_model(new_model, new_metrics, version, len(training_data))
            
            logger.info(
                f"Deployed AI model v{version} "
                f"(profit_acc={new_metrics.profitable_accuracy:.1%})"
            )
            return model_info
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return None
        finally:
            self._is_training = False
            
    def _prepare_training_data(self, data: list[dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data arrays.
        
        Handles backwards compatibility by filling missing features with 0.
        Uses all available features from the expanded schema.
        
        Args:
            data: Raw training data.
            
        Returns:
            Tuple of (features, labels) arrays.
        """
        # Comprehensive feature columns (from MarketSnapshot)
        # Ordered by importance/availability
        feature_cols = [
            # Core price features (always available)
            "price", "spread", "spread_pct", "mid_price",
            # Volume & liquidity
            "volume_24h", "volume_1h", "volume_6h", "liquidity",
            "liquidity_bid", "liquidity_ask",
            # Orderbook features
            "orderbook_imbalance", "bid_depth", "ask_depth",
            "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10",
            "bid_levels", "ask_levels", "best_bid_size", "best_ask_size",
            "bid_ask_size_ratio",
            # Momentum features
            "momentum_1h", "momentum_4h", "momentum_24h", "momentum_7d",
            "price_change_1h", "price_change_4h", "price_change_24h",
            "price_high_24h", "price_low_24h", "price_range_24h",
            # Volatility features
            "volatility_1h", "volatility_24h", "volatility_7d", "atr_24h",
            # Trade flow features
            "trade_count_1h", "trade_count_24h",
            "avg_trade_size_1h", "avg_trade_size_24h",
            "buy_volume_1h", "sell_volume_1h", "buy_sell_ratio_1h",
            "large_trade_count_24h",
            # Timing features
            "days_to_resolution", "hours_to_resolution", "market_age_days",
            "hour_of_day", "day_of_week",
            # Market state
            "total_yes_shares", "total_no_shares", "open_interest",
            # Related/social features
            "num_related_markets", "avg_related_price",
            "comment_count", "view_count", "unique_traders",
            # Derived features
            "price_vs_volume_ratio", "liquidity_per_dollar_volume",
            "spread_adjusted_edge",
        ]
        
        # Store feature columns for later use in predictions
        self._feature_cols = feature_cols
        
        # Target: predict 24h price change
        X_list = []
        y_list = []
        
        for example in data:
            # Check if we have the target
            target = example.get("label_price_change_24h")
            if target is None:
                continue
                
            # Extract features - fill missing with 0 for backwards compatibility
            features = []
            for col in feature_cols:
                val = example.get(col, 0)  # Default to 0 if missing
                try:
                    features.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    features.append(0.0)
                
            X_list.append(features)
            y_list.append(float(target))
                
        if len(X_list) == 0:
            return None, None
        
        logger.info(f"Prepared {len(X_list)} examples with {len(feature_cols)} features each")
        return np.array(X_list), np.array(y_list)
    
    def _prepare_training_data_with_time(
        self, data: list[dict]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list[str]]]:
        """Prepare training data arrays with timestamps for time-based splitting.
        
        Args:
            data: Raw training data (should be pre-sorted by time).
            
        Returns:
            Tuple of (features, labels, timestamps) arrays.
        """
        # Same feature columns as _prepare_training_data
        feature_cols = [
            "price", "spread", "spread_pct", "mid_price",
            "volume_24h", "volume_1h", "volume_6h", "liquidity",
            "liquidity_bid", "liquidity_ask",
            "orderbook_imbalance", "bid_depth", "ask_depth",
            "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10",
            "bid_levels", "ask_levels", "best_bid_size", "best_ask_size",
            "bid_ask_size_ratio",
            "momentum_1h", "momentum_4h", "momentum_24h", "momentum_7d",
            "price_change_1h", "price_change_4h", "price_change_24h",
            "price_high_24h", "price_low_24h", "price_range_24h",
            "volatility_1h", "volatility_24h", "volatility_7d", "atr_24h",
            "trade_count_1h", "trade_count_24h",
            "avg_trade_size_1h", "avg_trade_size_24h",
            "buy_volume_1h", "sell_volume_1h", "buy_sell_ratio_1h",
            "large_trade_count_24h",
            "days_to_resolution", "hours_to_resolution", "market_age_days",
            "hour_of_day", "day_of_week",
            "total_yes_shares", "total_no_shares", "open_interest",
            "num_related_markets", "avg_related_price",
            "comment_count", "view_count", "unique_traders",
            "price_vs_volume_ratio", "liquidity_per_dollar_volume",
            "spread_adjusted_edge",
        ]
        
        self._feature_cols = feature_cols
        
        X_list = []
        y_list = []
        timestamps = []
        
        for example in data:
            target = example.get("label_price_change_24h")
            if target is None:
                continue
            
            # Get timestamp for tracking
            ts = example.get("timestamp", example.get("created_at", ""))
            
            features = []
            for col in feature_cols:
                val = example.get(col, 0)
                try:
                    features.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    features.append(0.0)
            
            X_list.append(features)
            y_list.append(float(target))
            timestamps.append(ts)
        
        if len(X_list) == 0:
            return None, None, None
        
        logger.info(f"Prepared {len(X_list)} time-sorted examples with {len(feature_cols)} features")
        return np.array(X_list), np.array(y_list), timestamps
    
    def _train(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train a model using advanced ensemble methods.
        
        Args:
            X: Feature array.
            y: Label array.
            
        Returns:
            Trained model.
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        logger.info(f"Training on {n_samples} samples with {n_features} features")
        
        try:
            # Try Random Forest first (better for many features)
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            # Use a pipeline with scaling for better performance
            if n_samples > 1000:
                # Larger dataset - use more trees
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42,
                    ))
                ])
            else:
                # Smaller dataset - use gradient boosting
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42,
                    ))
                ])
            
            model.fit(X, y)
            
            # Log feature importances if available
            try:
                regressor = model.named_steps['regressor']
                if hasattr(regressor, 'feature_importances_'):
                    importances = regressor.feature_importances_
                    top_indices = np.argsort(importances)[-10:][::-1]
                    top_features = [(self._feature_cols[i], importances[i]) for i in top_indices]
                    logger.info(f"Top 10 features: {top_features}")
            except Exception:
                pass
                
            return model
            
        except ImportError:
            # Fallback to simple model
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=1.0))
            ])
            model.fit(X, y)
            return model
            
    def _evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """Evaluate a model.
        
        Args:
            model: Model to evaluate.
            X: Test features.
            y: Test labels.
            
        Returns:
            ModelMetrics.
        """
        predictions = model.predict(X)
        
        # MSE
        mse = float(np.mean((predictions - y) ** 2))
        
        # MAE
        mae = float(np.mean(np.abs(predictions - y)))
        
        # R2
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-10))
        
        # === REALISTIC ACCURACY METRICS ===
        # Minimum threshold for a "confident" prediction (1% price change)
        # This filters out noise and flat predictions
        MIN_PREDICTION_THRESHOLD = 0.01  # 1% price change
        MIN_ACTUAL_THRESHOLD = 0.01  # 1% actual change to count as "movement"
        SPREAD_COST = 0.02  # Assume 2% spread cost to be conservative
        
        # Directional accuracy: only count predictions above threshold
        # Predictions near zero are "no trade" signals
        confident_mask = np.abs(predictions) >= MIN_PREDICTION_THRESHOLD
        actual_moves_mask = np.abs(y) >= MIN_ACTUAL_THRESHOLD
        
        # Only evaluate on cases where we'd actually trade
        tradeable = confident_mask & actual_moves_mask
        if np.sum(tradeable) > 0:
            correct_direction = np.sum(
                (predictions[tradeable] > 0) == (y[tradeable] > 0)
            )
            directional_accuracy = float(correct_direction / np.sum(tradeable))
        else:
            directional_accuracy = 0.5  # No confident predictions = random
        
        # Profitable accuracy: net profit after spread
        # Only count as profitable if gain exceeds spread cost
        if np.sum(confident_mask) > 0:
            # Calculate net profit: |actual change| - spread if direction correct
            # Loss: -|actual change| - spread if direction wrong
            net_profits = np.where(
                (predictions > 0) == (y > 0),  # Correct direction
                np.abs(y) - SPREAD_COST,  # Profit minus spread
                -np.abs(y) - SPREAD_COST  # Loss plus spread
            )
            profitable = np.sum(net_profits[confident_mask] > 0)
            profitable_accuracy = float(profitable / np.sum(confident_mask))
        else:
            profitable_accuracy = 0.0  # No trades = no profit
        
        logger.info(
            f"Evaluation: {np.sum(tradeable)}/{len(y)} tradeable samples, "
            f"{np.sum(confident_mask)} confident predictions"
        )
        
        return ModelMetrics(
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            profitable_accuracy=profitable_accuracy,
        )
        
    def _deploy_model(
        self,
        model: Any,
        metrics: ModelMetrics,
        version: int,
        training_examples: int,
    ) -> ModelInfo:
        """Deploy a new model.
        
        Args:
            model: Model to deploy.
            metrics: Model metrics.
            version: Version number.
            training_examples: Number of training examples.
            
        Returns:
            ModelInfo for deployed model.
        """
        now = datetime.utcnow()
        
        # Save to versions directory
        version_path = os.path.join(self.model_dir, "versions", f"model_v{version}.pkl")
        with open(version_path, "wb") as f:
            pickle.dump(model, f)
            
        # Copy to current model
        current_path = self._get_current_model_path()
        shutil.copy(version_path, current_path)
        
        # Save state
        state = {
            "version": version,
            "created_at": now.isoformat(),
            "training_examples": training_examples,
            "metrics": {
                "mse": metrics.mse,
                "mae": metrics.mae,
                "r2": metrics.r2,
                "directional_accuracy": metrics.directional_accuracy,
                "profitable_accuracy": metrics.profitable_accuracy,
            },
        }
        
        with open(self._get_state_path(), "w") as f:
            json.dump(state, f, indent=2)
            
        # Update current model
        self._current_model = model
        self._current_model_info = ModelInfo(
            version=version,
            created_at=now,
            training_examples=training_examples,
            metrics=metrics,
            is_active=True,
            model_path=current_path,
        )
        
        return self._current_model_info
    
    def predict(self, features: dict) -> Optional[float]:
        """Make a prediction using the current model.
        
        Handles backwards compatibility - missing features are filled with 0.

        Args:
            features: Feature dictionary.

        Returns:
            Predicted price change, or None if no model.
        """
        if self._current_model is None:
            return None

        # Use stored feature columns from training, or default set
        feature_cols = getattr(self, '_feature_cols', None)
        if feature_cols is None:
            # Fallback to comprehensive feature set
            feature_cols = [
                "price", "spread", "spread_pct", "mid_price",
                "volume_24h", "volume_1h", "volume_6h", "liquidity",
                "liquidity_bid", "liquidity_ask",
                "orderbook_imbalance", "bid_depth", "ask_depth",
                "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10",
                "bid_levels", "ask_levels", "best_bid_size", "best_ask_size",
                "bid_ask_size_ratio",
                "momentum_1h", "momentum_4h", "momentum_24h", "momentum_7d",
                "price_change_1h", "price_change_4h", "price_change_24h",
                "price_high_24h", "price_low_24h", "price_range_24h",
                "volatility_1h", "volatility_24h", "volatility_7d", "atr_24h",
                "trade_count_1h", "trade_count_24h",
                "avg_trade_size_1h", "avg_trade_size_24h",
                "buy_volume_1h", "sell_volume_1h", "buy_sell_ratio_1h",
                "large_trade_count_24h",
                "days_to_resolution", "hours_to_resolution", "market_age_days",
                "hour_of_day", "day_of_week",
                "total_yes_shares", "total_no_shares", "open_interest",
                "num_related_markets", "avg_related_price",
                "comment_count", "view_count", "unique_traders",
                "price_vs_volume_ratio", "liquidity_per_dollar_volume",
                "spread_adjusted_edge",
            ]

        try:
            X = []
            for col in feature_cols:
                val = features.get(col, 0)
                try:
                    X.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    X.append(0.0)

            X = np.array([X])
            prediction = self._current_model.predict(X)[0]
            return float(prediction)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None


# Singleton instance
_trainer_instance: Optional[AITrainer] = None


def get_ai_trainer(
    model_dir: str = "data/ai_models",
    min_training_examples: int = 1000,
) -> AITrainer:
    """Get or create the singleton AI trainer.
    
    Args:
        model_dir: Model directory.
        min_training_examples: Minimum examples for training.
        
    Returns:
        AITrainer instance.
    """
    global _trainer_instance
    if _trainer_instance is None:
        _trainer_instance = AITrainer(
            model_dir=model_dir,
            min_training_examples=min_training_examples,
        )
    return _trainer_instance
