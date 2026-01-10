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
            
            # Prepare data
            X, y = self._prepare_training_data(training_data)
            
            if X is None or len(X) == 0:
                logger.warning("No valid training data after preparation")
                return None
                
            # Split into train/test
            split_idx = int(len(X) * (1 - self.benchmark_test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train new model
            new_model = self._train(X_train, y_train)
            
            # Evaluate new model
            new_metrics = self._evaluate(new_model, X_test, y_test)
            logger.info(
                f"New model metrics: score={new_metrics.score():.3f}, "
                f"dir_acc={new_metrics.directional_accuracy:.1%}, "
                f"profit_acc={new_metrics.profitable_accuracy:.1%}"
            )
            
            # Compare with current model
            should_deploy = True
            if self._current_model is not None:
                old_metrics = self._evaluate(self._current_model, X_test, y_test)
                improvement = (new_metrics.score() - old_metrics.score()) / max(0.001, old_metrics.score()) * 100
                
                logger.info(
                    f"Model comparison: old={old_metrics.score():.3f}, "
                    f"new={new_metrics.score():.3f}, improvement={improvement:.1f}%"
                )
                
                if improvement < self.min_improvement_pct:
                    logger.info(
                        f"New model not better enough ({improvement:.1f}% < {self.min_improvement_pct}%). "
                        "Keeping current model."
                    )
                    should_deploy = False
                    
            if should_deploy:
                # Deploy new model
                version = (self._current_model_info.version + 1) if self._current_model_info else 1
                model_info = self._deploy_model(new_model, new_metrics, version, len(training_data))
                
                logger.info(f"Deployed new AI model v{version}")
                return model_info
                
            return None
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return None
        finally:
            self._is_training = False
            
    def _prepare_training_data(self, data: list[dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data arrays.
        
        Args:
            data: Raw training data.
            
        Returns:
            Tuple of (features, labels) arrays.
        """
        # Feature columns to use
        feature_cols = [
            "price", "spread", "volume_24h", "liquidity",
            "orderbook_imbalance", "bid_depth", "ask_depth",
            "momentum_1h", "momentum_24h", "days_to_resolution"
        ]
        
        # Target: predict 24h price change
        X_list = []
        y_list = []
        
        for example in data:
            # Check if we have the target
            target = example.get("label_price_change_24h")
            if target is None:
                continue
                
            # Extract features
            features = []
            valid = True
            for col in feature_cols:
                val = example.get(col)
                if val is None:
                    valid = False
                    break
                features.append(float(val))
                
            if valid:
                X_list.append(features)
                y_list.append(float(target))
                
        if len(X_list) == 0:
            return None, None
            
        return np.array(X_list), np.array(y_list)
    
    def _train(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train a model.
        
        Args:
            X: Feature array.
            y: Label array.
            
        Returns:
            Trained model.
        """
        try:
            # Try gradient boosting first
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(X, y)
            return model
        except ImportError:
            # Fallback to simple model
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
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
        
        # Directional accuracy: did we predict the right direction?
        correct_direction = np.sum((predictions > 0) == (y > 0))
        directional_accuracy = float(correct_direction / len(y))
        
        # Profitable accuracy: would following this prediction be profitable?
        # If prediction > 0, buy; if actual > 0, profit
        profitable = np.sum(predictions * y > 0)  # Same sign = profit
        profitable_accuracy = float(profitable / len(y))
        
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
        
        Args:
            features: Feature dictionary.
            
        Returns:
            Predicted price change, or None if no model.
        """
        if self._current_model is None:
            return None
            
        feature_cols = [
            "price", "spread", "volume_24h", "liquidity",
            "orderbook_imbalance", "bid_depth", "ask_depth",
            "momentum_1h", "momentum_24h", "days_to_resolution"
        ]
        
        try:
            X = []
            for col in feature_cols:
                val = features.get(col, 0)
                X.append(float(val) if val is not None else 0.0)
                
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
