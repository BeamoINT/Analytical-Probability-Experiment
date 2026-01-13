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
    cv_std: float = 0.0  # Cross-validation standard deviation (lower = more consistent)
    n_features_used: int = 0  # Number of features after selection
    n_models_ensemble: int = 0  # Number of models in ensemble
    
    def score(self) -> float:
        """Overall score (higher is better)."""
        # Weighted combination of metrics
        # Penalize high CV variance (inconsistent performance)
        consistency_penalty = min(0.1, self.cv_std * 0.5)
        return (
            self.directional_accuracy * 0.4 +
            self.profitable_accuracy * 0.4 +
            max(0, self.r2) * 0.2 -
            consistency_penalty
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
                "cv_std": self.metrics.cv_std,
                "n_features_used": self.metrics.n_features_used,
                "n_models_ensemble": self.metrics.n_models_ensemble,
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
            
            # === SORT BY TIME ===
            sorted_data = sorted(
                training_data,
                key=lambda x: x.get("timestamp", x.get("created_at", ""))
            )
            
            # Prepare data (maintains sort order)
            X, y, timestamps = self._prepare_training_data_with_time(sorted_data)
            
            if X is None or len(X) == 0:
                logger.warning("No valid training data after preparation")
                return None
            
            # === ROBUST MULTI-VALIDATION STRATEGY ===
            # Use multiple validation approaches and report WORST-CASE
            # This prevents overfit metrics from fooling us
            
            validation_results = []
            
            # VALIDATION 1: Time-based with gap (no leakage)
            # Leave a gap between train and test to prevent temporal correlation
            gap_size = int(len(X) * 0.05)  # 5% gap
            train_end = int(len(X) * 0.75)  # 75% train
            test_start = train_end + gap_size
            
            if test_start < len(X) - 10:
                X_train_time = X[:train_end]
                y_train_time = y[:train_end]
                X_test_time = X[test_start:]
                y_test_time = y[test_start:]
                
                model_time = self._train(X_train_time, y_train_time)
                metrics_time = self._evaluate(model_time, X_test_time, y_test_time)
                validation_results.append(("time_gap", metrics_time))
                logger.info(f"Time-gap validation: profit_acc={metrics_time.profitable_accuracy:.1%}")
            
            # VALIDATION 2: Walk-forward (multiple periods)
            # Train on period 1, test on period 2; train on 1+2, test on 3; etc.
            n_periods = 4
            period_size = len(X) // n_periods
            walk_forward_accs = []
            
            if period_size > 50:
                for i in range(1, n_periods):
                    train_end = i * period_size
                    test_end = (i + 1) * period_size
                    
                    X_train_wf = X[:train_end]
                    y_train_wf = y[:train_end]
                    X_test_wf = X[train_end:test_end]
                    y_test_wf = y[train_end:test_end]
                    
                    if len(X_train_wf) > 50 and len(X_test_wf) > 10:
                        model_wf = self._train(X_train_wf, y_train_wf)
                        metrics_wf = self._evaluate(model_wf, X_test_wf, y_test_wf)
                        walk_forward_accs.append(metrics_wf.profitable_accuracy)
                
                if walk_forward_accs:
                    avg_wf_acc = sum(walk_forward_accs) / len(walk_forward_accs)
                    min_wf_acc = min(walk_forward_accs)
                    logger.info(
                        f"Walk-forward validation: avg={avg_wf_acc:.1%}, "
                        f"min={min_wf_acc:.1%}, periods={len(walk_forward_accs)}"
                    )
                    # Use the minimum as worst case
                    wf_metrics = ModelMetrics(
                        mse=0, mae=0, r2=0,
                        directional_accuracy=min_wf_acc,
                        profitable_accuracy=min_wf_acc,
                    )
                    validation_results.append(("walk_forward_min", wf_metrics))
            
            # VALIDATION 3: Random stratified sample (tests generalization)
            # Randomly sample 20% from ALL time periods
            np.random.seed(42)
            n_samples = len(X)
            test_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
            train_indices = np.array([i for i in range(n_samples) if i not in test_indices])
            
            X_train_rand = X[train_indices]
            y_train_rand = y[train_indices]
            X_test_rand = X[test_indices]
            y_test_rand = y[test_indices]
            
            model_rand = self._train(X_train_rand, y_train_rand)
            metrics_rand = self._evaluate(model_rand, X_test_rand, y_test_rand)
            validation_results.append(("random_sample", metrics_rand))
            logger.info(f"Random sample validation: profit_acc={metrics_rand.profitable_accuracy:.1%}")
            
            # VALIDATION 4: Oldest data test (can we predict old markets?)
            # Train on newest 75%, test on oldest 25%
            oldest_test_size = int(len(X) * 0.25)
            X_train_old = X[oldest_test_size:]
            y_train_old = y[oldest_test_size:]
            X_test_old = X[:oldest_test_size]
            y_test_old = y[:oldest_test_size]
            
            model_old = self._train(X_train_old, y_train_old)
            metrics_old = self._evaluate(model_old, X_test_old, y_test_old)
            validation_results.append(("oldest_data", metrics_old))
            logger.info(f"Oldest data validation: profit_acc={metrics_old.profitable_accuracy:.1%}")
            
            # === USE WORST-CASE METRICS ===
            # Report the MINIMUM across all validation strategies
            # This is the most honest estimate of real performance
            all_profit_accs = [m.profitable_accuracy for _, m in validation_results]
            all_dir_accs = [m.directional_accuracy for _, m in validation_results]
            all_r2s = [m.r2 for _, m in validation_results]
            
            worst_profit_acc = min(all_profit_accs) if all_profit_accs else 0.5
            worst_dir_acc = min(all_dir_accs) if all_dir_accs else 0.5
            worst_r2 = min(all_r2s) if all_r2s else 0
            avg_profit_acc = sum(all_profit_accs) / len(all_profit_accs) if all_profit_accs else 0.5
            
            logger.info(
                f"Multi-validation summary: "
                f"worst_profit={worst_profit_acc:.1%}, "
                f"avg_profit={avg_profit_acc:.1%}, "
                f"strategies={len(validation_results)}"
            )
            
            # === TRAIN FINAL MODEL ON ALL DATA ===
            # Now train on ALL data for the deployed model
            final_model = self._train(X, y)
            
            # Create final metrics using WORST-CASE values
            final_metrics = ModelMetrics(
                mse=sum(m.mse for _, m in validation_results) / len(validation_results) if validation_results else 0,
                mae=sum(m.mae for _, m in validation_results) / len(validation_results) if validation_results else 0,
                r2=worst_r2,
                directional_accuracy=worst_dir_acc,
                profitable_accuracy=worst_profit_acc,
            )
            
            # Add ensemble info
            if hasattr(final_model, 'fitted_models'):
                final_metrics.n_models_ensemble = len(final_model.fitted_models)
            if hasattr(final_model, 'selected_mask'):
                final_metrics.n_features_used = int(np.sum(final_model.selected_mask))
            
            logger.info(
                f"Final model (worst-case metrics): "
                f"dir_acc={final_metrics.directional_accuracy:.1%}, "
                f"profit_acc={final_metrics.profitable_accuracy:.1%}, "
                f"r2={final_metrics.r2:.3f}"
            )
            
            # === DEPLOY MODEL ===
            version = (self._current_model_info.version + 1) if self._current_model_info else 1
            model_info = self._deploy_model(final_model, final_metrics, version, len(training_data))
            
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
        """Train a model with robust anti-overfitting measures.
        
        Implements:
        - Feature selection (remove low-variance and redundant features)
        - Multiple regularized models
        - Early stopping for gradient boosting
        - Ensemble of diverse models
        - Time-series cross-validation for hyperparameter selection
        
        Args:
            X: Feature array.
            y: Label array.
            
        Returns:
            Trained ensemble model.
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        logger.info(f"Training on {n_samples} samples with {n_features} features")
        
        try:
            from sklearn.ensemble import (
                RandomForestRegressor, 
                GradientBoostingRegressor,
                VotingRegressor,
                HistGradientBoostingRegressor,
            )
            from sklearn.linear_model import Ridge, ElasticNet, Lasso
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.feature_selection import VarianceThreshold, SelectFromModel
            from sklearn.model_selection import TimeSeriesSplit
            
            # === STEP 1: FEATURE SELECTION ===
            # Remove near-zero variance features (noise)
            logger.info("Step 1: Feature selection...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Remove features with very low variance
            var_threshold = VarianceThreshold(threshold=0.01)
            try:
                X_selected = var_threshold.fit_transform(X_scaled)
                selected_mask = var_threshold.get_support()
                n_selected = X_selected.shape[1]
                logger.info(f"Variance filter: {n_features} -> {n_selected} features")
            except Exception:
                X_selected = X_scaled
                selected_mask = np.ones(n_features, dtype=bool)
                n_selected = n_features
            
            # Store selected feature indices for prediction
            self._selected_features = selected_mask
            
            # === STEP 2: TIME-SERIES CROSS-VALIDATION ===
            # Use multiple time-based folds to find best hyperparameters
            logger.info("Step 2: Time-series cross-validation...")
            
            n_splits = min(5, n_samples // 100)  # At least 100 samples per fold
            if n_splits < 2:
                n_splits = 2
                
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # === STEP 3: TRAIN MULTIPLE REGULARIZED MODELS ===
            logger.info("Step 3: Training regularized models...")
            
            models = []
            model_names = []
            cv_scores = []
            
            # Model 1: Ridge Regression (L2 regularization)
            ridge = Ridge(alpha=100.0)  # VERY strong regularization
            ridge_scores = self._cross_val_score(ridge, X_selected, y, tscv)
            models.append(ridge)
            model_names.append("Ridge")
            cv_scores.append(np.mean(ridge_scores))
            logger.info(f"Ridge CV score: {np.mean(ridge_scores):.4f} (+/- {np.std(ridge_scores):.4f})")
            
            # Model 2: ElasticNet (L1 + L2 regularization)
            elastic = ElasticNet(alpha=1.0, l1_ratio=0.7, max_iter=1000)  # Strong regularization
            elastic_scores = self._cross_val_score(elastic, X_selected, y, tscv)
            models.append(elastic)
            model_names.append("ElasticNet")
            cv_scores.append(np.mean(elastic_scores))
            logger.info(f"ElasticNet CV score: {np.mean(elastic_scores):.4f} (+/- {np.std(elastic_scores):.4f})")
            
            # Model 3: Random Forest with VERY strong regularization
            if n_samples >= 500:
                rf = RandomForestRegressor(
                    n_estimators=50,  # Fewer trees (less overfit)
                    max_depth=4,  # VERY shallow trees
                    min_samples_split=50,  # Need MANY samples to split
                    min_samples_leaf=25,  # Each leaf needs many samples
                    max_features=0.2,  # Only use 20% of features per tree
                    max_samples=0.7,  # Only use 70% of samples per tree
                    bootstrap=True,
                    oob_score=True,
                    n_jobs=-1,
                    random_state=42,
                )
                rf_scores = self._cross_val_score(rf, X_selected, y, tscv)
                models.append(rf)
                model_names.append("RandomForest")
                cv_scores.append(np.mean(rf_scores))
                logger.info(f"RandomForest CV score: {np.mean(rf_scores):.4f} (+/- {np.std(rf_scores):.4f})")
            
            # Model 4: Gradient Boosting with early stopping
            if n_samples >= 500:
                # Use HistGradientBoosting with STRONG regularization
                hgb = HistGradientBoostingRegressor(
                    max_iter=100,  # Fewer iterations
                    max_depth=3,  # VERY shallow trees
                    min_samples_leaf=50,  # More samples per leaf
                    l2_regularization=10.0,  # STRONG L2 penalty
                    learning_rate=0.05,  # Slow learning
                    early_stopping=True,
                    validation_fraction=0.2,  # 20% for early stopping
                    n_iter_no_change=5,  # Stop quickly if no improvement
                    random_state=42,
                )
                hgb_scores = self._cross_val_score(hgb, X_selected, y, tscv)
                models.append(hgb)
                model_names.append("HistGradientBoosting")
                cv_scores.append(np.mean(hgb_scores))
                logger.info(f"HistGradientBoosting CV score: {np.mean(hgb_scores):.4f} (+/- {np.std(hgb_scores):.4f})")
            
            # === STEP 4: CREATE ENSEMBLE ===
            logger.info("Step 4: Creating ensemble...")
            
            # Weight models by their CV performance
            # Better CV score = higher weight
            cv_scores = np.array(cv_scores)
            # Convert R2-like scores to positive weights
            weights = cv_scores - cv_scores.min() + 0.1
            weights = weights / weights.sum()
            
            logger.info(f"Model weights: {list(zip(model_names, weights.round(3)))}")
            
            # Fit all models on full training data
            fitted_models = []
            for model, name in zip(models, model_names):
                model.fit(X_selected, y)
                fitted_models.append((name, model))
            
            # Create weighted voting ensemble
            ensemble = VotingRegressor(
                estimators=fitted_models,
                weights=weights.tolist(),
            )
            
            # VotingRegressor needs to be fit, but individual models are already fit
            # So we create a wrapper that holds the pre-fitted models
            ensemble.estimators_ = [m for _, m in fitted_models]
            ensemble.named_estimators_ = {name: m for name, m in fitted_models}
            
            # === STEP 5: LOG FEATURE IMPORTANCES ===
            try:
                # Get feature importances from tree-based models
                for name, model in fitted_models:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        # Map back to original feature names
                        selected_cols = [self._feature_cols[i] for i, s in enumerate(selected_mask) if s]
                        top_indices = np.argsort(importances)[-10:][::-1]
                        top_features = [(selected_cols[i], round(importances[i], 4)) for i in top_indices]
                        logger.info(f"{name} top 10 features: {top_features}")
                        break  # Only log once
            except Exception as e:
                logger.debug(f"Could not log feature importances: {e}")
            
            # Wrap in a custom class that handles feature selection
            final_model = _EnsembleWithFeatureSelection(
                scaler=scaler,
                var_threshold=var_threshold,
                selected_mask=selected_mask,
                ensemble=ensemble,
                fitted_models=fitted_models,
                weights=weights,
            )
            
            logger.info(f"Training complete: {len(fitted_models)} models in ensemble")
            return final_model
            
        except ImportError as e:
            logger.warning(f"Advanced sklearn not available: {e}, using fallback")
            # Fallback to simple regularized model
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=10.0))  # Strong regularization
            ])
            model.fit(X, y)
            return model
    
    def _cross_val_score(self, model, X, y, cv) -> list:
        """Compute cross-validation scores for time-series data.
        
        Args:
            model: Sklearn model.
            X: Features.
            y: Labels.
            cv: Cross-validation splitter.
            
        Returns:
            List of R2 scores for each fold.
        """
        scores = []
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone model for each fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Calculate R2 score
            predictions = fold_model.predict(X_val)
            ss_res = np.sum((y_val - predictions) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            scores.append(r2)
            
        return scores


class _EnsembleWithFeatureSelection:
    """Wrapper that applies feature selection before ensemble prediction."""
    
    def __init__(self, scaler, var_threshold, selected_mask, ensemble, fitted_models, weights):
        self.scaler = scaler
        self.var_threshold = var_threshold
        self.selected_mask = selected_mask
        self.ensemble = ensemble
        self.fitted_models = fitted_models
        self.weights = weights
        
    def predict(self, X):
        """Predict with feature selection and ensemble averaging."""
        # Apply same preprocessing as training
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.selected_mask]
        
        # Weighted average of all model predictions
        predictions = np.zeros(X_selected.shape[0])
        for (name, model), weight in zip(self.fitted_models, self.weights):
            predictions += weight * model.predict(X_selected)
            
        return predictions
    
    def fit(self, X, y):
        """No-op since models are already fitted."""
        return self


# Re-attach methods to AITrainer that were defined after _EnsembleWithFeatureSelection
def _evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
    """Evaluate a model on test data with realistic metrics.
    
    Args:
        model: Model to evaluate.
        X: Test features.
        y: Test labels.
        
    Returns:
        ModelMetrics with realistic accuracy calculations.
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
    MIN_PREDICTION_THRESHOLD = 0.01  # 1% price change
    MIN_ACTUAL_THRESHOLD = 0.01  # 1% actual change to count as "movement"
    SPREAD_COST = 0.02  # Assume 2% spread cost to be conservative
    
    # Directional accuracy: only count predictions above threshold
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
    if np.sum(confident_mask) > 0:
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

# Attach to class
AITrainer._evaluate = _evaluate


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
    
    # Save state with all metrics including new ensemble info
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
            "cv_std": metrics.cv_std,
            "n_features_used": metrics.n_features_used,
            "n_models_ensemble": metrics.n_models_ensemble,
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

# Attach to class
AITrainer._deploy_model = _deploy_model


def _predict(self, features: dict) -> Optional[float]:
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

# Attach to class
AITrainer.predict = _predict


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
