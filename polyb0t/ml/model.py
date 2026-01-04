"""ML models for price prediction and strategy intelligence.

Implements gradient boosting and ensemble models for predicting
future price movements with proper validation and monitoring.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import LightGBM, gracefully handle if not installed
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not installed. ML features will use fallback.")
    LIGHTGBM_AVAILABLE = False

# Try to import sklearn for metrics
try:
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Using basic metrics.")
    SKLEARN_AVAILABLE = False


class PricePredictor:
    """Gradient boosting model for price prediction.
    
    Predicts future price changes using LightGBM with conservative
    parameters optimized for financial time series.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize predictor.
        
        Args:
            model_path: Path to load existing model from.
        """
        self.model = None
        self.feature_names: List[str] = []
        self.training_metrics: Dict = {}
        self.model_version = "1.0"
        
        if model_path and LIGHTGBM_AVAILABLE:
            self.load_model(model_path)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Train model on historical data.
        
        Args:
            X: Feature dataframe.
            y: Target series (future returns).
            validation_split: Fraction to use for validation.
            params: Optional custom parameters.
            
        Returns:
            Dictionary of training metrics.
        """
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM not available, cannot train model")
            return {"error": "lightgbm_not_installed"}
        
        # Split data (time-aware: train on earlier, validate on later)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training on {len(X_train)} examples, validating on {len(X_val)}")
        
        # Conservative parameters for trading
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,  # Conservative learning rate
                'num_leaves': 31,
                'max_depth': 5,  # Prevent overfitting
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,  # Random feature sampling
                'bagging_fraction': 0.8,  # Random row sampling
                'bagging_freq': 5,
                'lambda_l1': 0.1,  # L1 regularization
                'lambda_l2': 0.1,  # L2 regularization
                'verbose': -1,
                'seed': 42,
                'n_jobs': 4,
            }
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=list(X_train.columns),
            free_raw_data=False,
        )
        
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            free_raw_data=False,
        )
        
        # Train with early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress training logs
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks,
        )
        
        self.feature_names = list(X.columns)
        
        # Compute metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        metrics = self._compute_metrics(y_train, y_pred_train, y_val, y_pred_val)
        self.training_metrics = metrics
        
        logger.info(
            f"Training complete: val_r2={metrics['val_r2']:.4f}, "
            f"val_rmse={metrics['val_rmse']:.4f}"
        )
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> float:
        """Predict future price change for single example.
        
        Args:
            X: Feature dataframe (single row or multiple rows).
            
        Returns:
            Predicted price change (or mean of predictions if multiple rows).
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if not LIGHTGBM_AVAILABLE:
            return 0.0  # Fallback
        
        # Ensure features match training
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}, filling with 0")
                for feat in missing_features:
                    X[feat] = 0.0
            
            # Reorder to match training
            X = X[self.feature_names]
        
        predictions = self.model.predict(X)
        
        if isinstance(predictions, np.ndarray):
            return float(predictions[0] if len(predictions) == 1 else np.mean(predictions))
        else:
            return float(predictions)
    
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Predict for multiple examples.
        
        Args:
            X: Feature dataframe.
            
        Returns:
            Array of predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if not LIGHTGBM_AVAILABLE:
            return np.zeros(len(X))
        
        # Ensure features match
        if self.feature_names:
            for feat in self.feature_names:
                if feat not in X.columns:
                    X[feat] = 0.0
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_n: int = 20,
    ) -> Dict[str, float]:
        """Get feature importance scores.
        
        Args:
            importance_type: 'gain' or 'split'.
            top_n: Return top N features.
            
        Returns:
            Dictionary of feature -> importance.
        """
        if self.model is None or not LIGHTGBM_AVAILABLE:
            return {}
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        # Sort by importance
        feature_importance = dict(zip(self.feature_names, importance))
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
    
    def save_model(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: File path to save to.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if not LIGHTGBM_AVAILABLE:
            logger.warning("Cannot save model: LightGBM not available")
            return
        
        # Save model
        self.model.save_model(path)
        
        # Save metadata
        metadata_path = Path(path).with_suffix('.json')
        metadata = {
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'model_version': self.model_version,
            'num_features': len(self.feature_names),
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: File path to load from.
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("Cannot load model: LightGBM not available")
            return
        
        try:
            self.model = lgb.Booster(model_file=path)
            
            # Load metadata if available
            metadata_path = Path(path).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
                    self.training_metrics = metadata.get('training_metrics', {})
                    self.model_version = metadata.get('model_version', '1.0')
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise
    
    def _compute_metrics(
        self,
        y_train: pd.Series,
        y_pred_train: np.ndarray,
        y_val: pd.Series,
        y_pred_val: np.ndarray,
    ) -> Dict:
        """Compute comprehensive evaluation metrics."""
        if SKLEARN_AVAILABLE:
            train_r2 = r2_score(y_train, y_pred_train)
            val_r2 = r2_score(y_val, y_pred_val)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
        else:
            # Basic implementations
            train_r2 = self._r2(y_train.values, y_pred_train)
            val_r2 = self._r2(y_val.values, y_pred_val)
            train_rmse = np.sqrt(np.mean((y_train.values - y_pred_train) ** 2))
            val_rmse = np.sqrt(np.mean((y_val.values - y_pred_val) ** 2))
            train_mae = np.mean(np.abs(y_train.values - y_pred_train))
            val_mae = np.mean(np.abs(y_val.values - y_pred_val))
        
        # Direction accuracy (for trading: did we predict direction correctly?)
        train_direction_acc = np.mean((np.sign(y_train.values) == np.sign(y_pred_train)))
        val_direction_acc = np.mean((np.sign(y_val.values) == np.sign(y_pred_val)))
        
        return {
            'train_r2': float(train_r2),
            'val_r2': float(val_r2),
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse),
            'train_mae': float(train_mae),
            'val_mae': float(val_mae),
            'train_direction_acc': float(train_direction_acc),
            'val_direction_acc': float(val_direction_acc),
            'train_size': len(y_train),
            'val_size': len(y_val),
        }
    
    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score manually."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


class EnsemblePredictor:
    """Ensemble of multiple models for robust predictions.
    
    Combines predictions from multiple models (e.g., different
    time horizons, different regimes) for improved robustness.
    """
    
    def __init__(self, model_dir: Path):
        """Initialize ensemble.
        
        Args:
            model_dir: Directory containing individual models.
        """
        self.model_dir = model_dir
        self.models: Dict[str, PricePredictor] = {}
        self.weights: Dict[str, float] = {}
        
    def add_model(self, name: str, model: PricePredictor, weight: float = 1.0) -> None:
        """Add model to ensemble.
        
        Args:
            name: Model identifier.
            model: Trained PricePredictor.
            weight: Weight for ensemble averaging (higher = more influence).
        """
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model '{name}' to ensemble (weight={weight})")
    
    def predict(self, X: pd.DataFrame) -> float:
        """Predict using weighted ensemble.
        
        Args:
            X: Feature dataframe.
            
        Returns:
            Weighted average prediction.
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.weights[name])
            except Exception as e:
                logger.warning(f"Model '{name}' failed to predict: {e}")
        
        if not predictions:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_pred = sum(p * w for p, w in zip(predictions, weights)) / total_weight
        
        return weighted_pred
    
    def load_models(self, pattern: str = "model_*.txt") -> None:
        """Load all models matching pattern from model_dir.
        
        Args:
            pattern: Glob pattern for model files.
        """
        model_files = list(self.model_dir.glob(pattern))
        
        for model_file in model_files:
            name = model_file.stem
            try:
                model = PricePredictor(str(model_file))
                
                # Weight by validation R² (better models get more weight)
                r2 = model.training_metrics.get('val_r2', 0.5)
                weight = max(0.1, r2)  # Minimum weight 0.1
                
                self.add_model(name, model, weight)
                
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models into ensemble")

