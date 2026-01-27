"""Deep Ensemble combining Neural Network, XGBoost, and LightGBM.

This module implements a stacking ensemble with:
- Level 0: TabularMLP (NN), XGBoost, LightGBM
- Level 1: Ridge Logistic Regression meta-learner
- Final: Platt scaling calibration
"""

import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import torch
    from polyb0t.ml.moe.neural_expert import NeuralExpertTrainer, DEVICE
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    DEVICE = None

from polyb0t.ml.moe.augmentation import TabularAugmenter

logger = logging.getLogger(__name__)


# Default configurations
XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}

LIGHTGBM_CONFIG = {
    "n_estimators": 500,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

NEURAL_CONFIG = {
    "hidden_dims": [256, 128, 64],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 200,
    "early_stopping_patience": 15,
}


@dataclass
class EnsembleMetrics:
    """Metrics for the ensemble model."""

    nn_val_acc: float = 0.0
    xgb_val_acc: float = 0.0
    lgb_val_acc: float = 0.0
    ensemble_val_acc: float = 0.0
    meta_val_acc: float = 0.0

    nn_train_time: float = 0.0
    xgb_train_time: float = 0.0
    lgb_train_time: float = 0.0
    total_train_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nn_val_acc": self.nn_val_acc,
            "xgb_val_acc": self.xgb_val_acc,
            "lgb_val_acc": self.lgb_val_acc,
            "ensemble_val_acc": self.ensemble_val_acc,
            "meta_val_acc": self.meta_val_acc,
            "nn_train_time": self.nn_train_time,
            "xgb_train_time": self.xgb_train_time,
            "lgb_train_time": self.lgb_train_time,
            "total_train_time": self.total_train_time,
        }


class DeepExpertEnsemble:
    """Stacking ensemble with Neural Network, XGBoost, and LightGBM.

    Architecture:
        Level 0 (Base Models):
            - TabularMLP (PyTorch) - weight: 0.4
            - XGBoost - weight: 0.35
            - LightGBM - weight: 0.25

        Level 1 (Meta-Learner):
            - Ridge Logistic Regression on OOF predictions

        Final:
            - Platt Scaling Calibration
    """

    def __init__(
        self,
        nn_weight: float = 0.4,
        xgb_weight: float = 0.35,
        lgb_weight: float = 0.25,
        use_neural: bool = True,
        use_xgboost: bool = True,
        use_lightgbm: bool = True,
        use_stacking: bool = True,
        nn_config: Optional[Dict] = None,
        xgb_config: Optional[Dict] = None,
        lgb_config: Optional[Dict] = None,
    ):
        """Initialize ensemble.

        Args:
            nn_weight: Weight for neural network predictions
            xgb_weight: Weight for XGBoost predictions
            lgb_weight: Weight for LightGBM predictions
            use_neural: Whether to include neural network
            use_xgboost: Whether to include XGBoost
            use_lightgbm: Whether to include LightGBM
            use_stacking: Whether to use stacking meta-learner
            nn_config: Override neural network config
            xgb_config: Override XGBoost config
            lgb_config: Override LightGBM config
        """
        self.nn_weight = nn_weight
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight

        self.use_neural = use_neural and TORCH_AVAILABLE
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.use_lightgbm = use_lightgbm and LIGHTGBM_AVAILABLE
        self.use_stacking = use_stacking

        self.nn_config = {**NEURAL_CONFIG, **(nn_config or {})}
        self.xgb_config = {**XGBOOST_CONFIG, **(xgb_config or {})}
        self.lgb_config = {**LIGHTGBM_CONFIG, **(lgb_config or {})}

        # Models
        self.nn_trainer: Optional[NeuralExpertTrainer] = None
        self.xgb_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.meta_model: Optional[LogisticRegression] = None
        self.calibrator: Optional[CalibratedClassifierCV] = None

        # Preprocessing
        self.scaler = StandardScaler()
        self.augmenter = TabularAugmenter()

        # Metrics
        self.metrics = EnsembleMetrics()

        # Log availability
        logger.info(
            f"DeepExpertEnsemble initialized: "
            f"NN={self.use_neural}, XGB={self.use_xgboost}, LGB={self.use_lightgbm}"
        )

    def _normalize_weights(self) -> Tuple[float, float, float]:
        """Normalize weights based on which models are active."""
        weights = []
        if self.use_neural:
            weights.append(('nn', self.nn_weight))
        if self.use_xgboost:
            weights.append(('xgb', self.xgb_weight))
        if self.use_lightgbm:
            weights.append(('lgb', self.lgb_weight))

        if not weights:
            raise ValueError("No models enabled in ensemble")

        total = sum(w for _, w in weights)
        normalized = {name: w / total for name, w in weights}

        return (
            normalized.get('nn', 0),
            normalized.get('xgb', 0),
            normalized.get('lgb', 0),
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        val_fraction: float = 0.2,
    ) -> EnsembleMetrics:
        """Train the ensemble with all base models and meta-learner.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
            sample_weights: Optional sample weights
            val_fraction: Fraction of data for validation

        Returns:
            EnsembleMetrics with training results
        """
        import time
        start_time = time.time()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time-based split (train on older, validate on newer)
        n_samples = len(X)
        val_size = int(n_samples * val_fraction)
        train_idx = np.arange(n_samples - val_size)
        val_idx = np.arange(n_samples - val_size, n_samples)

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if sample_weights is not None:
            w_train = sample_weights[train_idx]
            w_val = sample_weights[val_idx]
        else:
            w_train = None
            w_val = None

        # Store OOF predictions for stacking
        oof_preds = np.zeros((len(X_train), 0))
        val_preds = np.zeros((len(X_val), 0))

        # Train Neural Network
        if self.use_neural:
            nn_start = time.time()
            self.nn_trainer = NeuralExpertTrainer(**self.nn_config)

            # Further split train for NN validation
            nn_val_size = int(len(X_train) * 0.15)
            X_nn_train = X_train[:-nn_val_size]
            y_nn_train = y_train[:-nn_val_size]
            X_nn_val = X_train[-nn_val_size:]
            y_nn_val = y_train[-nn_val_size:]
            w_nn_train = w_train[:-nn_val_size] if w_train is not None else None

            _, nn_metrics = self.nn_trainer.train(
                X_nn_train, y_nn_train, X_nn_val, y_nn_val, w_nn_train
            )

            nn_train_probs = self.nn_trainer.predict_proba(X_train)[:, 1:]
            nn_val_probs = self.nn_trainer.predict_proba(X_val)[:, 1:]

            oof_preds = np.hstack([oof_preds, nn_train_probs])
            val_preds = np.hstack([val_preds, nn_val_probs])

            self.metrics.nn_val_acc = (
                (self.nn_trainer.predict(X_val) == y_val).mean()
            )
            self.metrics.nn_train_time = time.time() - nn_start
            logger.info(f"NN trained: val_acc={self.metrics.nn_val_acc:.3f}")

        # Train XGBoost
        if self.use_xgboost:
            xgb_start = time.time()

            # Prepare eval set for early stopping
            eval_set = [(X_val, y_val)]

            # Remove early_stopping_rounds from config for fit
            xgb_fit_config = {k: v for k, v in self.xgb_config.items()
                             if k != 'early_stopping_rounds'}

            self.xgb_model = xgb.XGBClassifier(**xgb_fit_config)
            self.xgb_model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=eval_set,
                verbose=False,
            )

            xgb_train_probs = self.xgb_model.predict_proba(X_train)[:, 1:]
            xgb_val_probs = self.xgb_model.predict_proba(X_val)[:, 1:]

            oof_preds = np.hstack([oof_preds, xgb_train_probs])
            val_preds = np.hstack([val_preds, xgb_val_probs])

            self.metrics.xgb_val_acc = (
                (self.xgb_model.predict(X_val) == y_val).mean()
            )
            self.metrics.xgb_train_time = time.time() - xgb_start
            logger.info(f"XGBoost trained: val_acc={self.metrics.xgb_val_acc:.3f}")

        # Train LightGBM
        if self.use_lightgbm:
            lgb_start = time.time()

            self.lgb_model = lgb.LGBMClassifier(**self.lgb_config)
            self.lgb_model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            lgb_train_probs = self.lgb_model.predict_proba(X_train)[:, 1:]
            lgb_val_probs = self.lgb_model.predict_proba(X_val)[:, 1:]

            oof_preds = np.hstack([oof_preds, lgb_train_probs])
            val_preds = np.hstack([val_preds, lgb_val_probs])

            self.metrics.lgb_val_acc = (
                (self.lgb_model.predict(X_val) == y_val).mean()
            )
            self.metrics.lgb_train_time = time.time() - lgb_start
            logger.info(f"LightGBM trained: val_acc={self.metrics.lgb_val_acc:.3f}")

        # Compute weighted ensemble accuracy (before stacking)
        if val_preds.shape[1] > 0:
            nn_w, xgb_w, lgb_w = self._normalize_weights()

            weighted_val = np.zeros(len(X_val))
            col_idx = 0
            if self.use_neural:
                weighted_val += nn_w * val_preds[:, col_idx]
                col_idx += 1
            if self.use_xgboost:
                weighted_val += xgb_w * val_preds[:, col_idx]
                col_idx += 1
            if self.use_lightgbm:
                weighted_val += lgb_w * val_preds[:, col_idx]

            ensemble_preds = (weighted_val > 0.5).astype(int)
            self.metrics.ensemble_val_acc = (ensemble_preds == y_val).mean()

        # Train meta-learner (stacking)
        if self.use_stacking and oof_preds.shape[1] > 0:
            # Add original features to OOF predictions
            stacking_features_train = np.hstack([oof_preds, X_train])
            stacking_features_val = np.hstack([val_preds, X_val])

            self.meta_model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                solver='lbfgs',
                max_iter=1000,
            )
            self.meta_model.fit(stacking_features_train, y_train)

            meta_preds = self.meta_model.predict(stacking_features_val)
            self.metrics.meta_val_acc = (meta_preds == y_val).mean()
            logger.info(f"Meta-learner trained: val_acc={self.metrics.meta_val_acc:.3f}")

        self.metrics.total_train_time = time.time() - start_time
        logger.info(
            f"Ensemble training complete in {self.metrics.total_train_time:.1f}s. "
            f"Final val_acc: {max(self.metrics.meta_val_acc, self.metrics.ensemble_val_acc):.3f}"
        )

        return self.metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, 2)
        """
        X_scaled = self.scaler.transform(X)

        # Collect base model predictions
        base_preds = []

        if self.use_neural and self.nn_trainer is not None:
            nn_probs = self.nn_trainer.predict_proba(X_scaled)[:, 1]
            base_preds.append(nn_probs)

        if self.use_xgboost and self.xgb_model is not None:
            xgb_probs = self.xgb_model.predict_proba(X_scaled)[:, 1]
            base_preds.append(xgb_probs)

        if self.use_lightgbm and self.lgb_model is not None:
            lgb_probs = self.lgb_model.predict_proba(X_scaled)[:, 1]
            base_preds.append(lgb_probs)

        if not base_preds:
            # No models trained, return 0.5
            return np.full((len(X), 2), 0.5)

        base_preds = np.column_stack(base_preds)

        # Use meta-learner if available
        if self.use_stacking and self.meta_model is not None:
            stacking_features = np.hstack([base_preds, X_scaled])
            probs = self.meta_model.predict_proba(stacking_features)
            return probs

        # Otherwise use weighted average
        nn_w, xgb_w, lgb_w = self._normalize_weights()

        weighted = np.zeros(len(X))
        col_idx = 0
        if self.use_neural:
            weighted += nn_w * base_preds[:, col_idx]
            col_idx += 1
        if self.use_xgboost:
            weighted += xgb_w * base_preds[:, col_idx]
            col_idx += 1
        if self.use_lightgbm:
            weighted += lgb_w * base_preds[:, col_idx]

        # Return as 2-class probabilities
        probs = np.column_stack([1 - weighted, weighted])
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        state = {
            'nn_weight': self.nn_weight,
            'xgb_weight': self.xgb_weight,
            'lgb_weight': self.lgb_weight,
            'use_neural': self.use_neural,
            'use_xgboost': self.use_xgboost,
            'use_lightgbm': self.use_lightgbm,
            'use_stacking': self.use_stacking,
            'nn_config': self.nn_config,
            'xgb_config': self.xgb_config,
            'lgb_config': self.lgb_config,
            'scaler': self.scaler,
            'metrics': self.metrics.to_dict(),
        }

        # Save models separately
        if self.nn_trainer is not None and self.nn_trainer.model is not None:
            state['nn_state'] = {
                'model_state_dict': self.nn_trainer.model.state_dict(),
                'input_dim': self.nn_trainer.model.input_dim,
                'hidden_dims': self.nn_trainer.model.hidden_dims,
                'dropout': self.nn_trainer.model.dropout_rate,
            }

        if self.xgb_model is not None:
            state['xgb_model'] = self.xgb_model

        if self.lgb_model is not None:
            state['lgb_model'] = self.lgb_model

        if self.meta_model is not None:
            state['meta_model'] = self.meta_model

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "DeepExpertEnsemble":
        """Load ensemble from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        ensemble = cls(
            nn_weight=state['nn_weight'],
            xgb_weight=state['xgb_weight'],
            lgb_weight=state['lgb_weight'],
            use_neural=state['use_neural'],
            use_xgboost=state['use_xgboost'],
            use_lightgbm=state['use_lightgbm'],
            use_stacking=state['use_stacking'],
            nn_config=state.get('nn_config'),
            xgb_config=state.get('xgb_config'),
            lgb_config=state.get('lgb_config'),
        )

        ensemble.scaler = state['scaler']

        # Load neural network
        if 'nn_state' in state and TORCH_AVAILABLE:
            from polyb0t.ml.moe.neural_expert import TabularMLP, NeuralExpertTrainer

            nn_state = state['nn_state']
            ensemble.nn_trainer = NeuralExpertTrainer(
                input_dim=nn_state['input_dim'],
                hidden_dims=nn_state['hidden_dims'],
                dropout=nn_state['dropout'],
            )
            ensemble.nn_trainer.model = TabularMLP(
                input_dim=nn_state['input_dim'],
                hidden_dims=nn_state['hidden_dims'],
                dropout=nn_state['dropout'],
            )
            ensemble.nn_trainer.model.load_state_dict(nn_state['model_state_dict'])
            ensemble.nn_trainer.model.to(DEVICE)

        if 'xgb_model' in state:
            ensemble.xgb_model = state['xgb_model']

        if 'lgb_model' in state:
            ensemble.lgb_model = state['lgb_model']

        if 'meta_model' in state:
            ensemble.meta_model = state['meta_model']

        return ensemble


def get_deep_ensemble(
    nn_weight: float = 0.4,
    xgb_weight: float = 0.35,
    lgb_weight: float = 0.25,
    use_stacking: bool = True,
) -> DeepExpertEnsemble:
    """Factory function to create a deep ensemble."""
    return DeepExpertEnsemble(
        nn_weight=nn_weight,
        xgb_weight=xgb_weight,
        lgb_weight=lgb_weight,
        use_stacking=use_stacking,
    )
