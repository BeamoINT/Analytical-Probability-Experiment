"""Hyperparameter search using Optuna for MoE experts.

Implements Bayesian optimization with:
- TPE sampler for efficient search
- Median pruner for early stopping of bad trials
- Configurable search spaces for NN, XGBoost, LightGBM
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""

    n_trials: int = 50
    timeout: int = 3600  # 1 hour max
    n_jobs: int = 1  # Parallel trials (set > 1 for multiprocessing)
    study_name: str = "expert_hpo"
    storage: Optional[str] = None  # SQLite URL for persistence
    load_if_exists: bool = True
    direction: str = "maximize"  # maximize val accuracy

    # Pruning config
    n_startup_trials: int = 10  # Trials before pruning starts
    n_warmup_steps: int = 5  # Epochs before pruning within trial


@dataclass
class SearchSpace:
    """Search space for hyperparameter optimization."""

    # Neural network
    nn_hidden_dims_options: List[List[int]] = None
    nn_dropout_range: Tuple[float, float] = (0.1, 0.5)
    nn_learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    nn_weight_decay_range: Tuple[float, float] = (1e-6, 1e-3)
    nn_batch_size_options: List[int] = None

    # XGBoost
    xgb_n_estimators_options: List[int] = None
    xgb_max_depth_options: List[int] = None
    xgb_learning_rate_range: Tuple[float, float] = (0.01, 0.3)
    xgb_subsample_range: Tuple[float, float] = (0.6, 1.0)
    xgb_colsample_range: Tuple[float, float] = (0.6, 1.0)

    # LightGBM
    lgb_n_estimators_options: List[int] = None
    lgb_num_leaves_options: List[int] = None
    lgb_learning_rate_range: Tuple[float, float] = (0.01, 0.3)
    lgb_feature_fraction_range: Tuple[float, float] = (0.6, 1.0)
    lgb_bagging_fraction_range: Tuple[float, float] = (0.6, 1.0)

    def __post_init__(self):
        # Set defaults
        if self.nn_hidden_dims_options is None:
            self.nn_hidden_dims_options = [
                [256, 128, 64],
                [512, 256, 128],
                [256, 256, 128, 64],
                [128, 64, 32],
            ]
        if self.nn_batch_size_options is None:
            self.nn_batch_size_options = [32, 64, 128, 256]
        if self.xgb_n_estimators_options is None:
            self.xgb_n_estimators_options = [200, 500, 1000]
        if self.xgb_max_depth_options is None:
            self.xgb_max_depth_options = [4, 6, 8, 10]
        if self.lgb_n_estimators_options is None:
            self.lgb_n_estimators_options = [200, 500, 1000]
        if self.lgb_num_leaves_options is None:
            self.lgb_num_leaves_options = [31, 63, 127]


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimizer for MoE experts."""

    def __init__(
        self,
        config: Optional[HPOConfig] = None,
        search_space: Optional[SearchSpace] = None,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")

        self.config = config or HPOConfig()
        self.search_space = search_space or SearchSpace()
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None

    def _create_study(self) -> optuna.Study:
        """Create or load Optuna study."""
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(
            n_startup_trials=self.config.n_startup_trials,
            n_warmup_steps=self.config.n_warmup_steps,
        )

        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
        )

        return study

    def _sample_nn_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample neural network hyperparameters."""
        ss = self.search_space

        hidden_dims_idx = trial.suggest_categorical(
            "nn_hidden_dims_idx",
            list(range(len(ss.nn_hidden_dims_options)))
        )
        hidden_dims = ss.nn_hidden_dims_options[hidden_dims_idx]

        return {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float(
                "nn_dropout", ss.nn_dropout_range[0], ss.nn_dropout_range[1]
            ),
            "learning_rate": trial.suggest_float(
                "nn_learning_rate",
                ss.nn_learning_rate_range[0],
                ss.nn_learning_rate_range[1],
                log=True,
            ),
            "weight_decay": trial.suggest_float(
                "nn_weight_decay",
                ss.nn_weight_decay_range[0],
                ss.nn_weight_decay_range[1],
                log=True,
            ),
            "batch_size": trial.suggest_categorical(
                "nn_batch_size", ss.nn_batch_size_options
            ),
        }

    def _sample_xgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample XGBoost hyperparameters."""
        ss = self.search_space

        return {
            "n_estimators": trial.suggest_categorical(
                "xgb_n_estimators", ss.xgb_n_estimators_options
            ),
            "max_depth": trial.suggest_categorical(
                "xgb_max_depth", ss.xgb_max_depth_options
            ),
            "learning_rate": trial.suggest_float(
                "xgb_learning_rate",
                ss.xgb_learning_rate_range[0],
                ss.xgb_learning_rate_range[1],
            ),
            "subsample": trial.suggest_float(
                "xgb_subsample",
                ss.xgb_subsample_range[0],
                ss.xgb_subsample_range[1],
            ),
            "colsample_bytree": trial.suggest_float(
                "xgb_colsample",
                ss.xgb_colsample_range[0],
                ss.xgb_colsample_range[1],
            ),
        }

    def _sample_lgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample LightGBM hyperparameters."""
        ss = self.search_space

        return {
            "n_estimators": trial.suggest_categorical(
                "lgb_n_estimators", ss.lgb_n_estimators_options
            ),
            "num_leaves": trial.suggest_categorical(
                "lgb_num_leaves", ss.lgb_num_leaves_options
            ),
            "learning_rate": trial.suggest_float(
                "lgb_learning_rate",
                ss.lgb_learning_rate_range[0],
                ss.lgb_learning_rate_range[1],
            ),
            "feature_fraction": trial.suggest_float(
                "lgb_feature_fraction",
                ss.lgb_feature_fraction_range[0],
                ss.lgb_feature_fraction_range[1],
            ),
            "bagging_fraction": trial.suggest_float(
                "lgb_bagging_fraction",
                ss.lgb_bagging_fraction_range[0],
                ss.lgb_bagging_fraction_range[1],
            ),
        }

    def optimize_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization for the ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sample_weights: Optional sample weights

        Returns:
            Best hyperparameters found
        """
        from polyb0t.ml.moe.deep_ensemble import DeepExpertEnsemble

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            nn_params = self._sample_nn_params(trial)
            xgb_params = self._sample_xgb_params(trial)
            lgb_params = self._sample_lgb_params(trial)

            try:
                # Create ensemble with sampled params
                ensemble = DeepExpertEnsemble(
                    nn_config=nn_params,
                    xgb_config=xgb_params,
                    lgb_config=lgb_params,
                )

                # Train on subset for speed
                n_subset = min(len(X_train), 5000)
                idx = np.random.choice(len(X_train), n_subset, replace=False)
                X_subset = X_train[idx]
                y_subset = y_train[idx]
                w_subset = sample_weights[idx] if sample_weights is not None else None

                # Train
                metrics = ensemble.train(
                    X_subset, y_subset, w_subset, val_fraction=0.2
                )

                # Use meta-learner accuracy if available, otherwise ensemble
                score = max(metrics.meta_val_acc, metrics.ensemble_val_acc)

                # Report intermediate value for pruning
                trial.report(score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        # Create and run study
        self.study = self._create_study()

        logger.info(
            f"Starting HPO with {self.config.n_trials} trials, "
            f"timeout={self.config.timeout}s"
        )

        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        # Extract best params
        self.best_params = self._extract_best_params()

        logger.info(
            f"HPO complete. Best score: {self.study.best_value:.3f}, "
            f"trials: {len(self.study.trials)}"
        )

        return self.best_params

    def _extract_best_params(self) -> Dict[str, Any]:
        """Extract best parameters in usable format."""
        if self.study is None or len(self.study.trials) == 0:
            return {}

        best = self.study.best_params
        ss = self.search_space

        # Reconstruct hidden_dims from index
        hidden_dims_idx = best.get("nn_hidden_dims_idx", 0)
        hidden_dims = ss.nn_hidden_dims_options[hidden_dims_idx]

        return {
            "nn_config": {
                "hidden_dims": hidden_dims,
                "dropout": best.get("nn_dropout", 0.3),
                "learning_rate": best.get("nn_learning_rate", 1e-3),
                "weight_decay": best.get("nn_weight_decay", 1e-4),
                "batch_size": best.get("nn_batch_size", 128),
            },
            "xgb_config": {
                "n_estimators": best.get("xgb_n_estimators", 500),
                "max_depth": best.get("xgb_max_depth", 6),
                "learning_rate": best.get("xgb_learning_rate", 0.05),
                "subsample": best.get("xgb_subsample", 0.8),
                "colsample_bytree": best.get("xgb_colsample", 0.8),
            },
            "lgb_config": {
                "n_estimators": best.get("lgb_n_estimators", 500),
                "num_leaves": best.get("lgb_num_leaves", 63),
                "learning_rate": best.get("lgb_learning_rate", 0.05),
                "feature_fraction": best.get("lgb_feature_fraction", 0.8),
                "bagging_fraction": best.get("lgb_bagging_fraction", 0.8),
            },
            "best_score": self.study.best_value,
            "n_trials": len(self.study.trials),
        }

    def get_param_importances(self) -> Dict[str, float]:
        """Get parameter importances from the study."""
        if self.study is None:
            return {}

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception:
            return {}

    def save_best_params(self, path: str) -> None:
        """Save best parameters to JSON file."""
        import json

        if self.best_params is None:
            logger.warning("No best params to save")
            return

        # Convert numpy types to Python types
        params_clean = {}
        for k, v in self.best_params.items():
            if isinstance(v, dict):
                params_clean[k] = {
                    k2: (v2.item() if hasattr(v2, 'item') else v2)
                    for k2, v2 in v.items()
                }
            elif hasattr(v, 'item'):
                params_clean[k] = v.item()
            else:
                params_clean[k] = v

        with open(path, 'w') as f:
            json.dump(params_clean, f, indent=2)

    @staticmethod
    def load_best_params(path: str) -> Dict[str, Any]:
        """Load best parameters from JSON file."""
        import json

        with open(path, 'r') as f:
            return json.load(f)


class CachedHPO:
    """Hyperparameter optimizer with caching for reuse.

    Caches best params per expert type to avoid repeated searches.
    """

    def __init__(
        self,
        cache_dir: str = "data/hpo_cache",
        config: Optional[HPOConfig] = None,
    ):
        self.cache_dir = cache_dir
        self.config = config or HPOConfig()
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, expert_id: str) -> str:
        return os.path.join(self.cache_dir, f"{expert_id}_hpo.json")

    def get_or_optimize(
        self,
        expert_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        force_reoptimize: bool = False,
    ) -> Dict[str, Any]:
        """Get cached params or run optimization.

        Args:
            expert_id: Unique identifier for the expert
            X_train, y_train: Training data
            X_val, y_val: Validation data
            sample_weights: Optional sample weights
            force_reoptimize: Force re-running HPO even if cached

        Returns:
            Best hyperparameters
        """
        cache_path = self._get_cache_path(expert_id)

        # Check cache
        if not force_reoptimize and os.path.exists(cache_path):
            try:
                params = HyperparameterOptimizer.load_best_params(cache_path)
                logger.info(f"Loaded cached HPO params for {expert_id}")
                return params
            except Exception as e:
                logger.warning(f"Failed to load cached params: {e}")

        # Run HPO
        logger.info(f"Running HPO for {expert_id}")
        optimizer = HyperparameterOptimizer(config=self.config)
        best_params = optimizer.optimize_ensemble(
            X_train, y_train, X_val, y_val, sample_weights
        )

        # Cache results
        optimizer.save_best_params(cache_path)

        return best_params


def quick_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 20,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Quick hyperparameter optimization with reduced trials.

    For use during development or when speed is more important than
    finding the absolute best parameters.
    """
    config = HPOConfig(n_trials=n_trials, timeout=timeout)
    optimizer = HyperparameterOptimizer(config=config)
    return optimizer.optimize_ensemble(X_train, y_train, X_val, y_val)
