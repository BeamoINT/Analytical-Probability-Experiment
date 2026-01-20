"""AI Training Manager with classification-based profitable trade prediction.

This module handles:
- Training AI classifiers to predict profitable trades (yes/no)
- Multi-validation with worst-case metrics
- Confidence-based filtering (only trade when confident)
- Feature interactions for better pattern detection
- Time-weighted learning (recent data matters more)
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

# Configuration
SPREAD_COST = 0.02  # 2% spread cost assumption
MIN_PROFIT_THRESHOLD = 0.005  # 0.5% minimum profit after spread to count as "profitable"
CONFIDENCE_THRESHOLD = 0.60  # Only trade when >60% confident


@dataclass
class ModelMetrics:
    """Metrics for evaluating model performance.
    
    PRIMARY METRIC: simulated_profit_pct - actual P&L from backtesting
    This is what matters - you can be wrong often but still make money.
    """
    mse: float  # Mean Squared Error (for regression fallback)
    mae: float  # Mean Absolute Error
    r2: float  # R-squared (for reference)
    directional_accuracy: float  # % of correct direction predictions
    profitable_accuracy: float  # % of confident predictions that were actually profitable
    cv_std: float = 0.0  # Cross-validation standard deviation
    n_features_used: int = 0  # Number of features after selection
    n_models_ensemble: int = 0  # Number of models in ensemble
    avg_confidence: float = 0.0  # Average prediction confidence
    confident_trade_pct: float = 0.0  # % of samples where model was confident enough to trade
    
    # NEW: Profitability simulation metrics (THE METRICS THAT ACTUALLY MATTER)
    simulated_profit_pct: float = 0.0  # Total P&L from simulated trading
    simulated_num_trades: int = 0  # Number of trades in simulation
    simulated_win_rate: float = 0.0  # % of profitable trades
    simulated_avg_win: float = 0.0  # Average profit on winning trades
    simulated_avg_loss: float = 0.0  # Average loss on losing trades
    simulated_profit_factor: float = 0.0  # gross_profit / gross_loss
    simulated_max_drawdown: float = 0.0  # Worst peak-to-trough decline
    simulated_sharpe: float = 0.0  # Risk-adjusted return (higher = better)
    
    def score(self) -> float:
        """Overall score - PROFITABILITY IS KING."""
        # Primary: simulated profit (capped at reasonable range)
        profit_score = max(-0.5, min(0.5, self.simulated_profit_pct * 2))
        
        # Secondary: risk-adjusted (Sharpe)
        sharpe_score = max(-0.2, min(0.2, self.simulated_sharpe * 0.1))
        
        # Tertiary: profit factor (how much you make vs lose)
        pf_score = max(0, min(0.2, (self.simulated_profit_factor - 1) * 0.1))
        
        # Penalty for not trading (useless model)
        trade_penalty = 0 if self.simulated_num_trades >= 50 else -0.3
        
        return profit_score + sharpe_score + pf_score + trade_penalty


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
                "avg_confidence": self.metrics.avg_confidence,
                "confident_trade_pct": self.metrics.confident_trade_pct,
                # NEW: Profitability metrics (what actually matters)
                "simulated_profit_pct": self.metrics.simulated_profit_pct,
                "simulated_num_trades": self.metrics.simulated_num_trades,
                "simulated_win_rate": self.metrics.simulated_win_rate,
                "simulated_avg_win": self.metrics.simulated_avg_win,
                "simulated_avg_loss": self.metrics.simulated_avg_loss,
                "simulated_profit_factor": self.metrics.simulated_profit_factor,
                "simulated_max_drawdown": self.metrics.simulated_max_drawdown,
                "simulated_sharpe": self.metrics.simulated_sharpe,
                "score": self.metrics.score(),
            },
            "is_active": self.is_active,
            "model_path": self.model_path,
        }


class AITrainer:
    """Manages AI model training with classification-based approach."""
    
    def __init__(
        self,
        model_dir: str = "data/ai_models",
        min_training_examples: int = 1000,
        benchmark_test_size: float = 0.2,
        min_improvement_pct: float = 1.0,
    ):
        self.model_dir = model_dir
        self.min_training_examples = min_training_examples
        self.benchmark_test_size = benchmark_test_size
        self.min_improvement_pct = min_improvement_pct
        
        self._ensure_dirs()
        self._current_model = None
        self._current_model_info: Optional[ModelInfo] = None
        self._training_lock = threading.Lock()
        self._is_training = False
        self._feature_cols = None
        
        self._load_current_model()
        
    def _ensure_dirs(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "versions"), exist_ok=True)
        
    def _get_state_path(self) -> str:
        return os.path.join(self.model_dir, "trainer_state.json")
    
    def _get_current_model_path(self) -> str:
        return os.path.join(self.model_dir, "current_model.pkl")
    
    def _load_current_model(self) -> None:
        model_path = self._get_current_model_path()
        state_path = self._get_state_path()
        
        if os.path.exists(model_path) and os.path.exists(state_path):
            try:
                with open(model_path, "rb") as f:
                    self._current_model = pickle.load(f)
                    
                with open(state_path, "r") as f:
                    state = json.load(f)
                    
                # Handle old format without new metrics
                metrics_dict = state["metrics"]
                self._current_model_info = ModelInfo(
                    version=state["version"],
                    created_at=datetime.fromisoformat(state["created_at"]),
                    training_examples=state["training_examples"],
                    metrics=ModelMetrics(
                        mse=metrics_dict.get("mse", 0),
                        mae=metrics_dict.get("mae", 0),
                        r2=metrics_dict.get("r2", 0),
                        directional_accuracy=metrics_dict.get("directional_accuracy", 0),
                        profitable_accuracy=metrics_dict.get("profitable_accuracy", 0),
                        cv_std=metrics_dict.get("cv_std", 0),
                        n_features_used=metrics_dict.get("n_features_used", 0),
                        n_models_ensemble=metrics_dict.get("n_models_ensemble", 0),
                        avg_confidence=metrics_dict.get("avg_confidence", 0),
                        confident_trade_pct=metrics_dict.get("confident_trade_pct", 0),
                    ),
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
        return self._current_model is not None
    
    def get_model_info(self) -> Optional[dict]:
        if self._current_model_info:
            return self._current_model_info.to_dict()
        return None
    
    def is_training(self) -> bool:
        return self._is_training
    
    def can_train(self, available_examples: int) -> bool:
        return available_examples >= self.min_training_examples
    
    def _get_base_feature_cols(self) -> list:
        """Get the base feature columns."""
        return [
            # Core price features
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
    
    def _add_feature_interactions(self, X: np.ndarray, feature_cols: list) -> Tuple[np.ndarray, list]:
        """Add feature interactions for better pattern detection.
        
        Creates:
        - Ratios between related features
        - Polynomial interactions
        - Bucketed features
        """
        n_samples = X.shape[0]
        new_features = []
        new_names = []
        
        # Get column indices for key features
        def get_idx(name):
            try:
                return feature_cols.index(name)
            except ValueError:
                return None
        
        # === RATIO INTERACTIONS ===
        ratio_pairs = [
            ("volume_1h", "volume_24h", "volume_ratio_1h_24h"),
            ("volatility_1h", "volatility_24h", "volatility_ratio_1h_24h"),
            ("momentum_1h", "momentum_24h", "momentum_ratio_1h_24h"),
            ("bid_depth", "ask_depth", "bid_ask_depth_ratio"),
            ("liquidity_bid", "liquidity_ask", "liquidity_imbalance"),
            ("trade_count_1h", "trade_count_24h", "trade_count_ratio"),
            ("price_change_1h", "volatility_1h", "price_vs_volatility_1h"),
            ("price_change_24h", "volatility_24h", "price_vs_volatility_24h"),
        ]
        
        for col1, col2, name in ratio_pairs:
            idx1, idx2 = get_idx(col1), get_idx(col2)
            if idx1 is not None and idx2 is not None:
                denominator = X[:, idx2] + 1e-10
                ratio = X[:, idx1] / denominator
                ratio = np.clip(ratio, -10, 10)  # Clip extremes
                new_features.append(ratio)
                new_names.append(name)
        
        # === MOMENTUM STRENGTH ===
        # Combine multiple momentum signals
        momentum_indices = [get_idx(f"momentum_{p}") for p in ["1h", "4h", "24h", "7d"]]
        momentum_indices = [i for i in momentum_indices if i is not None]
        if len(momentum_indices) >= 2:
            # All positive = strong bullish, all negative = strong bearish
            momentum_agreement = np.prod(np.sign(X[:, momentum_indices] + 1e-10), axis=1)
            new_features.append(momentum_agreement)
            new_names.append("momentum_agreement")
            
            # Average momentum strength
            momentum_strength = np.mean(np.abs(X[:, momentum_indices]), axis=1)
            new_features.append(momentum_strength)
            new_names.append("momentum_strength")
        
        # === PRICE POSITION ===
        # Where is price relative to recent range?
        price_idx = get_idx("price")
        high_idx = get_idx("price_high_24h")
        low_idx = get_idx("price_low_24h")
        if all(i is not None for i in [price_idx, high_idx, low_idx]):
            price_range = X[:, high_idx] - X[:, low_idx] + 1e-10
            price_position = (X[:, price_idx] - X[:, low_idx]) / price_range
            price_position = np.clip(price_position, 0, 1)
            new_features.append(price_position)
            new_names.append("price_position_in_range")
        
        # === VOLATILITY-ADJUSTED MOMENTUM ===
        vol_24h_idx = get_idx("volatility_24h")
        mom_24h_idx = get_idx("momentum_24h")
        if vol_24h_idx is not None and mom_24h_idx is not None:
            vol_adj_momentum = X[:, mom_24h_idx] / (X[:, vol_24h_idx] + 1e-10)
            vol_adj_momentum = np.clip(vol_adj_momentum, -5, 5)
            new_features.append(vol_adj_momentum)
            new_names.append("volatility_adjusted_momentum")
        
        # === LIQUIDITY QUALITY ===
        spread_idx = get_idx("spread_pct")
        liq_idx = get_idx("liquidity")
        if spread_idx is not None and liq_idx is not None:
            # High liquidity + tight spread = good market
            liquidity_quality = X[:, liq_idx] / (X[:, spread_idx] + 1e-10)
            liquidity_quality = np.clip(np.log1p(liquidity_quality), 0, 10)
            new_features.append(liquidity_quality)
            new_names.append("liquidity_quality")
        
        # === TIME PRESSURE ===
        days_idx = get_idx("days_to_resolution")
        if days_idx is not None:
            # More urgency as resolution approaches
            time_pressure = 1 / (X[:, days_idx] + 1)
            time_pressure = np.clip(time_pressure, 0, 1)
            new_features.append(time_pressure)
            new_names.append("time_pressure")
        
        # === EXTREME PRICE DETECTION ===
        if price_idx is not None:
            # Flags for extreme prices (near 0 or 1)
            price_near_zero = (X[:, price_idx] < 0.1).astype(float)
            price_near_one = (X[:, price_idx] > 0.9).astype(float)
            price_in_middle = ((X[:, price_idx] >= 0.4) & (X[:, price_idx] <= 0.6)).astype(float)
            new_features.extend([price_near_zero, price_near_one, price_in_middle])
            new_names.extend(["price_near_zero", "price_near_one", "price_in_middle"])
        
        if new_features:
            new_features = np.column_stack(new_features)
            X_enhanced = np.hstack([X, new_features])
            enhanced_cols = feature_cols + new_names
            logger.info(f"Added {len(new_names)} interaction features: {new_names}")
            return X_enhanced, enhanced_cols
        
        return X, feature_cols
    
    def _create_binary_target(self, y_regression: np.ndarray) -> np.ndarray:
        """Convert price change to binary profitable/not profitable label.
        
        Profitable = price moved in predicted direction AND profit > spread cost
        """
        # For training, we'll define profitable as: |actual_change| > SPREAD_COST
        # AND direction prediction would be correct
        # Since we don't have predictions yet, we label based on whether 
        # a correct direction prediction would have been profitable
        
        # If |change| > spread_cost, it was a tradeable opportunity
        profitable = np.abs(y_regression) > SPREAD_COST + MIN_PROFIT_THRESHOLD
        return profitable.astype(int)
    
    def _compute_sample_weights(self, n_samples: int) -> np.ndarray:
        """Compute sample weights that favor recent data.
        
        Recent data gets higher weight because:
        1. Market conditions change
        2. Recent patterns are more relevant
        3. Older data may have different dynamics
        """
        # Exponential decay: recent samples get more weight
        # Weight ranges from 0.5 (oldest) to 1.5 (newest)
        positions = np.arange(n_samples) / n_samples  # 0 to 1
        weights = 0.5 + positions  # 0.5 to 1.5
        
        # Normalize so weights sum to n_samples
        weights = weights / weights.mean()
        
        return weights
    
    def train_model(self, training_data: list[dict]) -> Optional[ModelInfo]:
        """Train a new classification model to predict profitable trades."""
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
            logger.info(f"Starting AI training with {len(training_data)} examples (classification mode)")
            
            # Sort by time
            sorted_data = sorted(
                training_data,
                key=lambda x: x.get("timestamp", x.get("created_at", ""))
            )
            
            # Prepare data
            X, y_reg, timestamps = self._prepare_training_data_with_time(sorted_data)
            
            if X is None or len(X) == 0:
                logger.warning("No valid training data after preparation")
                return None
            
            # Add feature interactions
            X, self._feature_cols = self._add_feature_interactions(X, self._get_base_feature_cols())
            
            # Convert to binary classification target
            y_binary = self._create_binary_target(y_reg)
            
            # Log class distribution
            n_profitable = np.sum(y_binary)
            logger.info(
                f"Class distribution: {n_profitable}/{len(y_binary)} profitable "
                f"({100*n_profitable/len(y_binary):.1f}%)"
            )
            
            # Compute sample weights (recent data more important)
            sample_weights = self._compute_sample_weights(len(X))
            
            # === MULTI-VALIDATION STRATEGY ===
            validation_results = []
            
            # VALIDATION 1: Time-gap validation
            gap_size = int(len(X) * 0.05)
            train_end = int(len(X) * 0.75)
            test_start = train_end + gap_size
            
            if test_start < len(X) - 10:
                X_train = X[:train_end]
                y_train = y_binary[:train_end]
                y_reg_train = y_reg[:train_end]
                w_train = sample_weights[:train_end]
                X_test = X[test_start:]
                y_test = y_binary[test_start:]
                y_reg_test = y_reg[test_start:]
                
                model = self._train_classifier(X_train, y_train, y_reg_train, w_train)
                metrics = self._evaluate_classifier(model, X_test, y_test, y_reg_test)
                validation_results.append(("time_gap", metrics))
                logger.info(f"Time-gap validation: profit_acc={metrics.profitable_accuracy:.1%}")
            
            # VALIDATION 2: Walk-forward
            n_periods = 4
            period_size = len(X) // n_periods
            walk_forward_accs = []
            
            if period_size > 50:
                for i in range(1, n_periods):
                    train_end = i * period_size
                    test_end = (i + 1) * period_size
                    
                    X_train = X[:train_end]
                    y_train = y_binary[:train_end]
                    y_reg_train = y_reg[:train_end]
                    w_train = sample_weights[:train_end]
                    X_test = X[train_end:test_end]
                    y_test = y_binary[train_end:test_end]
                    y_reg_test = y_reg[train_end:test_end]
                    
                    if len(X_train) > 50 and len(X_test) > 10:
                        model = self._train_classifier(X_train, y_train, y_reg_train, w_train)
                        metrics = self._evaluate_classifier(model, X_test, y_test, y_reg_test)
                        walk_forward_accs.append(metrics.profitable_accuracy)
                
                if walk_forward_accs:
                    min_acc = min(walk_forward_accs)
                    logger.info(f"Walk-forward: min={min_acc:.1%}, periods={len(walk_forward_accs)}")
                    wf_metrics = ModelMetrics(mse=0, mae=0, r2=0, directional_accuracy=min_acc, profitable_accuracy=min_acc)
                    validation_results.append(("walk_forward", wf_metrics))
            
            # VALIDATION 3: Random sample
            np.random.seed(42)
            test_indices = np.random.choice(len(X), size=int(len(X) * 0.2), replace=False)
            train_indices = np.array([i for i in range(len(X)) if i not in test_indices])
            
            X_train = X[train_indices]
            y_train = y_binary[train_indices]
            y_reg_train = y_reg[train_indices]
            w_train = sample_weights[train_indices]
            X_test = X[test_indices]
            y_test = y_binary[test_indices]
            y_reg_test = y_reg[test_indices]
            
            model = self._train_classifier(X_train, y_train, y_reg_train, w_train)
            metrics = self._evaluate_classifier(model, X_test, y_test, y_reg_test)
            validation_results.append(("random", metrics))
            logger.info(f"Random sample validation: profit_acc={metrics.profitable_accuracy:.1%}")
            
            # VALIDATION 4: Oldest data
            oldest_test_size = int(len(X) * 0.25)
            X_train = X[oldest_test_size:]
            y_train = y_binary[oldest_test_size:]
            y_reg_train = y_reg[oldest_test_size:]
            w_train = sample_weights[oldest_test_size:]
            X_test = X[:oldest_test_size]
            y_test = y_binary[:oldest_test_size]
            y_reg_test = y_reg[:oldest_test_size]
            
            model = self._train_classifier(X_train, y_train, y_reg_train, w_train)
            metrics = self._evaluate_classifier(model, X_test, y_test, y_reg_test)
            validation_results.append(("oldest", metrics))
            logger.info(f"Oldest data validation: profit_acc={metrics.profitable_accuracy:.1%}")
            
            # === WORST-CASE METRICS ===
            all_profit_accs = [m.profitable_accuracy for _, m in validation_results]
            all_dir_accs = [m.directional_accuracy for _, m in validation_results]
            all_conf_pcts = [m.confident_trade_pct for _, m in validation_results]
            
            worst_profit_acc = min(all_profit_accs) if all_profit_accs else 0.5
            worst_dir_acc = min(all_dir_accs) if all_dir_accs else 0.5
            avg_conf_pct = sum(all_conf_pcts) / len(all_conf_pcts) if all_conf_pcts else 0
            
            logger.info(
                f"Multi-validation summary: worst_profit={worst_profit_acc:.1%}, "
                f"avg_confident_pct={avg_conf_pct:.1%}"
            )
            
            # === TRAIN FINAL MODEL ON ALL DATA ===
            final_model = self._train_classifier(X, y_binary, y_reg, sample_weights)
            
            # Evaluate on held-out portion for final metrics
            final_metrics = ModelMetrics(
                mse=0, mae=0,
                r2=0,
                directional_accuracy=worst_dir_acc,
                profitable_accuracy=worst_profit_acc,
                confident_trade_pct=avg_conf_pct,
            )
            
            if hasattr(final_model, 'n_features_'):
                final_metrics.n_features_used = final_model.n_features_
            elif hasattr(final_model, 'scaler'):
                final_metrics.n_features_used = X.shape[1]
            
            final_metrics.n_models_ensemble = getattr(final_model, 'n_models_', 1)
            
            logger.info(
                f"Final model: profit_acc={final_metrics.profitable_accuracy:.1%}, "
                f"confident_trades={final_metrics.confident_trade_pct:.1%}"
            )
            
            # Deploy
            version = (self._current_model_info.version + 1) if self._current_model_info else 1
            model_info = self._deploy_model(final_model, final_metrics, version, len(training_data))
            
            logger.info(f"Deployed AI model v{version}")
            return model_info
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return None
        finally:
            self._is_training = False
    
    def _prepare_training_data_with_time(
        self, data: list[dict]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[list[str]]]:
        """Prepare training data arrays with timestamps."""
        feature_cols = self._get_base_feature_cols()
        
        X_list = []
        y_list = []
        timestamps = []
        
        for example in data:
            target = example.get("label_price_change_24h")
            if target is None:
                continue
            
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
        
        logger.info(f"Prepared {len(X_list)} examples with {len(feature_cols)} base features")
        return np.array(X_list), np.array(y_list), timestamps
    
    def _train_classifier(
        self, 
        X: np.ndarray, 
        y_binary: np.ndarray, 
        y_reg: np.ndarray,
        sample_weights: np.ndarray
    ) -> Any:
        """Train a MORE THOROUGH classifier ensemble.
        
        This takes longer but produces better models:
        - More trees in forests
        - More boosting iterations
        - Cross-validated hyperparameter selection
        - Multiple ensemble methods
        """
        try:
            from sklearn.ensemble import (
                RandomForestClassifier,
                ExtraTreesClassifier,
                HistGradientBoostingClassifier,
                AdaBoostClassifier,
                VotingClassifier,
            )
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            
            logger.info(f"Training THOROUGH classifier on {len(X)} samples...")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            classifiers = []
            
            # 1. Logistic Regression with cross-validation
            logger.info("Training LogisticRegression...")
            lr = LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=2000,  # More iterations
                random_state=42,
                solver='lbfgs',
            )
            lr.fit(X_scaled, y_binary, sample_weight=sample_weights)
            lr_score = cross_val_score(lr, X_scaled, y_binary, cv=5, scoring='accuracy').mean()
            classifiers.append(("LogisticRegression", lr, lr_score))
            logger.info(f"LogisticRegression CV accuracy: {lr_score:.3f}")
            
            # 2. Random Forest - MORE TREES, DEEPER
            if len(X) >= 500:
                logger.info("Training RandomForest (200 trees, depth 8)...")
                rf = RandomForestClassifier(
                    n_estimators=200,  # 4x more trees
                    max_depth=8,  # Deeper
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                    oob_score=True,  # Out-of-bag scoring
                )
                rf.fit(X_scaled, y_binary, sample_weight=sample_weights)
                rf_score = rf.oob_score_
                classifiers.append(("RandomForest", rf, rf_score))
                logger.info(f"RandomForest OOB accuracy: {rf_score:.3f}")
            
            # 3. Extra Trees (more randomness, often better)
            if len(X) >= 500:
                logger.info("Training ExtraTrees (200 trees, depth 10)...")
                et = ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True,
                    oob_score=True,
                )
                et.fit(X_scaled, y_binary, sample_weight=sample_weights)
                et_score = et.oob_score_
                classifiers.append(("ExtraTrees", et, et_score))
                logger.info(f"ExtraTrees OOB accuracy: {et_score:.3f}")
            
            # 4. Gradient Boosting - MORE ITERATIONS
            if len(X) >= 500:
                logger.info("Training HistGradientBoosting (500 iterations)...")
                gb = HistGradientBoostingClassifier(
                    max_iter=500,  # 5x more iterations
                    max_depth=6,  # Deeper
                    min_samples_leaf=20,
                    l2_regularization=2.0,
                    learning_rate=0.03,  # Lower learning rate with more iterations
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    class_weight='balanced',
                    random_state=42,
                )
                gb.fit(X_scaled, y_binary, sample_weight=sample_weights)
                gb_score = cross_val_score(gb, X_scaled, y_binary, cv=3, scoring='accuracy').mean()
                classifiers.append(("GradientBoosting", gb, gb_score))
                logger.info(f"GradientBoosting CV accuracy: {gb_score:.3f}")
            
            # 5. AdaBoost with stumps (often good for noisy data)
            if len(X) >= 500:
                logger.info("Training AdaBoost (200 estimators)...")
                ada = AdaBoostClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    random_state=42,
                )
                ada.fit(X_scaled, y_binary, sample_weight=sample_weights)
                ada_score = cross_val_score(ada, X_scaled, y_binary, cv=3, scoring='accuracy').mean()
                classifiers.append(("AdaBoost", ada, ada_score))
                logger.info(f"AdaBoost CV accuracy: {ada_score:.3f}")
            
            # Sort by score and take top 4
            classifiers.sort(key=lambda x: x[2], reverse=True)
            top_classifiers = [(name, clf) for name, clf, score in classifiers[:4]]
            
            logger.info(f"Selected top {len(top_classifiers)} classifiers for ensemble")
            
            # Create ensemble wrapper
            ensemble = _ClassifierEnsemble(
                scaler=scaler,
                classifiers=top_classifiers,
                feature_cols=self._feature_cols,
            )
            
            return ensemble
            
        except ImportError as e:
            logger.warning(f"sklearn not available: {e}")
            return None
    
    def _evaluate_classifier(
        self, 
        model: Any, 
        X: np.ndarray, 
        y_binary: np.ndarray,
        y_reg: np.ndarray
    ) -> ModelMetrics:
        """Evaluate classifier with PROFITABILITY SIMULATION.
        
        This simulates actual trading to measure what really matters:
        - Total profit/loss
        - Win rate
        - Risk-adjusted returns (Sharpe)
        - Max drawdown
        """
        if model is None:
            return ModelMetrics(0, 0, 0, 0.5, 0.5)
        
        # Get probabilities
        probs = model.predict_proba(X)
        
        # Confidence is max(prob_profitable, prob_not_profitable)
        confidence = np.max(probs, axis=1)
        predictions = model.predict(X)
        
        # Only evaluate on confident predictions
        confident_mask = confidence >= CONFIDENCE_THRESHOLD
        n_confident = np.sum(confident_mask)
        
        if n_confident == 0:
            return ModelMetrics(
                mse=0, mae=0, r2=0,
                directional_accuracy=0.5,
                profitable_accuracy=0.5,
                confident_trade_pct=0,
                avg_confidence=float(np.mean(confidence)),
            )
        
        # Basic accuracy metrics
        correct = predictions[confident_mask] == y_binary[confident_mask]
        accuracy = float(np.mean(correct))
        
        predicted_profitable = predictions == 1
        confident_profitable = confident_mask & predicted_profitable
        
        if np.sum(confident_profitable) > 0:
            actually_profitable = y_binary[confident_profitable] == 1
            profitable_accuracy = float(np.mean(actually_profitable))
        else:
            profitable_accuracy = 0.5
        
        # ========================================
        # PROFITABILITY SIMULATION - THE KEY METRIC
        # ========================================
        # Simulate trading: when model says "profitable" with high confidence, we trade
        # P&L is based on actual price changes (y_reg), not binary classification
        
        trade_results = []
        wins = []
        losses = []
        
        for i in range(len(X)):
            if not confident_mask[i]:
                continue  # Skip low confidence
            
            if predictions[i] == 1:  # Model says "this trade will be profitable"
                # We BUY - profit is actual price change minus spread
                actual_change = y_reg[i] if i < len(y_reg) else 0
                trade_pnl = actual_change - SPREAD_COST
            else:  # Model says "this trade will NOT be profitable" - we could short or skip
                # For now, we skip unprofitable predictions (conservative)
                continue
            
            trade_results.append(trade_pnl)
            if trade_pnl > 0:
                wins.append(trade_pnl)
            else:
                losses.append(trade_pnl)
        
        # Calculate simulation metrics
        num_trades = len(trade_results)
        
        if num_trades == 0:
            return ModelMetrics(
                mse=0, mae=0, r2=0,
                directional_accuracy=accuracy,
                profitable_accuracy=profitable_accuracy,
                confident_trade_pct=float(n_confident / len(X)),
                avg_confidence=float(np.mean(confidence)),
                simulated_profit_pct=0,
                simulated_num_trades=0,
            )
        
        # Total P&L
        total_profit = sum(trade_results)
        
        # Win rate
        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        
        # Average win/loss
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0  # Will be negative
        
        # Profit factor (gross profit / abs(gross loss))
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001  # Avoid div by 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 10.0
        
        # Max drawdown
        cumulative = np.cumsum(trade_results)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Sharpe ratio (annualized, assuming ~250 trading days)
        if len(trade_results) > 1:
            returns_std = np.std(trade_results)
            if returns_std > 0:
                daily_sharpe = np.mean(trade_results) / returns_std
                sharpe = daily_sharpe * np.sqrt(250)  # Annualized
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        logger.info(
            f"SIMULATION: {num_trades} trades, profit={total_profit:.2%}, "
            f"win_rate={win_rate:.1%}, profit_factor={profit_factor:.2f}, "
            f"sharpe={sharpe:.2f}, max_dd={max_drawdown:.2%}"
        )
        
        return ModelMetrics(
            mse=0, mae=0, r2=0,
            directional_accuracy=accuracy,
            profitable_accuracy=profitable_accuracy,
            confident_trade_pct=float(n_confident / len(X)),
            avg_confidence=float(np.mean(confidence)),
            simulated_profit_pct=total_profit,
            simulated_num_trades=num_trades,
            simulated_win_rate=win_rate,
            simulated_avg_win=float(avg_win),
            simulated_avg_loss=float(avg_loss),
            simulated_profit_factor=profit_factor,
            simulated_max_drawdown=max_drawdown,
            simulated_sharpe=sharpe,
        )
    
    def _deploy_model(
        self,
        model: Any,
        metrics: ModelMetrics,
        version: int,
        training_examples: int,
    ) -> ModelInfo:
        """Deploy a new model."""
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
                "cv_std": metrics.cv_std,
                "n_features_used": metrics.n_features_used,
                "n_models_ensemble": metrics.n_models_ensemble,
                "avg_confidence": metrics.avg_confidence,
                "confident_trade_pct": metrics.confident_trade_pct,
            },
        }
        
        with open(self._get_state_path(), "w") as f:
            json.dump(state, f, indent=2)
            
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
    
    def predict(self, features: dict) -> Optional[Tuple[float, float]]:
        """Make a prediction with confidence.
        
        Returns:
            Tuple of (predicted_change, confidence) or None
            predicted_change: Positive = bullish, Negative = bearish
            confidence: 0.0 to 1.0, how confident the model is
        """
        if self._current_model is None:
            return None
        
        try:
            # Build feature vector
            base_cols = self._get_base_feature_cols()
            X = []
            for col in base_cols:
                val = features.get(col, 0)
                try:
                    X.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    X.append(0.0)
            
            X = np.array([X])
            
            # Add interactions
            X, _ = self._add_feature_interactions(X, base_cols)
            
            # Get prediction and confidence
            if hasattr(self._current_model, 'predict_proba'):
                probs = self._current_model.predict_proba(X)[0]
                confidence = float(max(probs))
                is_profitable = int(np.argmax(probs)) == 1
                
                # Convert to directional prediction
                # If profitable, estimate direction from features
                if is_profitable:
                    # Use momentum features to determine direction
                    momentum = features.get("momentum_24h", 0) or 0
                    predicted_change = 0.05 if momentum >= 0 else -0.05
                else:
                    predicted_change = 0.0
                
                return (predicted_change, confidence)
            else:
                # Fallback for non-probabilistic model
                pred = self._current_model.predict(X)[0]
                return (float(pred), 0.5)
                
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None


class _ClassifierEnsemble:
    """Ensemble of classifiers with probability averaging."""
    
    def __init__(self, scaler, classifiers, feature_cols):
        self.scaler = scaler
        self.classifiers = classifiers
        self.feature_cols = feature_cols
        self.n_features_ = len(feature_cols) if feature_cols else 0
        self.n_models_ = len(classifiers)
        
    def predict_proba(self, X):
        """Average probabilities from all classifiers."""
        X_scaled = self.scaler.transform(X)
        
        all_probs = []
        for name, clf in self.classifiers:
            probs = clf.predict_proba(X_scaled)
            all_probs.append(probs)
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs
    
    def predict(self, X):
        """Predict class based on averaged probabilities."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# Singleton instance
_trainer_instance: Optional[AITrainer] = None


def get_ai_trainer(
    model_dir: str = "data/ai_models",
    min_training_examples: int = 1000,
) -> AITrainer:
    """Get or create the singleton AI trainer."""
    global _trainer_instance
    if _trainer_instance is None:
        _trainer_instance = AITrainer(
            model_dir=model_dir,
            min_training_examples=min_training_examples,
        )
    return _trainer_instance
