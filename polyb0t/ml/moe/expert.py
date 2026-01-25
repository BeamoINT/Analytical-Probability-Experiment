"""Expert model class for MoE architecture.

Each expert is a self-contained classifier ensemble that specializes
in a particular domain (category, risk level, time horizon).

Includes versioning support for rollback and smart state management.
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from polyb0t.ml.moe.versioning import (
    ExpertState,
    ExpertVersion,
    ExpertVersionManager,
)

logger = logging.getLogger(__name__)

# Constants for profitability calculation
SPREAD_COST = 0.02  # 2% spread
MIN_PROFIT_THRESHOLD = 0.005  # 0.5% minimum profit to be considered profitable
POSITION_SIZE = 0.05  # 5% of portfolio per trade
CONFIDENCE_THRESHOLD = 0.60  # Only trade when 60%+ confident


@dataclass
class ExpertMetrics:
    """Metrics for an expert's performance."""
    
    # Profitability metrics (PRIMARY - what actually matters)
    simulated_profit_pct: float = 0.0
    simulated_num_trades: int = 0
    simulated_win_rate: float = 0.0
    simulated_avg_win: float = 0.0
    simulated_avg_loss: float = 0.0
    simulated_profit_factor: float = 0.0
    simulated_max_drawdown: float = 0.0
    simulated_sharpe: float = 0.0
    
    # Secondary metrics (for debugging only)
    directional_accuracy: float = 0.0
    profitable_accuracy: float = 0.0
    confident_trade_pct: float = 0.0
    avg_confidence: float = 0.0
    
    # Training info
    n_training_examples: int = 0
    n_features_used: int = 0
    last_trained: Optional[datetime] = None
    
    def score(self) -> float:
        """Calculate overall expert score based on PROFITABILITY ONLY.
        
        Accuracy is NOT a factor - only profit matters.
        """
        # Primary: simulated profit (capped to avoid outliers)
        profit_score = max(-0.5, min(0.5, self.simulated_profit_pct * 2))
        
        # Secondary: risk-adjusted returns (Sharpe)
        sharpe_bonus = max(-0.1, min(0.1, self.simulated_sharpe * 0.05))
        
        # Bonus for good profit factor
        pf_bonus = 0.05 if self.simulated_profit_factor > 1.5 else 0
        
        # Penalty for too few trades (not learning)
        trade_penalty = -0.1 if self.simulated_num_trades < 20 else 0
        
        return profit_score + sharpe_bonus + pf_bonus + trade_penalty
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "simulated_profit_pct": self.simulated_profit_pct,
            "simulated_num_trades": self.simulated_num_trades,
            "simulated_win_rate": self.simulated_win_rate,
            "simulated_avg_win": self.simulated_avg_win,
            "simulated_avg_loss": self.simulated_avg_loss,
            "simulated_profit_factor": self.simulated_profit_factor,
            "simulated_max_drawdown": self.simulated_max_drawdown,
            "simulated_sharpe": self.simulated_sharpe,
            "directional_accuracy": self.directional_accuracy,
            "profitable_accuracy": self.profitable_accuracy,
            "confident_trade_pct": self.confident_trade_pct,
            "avg_confidence": self.avg_confidence,
            "n_training_examples": self.n_training_examples,
            "n_features_used": self.n_features_used,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertMetrics":
        """Create from dictionary."""
        last_trained = None
        if data.get("last_trained"):
            try:
                last_trained = datetime.fromisoformat(data["last_trained"])
            except:
                pass
        
        return cls(
            simulated_profit_pct=data.get("simulated_profit_pct", 0.0),
            simulated_num_trades=data.get("simulated_num_trades", 0),
            simulated_win_rate=data.get("simulated_win_rate", 0.0),
            simulated_avg_win=data.get("simulated_avg_win", 0.0),
            simulated_avg_loss=data.get("simulated_avg_loss", 0.0),
            simulated_profit_factor=data.get("simulated_profit_factor", 0.0),
            simulated_max_drawdown=data.get("simulated_max_drawdown", 0.0),
            simulated_sharpe=data.get("simulated_sharpe", 0.0),
            directional_accuracy=data.get("directional_accuracy", 0.0),
            profitable_accuracy=data.get("profitable_accuracy", 0.0),
            confident_trade_pct=data.get("confident_trade_pct", 0.0),
            avg_confidence=data.get("avg_confidence", 0.0),
            n_training_examples=data.get("n_training_examples", 0),
            n_features_used=data.get("n_features_used", 0),
            last_trained=last_trained,
        )


class _ExpertEnsemble:
    """Ensemble of classifiers for a single expert."""
    
    def __init__(self, classifiers: List[Any], scaler: StandardScaler):
        self.classifiers = classifiers
        self.scaler = scaler
        self.n_models_ = len(classifiers)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions averaged across all classifiers."""
        X_scaled = self.scaler.transform(X)
        
        all_probs = []
        for clf in self.classifiers:
            try:
                probs = clf.predict_proba(X_scaled)
                all_probs.append(probs)
            except Exception:
                continue
        
        if not all_probs:
            # Return 50/50 if no classifiers work
            return np.full((len(X), 2), 0.5)
        
        # Average probabilities
        return np.mean(all_probs, axis=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


@dataclass
class Expert:
    """A specialized expert for a particular market domain.
    
    Attributes:
        expert_id: Unique identifier
        expert_type: 'category', 'risk', 'time', 'volume', 'volatility', 'timing', or 'dynamic'
        domain: Specific domain (e.g., 'sports', 'low_risk', 'short_term')
    """
    
    expert_id: str
    expert_type: str  # 'category', 'risk', 'time', 'volume', 'volatility', 'timing', 'dynamic'
    domain: str  # e.g., 'sports', 'low_risk', 'short_term'
    
    # State (use versioning system)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Model (set after training)
    _model: Optional[_ExpertEnsemble] = field(default=None, repr=False)
    _feature_cols: List[str] = field(default_factory=list, repr=False)
    
    # Performance metrics
    metrics: ExpertMetrics = field(default_factory=ExpertMetrics)
    
    # Training history (keep last 20 for trend analysis)
    training_history: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    
    # Versioning manager
    _version_manager: Optional[ExpertVersionManager] = field(default=None, repr=False)
    _versions_dir: str = field(default="data/moe_models/versions", repr=False)
    
    # Confidence multiplier (new experts start lower)
    confidence_multiplier: float = 0.5
    
    def __post_init__(self):
        """Initialize feature columns and version manager."""
        if not self._feature_cols:
            self._feature_cols = self._get_base_feature_cols()
        
        # Initialize version manager
        if self._version_manager is None:
            versions_dir = os.path.join(self._versions_dir, self.expert_id)
            self._version_manager = ExpertVersionManager(
                self.expert_id, versions_dir
            )
    
    @property
    def state(self) -> ExpertState:
        """Get the current state from version manager."""
        if self._version_manager:
            return self._version_manager.state
        return ExpertState.UNTRAINED
    
    @property
    def is_active(self) -> bool:
        """Check if expert is active (trading enabled)."""
        if self._version_manager:
            return self._version_manager.is_trading_enabled()
        return False
    
    @property
    def is_deprecated(self) -> bool:
        """Check if expert is deprecated."""
        return self.state == ExpertState.DEPRECATED
    
    @property
    def current_version(self) -> Optional[int]:
        """Get current version ID."""
        if self._version_manager:
            return self._version_manager.current_version_id
        return None
    
    def _get_base_feature_cols(self) -> List[str]:
        """Get the base feature columns for training."""
        return [
            # Price features
            "price", "bid", "ask", "spread", "spread_pct", "mid_price",
            # Volume
            "volume_24h", "volume_1h", "volume_6h",
            # Liquidity
            "liquidity", "liquidity_bid", "liquidity_ask",
            # Orderbook
            "orderbook_imbalance", "bid_depth", "ask_depth",
            "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10",
            "best_bid_size", "best_ask_size", "bid_ask_size_ratio",
            # Momentum
            "momentum_1h", "momentum_4h", "momentum_24h", "momentum_7d",
            # Volatility
            "volatility_1h", "volatility_24h", "volatility_7d",
            # Trade flow
            "trade_count_1h", "trade_count_24h",
            "avg_trade_size_1h", "avg_trade_size_24h",
            "buy_sell_ratio_1h",
            # Market metadata
            "days_to_resolution", "hours_to_resolution", "market_age_days",
            # Timing
            "hour_of_day", "day_of_week", "is_weekend",
            # Market state
            "open_interest", "unique_traders",
        ]
    
    def _add_interaction_features(self, X: np.ndarray) -> np.ndarray:
        """Add interaction features to improve pattern detection."""
        # Create interaction features
        interactions = []
        
        try:
            # Volume ratios
            vol_1h_idx = self._feature_cols.index("volume_1h") if "volume_1h" in self._feature_cols else -1
            vol_24h_idx = self._feature_cols.index("volume_24h") if "volume_24h" in self._feature_cols else -1
            
            if vol_1h_idx >= 0 and vol_24h_idx >= 0:
                vol_ratio = np.where(
                    X[:, vol_24h_idx] > 0,
                    X[:, vol_1h_idx] / (X[:, vol_24h_idx] + 1e-8),
                    0
                )
                interactions.append(vol_ratio.reshape(-1, 1))
            
            # Volatility ratio
            volat_1h_idx = self._feature_cols.index("volatility_1h") if "volatility_1h" in self._feature_cols else -1
            volat_24h_idx = self._feature_cols.index("volatility_24h") if "volatility_24h" in self._feature_cols else -1
            
            if volat_1h_idx >= 0 and volat_24h_idx >= 0:
                volat_ratio = np.where(
                    X[:, volat_24h_idx] > 0,
                    X[:, volat_1h_idx] / (X[:, volat_24h_idx] + 1e-8),
                    1
                )
                interactions.append(volat_ratio.reshape(-1, 1))
            
            # Momentum agreement (1h and 24h same direction)
            mom_1h_idx = self._feature_cols.index("momentum_1h") if "momentum_1h" in self._feature_cols else -1
            mom_24h_idx = self._feature_cols.index("momentum_24h") if "momentum_24h" in self._feature_cols else -1
            
            if mom_1h_idx >= 0 and mom_24h_idx >= 0:
                mom_agreement = (np.sign(X[:, mom_1h_idx]) == np.sign(X[:, mom_24h_idx])).astype(float)
                interactions.append(mom_agreement.reshape(-1, 1))
                
                # Combined momentum strength
                mom_strength = np.abs(X[:, mom_1h_idx]) + np.abs(X[:, mom_24h_idx])
                interactions.append(mom_strength.reshape(-1, 1))
            
            # Price position (where in 0-1 range)
            price_idx = self._feature_cols.index("price") if "price" in self._feature_cols else -1
            if price_idx >= 0:
                price_extreme = np.abs(X[:, price_idx] - 0.5) * 2  # 0 at 0.5, 1 at extremes
                interactions.append(price_extreme.reshape(-1, 1))
            
            # Liquidity quality
            liq_bid_idx = self._feature_cols.index("liquidity_bid") if "liquidity_bid" in self._feature_cols else -1
            liq_ask_idx = self._feature_cols.index("liquidity_ask") if "liquidity_ask" in self._feature_cols else -1
            
            if liq_bid_idx >= 0 and liq_ask_idx >= 0:
                liq_balance = X[:, liq_bid_idx] / (X[:, liq_bid_idx] + X[:, liq_ask_idx] + 1e-8)
                interactions.append(liq_balance.reshape(-1, 1))
            
            # Time pressure (approaching resolution)
            days_idx = self._feature_cols.index("days_to_resolution") if "days_to_resolution" in self._feature_cols else -1
            if days_idx >= 0:
                time_pressure = 1 / (X[:, days_idx] + 1)  # Higher when closer to resolution
                interactions.append(time_pressure.reshape(-1, 1))
            
        except Exception as e:
            logger.debug(f"Error adding interaction features: {e}")
        
        if interactions:
            return np.hstack([X] + interactions)
        return X
    
    def train(
        self,
        X: np.ndarray,
        y_binary: np.ndarray,
        y_reg: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> ExpertMetrics:
        """Train this expert on the provided data.
        
        Args:
            X: Feature matrix
            y_binary: Binary target (1=profitable, 0=not profitable)
            y_reg: Regression target (actual price change) for simulation
            sample_weights: Optional sample weights
        
        Returns:
            ExpertMetrics with profitability results
        """
        if len(X) < 50:
            logger.warning(f"Expert {self.expert_id}: Not enough data ({len(X)} samples)")
            return self.metrics
        
        logger.info(f"Expert {self.expert_id} ({self.domain}): Training on {len(X)} samples...")
        
        # Add interaction features
        X_enhanced = self._add_interaction_features(X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_enhanced)
        
        # Train multiple classifiers and select best
        classifiers = []
        classifier_configs = [
            ("LogisticRegression", LogisticRegression(
                C=0.1, class_weight='balanced', solver='lbfgs', max_iter=1000
            )),
            ("RandomForest", RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight='balanced',
                random_state=42, n_jobs=-1
            )),
            ("ExtraTrees", ExtraTreesClassifier(
                n_estimators=100, max_depth=8, class_weight='balanced',
                random_state=42, n_jobs=-1
            )),
            ("HistGradientBoosting", HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.05,
                early_stopping=True, validation_fraction=0.1,
                random_state=42
            )),
        ]
        
        # Cross-validate and select top 3 classifiers
        cv_scores = []
        for name, clf in classifier_configs:
            try:
                if sample_weights is not None and hasattr(clf, 'fit'):
                    clf.fit(X_scaled, y_binary, sample_weight=sample_weights)
                else:
                    clf.fit(X_scaled, y_binary)
                
                # Get CV score
                scores = cross_val_score(clf, X_scaled, y_binary, cv=3, scoring='roc_auc')
                cv_scores.append((name, clf, np.mean(scores)))
            except Exception as e:
                logger.debug(f"Expert {self.expert_id}: Failed to train {name}: {e}")
                continue
        
        if not cv_scores:
            logger.error(f"Expert {self.expert_id}: All classifiers failed")
            return self.metrics
        
        # Sort by CV score and take top 3
        cv_scores.sort(key=lambda x: x[2], reverse=True)
        classifiers = [clf for _, clf, _ in cv_scores[:3]]
        
        # Create ensemble
        self._model = _ExpertEnsemble(classifiers, scaler)
        
        # Evaluate with profitability simulation
        self.metrics = self._evaluate_profitability(X_enhanced, y_binary, y_reg)
        self.metrics.n_training_examples = len(X)
        self.metrics.n_features_used = X_enhanced.shape[1]
        self.metrics.last_trained = datetime.utcnow()
        
        # Log results
        logger.info(
            f"Expert {self.expert_id} ({self.domain}): "
            f"profit={self.metrics.simulated_profit_pct:+.1%}, "
            f"trades={self.metrics.simulated_num_trades}, "
            f"win_rate={self.metrics.simulated_win_rate:.1%}, "
            f"PF={self.metrics.simulated_profit_factor:.2f}"
        )
        
        # Record training history
        self.training_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "n_samples": len(X),
            "profit_pct": self.metrics.simulated_profit_pct,
            "num_trades": self.metrics.simulated_num_trades,
            "win_rate": self.metrics.simulated_win_rate,
        })
        
        return self.metrics
    
    def _evaluate_profitability(
        self,
        X: np.ndarray,
        y_binary: np.ndarray,
        y_reg: np.ndarray,
    ) -> ExpertMetrics:
        """Evaluate expert using PROFITABILITY SIMULATION.
        
        This is the key metric - we simulate actual trading and measure P&L.
        """
        if self._model is None:
            return ExpertMetrics()
        
        # Get predictions and confidence
        probs = self._model.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        
        # Binary metrics for reference
        accuracy = np.mean(predictions == y_binary)
        
        # Profitable accuracy
        profitable_mask = np.abs(y_reg) > (SPREAD_COST + MIN_PROFIT_THRESHOLD)
        if np.sum(profitable_mask) > 0:
            profitable_accuracy = np.mean(predictions[profitable_mask] == y_binary[profitable_mask])
        else:
            profitable_accuracy = 0.5
        
        # Confident predictions
        confident_mask = confidence >= CONFIDENCE_THRESHOLD
        n_confident = np.sum(confident_mask)
        
        # === PROFITABILITY SIMULATION ===
        portfolio = 1.0
        peak_portfolio = 1.0
        max_drawdown = 0.0
        
        trade_returns = []
        wins = []
        losses = []
        
        for i in range(len(X)):
            if not confident_mask[i]:
                continue
            
            if predictions[i] == 1:  # Model says profitable
                actual_change = y_reg[i] if i < len(y_reg) else 0
                trade_return = actual_change - SPREAD_COST
                
                portfolio_return = POSITION_SIZE * trade_return
                portfolio = portfolio * (1 + portfolio_return)
                trade_returns.append(portfolio_return)
                
                if portfolio_return > 0:
                    wins.append(portfolio_return)
                else:
                    losses.append(portfolio_return)
                
                if portfolio > peak_portfolio:
                    peak_portfolio = portfolio
                current_dd = (peak_portfolio - portfolio) / peak_portfolio
                if current_dd > max_drawdown:
                    max_drawdown = current_dd
        
        num_trades = len(trade_returns)
        
        if num_trades == 0:
            return ExpertMetrics(
                directional_accuracy=accuracy,
                profitable_accuracy=profitable_accuracy,
                confident_trade_pct=float(n_confident / len(X)) if len(X) > 0 else 0,
                avg_confidence=float(np.mean(confidence)),
            )
        
        total_profit = portfolio - 1.0
        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 10.0
        
        # Sharpe ratio
        if len(trade_returns) > 1:
            returns_std = np.std(trade_returns)
            if returns_std > 0:
                trades_per_year = num_trades * (365 / 7)
                sharpe = (np.mean(trade_returns) / returns_std) * np.sqrt(trades_per_year)
            else:
                sharpe = 0
        else:
            sharpe = 0
        
        return ExpertMetrics(
            simulated_profit_pct=total_profit,
            simulated_num_trades=num_trades,
            simulated_win_rate=win_rate,
            simulated_avg_win=float(avg_win),
            simulated_avg_loss=float(avg_loss),
            simulated_profit_factor=profit_factor,
            simulated_max_drawdown=max_drawdown,
            simulated_sharpe=sharpe,
            directional_accuracy=accuracy,
            profitable_accuracy=profitable_accuracy,
            confident_trade_pct=float(n_confident / len(X)) if len(X) > 0 else 0,
            avg_confidence=float(np.mean(confidence)),
        )
    
    def predict(self, features: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Make a prediction with confidence.
        
        Returns:
            Tuple of (predicted_profitable, confidence) or None
            predicted_profitable: 1.0 = profitable, 0.0 = not profitable
            confidence: 0.0 to 1.0
        """
        if self._model is None:
            return None
        
        try:
            # Build feature vector
            X = []
            for col in self._feature_cols:
                val = features.get(col, 0)
                try:
                    X.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    X.append(0.0)
            
            X = np.array(X).reshape(1, -1)
            X = self._add_interaction_features(X)
            
            # Get prediction
            probs = self._model.predict_proba(X)[0]
            prediction = int(np.argmax(probs))
            confidence = float(np.max(probs))
            
            return (float(prediction), confidence)
            
        except Exception as e:
            logger.debug(f"Expert {self.expert_id} prediction error: {e}")
            return None
    
    def should_deprecate(self) -> bool:
        """Check if this expert should be deprecated.
        
        Now uses the state machine - only deprecated state means truly deprecated.
        """
        return self.state == ExpertState.DEPRECATED
    
    def update_state_after_training(self) -> None:
        """Update state after training using version manager."""
        if self._version_manager:
            # Create new version with current metrics (pass model for first version)
            self._version_manager.update_version_metrics(self.metrics, model=self._model)
            
            # Update confidence multiplier based on state
            self._update_confidence_multiplier()
    
    def _update_confidence_multiplier(self) -> None:
        """Update confidence multiplier based on performance history."""
        if self.state == ExpertState.ACTIVE:
            # Increase confidence for good performance
            good_cycles = sum(
                1 for h in self.training_history[-5:] 
                if h.get("profit_pct", 0) > 0
            )
            # 0.5 base + 0.1 per good cycle, max 1.0
            self.confidence_multiplier = min(1.0, 0.5 + 0.1 * good_cycles)
        elif self.state == ExpertState.PROBATION:
            self.confidence_multiplier = 0.5
        elif self.state == ExpertState.SUSPENDED:
            self.confidence_multiplier = 0.0  # No trading
        else:
            self.confidence_multiplier = 0.3
    
    def calculate_trend(self, window: int = 5) -> float:
        """Calculate performance trend over last N training cycles.
        
        Returns:
            Positive = improving, negative = declining
        """
        if len(self.training_history) < window:
            return 0.0
        
        recent = self.training_history[-window:]
        profits = [r.get("profit_pct", 0) for r in recent]
        
        if len(profits) < 2:
            return 0.0
        
        # Linear trend
        return (profits[-1] - profits[0]) / window
    
    def deprecate(self, reason: str = ""):
        """Mark this expert as deprecated via state machine."""
        if self._version_manager:
            self._version_manager.state = ExpertState.DEPRECATED
            self._version_manager._consecutive_bad_cycles = 999
        logger.info(f"Expert {self.expert_id} ({self.domain}) DEPRECATED: {reason}")
    
    def suspend(self, reason: str = ""):
        """Temporarily suspend trading for this expert."""
        if self._version_manager:
            self._version_manager.state = ExpertState.SUSPENDED
        self.confidence_multiplier = 0.0
        logger.info(f"Expert {self.expert_id} ({self.domain}) SUSPENDED: {reason}")
    
    def save(self, path: str):
        """Save expert to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            "expert_id": self.expert_id,
            "expert_type": self.expert_type,
            "domain": self.domain,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics.to_dict(),
            "feature_cols": self._feature_cols,
            "training_history": self.training_history[-20:],  # Keep last 20
            "confidence_multiplier": self.confidence_multiplier,
        }
        
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump(state, f)
        
        # Save version manager state
        if self._version_manager:
            version_state_path = path + ".versions.json"
            self._version_manager.save_state(version_state_path)
        
        if self._model is not None:
            with open(path + ".model.pkl", "wb") as f:
                pickle.dump(self._model, f)
    
    @classmethod
    def load(cls, path: str) -> Optional["Expert"]:
        """Load expert from disk."""
        try:
            with open(path + ".meta.pkl", "rb") as f:
                state = pickle.load(f)
            
            expert = cls(
                expert_id=state["expert_id"],
                expert_type=state["expert_type"],
                domain=state["domain"],
                created_at=datetime.fromisoformat(state["created_at"]),
            )
            
            expert.metrics = ExpertMetrics.from_dict(state.get("metrics", {}))
            expert._feature_cols = state.get("feature_cols", expert._get_base_feature_cols())
            expert.training_history = state.get("training_history", [])
            expert.confidence_multiplier = state.get("confidence_multiplier", 0.5)
            
            # Load version manager state
            version_state_path = path + ".versions.json"
            if os.path.exists(version_state_path):
                versions_dir = os.path.join("data/moe_models/versions", expert.expert_id)
                expert._version_manager = ExpertVersionManager.load_state(
                    version_state_path, expert.expert_id, versions_dir
                )
            
            # Load model if exists
            model_path = path + ".model.pkl"
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    expert._model = pickle.load(f)
            
            return expert
            
        except Exception as e:
            logger.error(f"Failed to load expert from {path}: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for status display."""
        version_info = None
        if self._version_manager:
            version_info = self._version_manager.to_dict()
        
        return {
            "expert_id": self.expert_id,
            "expert_type": self.expert_type,
            "domain": self.domain,
            "state": self.state.value,
            "is_active": self.is_active,
            "is_deprecated": self.is_deprecated,
            "created_at": self.created_at.isoformat(),
            "has_model": self._model is not None,
            "current_version": self.current_version,
            "confidence_multiplier": self.confidence_multiplier,
            "trend": self.calculate_trend(),
            "metrics": self.metrics.to_dict(),
            "score": self.metrics.score(),
            "version_info": version_info,
        }
