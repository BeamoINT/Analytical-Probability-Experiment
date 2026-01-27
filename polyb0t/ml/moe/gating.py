"""Gating Network for expert routing.

The gating network learns which expert to trust for each market type,
routing trades to the most profitable expert for the given market characteristics.

Supports both gradient boosting (fast) and neural attention-based gating (more accurate).
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Try to import neural gating
try:
    from polyb0t.ml.moe.neural_gating import NeuralGatingTrainer, HybridGating
    NEURAL_GATING_AVAILABLE = True
except ImportError:
    NEURAL_GATING_AVAILABLE = False
    NeuralGatingTrainer = None
    HybridGating = None

logger = logging.getLogger(__name__)


@dataclass
class GatingMetrics:
    """Metrics for gating network performance."""
    
    routing_accuracy: float = 0.0  # How often it routes to the best expert
    profit_vs_random: float = 0.0  # Profit improvement over random routing
    n_samples_trained: int = 0
    last_trained: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_accuracy": self.routing_accuracy,
            "profit_vs_random": self.profit_vs_random,
            "n_samples_trained": self.n_samples_trained,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GatingMetrics":
        last_trained = None
        if data.get("last_trained"):
            try:
                last_trained = datetime.fromisoformat(data["last_trained"])
            except:
                pass
        
        return cls(
            routing_accuracy=data.get("routing_accuracy", 0.0),
            profit_vs_random=data.get("profit_vs_random", 0.0),
            n_samples_trained=data.get("n_samples_trained", 0),
            last_trained=last_trained,
        )


class GatingNetwork:
    """Gating network that routes markets to the best expert.

    The gating network learns to predict which expert will be most
    profitable for a given market, based on market characteristics.

    Supports both gradient boosting (fast) and neural attention-based gating.
    """

    def __init__(self, expert_ids: List[str], use_neural: Optional[bool] = None):
        """Initialize gating network.

        Args:
            expert_ids: List of expert IDs this network can route to
            use_neural: Use neural gating (None = check settings)
        """
        self.expert_ids = expert_ids
        self._model: Optional[GradientBoostingClassifier] = None
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(expert_ids)

        self._feature_names = self._get_gating_features()
        self.metrics = GatingMetrics()

        # Expert profit history for training
        self._expert_profits: Dict[str, List[float]] = {eid: [] for eid in expert_ids}

        # Neural gating support
        if use_neural is None:
            try:
                from polyb0t.config.settings import get_settings
                settings = get_settings()
                use_neural = settings.ai_use_neural_gating
            except Exception:
                use_neural = False

        self._use_neural = use_neural and NEURAL_GATING_AVAILABLE
        self._neural_trainer: Optional[NeuralGatingTrainer] = None
        self._hybrid_gating: Optional[HybridGating] = None
    
    def _get_gating_features(self) -> List[str]:
        """Get feature names used for gating decisions."""
        return [
            # Category (will be one-hot encoded externally)
            "category_sports", "category_politics_us", "category_politics_intl",
            "category_crypto", "category_economics", "category_entertainment",
            "category_tech", "category_weather", "category_science",
            "category_legal", "category_other",
            # Risk level
            "is_low_risk", "is_medium_risk", "is_high_risk",
            # Time horizon
            "is_short_term", "is_medium_term", "is_long_term",
            # Market characteristics
            "price", "volatility_24h", "volume_24h", "liquidity",
            "days_to_resolution", "market_age_days",
            "spread_pct", "orderbook_imbalance",
        ]
    
    def extract_gating_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract features for gating decision.
        
        Args:
            features: Full feature dict from market
        
        Returns:
            Feature vector for gating network
        """
        X = []
        
        # Category one-hot encoding
        category = features.get("category", "other")
        categories = [
            "sports", "politics_us", "politics_intl", "crypto", "economics",
            "entertainment", "tech", "weather", "science", "legal", "other"
        ]
        for cat in categories:
            X.append(1.0 if category == cat else 0.0)
        
        # Risk level based on price
        price = features.get("price", 0.5)
        is_low_risk = 1.0 if (price > 0.85 or price < 0.15) else 0.0
        is_high_risk = 1.0 if (0.4 < price < 0.6) else 0.0
        is_medium_risk = 1.0 if not (is_low_risk or is_high_risk) else 0.0
        X.extend([is_low_risk, is_medium_risk, is_high_risk])
        
        # Time horizon based on days to resolution
        days = features.get("days_to_resolution", 30)
        is_short_term = 1.0 if days < 7 else 0.0
        is_long_term = 1.0 if days > 30 else 0.0
        is_medium_term = 1.0 if not (is_short_term or is_long_term) else 0.0
        X.extend([is_short_term, is_medium_term, is_long_term])
        
        # Numeric features
        numeric_features = [
            "price", "volatility_24h", "volume_24h", "liquidity",
            "days_to_resolution", "market_age_days",
            "spread_pct", "orderbook_imbalance",
        ]
        for feat in numeric_features:
            val = features.get(feat, 0)
            try:
                X.append(float(val) if val is not None else 0.0)
            except (ValueError, TypeError):
                X.append(0.0)
        
        return np.array(X)
    
    def train(
        self,
        samples: List[Dict[str, Any]],
        expert_results: Dict[str, List[float]],
    ) -> GatingMetrics:
        """Train gating network to route to most profitable expert.

        Args:
            samples: List of market feature dicts
            expert_results: Dict mapping expert_id -> list of profits for each sample

        Returns:
            GatingMetrics
        """
        if len(samples) < 50:
            logger.warning(f"Gating: Not enough samples ({len(samples)})")
            return self.metrics

        logger.info(f"Training gating network on {len(samples)} samples...")

        # Build training data
        X_list = []
        y_list = []  # Best expert for each sample
        expert_perf_matrix = []  # Performance of each expert on each sample

        for i, sample in enumerate(samples):
            X_list.append(self.extract_gating_features(sample))

            # Find best expert for this sample (highest profit)
            best_expert = None
            best_profit = float('-inf')
            sample_perfs = []

            for expert_id in self.expert_ids:
                if expert_id in expert_results and i < len(expert_results[expert_id]):
                    profit = expert_results[expert_id][i]
                    sample_perfs.append(profit)
                    if profit > best_profit:
                        best_profit = profit
                        best_expert = expert_id
                else:
                    sample_perfs.append(0.0)

            if best_expert is None:
                best_expert = self.expert_ids[0]  # Default

            y_list.append(best_expert)
            expert_perf_matrix.append(sample_perfs)

        X = np.array(X_list)
        y = self._label_encoder.transform(y_list)
        expert_performance = np.array(expert_perf_matrix)

        # Normalize expert performance to probabilities for soft labels
        if expert_performance.shape[1] > 0:
            # Shift to positive and normalize
            shifted = expert_performance - expert_performance.min(axis=1, keepdims=True) + 0.01
            soft_labels = shifted / shifted.sum(axis=1, keepdims=True)
        else:
            soft_labels = None

        # Train neural gating if available and configured
        if self._use_neural and NEURAL_GATING_AVAILABLE and len(X) >= 500:
            return self._train_neural(X, y, soft_labels, samples, expert_results)
        else:
            return self._train_gradient_boosting(X, y, samples, expert_results)

    def _train_gradient_boosting(
        self,
        X: np.ndarray,
        y: np.ndarray,
        samples: List[Dict[str, Any]],
        expert_results: Dict[str, List[float]],
    ) -> GatingMetrics:
        """Train using gradient boosting (fast baseline)."""
        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Train gradient boosting classifier
        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._model.fit(X_scaled, y)

        # Evaluate routing accuracy
        predictions = self._model.predict(X_scaled)
        routing_accuracy = np.mean(predictions == y)

        # Calculate profit improvement vs random
        random_profits = []
        gated_profits = []

        for i in range(len(samples)):
            # Random selection
            random_expert = np.random.choice(self.expert_ids)
            if random_expert in expert_results and i < len(expert_results[random_expert]):
                random_profits.append(expert_results[random_expert][i])

            # Gated selection
            gated_expert = self._label_encoder.inverse_transform([predictions[i]])[0]
            if gated_expert in expert_results and i < len(expert_results[gated_expert]):
                gated_profits.append(expert_results[gated_expert][i])

        profit_improvement = np.mean(gated_profits) - np.mean(random_profits) if random_profits else 0

        self.metrics = GatingMetrics(
            routing_accuracy=routing_accuracy,
            profit_vs_random=profit_improvement,
            n_samples_trained=len(samples),
            last_trained=datetime.utcnow(),
        )

        logger.info(
            f"Gating network (GB) trained: routing_acc={routing_accuracy:.1%}, "
            f"profit_vs_random={profit_improvement:+.2%}"
        )

        return self.metrics

    def _train_neural(
        self,
        X: np.ndarray,
        y: np.ndarray,
        soft_labels: Optional[np.ndarray],
        samples: List[Dict[str, Any]],
        expert_results: Dict[str, List[float]],
    ) -> GatingMetrics:
        """Train using neural attention-based gating."""
        import time
        start_time = time.time()

        logger.info(f"Training NEURAL gating network on {len(X)} samples...")

        # Get settings for configuration
        try:
            from polyb0t.config.settings import get_settings
            settings = get_settings()
            hidden_dim = settings.ai_gating_hidden_dim
            num_heads = settings.ai_gating_num_heads
        except Exception:
            hidden_dim = 64
            num_heads = 4

        # Time-based split for validation
        n_samples = len(X)
        val_fraction = 0.2
        train_end = int(n_samples * (1 - val_fraction))

        X_train, X_val = X[:train_end], X[train_end:]
        y_train, y_val = y[:train_end], y[train_end:]
        soft_train = soft_labels[:train_end] if soft_labels is not None else None

        try:
            # Initialize and train neural gating
            self._neural_trainer = NeuralGatingTrainer(
                input_dim=X.shape[1],
                num_experts=len(self.expert_ids),
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )

            _, neural_metrics = self._neural_trainer.train(
                X_train, y_train, X_val, y_val, soft_train
            )

            # Also train GB for hybrid/fallback
            X_scaled = self._scaler.fit_transform(X)
            self._model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            self._model.fit(X_scaled, y)

            # Use neural predictions for evaluation
            neural_weights = self._neural_trainer.predict(X)
            predictions = np.argmax(neural_weights, axis=1)
            routing_accuracy = np.mean(predictions == y)

        except Exception as e:
            logger.warning(f"Neural gating failed ({e}), falling back to GB")
            return self._train_gradient_boosting(X, y, samples, expert_results)

        # Calculate profit improvement vs random
        random_profits = []
        gated_profits = []

        for i in range(len(samples)):
            # Random selection
            random_expert = np.random.choice(self.expert_ids)
            if random_expert in expert_results and i < len(expert_results[random_expert]):
                random_profits.append(expert_results[random_expert][i])

            # Neural gated selection
            pred_idx = predictions[i]
            if pred_idx < len(self.expert_ids):
                gated_expert = self.expert_ids[pred_idx]
                if gated_expert in expert_results and i < len(expert_results[gated_expert]):
                    gated_profits.append(expert_results[gated_expert][i])

        profit_improvement = np.mean(gated_profits) - np.mean(random_profits) if random_profits else 0

        training_time = time.time() - start_time
        self.metrics = GatingMetrics(
            routing_accuracy=routing_accuracy,
            profit_vs_random=profit_improvement,
            n_samples_trained=len(samples),
            last_trained=datetime.utcnow(),
        )

        logger.info(
            f"Gating network (NEURAL) trained in {training_time:.1f}s: "
            f"routing_acc={routing_accuracy:.1%}, "
            f"profit_vs_random={profit_improvement:+.2%}"
        )

        return self.metrics
    
    def get_weights(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Get expert weights for a given market.

        Args:
            features: Market feature dict

        Returns:
            Dict mapping expert_id to weight (sum to 1.0)
        """
        # Check if neural gating is trained and available
        if self._use_neural and self._neural_trainer is not None:
            try:
                X = self.extract_gating_features(features).reshape(1, -1)
                neural_weights = self._neural_trainer.predict(X)[0]

                # Map to expert IDs
                weights = {}
                for i, expert_id in enumerate(self.expert_ids):
                    if i < len(neural_weights):
                        weights[expert_id] = float(neural_weights[i])
                    else:
                        weights[expert_id] = 0.0

                # Normalize
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}

                return weights

            except Exception as e:
                logger.debug(f"Neural gating error ({e}), falling back to GB")
                # Fall through to GB

        if self._model is None:
            # Uniform weights if not trained
            weight = 1.0 / len(self.expert_ids)
            return {eid: weight for eid in self.expert_ids}

        try:
            X = self.extract_gating_features(features).reshape(1, -1)
            X_scaled = self._scaler.transform(X)

            # Get class probabilities
            probs = self._model.predict_proba(X_scaled)[0]

            # Map to expert IDs
            weights = {}
            for i, expert_id in enumerate(self._label_encoder.classes_):
                if i < len(probs):
                    weights[expert_id] = float(probs[i])
                else:
                    weights[expert_id] = 0.0

            # Normalize to sum to 1
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

            return weights

        except Exception as e:
            logger.debug(f"Gating weight error: {e}")
            weight = 1.0 / len(self.expert_ids)
            return {eid: weight for eid in self.expert_ids}
    
    def get_best_expert(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """Get the best expert for a given market.
        
        Args:
            features: Market feature dict
        
        Returns:
            Tuple of (expert_id, confidence)
        """
        weights = self.get_weights(features)
        best_expert = max(weights.keys(), key=lambda k: weights[k])
        return best_expert, weights[best_expert]
    
    def update_expert_ids(self, expert_ids: List[str]):
        """Update the list of expert IDs (when experts are added/removed)."""
        # Add new experts
        for eid in expert_ids:
            if eid not in self.expert_ids:
                self.expert_ids.append(eid)
                self._expert_profits[eid] = []
        
        # Mark removed experts (don't delete to preserve training)
        for eid in self.expert_ids:
            if eid not in expert_ids:
                self._expert_profits[eid] = []  # Clear history
        
        # Re-fit label encoder
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(expert_ids)
        
        # Model needs retraining
        self._model = None
    
    def save(self, path: str):
        """Save gating network to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            "expert_ids": self.expert_ids,
            "feature_names": self._feature_names,
            "metrics": self.metrics.to_dict(),
            "expert_profits": self._expert_profits,
            "use_neural": self._use_neural,
        }

        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump(state, f)

        if self._model is not None:
            model_state = {
                "model": self._model,
                "scaler": self._scaler,
                "label_encoder": self._label_encoder,
            }
            with open(path + ".model.pkl", "wb") as f:
                pickle.dump(model_state, f)

        # Save neural gating if trained
        if self._neural_trainer is not None:
            try:
                neural_path = path + ".neural.pt"
                self._neural_trainer.save(neural_path)
                logger.debug(f"Saved neural gating to {neural_path}")
            except Exception as e:
                logger.debug(f"Could not save neural gating: {e}")

    @classmethod
    def load(cls, path: str) -> Optional["GatingNetwork"]:
        """Load gating network from disk."""
        try:
            with open(path + ".meta.pkl", "rb") as f:
                state = pickle.load(f)

            use_neural = state.get("use_neural", False)
            gating = cls(expert_ids=state["expert_ids"], use_neural=use_neural)
            gating._feature_names = state.get("feature_names", gating._get_gating_features())
            gating.metrics = GatingMetrics.from_dict(state.get("metrics", {}))
            gating._expert_profits = state.get("expert_profits", {})

            # Load GB model if exists
            model_path = path + ".model.pkl"
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model_state = pickle.load(f)
                gating._model = model_state["model"]
                gating._scaler = model_state["scaler"]
                gating._label_encoder = model_state["label_encoder"]

            # Load neural gating if exists
            neural_path = path + ".neural.pt"
            if os.path.exists(neural_path) and NEURAL_GATING_AVAILABLE:
                try:
                    gating._neural_trainer = NeuralGatingTrainer.load(neural_path)
                    logger.debug(f"Loaded neural gating from {neural_path}")
                except Exception as e:
                    logger.debug(f"Could not load neural gating: {e}")

            return gating

        except Exception as e:
            logger.error(f"Failed to load gating network from {path}: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for status display."""
        return {
            "n_experts": len(self.expert_ids),
            "expert_ids": self.expert_ids,
            "is_trained": self._model is not None,
            "use_neural": self._use_neural,
            "neural_trained": self._neural_trainer is not None,
            "metrics": self.metrics.to_dict(),
        }
