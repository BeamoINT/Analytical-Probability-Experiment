"""Meta-Controller AI System for Expert Mixing.

The Meta-Controller learns optimal combinations of experts for different
market types. It uses ensemble predictions from ALL active experts.

Key features:
- Ensemble predictions using all active experts (not single routing)
- Softmax-based weighting using expert performance history
- Synergy bonuses when expert pairs have positive correlation
- Confidence based on expert agreement and performance
- Dynamic weight adjustment based on P&L
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Configuration
STATE_FILE = "data/meta_controller_state.json"
MIXTURE_HISTORY_FILE = "data/mixture_history.json"
MAX_SUPPORTING_EXPERTS = 4
MIN_TRADES_FOR_LEARNING = 20
LEARNING_RATE = 0.1
DECAY_FACTOR = 0.95  # Weight decay for older performance


class CombinationStrategy(Enum):
    """How to combine expert predictions."""
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    STACKED = "stacked"


@dataclass
class MixtureConfig:
    """Configuration for an expert mixture."""
    
    primary_expert_id: str
    supporting_experts: List[Tuple[str, float]] = field(default_factory=list)
    combination_strategy: CombinationStrategy = CombinationStrategy.WEIGHTED_AVERAGE
    confidence_boost: float = 1.0  # Multiplier for mixture confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_expert_id": self.primary_expert_id,
            "supporting_experts": self.supporting_experts,
            "combination_strategy": self.combination_strategy.value,
            "confidence_boost": self.confidence_boost,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixtureConfig":
        return cls(
            primary_expert_id=data["primary_expert_id"],
            supporting_experts=[(e[0], e[1]) for e in data.get("supporting_experts", [])],
            combination_strategy=CombinationStrategy(data.get("combination_strategy", "weighted_average")),
            confidence_boost=data.get("confidence_boost", 1.0),
        )


@dataclass
class MixtureOutcome:
    """Tracks the outcome of a mixture prediction."""
    
    mixture_id: str
    timestamp: datetime
    market_id: str
    primary_expert: str
    supporting_experts: List[str]
    expert_weights: Dict[str, float]
    prediction: float
    confidence: float
    actual_outcome: Optional[float] = None  # Set when trade resolves
    profit_pct: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mixture_id": self.mixture_id,
            "timestamp": self.timestamp.isoformat(),
            "market_id": self.market_id,
            "primary_expert": self.primary_expert,
            "supporting_experts": self.supporting_experts,
            "expert_weights": self.expert_weights,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "actual_outcome": self.actual_outcome,
            "profit_pct": self.profit_pct,
        }


@dataclass
class ExpertPairPerformance:
    """Tracks how well two experts work together."""
    
    expert_a: str
    expert_b: str
    combined_trades: int = 0
    combined_profit: float = 0.0
    agreement_rate: float = 0.0  # How often they agree
    synergy_score: float = 0.0  # Does combining improve performance?
    
    def update(self, profit: float, agreed: bool) -> None:
        """Update pair performance with new trade."""
        self.combined_trades += 1
        self.combined_profit += profit
        
        # Update agreement rate with exponential moving average
        alpha = 0.1
        agree_val = 1.0 if agreed else 0.0
        self.agreement_rate = alpha * agree_val + (1 - alpha) * self.agreement_rate
        
        # Update synergy score
        if self.combined_trades >= MIN_TRADES_FOR_LEARNING:
            avg_profit = self.combined_profit / self.combined_trades
            self.synergy_score = avg_profit * (1 + 0.5 * (1 - self.agreement_rate))


# Rules-based expert hints for different market types
CATEGORY_PRIMARY_EXPERTS = {
    "sports": "sports",
    "politics_us": "politics_us",
    "politics_intl": "politics_intl",
    "crypto": "crypto",
    "economics": "economics",
    "entertainment": "entertainment",
    "tech": "tech",
    "weather": "weather",
    "science": "science",
    "legal": "legal",
}

# Default supporting experts for each category (rules-based starting point)
DEFAULT_SUPPORTERS = {
    "sports": ["high_volume", "short_term", "weekend_trader"],
    "politics_us": ["medium_risk", "high_volatility", "long_term"],
    "politics_intl": ["high_risk", "long_term", "low_volume"],
    "crypto": ["high_volatility", "momentum_strong", "short_term"],
    "economics": ["low_risk", "medium_term", "high_liquidity"],
    "entertainment": ["medium_risk", "short_term", "high_volume"],
    "tech": ["medium_risk", "momentum_strong", "medium_term"],
    "weather": ["short_term", "low_volatility", "low_risk"],
    "science": ["long_term", "low_risk", "low_volume"],
    "legal": ["long_term", "low_risk", "low_volatility"],
}


class MixtureLearner:
    """Learns optimal expert combinations from trade outcomes."""
    
    def __init__(self):
        self.pair_performance: Dict[str, ExpertPairPerformance] = {}
        self.expert_weights: Dict[str, Dict[str, float]] = {}  # primary -> {support: weight}
        self.category_performance: Dict[str, Dict[str, float]] = {}  # category -> {mixture_key: profit}
        
    def _pair_key(self, a: str, b: str) -> str:
        """Create consistent key for expert pair."""
        return f"{min(a, b)}_{max(a, b)}"
    
    def record_outcome(
        self,
        primary: str,
        supporters: List[str],
        predictions: Dict[str, float],
        profit: float,
        category: str,
    ) -> None:
        """Record the outcome of a mixture prediction."""
        # Update pair performance for each supporter
        for supporter in supporters:
            pair_key = self._pair_key(primary, supporter)
            
            if pair_key not in self.pair_performance:
                self.pair_performance[pair_key] = ExpertPairPerformance(
                    expert_a=primary, expert_b=supporter
                )
            
            # Check if they agreed
            primary_pred = predictions.get(primary, 0.5)
            support_pred = predictions.get(supporter, 0.5)
            agreed = (primary_pred > 0.5) == (support_pred > 0.5)
            
            self.pair_performance[pair_key].update(profit, agreed)
        
        # Update category-specific learning
        if category not in self.category_performance:
            self.category_performance[category] = {}
        
        mixture_key = f"{primary}_{','.join(sorted(supporters))}"
        if mixture_key not in self.category_performance[category]:
            self.category_performance[category][mixture_key] = 0.0
        
        # Exponential moving average of profit
        alpha = LEARNING_RATE
        self.category_performance[category][mixture_key] = (
            alpha * profit + (1 - alpha) * self.category_performance[category][mixture_key]
        )
        
        # Update expert weights for this primary
        self._update_weights(primary, supporters, profit)
    
    def _update_weights(self, primary: str, supporters: List[str], profit: float) -> None:
        """Update learned weights for supporting experts."""
        if primary not in self.expert_weights:
            self.expert_weights[primary] = {}
        
        for supporter in supporters:
            if supporter not in self.expert_weights[primary]:
                self.expert_weights[primary][supporter] = 0.5  # Start at neutral
            
            # Adjust weight based on profit
            current = self.expert_weights[primary][supporter]
            delta = LEARNING_RATE * profit  # Positive profit increases weight
            new_weight = max(0.0, min(1.0, current + delta))
            self.expert_weights[primary][supporter] = new_weight
    
    def get_best_supporters(
        self,
        primary: str,
        available_experts: List[str],
        max_supporters: int = MAX_SUPPORTING_EXPERTS,
    ) -> List[Tuple[str, float]]:
        """Get the best supporting experts for a primary expert."""
        candidates = []
        
        for expert in available_experts:
            if expert == primary:
                continue
            
            # Get learned weight
            weight = 0.5  # Default
            if primary in self.expert_weights:
                weight = self.expert_weights[primary].get(expert, 0.5)
            
            # Check pair synergy
            pair_key = self._pair_key(primary, expert)
            if pair_key in self.pair_performance:
                synergy = self.pair_performance[pair_key].synergy_score
                weight = (weight + synergy) / 2
            
            candidates.append((expert, weight))
        
        # Sort by weight and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_supporters]
    
    def get_synergy_matrix(self, experts: List[str]) -> Dict[str, Dict[str, float]]:
        """Get synergy scores between all expert pairs."""
        matrix = {}
        for expert in experts:
            matrix[expert] = {}
            for other in experts:
                if expert == other:
                    matrix[expert][other] = 1.0
                else:
                    pair_key = self._pair_key(expert, other)
                    if pair_key in self.pair_performance:
                        matrix[expert][other] = self.pair_performance[pair_key].synergy_score
                    else:
                        matrix[expert][other] = 0.0
        return matrix
    
    def save_state(self, path: str = STATE_FILE) -> None:
        """Save learner state."""
        state = {
            "pair_performance": {
                k: {
                    "expert_a": v.expert_a,
                    "expert_b": v.expert_b,
                    "combined_trades": v.combined_trades,
                    "combined_profit": v.combined_profit,
                    "agreement_rate": v.agreement_rate,
                    "synergy_score": v.synergy_score,
                }
                for k, v in self.pair_performance.items()
            },
            "expert_weights": self.expert_weights,
            "category_performance": self.category_performance,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str = STATE_FILE) -> None:
        """Load learner state."""
        if not os.path.exists(path):
            return
        
        try:
            with open(path, "r") as f:
                state = json.load(f)
            
            self.pair_performance = {}
            for k, v in state.get("pair_performance", {}).items():
                self.pair_performance[k] = ExpertPairPerformance(
                    expert_a=v["expert_a"],
                    expert_b=v["expert_b"],
                    combined_trades=v["combined_trades"],
                    combined_profit=v["combined_profit"],
                    agreement_rate=v["agreement_rate"],
                    synergy_score=v["synergy_score"],
                )
            
            self.expert_weights = state.get("expert_weights", {})
            self.category_performance = state.get("category_performance", {})
            
        except Exception as e:
            logger.error(f"Failed to load learner state: {e}")


class MetaController:
    """Main controller that orchestrates expert mixing.
    
    Uses rules-based hints combined with learned optimal combinations
    to mix multiple experts for each prediction.
    """
    
    def __init__(self, expert_pool: Any = None):
        self.expert_pool = expert_pool
        self.learner = MixtureLearner()
        self.mixture_history: List[MixtureOutcome] = []
        self._next_mixture_id = 0
        
        # Load saved state
        self.learner.load_state()
        self._load_history()
    
    def _load_history(self, path: str = MIXTURE_HISTORY_FILE) -> None:
        """Load mixture history from file."""
        if not os.path.exists(path):
            return
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            self._next_mixture_id = data.get("next_id", 0)
            # Keep only last 1000 outcomes
            for item in data.get("outcomes", [])[-1000:]:
                outcome = MixtureOutcome(
                    mixture_id=item["mixture_id"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    market_id=item["market_id"],
                    primary_expert=item["primary_expert"],
                    supporting_experts=item["supporting_experts"],
                    expert_weights=item["expert_weights"],
                    prediction=item["prediction"],
                    confidence=item["confidence"],
                    actual_outcome=item.get("actual_outcome"),
                    profit_pct=item.get("profit_pct"),
                )
                self.mixture_history.append(outcome)
                
        except Exception as e:
            logger.error(f"Failed to load mixture history: {e}")
    
    def _save_history(self, path: str = MIXTURE_HISTORY_FILE) -> None:
        """Save mixture history to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "next_id": self._next_mixture_id,
            "outcomes": [o.to_dict() for o in self.mixture_history[-1000:]],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_primary_expert(self, features: Dict[str, Any]) -> str:
        """Get the primary expert for a market based on rules."""
        category = features.get("category", "other")
        return CATEGORY_PRIMARY_EXPERTS.get(category, "other")
    
    def get_mixture_config(self, features: Dict[str, Any]) -> MixtureConfig:
        """Get the optimal mixture configuration for a market."""
        category = features.get("category", "other")
        primary = self.get_primary_expert(features)
        
        # Get available experts
        if self.expert_pool:
            available = [e.expert_id for e in self.expert_pool.get_active_experts()]
        else:
            available = list(CATEGORY_PRIMARY_EXPERTS.values())
        
        # Get learned best supporters
        learned_supporters = self.learner.get_best_supporters(primary, available)
        
        # If no learned data, use defaults
        if not learned_supporters or all(w < 0.3 for _, w in learned_supporters):
            default = DEFAULT_SUPPORTERS.get(category, ["medium_risk", "medium_term"])
            learned_supporters = [(e, 0.5) for e in default if e in available]
        
        # Filter to only include supporters with weight > 0.2
        supporters = [(e, w) for e, w in learned_supporters if w > 0.2]
        
        # Calculate confidence boost based on synergy
        synergy_sum = sum(w for _, w in supporters)
        confidence_boost = 1.0 + 0.1 * min(synergy_sum, 2.0)
        
        return MixtureConfig(
            primary_expert_id=primary,
            supporting_experts=supporters,
            combination_strategy=CombinationStrategy.WEIGHTED_AVERAGE,
            confidence_boost=confidence_boost,
        )
    
    def predict_with_mixture(
        self,
        features: Dict[str, Any],
        config: Optional[MixtureConfig] = None,
        use_all_experts: bool = True,  # Always use ensemble by default
        use_two_stage: bool = True,  # Enable cross-expert awareness
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Make a prediction using two-stage cross-expert aware ensemble.
        
        Two-Stage Prediction:
        1. Stage 1: Collect independent predictions from all experts
        2. Compute consensus features (mean, std, agreement, bullish ratio)
        3. Stage 2: Re-predict with cross-expert awareness features
        
        Each expert's weight is based on:
        - Recent profit performance
        - Prediction confidence
        - Expert confidence multiplier
        
        Returns:
            Tuple of (prediction, confidence, metadata)
        """
        if self.expert_pool is None:
            return 0.5, 0.0, {"error": "No expert pool"}
        
        # Get ALL active experts
        active_experts = self.expert_pool.get_active_experts()
        
        if not active_experts:
            # Fallback to old routing-based approach
            if config is None:
                config = self.get_mixture_config(features)
            use_all_experts = False
        
        predictions = {}
        weights = {}
        expert_scores = {}  # For softmax weighting
        first_round_predictions = {}  # For two-stage prediction
        first_round_confidences = {}
        
        if use_all_experts:
            # === STAGE 1: Collect independent predictions from ALL experts ===
            for expert in active_experts:
                result = expert.predict(features)
                if result:
                    pred, conf = result
                    first_round_predictions[expert.expert_id] = pred
                    first_round_confidences[expert.expert_id] = conf
            
            # === COMPUTE CONSENSUS FEATURES ===
            consensus_features = self._compute_consensus_features(
                first_round_predictions, first_round_confidences
            )
            
            # === STAGE 2: Re-predict with cross-expert awareness ===
            if use_two_stage and first_round_predictions:
                # Enhance features with consensus
                enhanced_features = {**features, **consensus_features}
                
                # Re-predict with cross-expert awareness
                for expert in active_experts:
                    result = expert.predict(enhanced_features)
                    if result:
                        pred, conf = result
                        predictions[expert.expert_id] = pred
                        
                        # Score = based on profit history + confidence
                        profit_score = 1.0 + max(-0.5, min(1.0, expert.metrics.simulated_profit_pct))
                        expert_scores[expert.expert_id] = profit_score * conf * expert.confidence_multiplier
            else:
                # Use first-round predictions directly
                predictions = first_round_predictions
                for expert_id, pred in predictions.items():
                    expert = self.expert_pool.experts.get(expert_id)
                    if expert:
                        conf = first_round_confidences.get(expert_id, 0.5)
                        profit_score = 1.0 + max(-0.5, min(1.0, expert.metrics.simulated_profit_pct))
                        expert_scores[expert_id] = profit_score * conf * expert.confidence_multiplier
            
            # Apply softmax to normalize weights
            if expert_scores:
                weights = self._softmax_weights(expert_scores)
        else:
            # === FALLBACK: Use config-based routing ===
            if config is None:
                config = self.get_mixture_config(features)
                
            # Primary expert gets highest weight (1.0)
            primary = self.expert_pool.experts.get(config.primary_expert_id)
            if primary and primary.is_active:
                result = primary.predict(features)
                if result:
                    pred, conf = result
                    predictions[config.primary_expert_id] = pred
                    weights[config.primary_expert_id] = conf * primary.confidence_multiplier
            
            # Supporting experts
            for expert_id, support_weight in config.supporting_experts:
                expert = self.expert_pool.experts.get(expert_id)
                if expert and expert.is_active:
                    result = expert.predict(features)
                    if result:
                        pred, conf = result
                        predictions[expert_id] = pred
                        weights[expert_id] = support_weight * conf * expert.confidence_multiplier
        
        if not predictions:
            return 0.5, 0.0, {"error": "No predictions from ensemble"}
        
        # Combine predictions using weighted average with ensemble confidence
        final_pred, final_conf = self._weighted_average(predictions, weights)
        
        # For ensemble mode, enhance confidence with performance awareness
        if use_all_experts and self.expert_pool:
            expert_performances = {}
            for expert in self.expert_pool.get_active_experts():
                expert_performances[expert.expert_id] = expert.metrics.simulated_profit_pct
            
            enhanced_conf = self._calculate_ensemble_confidence(
                predictions, weights, expert_performances
            )
            final_conf = enhanced_conf
        
        # Record for learning
        mixture_id = f"mix_{self._next_mixture_id}"
        self._next_mixture_id += 1
        
        # Determine primary expert (highest weight) for history tracking
        if weights:
            primary_expert = max(weights.keys(), key=lambda k: weights[k])
            supporting = [e for e in weights.keys() if e != primary_expert]
        else:
            primary_expert = "ensemble"
            supporting = list(predictions.keys())
        
        outcome = MixtureOutcome(
            mixture_id=mixture_id,
            timestamp=datetime.utcnow(),
            market_id=features.get("market_id", "unknown"),
            primary_expert=primary_expert,
            supporting_experts=supporting,
            expert_weights=weights,
            prediction=final_pred,
            confidence=final_conf,
        )
        self.mixture_history.append(outcome)
        
        # Trim history
        if len(self.mixture_history) > 1000:
            self.mixture_history = self.mixture_history[-1000:]
        
        # Include consensus features in metadata for tracking
        consensus_info = {}
        if use_all_experts and first_round_predictions:
            consensus_info = self._compute_consensus_features(
                first_round_predictions, first_round_confidences
            )
        
        metadata = {
            "mixture_id": mixture_id,
            "ensemble_mode": use_all_experts,
            "two_stage": use_two_stage,
            "num_experts_used": len(predictions),
            "primary_expert": primary_expert,
            "supporting_experts": supporting,
            "expert_predictions": predictions,
            "expert_weights": weights,
            "first_round_predictions": first_round_predictions,
            "consensus_features": consensus_info,
            "combination_strategy": "two_stage_ensemble" if (use_all_experts and use_two_stage) else ("ensemble" if use_all_experts else "routed"),
        }
        
        return final_pred, final_conf, metadata
    
    def _weighted_average(
        self,
        predictions: Dict[str, float],
        weights: Dict[str, float],
    ) -> Tuple[float, float]:
        """Combine predictions using weighted average."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.5, 0.0
        
        weighted_pred = sum(
            predictions[e] * weights.get(e, 0) 
            for e in predictions
        ) / total_weight
        
        # Confidence is based on agreement and total weight
        preds = list(predictions.values())
        agreement = 1.0 - np.std(preds) * 2  # Lower std = higher agreement
        avg_weight = total_weight / len(predictions)
        
        confidence = min(1.0, max(0.0, agreement * avg_weight))
        
        return weighted_pred, confidence
    
    def _voting(
        self,
        predictions: Dict[str, float],
        weights: Dict[str, float],
    ) -> Tuple[float, float]:
        """Combine predictions using weighted voting."""
        votes_yes = 0.0
        votes_no = 0.0
        
        for expert, pred in predictions.items():
            weight = weights.get(expert, 1.0)
            if pred > 0.5:
                votes_yes += weight
            else:
                votes_no += weight
        
        total = votes_yes + votes_no
        if total == 0:
            return 0.5, 0.0
        
        prediction = votes_yes / total
        confidence = abs(votes_yes - votes_no) / total
        
        return prediction, confidence
    
    def _softmax_weights(
        self,
        scores: Dict[str, float],
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Apply softmax to convert scores to normalized weights.
        
        Args:
            scores: Expert ID -> raw score (higher = better)
            temperature: Controls sharpness (lower = more peaked)
            
        Returns:
            Expert ID -> normalized weight (sums to 1.0)
        """
        if not scores:
            return {}
        
        # Apply temperature scaling
        scaled = {k: v / temperature for k, v in scores.items()}
        
        # Softmax: exp(x) / sum(exp(x))
        max_score = max(scaled.values())  # For numerical stability
        exp_scores = {k: np.exp(v - max_score) for k, v in scaled.items()}
        total = sum(exp_scores.values())
        
        if total == 0:
            # Uniform weights if all zeros
            n = len(scores)
            return {k: 1.0 / n for k in scores}
        
        return {k: v / total for k, v in exp_scores.items()}
    
    def _compute_consensus_features(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute consensus features from first-round expert predictions.
        
        These features enable cross-expert awareness - each expert can see
        what the collective is predicting and adjust accordingly.
        
        Args:
            predictions: Expert ID -> prediction value (0-1)
            confidences: Expert ID -> confidence value (0-1)
            
        Returns:
            Dict of consensus feature name -> value
        """
        if not predictions:
            return {
                "expert_consensus_mean": 0.5,
                "expert_consensus_std": 0.0,
                "expert_agreement_score": 0.5,
                "expert_bullish_ratio": 0.5,
                "top_3_consensus": 0.5,
                "expert_confidence_mean": 0.5,
                "expert_count": 0,
            }
        
        preds = list(predictions.values())
        confs = list(confidences.values())
        
        # Basic statistics
        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds)) if len(preds) > 1 else 0.0
        
        # Agreement score: 1 = all agree, 0 = maximum disagreement
        # Scaled so 0.5 std = 0 agreement
        agreement_score = max(0.0, min(1.0, 1.0 - std_pred * 2))
        
        # Bullish ratio: % of experts predicting > 0.5
        bullish_count = sum(1 for p in preds if p > 0.5)
        bullish_ratio = bullish_count / len(preds)
        
        # Top 3 by confidence consensus
        if len(confidences) >= 3:
            top_3_ids = sorted(confidences.keys(), key=lambda k: confidences[k], reverse=True)[:3]
            top_3_preds = [predictions[eid] for eid in top_3_ids]
            top_3_consensus = float(np.mean(top_3_preds))
        else:
            top_3_consensus = mean_pred
        
        # Mean confidence
        mean_conf = float(np.mean(confs)) if confs else 0.5
        
        return {
            "expert_consensus_mean": mean_pred,
            "expert_consensus_std": std_pred,
            "expert_agreement_score": agreement_score,
            "expert_bullish_ratio": bullish_ratio,
            "top_3_consensus": top_3_consensus,
            "expert_confidence_mean": mean_conf,
            "expert_count": len(predictions),
        }
    
    def _calculate_ensemble_confidence(
        self,
        predictions: Dict[str, float],
        weights: Dict[str, float],
        expert_performances: Dict[str, float],
    ) -> float:
        """Calculate ensemble confidence based on agreement and performance.
        
        Confidence formula:
        - Base: agreement between experts (1 - std of predictions)
        - Boost: if majority of high-performing experts agree
        - Penalty: if expert with best recent performance disagrees
        
        Args:
            predictions: Expert ID -> prediction value
            weights: Expert ID -> weight
            expert_performances: Expert ID -> recent profit (for boost/penalty)
            
        Returns:
            Confidence score 0 to 1
        """
        if not predictions:
            return 0.0
        
        preds = list(predictions.values())
        
        # Base confidence: agreement (lower std = higher agreement)
        std = np.std(preds) if len(preds) > 1 else 0
        agreement = max(0, 1.0 - std * 2)
        
        # Find weighted average prediction
        total_weight = sum(weights.values())
        if total_weight == 0:
            return agreement * 0.5
        
        weighted_pred = sum(predictions[e] * weights[e] for e in predictions) / total_weight
        
        # Boost: if best performers agree with ensemble prediction
        if expert_performances:
            top_performers = sorted(
                expert_performances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 performers
            
            agreeing = 0
            for expert_id, _ in top_performers:
                if expert_id in predictions:
                    expert_pred = predictions[expert_id]
                    # Check if prediction is in same direction
                    if (expert_pred > 0.5 and weighted_pred > 0.5) or \
                       (expert_pred < 0.5 and weighted_pred < 0.5):
                        agreeing += 1
            
            # Boost if majority of top performers agree
            boost = 1.0 + 0.15 * (agreeing / len(top_performers))
        else:
            boost = 1.0
        
        confidence = min(1.0, agreement * boost)
        return confidence
    
    def record_trade_outcome(
        self,
        mixture_id: str,
        profit_pct: float,
        category: str,
    ) -> None:
        """Record the outcome of a trade for learning."""
        # Find the mixture outcome
        for outcome in self.mixture_history:
            if outcome.mixture_id == mixture_id:
                outcome.profit_pct = profit_pct
                outcome.actual_outcome = 1.0 if profit_pct > 0 else 0.0
                
                # Update learner
                self.learner.record_outcome(
                    primary=outcome.primary_expert,
                    supporters=outcome.supporting_experts,
                    predictions={},  # We don't have individual predictions stored
                    profit=profit_pct,
                    category=category,
                )
                break
        
        # Save updated state
        self.learner.save_state()
        self._save_history()
    
    def get_status(self) -> Dict[str, Any]:
        """Get Meta-Controller status for dashboard."""
        # Calculate overall mixture performance
        resolved = [o for o in self.mixture_history if o.profit_pct is not None]
        
        total_profit = sum(o.profit_pct for o in resolved) if resolved else 0
        win_rate = sum(1 for o in resolved if o.profit_pct > 0) / len(resolved) if resolved else 0
        
        # Get most used mixtures
        mixture_counts: Dict[str, int] = {}
        for outcome in self.mixture_history[-100:]:
            key = f"{outcome.primary_expert}+{','.join(outcome.supporting_experts)}"
            mixture_counts[key] = mixture_counts.get(key, 0) + 1
        
        top_mixtures = sorted(mixture_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Get synergy insights
        synergy_insights = []
        for pair_key, perf in self.learner.pair_performance.items():
            if perf.combined_trades >= MIN_TRADES_FOR_LEARNING:
                synergy_insights.append({
                    "pair": pair_key,
                    "trades": perf.combined_trades,
                    "synergy": perf.synergy_score,
                    "agreement": perf.agreement_rate,
                })
        
        synergy_insights.sort(key=lambda x: x["synergy"], reverse=True)
        
        return {
            "total_mixtures": len(self.mixture_history),
            "resolved_mixtures": len(resolved),
            "total_profit_pct": total_profit,
            "win_rate": win_rate,
            "top_mixtures": top_mixtures,
            "synergy_insights": synergy_insights[:10],
            "learned_weights": self.learner.expert_weights,
            "category_performance": self.learner.category_performance,
        }
    
    def save(self) -> None:
        """Save all state."""
        self.learner.save_state()
        self._save_history()


# Singleton instance
_meta_controller: Optional[MetaController] = None


def get_meta_controller(expert_pool: Any = None) -> MetaController:
    """Get or create the singleton MetaController."""
    global _meta_controller
    if _meta_controller is None:
        _meta_controller = MetaController(expert_pool)
    elif expert_pool is not None and _meta_controller.expert_pool is None:
        _meta_controller.expert_pool = expert_pool
    return _meta_controller
