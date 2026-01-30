"""Expert Pool manager for MoE architecture.

Manages the lifecycle of all 24 experts: creation, training, state management, and persistence.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from polyb0t.ml.moe.expert import Expert, ExpertMetrics
from polyb0t.ml.moe.gating import GatingNetwork
from polyb0t.ml.moe.versioning import ExpertState

logger = logging.getLogger(__name__)


# ============================================================================
# 24 EXPERT CONFIGURATIONS
# ============================================================================

# Category Experts (10)
CATEGORY_EXPERTS = [
    ("sports", "category", "sports"),
    ("politics_us", "category", "politics_us"),
    ("politics_intl", "category", "politics_intl"),
    ("crypto", "category", "crypto"),
    ("economics", "category", "economics"),
    ("entertainment", "category", "entertainment"),
    ("tech", "category", "tech"),
    ("weather", "category", "weather"),
    ("science", "category", "science"),
    ("legal", "category", "legal"),
]

# Risk Experts (3)
RISK_EXPERTS = [
    ("low_risk", "risk", "low_risk"),        # price 85-100% or 0-15%
    ("medium_risk", "risk", "medium_risk"),  # price 15-35% or 65-85%
    ("high_risk", "risk", "high_risk"),      # price 35-65%
]

# Time Horizon Experts (3)
TIME_EXPERTS = [
    ("short_term", "time", "short_term"),    # < 3 days
    ("medium_term", "time", "medium_term"),  # 3-14 days
    ("long_term", "time", "long_term"),      # > 14 days
]

# Volume/Liquidity Experts (3)
VOLUME_EXPERTS = [
    ("high_volume", "volume", "high_volume"),      # top 20% by volume
    ("low_volume", "volume", "low_volume"),        # bottom 20% by volume
    ("high_liquidity", "volume", "high_liquidity"),  # top 20% by liquidity
]

# Volatility Experts (3)
VOLATILITY_EXPERTS = [
    ("high_volatility", "volatility", "high_volatility"),    # volatility_24h > 5%
    ("low_volatility", "volatility", "low_volatility"),      # volatility_24h < 1%
    ("momentum_strong", "volatility", "momentum_strong"),    # momentum_24h > 10%
]

# Timing Experts (2)
TIMING_EXPERTS = [
    ("weekend_trader", "timing", "weekend_trader"),  # is_weekend = True
    ("market_close", "timing", "market_close"),      # days_to_resolution < 1
]

# Prediction Horizon Experts (3) - different prediction timeframes
HORIZON_EXPERTS = [
    ("horizon_1h", "horizon", "1h"),    # Predict 1-hour price changes
    ("horizon_4h", "horizon", "4h"),    # Predict 4-hour price changes
    ("horizon_24h", "horizon", "24h"),  # Predict 24-hour price changes
]

# News/Sentiment Experts (3) - markets with news activity
NEWS_EXPERTS = [
    ("news_driven", "news", "news_driven"),      # Markets with recent news articles
    ("sentiment_positive", "news", "positive"),   # Positive sentiment signals
    ("sentiment_negative", "news", "negative"),   # Negative sentiment signals
]

# Smart Money Experts (2) - follow smart wallet activity
SMART_MONEY_EXPERTS = [
    ("smart_accumulation", "smart_money", "accumulation"),  # Smart wallets buying
    ("smart_distribution", "smart_money", "distribution"),  # Smart wallets selling
]

# All default experts (32 total)
ALL_DEFAULT_EXPERTS = (
    CATEGORY_EXPERTS +
    RISK_EXPERTS +
    TIME_EXPERTS +
    VOLUME_EXPERTS +
    VOLATILITY_EXPERTS +
    TIMING_EXPERTS +
    HORIZON_EXPERTS +
    NEWS_EXPERTS +
    SMART_MONEY_EXPERTS
)


@dataclass
class ExpertPoolState:
    """State of the expert pool."""
    
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_training: Optional[datetime] = None
    total_training_cycles: int = 0
    
    # Overall metrics
    total_profit_pct: float = 0.0
    total_trades: int = 0
    best_expert_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "total_training_cycles": self.total_training_cycles,
            "total_profit_pct": self.total_profit_pct,
            "total_trades": self.total_trades,
            "best_expert_id": self.best_expert_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertPoolState":
        return cls(
            version=data.get("version", 1),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            last_training=datetime.fromisoformat(data["last_training"]) if data.get("last_training") else None,
            total_training_cycles=data.get("total_training_cycles", 0),
            total_profit_pct=data.get("total_profit_pct", 0.0),
            total_trades=data.get("total_trades", 0),
            best_expert_id=data.get("best_expert_id"),
        )


class ExpertPool:
    """Manages all experts and the gating network.
    
    Handles:
    - Creating default experts
    - Training all experts
    - Deprecating failing experts
    - Creating new dynamic experts
    - Routing predictions through gating network
    """
    
    def __init__(self, model_dir: str = "data/moe_models"):
        self.model_dir = model_dir
        self.experts_dir = os.path.join(model_dir, "experts")
        
        os.makedirs(self.experts_dir, exist_ok=True)
        
        self.experts: Dict[str, Expert] = {}
        self.gating: Optional[GatingNetwork] = None
        self.state = ExpertPoolState()
        
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing pool or create default experts."""
        state_path = os.path.join(self.model_dir, "pool_state.json")
        
        if os.path.exists(state_path):
            self._load_state()
        else:
            self._create_default_experts()
    
    def _create_default_experts(self):
        """Create the default set of 24 experts."""
        logger.info("Creating default expert pool with 24 experts...")
        
        # Create all 24 default experts
        for expert_id, expert_type, domain in ALL_DEFAULT_EXPERTS:
            self.experts[expert_id] = Expert(
                expert_id=expert_id,
                expert_type=expert_type,
                domain=domain,
            )
        
        # Create gating network
        self.gating = GatingNetwork(list(self.experts.keys()))
        
        logger.info(f"Created {len(self.experts)} default experts")
        self._save_state()
    
    def _load_state(self):
        """Load pool state and all experts."""
        state_path = os.path.join(self.model_dir, "pool_state.json")
        
        try:
            with open(state_path, "r") as f:
                data = json.load(f)
            
            self.state = ExpertPoolState.from_dict(data.get("state", {}))
            
            # Load experts
            expert_ids = data.get("expert_ids", [])
            for expert_id in expert_ids:
                expert_path = os.path.join(self.experts_dir, expert_id)
                expert = Expert.load(expert_path)
                if expert:
                    self.experts[expert_id] = expert
                else:
                    logger.warning(f"Failed to load expert: {expert_id}")
            
            # Load gating network
            gating_path = os.path.join(self.model_dir, "gating")
            self.gating = GatingNetwork.load(gating_path)
            if self.gating is None:
                self.gating = GatingNetwork(list(self.experts.keys()))
            
            logger.info(f"Loaded expert pool: {len(self.experts)} experts")
            
        except Exception as e:
            logger.error(f"Failed to load pool state: {e}")
            self._create_default_experts()
    
    def _save_state(self):
        """Save pool state and all experts."""
        state_path = os.path.join(self.model_dir, "pool_state.json")
        
        try:
            data = {
                "state": self.state.to_dict(),
                "expert_ids": list(self.experts.keys()),
            }
            
            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)
            
            # Save all experts
            for expert_id, expert in self.experts.items():
                expert_path = os.path.join(self.experts_dir, expert_id)
                expert.save(expert_path)
            
            # Save gating network
            if self.gating:
                gating_path = os.path.join(self.model_dir, "gating")
                self.gating.save(gating_path)
                
        except Exception as e:
            logger.error(f"Failed to save pool state: {e}")
    
    def get_expert(self, expert_id: str) -> Optional[Expert]:
        """Get an expert by ID."""
        return self.experts.get(expert_id)
    
    def get_active_experts(self) -> List[Expert]:
        """Get all active (trading-enabled) experts."""
        return [e for e in self.experts.values() if e.is_active and not e.is_deprecated]

    def get_trainable_experts(self) -> List[Expert]:
        """Get all non-deprecated experts that have trained models.

        This includes ACTIVE, SUSPENDED, and PROBATION experts - any expert
        that has been trained and could potentially make predictions.
        Used in dry-run/evaluation mode to gather performance data from all experts.
        """
        return [
            e for e in self.experts.values()
            if not e.is_deprecated and e._model is not None
        ]

    def get_deprecated_experts(self) -> List[Expert]:
        """Get all deprecated experts."""
        return [e for e in self.experts.values() if e.is_deprecated]
    
    def get_experts_for_market(
        self,
        features: Dict[str, Any],
        include_inactive: bool = False,
    ) -> List[Tuple[Expert, float]]:
        """Get relevant experts for a market with their weights.

        Args:
            features: Market feature dict
            include_inactive: If True, include SUSPENDED/PROBATION experts
                (for dry-run/evaluation mode to gather performance data)

        Returns:
            List of (expert, weight) tuples sorted by weight descending
        """
        # Select expert pool based on mode
        experts = self.get_trainable_experts() if include_inactive else self.get_active_experts()

        if not experts:
            return []

        if self.gating is None:
            return [(e, 1.0 / len(experts)) for e in experts]

        weights = self.gating.get_weights(features)

        result = []
        for expert in experts:
            weight = weights.get(expert.expert_id, 0.0)
            if weight > 0.01:  # Ignore very low weights
                result.append((expert, weight))

        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def predict(
        self,
        features: Dict[str, Any],
        include_inactive: bool = False,
    ) -> Optional[Tuple[float, float, str]]:
        """Make a weighted prediction using experts.

        Args:
            features: Market feature dict
            include_inactive: If True, include SUSPENDED/PROBATION experts
                (for dry-run/evaluation mode)

        Returns:
            Tuple of (prediction, confidence, best_expert_id) or None
        """
        expert_weights = self.get_experts_for_market(features, include_inactive=include_inactive)

        if not expert_weights:
            return None

        weighted_pred = 0.0
        weighted_conf = 0.0
        total_weight = 0.0
        best_expert_id = None
        best_weight = 0.0

        for expert, weight in expert_weights:
            result = expert.predict(features)
            if result is None:
                continue

            pred, conf = result
            weighted_pred += pred * weight * conf
            weighted_conf += conf * weight
            total_weight += weight

            if weight > best_weight:
                best_weight = weight
                best_expert_id = expert.expert_id

        if total_weight == 0:
            return None

        final_pred = weighted_pred / total_weight
        final_conf = weighted_conf / total_weight

        return (final_pred, final_conf, best_expert_id)
    
    def predict_with_mixture(
        self,
        features: Dict[str, Any],
        mixture_config: Optional[Any] = None,
        include_inactive: bool = False,
    ) -> Optional[Tuple[float, float, Dict[str, Any]]]:
        """Make a prediction using expert mixture from MetaController.

        Args:
            features: Market features
            mixture_config: Optional MixtureConfig from MetaController
            include_inactive: If True, include SUSPENDED/PROBATION experts
                (for dry-run/evaluation mode)

        Returns:
            Tuple of (prediction, confidence, metadata) or None
        """
        from polyb0t.ml.moe.meta_controller import get_meta_controller

        meta = get_meta_controller(self)
        return meta.predict_with_mixture(features, mixture_config, include_inactive=include_inactive)
    
    def add_expert(self, expert: Expert) -> bool:
        """Add a new expert to the pool.
        
        Args:
            expert: Expert to add
        
        Returns:
            True if added successfully
        """
        if expert.expert_id in self.experts:
            logger.warning(f"Expert {expert.expert_id} already exists")
            return False
        
        self.experts[expert.expert_id] = expert
        
        # Update gating network
        if self.gating:
            self.gating.update_expert_ids(list(self.experts.keys()))
        
        logger.info(f"Added expert: {expert.expert_id} ({expert.domain})")
        self._save_state()
        return True
    
    def deprecate_expert(self, expert_id: str, reason: str = "") -> bool:
        """Deprecate an expert.
        
        Args:
            expert_id: ID of expert to deprecate
            reason: Reason for deprecation
        
        Returns:
            True if deprecated successfully
        """
        if expert_id not in self.experts:
            return False
        
        expert = self.experts[expert_id]
        expert.deprecate(reason)
        
        # Update gating network
        if self.gating:
            active_ids = [e.expert_id for e in self.get_active_experts()]
            self.gating.update_expert_ids(active_ids)
        
        self._save_state()
        return True
    
    def get_category_for_market(self, features: Dict[str, Any]) -> str:
        """Get the category for a market."""
        return features.get("category", "other")
    
    def get_risk_level(self, features: Dict[str, Any]) -> str:
        """Get the risk level for a market based on price."""
        price = features.get("price", 0.5)
        
        if price > 0.85 or price < 0.15:
            return "low_risk"
        elif 0.35 < price < 0.65:
            return "high_risk"
        else:
            return "medium_risk"
    
    def get_time_horizon(self, features: Dict[str, Any]) -> str:
        """Get the time horizon for a market."""
        days = features.get("days_to_resolution", 30)
        
        if days < 3:
            return "short_term"
        elif days > 14:
            return "long_term"
        else:
            return "medium_term"
    
    def get_volume_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Get volume/liquidity level for a market."""
        volume = features.get("volume_24h", 0)
        liquidity = features.get("liquidity", 0)
        
        # These thresholds will be adjusted based on actual data distribution
        if volume > 10000:
            return "high_volume"
        elif volume < 1000:
            return "low_volume"
        elif liquidity > 50000:
            return "high_liquidity"
        return None
    
    def get_volatility_level(self, features: Dict[str, Any]) -> Optional[str]:
        """Get volatility level for a market."""
        volatility = features.get("volatility_24h", 0)
        momentum = abs(features.get("momentum_24h", 0))
        
        if volatility > 0.05:
            return "high_volatility"
        elif volatility < 0.01:
            return "low_volatility"
        elif momentum > 0.10:
            return "momentum_strong"
        return None
    
    def get_timing_flag(self, features: Dict[str, Any]) -> Optional[str]:
        """Get timing flag for a market."""
        is_weekend = features.get("is_weekend", False)
        days_to_resolution = features.get("days_to_resolution", 30)
        
        if is_weekend:
            return "weekend_trader"
        elif days_to_resolution < 1:
            return "market_close"
        return None
    
    def filter_data_for_expert(
        self,
        expert: Expert,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter training data relevant to an expert's domain.
        
        Args:
            expert: Expert to filter for
            data: Full training data
        
        Returns:
            Filtered data relevant to this expert
        """
        filtered = []
        
        for sample in data:
            features = sample.get("features", sample)
            if isinstance(features, str):
                import json as json_module
                try:
                    features = json_module.loads(features)
                except (json_module.JSONDecodeError, TypeError):
                    features = {}
            
            match = False
            
            if expert.expert_type == "category":
                category = self.get_category_for_market(features)
                # 'other' expert gets anything not matching the main categories
                if expert.domain == "other":
                    match = category not in [
                        "sports", "politics_us", "politics_intl", "crypto",
                        "economics", "entertainment", "tech", "weather",
                        "science", "legal"
                    ]
                else:
                    match = (category == expert.domain)
                    
            elif expert.expert_type == "risk":
                risk_level = self.get_risk_level(features)
                match = (risk_level == expert.domain)
                    
            elif expert.expert_type == "time":
                time_horizon = self.get_time_horizon(features)
                match = (time_horizon == expert.domain)
                    
            elif expert.expert_type == "volume":
                volume_level = self.get_volume_level(features)
                match = (volume_level == expert.domain)
                    
            elif expert.expert_type == "volatility":
                volatility_level = self.get_volatility_level(features)
                match = (volatility_level == expert.domain)
                    
            elif expert.expert_type == "timing":
                timing_flag = self.get_timing_flag(features)
                match = (timing_flag == expert.domain)
                    
            elif expert.expert_type == "dynamic":
                # Dynamic experts get all data for now
                match = True

            elif expert.expert_type == "news":
                # News-based experts filter by sentiment features
                news_count = features.get("news_article_count", 0)
                sentiment = features.get("news_sentiment_score", 0)

                if expert.domain == "news_driven":
                    # Any market with recent news activity
                    match = news_count > 0
                elif expert.domain == "positive":
                    # Markets with positive sentiment
                    match = sentiment > 0.3
                elif expert.domain == "negative":
                    # Markets with negative sentiment
                    match = sentiment < -0.3

            elif expert.expert_type == "smart_money":
                # Smart money experts filter by insider tracking features
                direction_24h = features.get("smart_wallet_net_direction_24h", 0)
                buy_count = features.get("smart_wallet_buy_count_24h", 0)
                sell_count = features.get("smart_wallet_sell_count_24h", 0)
                total_activity = buy_count + sell_count

                if expert.domain == "accumulation":
                    # Smart wallets accumulating (net buying)
                    match = total_activity > 0 and direction_24h > 0.3
                elif expert.domain == "distribution":
                    # Smart wallets distributing (net selling)
                    match = total_activity > 0 and direction_24h < -0.3

            if match:
                filtered.append(sample)
        
        return filtered
    
    def get_top_experts(self, n: int = 5) -> List[Expert]:
        """Get the top N performing experts by profitability."""
        active = self.get_active_experts()
        active.sort(key=lambda e: e.metrics.simulated_profit_pct, reverse=True)
        return active[:n]
    
    def get_suspended_experts(self) -> List[Expert]:
        """Get all suspended experts (still training, not trading)."""
        return [e for e in self.experts.values() if e.state == ExpertState.SUSPENDED]
    
    def get_probation_experts(self) -> List[Expert]:
        """Get all experts in probation."""
        return [e for e in self.experts.values() if e.state == ExpertState.PROBATION]
    
    def get_experts_by_state(self) -> Dict[str, List[Expert]]:
        """Get experts grouped by state."""
        result = {state.value: [] for state in ExpertState}
        for expert in self.experts.values():
            result[expert.state.value].append(expert)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics for dashboard."""
        active = self.get_active_experts()
        deprecated = self.get_deprecated_experts()
        suspended = self.get_suspended_experts()
        probation = self.get_probation_experts()
        
        # Calculate overall profit from active experts
        total_profit = sum(e.metrics.simulated_profit_pct for e in active)
        total_trades = sum(e.metrics.simulated_num_trades for e in active)
        
        # Find best expert
        best_expert = None
        best_profit = float('-inf')
        for e in active:
            if e.metrics.simulated_profit_pct > best_profit:
                best_profit = e.metrics.simulated_profit_pct
                best_expert = e
        
        # Count by state
        state_counts = {}
        for state in ExpertState:
            state_counts[state.value] = sum(1 for e in self.experts.values() if e.state == state)
        
        return {
            "total_experts": len(self.experts),
            "active_experts": len(active),
            "suspended_experts": len(suspended),
            "probation_experts": len(probation),
            "deprecated_experts": len(deprecated),
            "state_counts": state_counts,
            "total_profit_pct": total_profit,
            "total_trades": total_trades,
            "best_expert": best_expert.to_dict() if best_expert else None,
            "gating": self.gating.to_dict() if self.gating else None,
            "state": self.state.to_dict(),
            "experts": [e.to_dict() for e in active],
            "suspended": [e.to_dict() for e in suspended],
            "probation": [e.to_dict() for e in probation],
            "deprecated": [e.to_dict() for e in deprecated],
        }
    
    def update_state_after_training(self):
        """Update pool state after a training cycle."""
        self.state.total_training_cycles += 1
        self.state.last_training = datetime.utcnow()
        
        # Update aggregated metrics
        active = self.get_active_experts()
        self.state.total_profit_pct = sum(e.metrics.simulated_profit_pct for e in active)
        self.state.total_trades = sum(e.metrics.simulated_num_trades for e in active)
        
        # Find best expert
        best = None
        best_profit = float('-inf')
        for e in active:
            if e.metrics.simulated_profit_pct > best_profit:
                best_profit = e.metrics.simulated_profit_pct
                best = e
        
        if best:
            self.state.best_expert_id = best.expert_id
        
        self._save_state()
    
    def save(self):
        """Explicitly save the pool state."""
        self._save_state()


# Singleton instance
_expert_pool: Optional[ExpertPool] = None


def get_expert_pool() -> ExpertPool:
    """Get or create the singleton expert pool."""
    global _expert_pool
    if _expert_pool is None:
        _expert_pool = ExpertPool()
    return _expert_pool
