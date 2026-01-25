"""Expert Pool manager for MoE architecture.

Manages the lifecycle of all experts: creation, training, deprecation, and persistence.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from polyb0t.ml.moe.expert import Expert, ExpertMetrics
from polyb0t.ml.moe.gating import GatingNetwork

logger = logging.getLogger(__name__)


# Default expert configurations
CATEGORY_EXPERTS = [
    ("sports", "category", "sports"),
    ("politics_us", "category", "politics_us"),
    ("politics_intl", "category", "politics_intl"),
    ("crypto", "category", "crypto"),
    ("economics", "category", "economics"),
    ("entertainment", "category", "entertainment"),
    ("other", "category", "other"),
]

RISK_EXPERTS = [
    ("low_risk", "risk", "low_risk"),
    ("medium_risk", "risk", "medium_risk"),
    ("high_risk", "risk", "high_risk"),
]

TIME_EXPERTS = [
    ("short_term", "time", "short_term"),
    ("long_term", "time", "long_term"),
]


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
        """Create the default set of experts."""
        logger.info("Creating default expert pool...")
        
        # Category experts
        for expert_id, expert_type, domain in CATEGORY_EXPERTS:
            self.experts[expert_id] = Expert(
                expert_id=expert_id,
                expert_type=expert_type,
                domain=domain,
            )
        
        # Risk experts
        for expert_id, expert_type, domain in RISK_EXPERTS:
            self.experts[expert_id] = Expert(
                expert_id=expert_id,
                expert_type=expert_type,
                domain=domain,
            )
        
        # Time horizon experts
        for expert_id, expert_type, domain in TIME_EXPERTS:
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
        """Get all active (non-deprecated) experts."""
        return [e for e in self.experts.values() if e.is_active and not e.is_deprecated]
    
    def get_deprecated_experts(self) -> List[Expert]:
        """Get all deprecated experts."""
        return [e for e in self.experts.values() if e.is_deprecated]
    
    def get_experts_for_market(self, features: Dict[str, Any]) -> List[Tuple[Expert, float]]:
        """Get relevant experts for a market with their weights.
        
        Args:
            features: Market feature dict
        
        Returns:
            List of (expert, weight) tuples sorted by weight descending
        """
        if self.gating is None:
            return [(e, 1.0 / len(self.experts)) for e in self.get_active_experts()]
        
        weights = self.gating.get_weights(features)
        
        result = []
        for expert in self.get_active_experts():
            weight = weights.get(expert.expert_id, 0.0)
            if weight > 0.01:  # Ignore very low weights
                result.append((expert, weight))
        
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def predict(self, features: Dict[str, Any]) -> Optional[Tuple[float, float, str]]:
        """Make a weighted prediction using all experts.
        
        Args:
            features: Market feature dict
        
        Returns:
            Tuple of (prediction, confidence, best_expert_id) or None
        """
        expert_weights = self.get_experts_for_market(features)
        
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
        
        if days < 7:
            return "short_term"
        elif days > 30:
            return "long_term"
        else:
            return "medium_term"
    
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
            
            if expert.expert_type == "category":
                # Category experts get data from their category
                category = self.get_category_for_market(features)
                if category == expert.domain or expert.domain == "other":
                    filtered.append(sample)
                    
            elif expert.expert_type == "risk":
                # Risk experts get data matching their risk level
                risk_level = self.get_risk_level(features)
                if risk_level == expert.domain:
                    filtered.append(sample)
                    
            elif expert.expert_type == "time":
                # Time experts get data matching their time horizon
                time_horizon = self.get_time_horizon(features)
                if time_horizon == expert.domain:
                    filtered.append(sample)
                    
            elif expert.expert_type == "dynamic":
                # Dynamic experts have custom filters (stored in domain)
                # For now, give them a sample of all data
                filtered.append(sample)
        
        return filtered
    
    def get_top_experts(self, n: int = 5) -> List[Expert]:
        """Get the top N performing experts by profitability."""
        active = self.get_active_experts()
        active.sort(key=lambda e: e.metrics.simulated_profit_pct, reverse=True)
        return active[:n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics for dashboard."""
        active = self.get_active_experts()
        deprecated = self.get_deprecated_experts()
        
        # Calculate overall profit
        total_profit = sum(e.metrics.simulated_profit_pct for e in active)
        total_trades = sum(e.metrics.simulated_num_trades for e in active)
        
        # Find best expert
        best_expert = None
        best_profit = float('-inf')
        for e in active:
            if e.metrics.simulated_profit_pct > best_profit:
                best_profit = e.metrics.simulated_profit_pct
                best_expert = e
        
        return {
            "total_experts": len(self.experts),
            "active_experts": len(active),
            "deprecated_experts": len(deprecated),
            "total_profit_pct": total_profit,
            "total_trades": total_trades,
            "best_expert": best_expert.to_dict() if best_expert else None,
            "gating": self.gating.to_dict() if self.gating else None,
            "state": self.state.to_dict(),
            "experts": [e.to_dict() for e in active],
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
