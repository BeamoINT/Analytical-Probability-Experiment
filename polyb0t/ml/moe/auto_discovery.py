"""Auto-discovery module for spawning new experts.

This module analyzes trading data to find profitable market subsets
that aren't well-served by existing experts, and spawns new dynamic
experts to capture those opportunities.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import numpy as np

from polyb0t.ml.moe.expert import Expert, SPREAD_COST

logger = logging.getLogger(__name__)

# Thresholds for expert creation
MIN_CLUSTER_SIZE = 50  # Minimum samples to justify new expert
MIN_PROFIT_POTENTIAL = 0.05  # 5% minimum profit potential
MIN_PROFIT_IMPROVEMENT = 0.03  # 3% improvement over existing experts


@dataclass
class ExpertCandidate:
    """A candidate for a new expert."""
    
    name: str
    description: str
    filter_type: str  # 'keyword', 'combined', 'pattern'
    filter_value: Any
    n_samples: int
    potential_profit: float
    current_best_profit: float  # Profit from best existing expert
    improvement: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "filter_type": self.filter_type,
            "n_samples": self.n_samples,
            "potential_profit": self.potential_profit,
            "current_best_profit": self.current_best_profit,
            "improvement": self.improvement,
        }


class AutoDiscovery:
    """Discovers new expert opportunities from trading data.
    
    Analyzes:
    1. Keyword patterns in market titles
    2. Time-based patterns (weekend, specific hours)
    3. Combined category + condition patterns
    """
    
    def __init__(self, pool: Any):  # ExpertPool type hint avoided for circular import
        self.pool = pool
        self.candidates: List[ExpertCandidate] = []
        self._created_experts: Set[str] = set()  # Track created expert names
    
    def discover(
        self,
        data: List[Dict[str, Any]],
        X: np.ndarray,
        y_binary: np.ndarray,
        y_reg: np.ndarray,
        categories: List[str],
    ) -> List[Expert]:
        """Discover and create new experts from profitable patterns.
        
        Args:
            data: Raw training data
            X: Feature matrix
            y_binary: Binary labels
            y_reg: Regression labels (actual price changes)
            categories: Category for each sample
        
        Returns:
            List of newly created experts
        """
        logger.info("Running auto-discovery for new experts...")
        
        new_experts = []
        self.candidates = []
        
        # 1. Find keyword-based patterns
        keyword_candidates = self._find_keyword_patterns(data, y_reg)
        self.candidates.extend(keyword_candidates)
        
        # 2. Find time-based patterns
        time_candidates = self._find_time_patterns(data, y_reg)
        self.candidates.extend(time_candidates)
        
        # 3. Find combined patterns (category + condition)
        combined_candidates = self._find_combined_patterns(data, y_reg, categories)
        self.candidates.extend(combined_candidates)
        
        # Sort by improvement over existing experts
        self.candidates.sort(key=lambda c: c.improvement, reverse=True)
        
        # Log candidates
        if self.candidates:
            logger.info(f"Found {len(self.candidates)} expert candidates:")
            for c in self.candidates[:5]:
                logger.info(
                    f"  - {c.name}: {c.n_samples} samples, "
                    f"potential={c.potential_profit:+.1%}, "
                    f"improvement={c.improvement:+.1%}"
                )
        
        # Create top candidates that pass thresholds
        for candidate in self.candidates[:3]:  # Max 3 new experts per cycle
            if self._should_create_expert(candidate):
                expert = self._create_expert_from_candidate(candidate)
                if expert:
                    new_experts.append(expert)
        
        return new_experts
    
    def _find_keyword_patterns(
        self,
        data: List[Dict[str, Any]],
        y_reg: np.ndarray,
    ) -> List[ExpertCandidate]:
        """Find profitable keyword patterns in market titles."""
        candidates = []
        
        # Extract keywords and track profitability
        keyword_profits = defaultdict(list)
        
        for i, sample in enumerate(data):
            if i >= len(y_reg):
                break
            
            features = sample.get("features", sample)
            if isinstance(features, str):
                import json
                try:
                    features = json.loads(features)
                except:
                    features = {}
            
            title = features.get("market_title", sample.get("market_title", ""))
            if not title:
                continue
            
            # Extract keywords (simple word-based)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
            profit = y_reg[i] - SPREAD_COST
            
            for word in set(words):  # Unique words only
                keyword_profits[word].append(profit)
        
        # Find profitable keywords
        for keyword, profits in keyword_profits.items():
            if len(profits) < MIN_CLUSTER_SIZE:
                continue
            
            avg_profit = np.mean(profits)
            if avg_profit < MIN_PROFIT_POTENTIAL:
                continue
            
            # Skip common words
            if keyword in ['will', 'the', 'this', 'that', 'with', 'from', 'have', 'been']:
                continue
            
            # Skip if we already have an expert for this
            if f"keyword_{keyword}" in self._created_experts:
                continue
            
            candidates.append(ExpertCandidate(
                name=f"keyword_{keyword}",
                description=f"Markets containing '{keyword}'",
                filter_type="keyword",
                filter_value=keyword,
                n_samples=len(profits),
                potential_profit=avg_profit,
                current_best_profit=0.0,  # Will be updated
                improvement=avg_profit,
            ))
        
        return candidates
    
    def _find_time_patterns(
        self,
        data: List[Dict[str, Any]],
        y_reg: np.ndarray,
    ) -> List[ExpertCandidate]:
        """Find profitable time-based patterns."""
        candidates = []
        
        # Track profits by time patterns
        weekend_profits = []
        weekday_profits = []
        night_profits = []  # 10pm - 6am
        morning_profits = []  # 6am - 12pm
        
        for i, sample in enumerate(data):
            if i >= len(y_reg):
                break
            
            features = sample.get("features", sample)
            if isinstance(features, str):
                import json
                try:
                    features = json.loads(features)
                except:
                    features = {}
            
            profit = y_reg[i] - SPREAD_COST
            is_weekend = features.get("is_weekend", 0)
            hour = features.get("hour_of_day", 12)
            
            if is_weekend:
                weekend_profits.append(profit)
            else:
                weekday_profits.append(profit)
            
            if 22 <= hour or hour < 6:
                night_profits.append(profit)
            elif 6 <= hour < 12:
                morning_profits.append(profit)
        
        # Check weekend pattern
        if len(weekend_profits) >= MIN_CLUSTER_SIZE:
            avg_weekend = np.mean(weekend_profits)
            avg_weekday = np.mean(weekday_profits) if weekday_profits else 0
            if avg_weekend > MIN_PROFIT_POTENTIAL and avg_weekend > avg_weekday + MIN_PROFIT_IMPROVEMENT:
                if "time_weekend" not in self._created_experts:
                    candidates.append(ExpertCandidate(
                        name="time_weekend",
                        description="Weekend trading",
                        filter_type="time",
                        filter_value="weekend",
                        n_samples=len(weekend_profits),
                        potential_profit=avg_weekend,
                        current_best_profit=avg_weekday,
                        improvement=avg_weekend - avg_weekday,
                    ))
        
        # Check night pattern
        if len(night_profits) >= MIN_CLUSTER_SIZE:
            avg_night = np.mean(night_profits)
            overall_avg = np.mean(y_reg) - SPREAD_COST if len(y_reg) > 0 else 0
            if avg_night > MIN_PROFIT_POTENTIAL and avg_night > overall_avg + MIN_PROFIT_IMPROVEMENT:
                if "time_night" not in self._created_experts:
                    candidates.append(ExpertCandidate(
                        name="time_night",
                        description="Night trading (10pm-6am)",
                        filter_type="time",
                        filter_value="night",
                        n_samples=len(night_profits),
                        potential_profit=avg_night,
                        current_best_profit=overall_avg,
                        improvement=avg_night - overall_avg,
                    ))
        
        return candidates
    
    def _find_combined_patterns(
        self,
        data: List[Dict[str, Any]],
        y_reg: np.ndarray,
        categories: List[str],
    ) -> List[ExpertCandidate]:
        """Find profitable combined patterns (category + condition)."""
        candidates = []
        
        # Track category + weekend combinations
        category_weekend_profits = defaultdict(list)
        category_high_vol_profits = defaultdict(list)
        
        for i, sample in enumerate(data):
            if i >= len(y_reg) or i >= len(categories):
                break
            
            features = sample.get("features", sample)
            if isinstance(features, str):
                import json
                try:
                    features = json.loads(features)
                except:
                    features = {}
            
            category = categories[i]
            profit = y_reg[i] - SPREAD_COST
            is_weekend = features.get("is_weekend", 0)
            volatility = features.get("volatility_24h", 0)
            
            # Category + weekend
            if is_weekend:
                category_weekend_profits[category].append(profit)
            
            # Category + high volatility
            if volatility > 0.05:  # 5% daily volatility
                category_high_vol_profits[category].append(profit)
        
        # Check category + weekend patterns
        for category, profits in category_weekend_profits.items():
            if len(profits) < MIN_CLUSTER_SIZE:
                continue
            
            avg_profit = np.mean(profits)
            if avg_profit < MIN_PROFIT_POTENTIAL:
                continue
            
            expert_name = f"combo_{category}_weekend"
            if expert_name not in self._created_experts:
                candidates.append(ExpertCandidate(
                    name=expert_name,
                    description=f"{category} markets on weekends",
                    filter_type="combined",
                    filter_value={"category": category, "weekend": True},
                    n_samples=len(profits),
                    potential_profit=avg_profit,
                    current_best_profit=0.0,
                    improvement=avg_profit,
                ))
        
        # Check category + high volatility patterns
        for category, profits in category_high_vol_profits.items():
            if len(profits) < MIN_CLUSTER_SIZE:
                continue
            
            avg_profit = np.mean(profits)
            if avg_profit < MIN_PROFIT_POTENTIAL:
                continue
            
            expert_name = f"combo_{category}_highvol"
            if expert_name not in self._created_experts:
                candidates.append(ExpertCandidate(
                    name=expert_name,
                    description=f"High volatility {category} markets",
                    filter_type="combined",
                    filter_value={"category": category, "high_volatility": True},
                    n_samples=len(profits),
                    potential_profit=avg_profit,
                    current_best_profit=0.0,
                    improvement=avg_profit,
                ))
        
        return candidates
    
    def _should_create_expert(self, candidate: ExpertCandidate) -> bool:
        """Check if a candidate should become an expert."""
        # Must have enough samples
        if candidate.n_samples < MIN_CLUSTER_SIZE:
            return False
        
        # Must show profit potential
        if candidate.potential_profit < MIN_PROFIT_POTENTIAL:
            return False
        
        # Must improve over existing experts
        if candidate.improvement < MIN_PROFIT_IMPROVEMENT:
            return False
        
        # Check we haven't created too many dynamic experts
        dynamic_count = sum(
            1 for e in self.pool.get_active_experts()
            if e.expert_type == "dynamic"
        )
        if dynamic_count >= 10:
            logger.info(f"Skipping {candidate.name}: too many dynamic experts")
            return False
        
        return True
    
    def _create_expert_from_candidate(
        self, candidate: ExpertCandidate
    ) -> Optional[Expert]:
        """Create a new expert from a candidate."""
        try:
            expert_id = f"dynamic_{candidate.name}_{uuid4().hex[:8]}"
            
            expert = Expert(
                expert_id=expert_id,
                expert_type="dynamic",
                domain=candidate.name,
            )
            
            # Add to pool
            if self.pool.add_expert(expert):
                self._created_experts.add(candidate.name)
                logger.info(
                    f"Created new expert: {expert_id} "
                    f"({candidate.description}, {candidate.n_samples} samples)"
                )
                return expert
            
        except Exception as e:
            logger.error(f"Failed to create expert from {candidate.name}: {e}")
        
        return None
    
    def get_candidates(self) -> List[Dict[str, Any]]:
        """Get current expert candidates for dashboard display."""
        return [c.to_dict() for c in self.candidates]
