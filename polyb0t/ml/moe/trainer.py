"""MoE Trainer - trains all experts and gating network jointly.

This module handles the complete training loop for the Mixture of Experts
architecture, optimizing for profitability rather than accuracy.
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from polyb0t.ml.moe.expert import Expert, ExpertMetrics, SPREAD_COST, MIN_PROFIT_THRESHOLD
from polyb0t.ml.moe.expert_pool import ExpertPool, get_expert_pool
from polyb0t.ml.moe.auto_discovery import AutoDiscovery
from polyb0t.ml.moe.versioning import ExpertState

logger = logging.getLogger(__name__)

# Training constants
MIN_TRAINING_SAMPLES = 100
TRAINING_INTERVAL_HOURS = 6
HOLDOUT_FRACTION = 0.2
SAMPLE_WEIGHT_DECAY = 0.5  # Recent data weighted more
MIN_EXPERT_SAMPLES = 30  # Minimum samples for an expert to train


class MoETrainer:
    """Trainer for the Mixture of Experts system.
    
    Handles:
    1. Loading and preparing training data
    2. Training each expert on its relevant data subset
    3. Training the gating network
    4. Running profitability simulation
    5. Auto-deprecating failing experts
    6. Discovering new expert opportunities
    """
    
    def __init__(
        self,
        pool: Optional[ExpertPool] = None,
        db_path: str = "data/ai_training.db",
    ):
        self.pool = pool or get_expert_pool()
        self.db_path = db_path
        self.auto_discovery = AutoDiscovery(self.pool)
        
        self._last_training: Optional[datetime] = None
        self._is_training = False
    
    def should_train(self) -> bool:
        """Check if training should run now."""
        if self._is_training:
            return False
        
        if self._last_training is None:
            return True
        
        elapsed = datetime.utcnow() - self._last_training
        return elapsed.total_seconds() >= TRAINING_INTERVAL_HOURS * 3600
    
    def train(self) -> Optional[Dict[str, Any]]:
        """Run complete MoE training cycle.
        
        Returns:
            Training results summary or None if training failed/skipped
        """
        if self._is_training:
            logger.warning("MoE training already in progress")
            return None
        
        self._is_training = True
        start_time = datetime.utcnow()
        
        try:
            logger.info("=" * 60)
            logger.info("STARTING MOE TRAINING CYCLE")
            logger.info("=" * 60)
            
            # 1. Load training data
            training_data = self._load_training_data()
            if len(training_data) < MIN_TRAINING_SAMPLES:
                logger.warning(f"Not enough training data: {len(training_data)} < {MIN_TRAINING_SAMPLES}")
                return None
            
            logger.info(f"Loaded {len(training_data)} training samples")
            
            # 2. Prepare features and labels
            X, y_binary, y_reg, timestamps, categories = self._prepare_data(training_data)
            if X is None:
                logger.error("Failed to prepare training data")
                return None
            
            logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
            
            # 3. Calculate sample weights (recent data weighted more)
            sample_weights = self._calculate_sample_weights(timestamps)
            
            # 4. Train each expert on its relevant data
            # Train ALL non-deprecated experts (including suspended ones)
            expert_results = {}  # expert_id -> list of profits per sample
            experts_to_train = [
                e for e in self.pool.experts.values() 
                if e.state != ExpertState.DEPRECATED
            ]
            
            logger.info(f"Training {len(experts_to_train)} experts...")
            
            for expert in experts_to_train:
                state_str = expert.state.value
                logger.info(f"Training expert: {expert.expert_id} ({expert.domain}) [{state_str}]...")
                
                # Filter data for this expert
                indices = self._get_expert_data_indices(
                    expert, training_data, categories
                )
                
                if len(indices) < MIN_EXPERT_SAMPLES:
                    logger.info(f"  Skipping - only {len(indices)} samples (need {MIN_EXPERT_SAMPLES})")
                    continue
                
                # Extract subset
                X_expert = X[indices]
                y_binary_expert = y_binary[indices]
                y_reg_expert = y_reg[indices]
                weights_expert = sample_weights[indices]
                
                # Train
                metrics = expert.train(
                    X_expert, y_binary_expert, y_reg_expert, weights_expert
                )
                
                # Update state after training (handles versioning, state transitions)
                expert.update_state_after_training()
                
                # Record per-sample profits for gating training
                expert_results[expert.expert_id] = self._calculate_per_sample_profits(
                    expert, X, y_reg
                )
                
                # Log state transition
                logger.info(
                    f"  {expert.expert_id}: state={expert.state.value}, "
                    f"profit={metrics.simulated_profit_pct:+.1%}, "
                    f"conf_mult={expert.confidence_multiplier:.2f}"
                )
            
            # 5. Train gating network (only use active experts)
            active_results = {
                k: v for k, v in expert_results.items()
                if self.pool.experts.get(k) and self.pool.experts[k].is_active
            }
            if self.pool.gating and len(active_results) > 1:
                logger.info(f"Training gating network on {len(active_results)} active experts...")
                self.pool.gating.train(training_data, active_results)
            
            # 6. Log expert state summary
            state_summary = self._get_state_summary()
            logger.info(f"Expert states: {state_summary}")
            
            # 7. Run auto-discovery for new experts
            new_experts = self.auto_discovery.discover(
                training_data, X, y_binary, y_reg, categories
            )
            
            # 8. Update pool state
            self.pool.update_state_after_training()
            self._last_training = datetime.utcnow()
            
            # Calculate training time
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Count experts by state
            n_deprecated = sum(1 for e in self.pool.experts.values() if e.state == ExpertState.DEPRECATED)
            n_suspended = sum(1 for e in self.pool.experts.values() if e.state == ExpertState.SUSPENDED)
            n_active = sum(1 for e in self.pool.experts.values() if e.state == ExpertState.ACTIVE)
            
            # Build results summary
            results = {
                "success": True,
                "training_time_seconds": training_time,
                "n_samples": len(training_data),
                "n_experts_trained": len(expert_results),
                "n_active": n_active,
                "n_suspended": n_suspended,
                "n_deprecated": n_deprecated,
                "n_new_experts": len(new_experts),
                "state_summary": state_summary,
                "pool_stats": self.pool.get_stats(),
            }
            
            logger.info("=" * 60)
            logger.info(f"MOE TRAINING COMPLETE in {training_time:.1f}s")
            logger.info(f"  Experts trained: {len(expert_results)}")
            logger.info(f"  Active: {n_active}, Suspended: {n_suspended}, Deprecated: {n_deprecated}")
            logger.info(f"  New experts created: {len(new_experts)}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"MoE training failed: {e}", exc_info=True)
            return None
            
        finally:
            self._is_training = False
    
    def _load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from database."""
        if not os.path.exists(self.db_path):
            logger.error(f"Training database not found: {self.db_path}")
            return []
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Load labeled examples (those with 24h price change)
            cursor.execute("""
                SELECT * FROM training_examples
                WHERE price_change_24h IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 100000
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dicts
            data = []
            for row in rows:
                d = dict(row)
                # Parse features JSON if stored as string
                if "features" in d and isinstance(d["features"], str):
                    import json
                    try:
                        d["features"] = json.loads(d["features"])
                    except:
                        d["features"] = {}
                data.append(d)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return []
    
    def _prepare_data(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[List[str]]]:
        """Prepare feature matrix and labels.
        
        Returns:
            Tuple of (X, y_binary, y_reg, timestamps, categories)
        """
        feature_cols = self._get_feature_cols()
        
        X_list = []
        y_binary_list = []
        y_reg_list = []
        timestamps = []
        categories = []
        
        for sample in data:
            # Get features
            features = sample.get("features", sample)
            if isinstance(features, str):
                import json
                try:
                    features = json.loads(features)
                except:
                    features = {}
            
            # Extract feature vector
            row = []
            for col in feature_cols:
                val = features.get(col, 0)
                try:
                    row.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    row.append(0.0)
            
            # Get labels
            price_change_24h = sample.get("price_change_24h", 0)
            if price_change_24h is None:
                continue
            
            # Binary: is this trade profitable after spread?
            is_profitable = abs(price_change_24h) > (SPREAD_COST + MIN_PROFIT_THRESHOLD)
            
            X_list.append(row)
            y_binary_list.append(1 if is_profitable else 0)
            y_reg_list.append(float(price_change_24h))
            timestamps.append(sample.get("created_at", ""))
            categories.append(features.get("category", sample.get("category", "other")))
        
        if not X_list:
            return None, None, None, None, None
        
        return (
            np.array(X_list),
            np.array(y_binary_list),
            np.array(y_reg_list),
            timestamps,
            categories,
        )
    
    def _get_feature_cols(self) -> List[str]:
        """Get feature column names."""
        return [
            "price", "bid", "ask", "spread", "spread_pct", "mid_price",
            "volume_24h", "volume_1h", "volume_6h",
            "liquidity", "liquidity_bid", "liquidity_ask",
            "orderbook_imbalance", "bid_depth", "ask_depth",
            "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10",
            "best_bid_size", "best_ask_size", "bid_ask_size_ratio",
            "momentum_1h", "momentum_4h", "momentum_24h", "momentum_7d",
            "volatility_1h", "volatility_24h", "volatility_7d",
            "trade_count_1h", "trade_count_24h",
            "avg_trade_size_1h", "avg_trade_size_24h",
            "buy_sell_ratio_1h",
            "days_to_resolution", "hours_to_resolution", "market_age_days",
            "hour_of_day", "day_of_week", "is_weekend",
            "open_interest", "unique_traders",
        ]
    
    def _calculate_sample_weights(self, timestamps: List[str]) -> np.ndarray:
        """Calculate sample weights based on recency."""
        n_samples = len(timestamps)
        
        # Parse timestamps and sort indices
        times = []
        for ts in timestamps:
            try:
                t = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                times.append(t.timestamp())
            except:
                times.append(0)
        
        times = np.array(times)
        
        if times.max() == times.min():
            return np.ones(n_samples)
        
        # Normalize to 0-1 range (0 = oldest, 1 = newest)
        normalized = (times - times.min()) / (times.max() - times.min() + 1e-8)
        
        # Apply weight decay: recent samples get higher weight
        weights = SAMPLE_WEIGHT_DECAY + (1 - SAMPLE_WEIGHT_DECAY) * normalized
        
        # Normalize to sum to n_samples
        weights = weights * n_samples / weights.sum()
        
        return weights
    
    def _get_expert_data_indices(
        self,
        expert: Expert,
        data: List[Dict[str, Any]],
        categories: List[str],
    ) -> np.ndarray:
        """Get indices of training data relevant to an expert."""
        indices = []
        
        for i, (sample, category) in enumerate(zip(data, categories)):
            features = sample.get("features", sample)
            if isinstance(features, str):
                import json
                try:
                    features = json.loads(features)
                except:
                    features = {}
            
            if expert.expert_type == "category":
                if category == expert.domain:
                    indices.append(i)
                elif expert.domain == "other" and category not in [
                    "sports", "politics_us", "politics_intl", "crypto",
                    "economics", "entertainment"
                ]:
                    indices.append(i)
                    
            elif expert.expert_type == "risk":
                price = features.get("price", 0.5)
                risk_level = self.pool.get_risk_level(features)
                if risk_level == expert.domain:
                    indices.append(i)
                    
            elif expert.expert_type == "time":
                time_horizon = self.pool.get_time_horizon(features)
                if time_horizon == expert.domain:
                    indices.append(i)
                    
            elif expert.expert_type == "dynamic":
                # Dynamic experts get a sample of all data for now
                indices.append(i)
        
        return np.array(indices)
    
    def _calculate_per_sample_profits(
        self,
        expert: Expert,
        X: np.ndarray,
        y_reg: np.ndarray,
    ) -> List[float]:
        """Calculate profit for each sample using this expert."""
        profits = []
        
        if expert._model is None:
            return [0.0] * len(X)
        
        try:
            # Add interaction features
            X_enhanced = expert._add_interaction_features(X)
            
            # Get predictions
            probs = expert._model.predict_proba(X_enhanced)
            predictions = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)
            
            for i in range(len(X)):
                if confidence[i] < 0.6:
                    profits.append(0.0)  # No trade
                elif predictions[i] == 1:  # Predicted profitable
                    actual_change = y_reg[i]
                    profit = actual_change - SPREAD_COST
                    profits.append(profit * 0.05)  # 5% position
                else:
                    profits.append(0.0)  # Skipped
                    
        except Exception as e:
            logger.debug(f"Error calculating profits for {expert.expert_id}: {e}")
            profits = [0.0] * len(X)
        
        return profits
    
    def _get_state_summary(self) -> Dict[str, int]:
        """Get count of experts by state."""
        summary = {state.value: 0 for state in ExpertState}
        for expert in self.pool.experts.values():
            summary[expert.state.value] += 1
        return summary
    
    def _check_deprecations(self) -> List[str]:
        """Check for and deprecate failing experts.
        
        Note: With the new state machine, deprecation happens automatically
        via update_state_after_training(). This method is kept for
        additional manual deprecation if needed.
        """
        deprecated = []
        
        for expert in self.pool.experts.values():
            if expert.should_deprecate():
                deprecated.append(expert.expert_id)
        
        return deprecated
    
    def get_expert_trends(self) -> Dict[str, float]:
        """Get performance trends for all experts."""
        trends = {}
        for expert in self.pool.experts.values():
            trends[expert.expert_id] = expert.calculate_trend()
        return trends
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        state_summary = self._get_state_summary()
        trends = self.get_expert_trends()
        
        # Find improving/declining experts
        improving = [k for k, v in trends.items() if v > 0.01]
        declining = [k for k, v in trends.items() if v < -0.01]
        
        return {
            "last_training": self._last_training.isoformat() if self._last_training else None,
            "is_training": self._is_training,
            "state_summary": state_summary,
            "improving_experts": improving,
            "declining_experts": declining,
            "pool_stats": self.pool.get_stats(),
        }


# Singleton instance
_moe_trainer: Optional[MoETrainer] = None


def get_moe_trainer() -> MoETrainer:
    """Get or create the singleton MoE trainer."""
    global _moe_trainer
    if _moe_trainer is None:
        _moe_trainer = MoETrainer()
    return _moe_trainer
