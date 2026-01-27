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
TRAINING_INTERVAL_HOURS = 2  # Train every 2 hours for faster learning
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
                
                # Determine which horizon labels to use for this expert
                horizon = self._get_expert_horizon(expert)
                y_binary_horizon = y_binary[horizon]
                y_reg_horizon = y_reg[horizon]
                
                # Filter data for this expert
                indices = self._get_expert_data_indices(
                    expert, training_data, categories
                )
                
                if len(indices) < MIN_EXPERT_SAMPLES:
                    logger.info(f"  Skipping - only {len(indices)} samples (need {MIN_EXPERT_SAMPLES})")
                    continue
                
                # Extract subset
                X_expert = X[indices]
                y_binary_expert = y_binary_horizon[indices]
                y_reg_expert = y_reg_horizon[indices]
                weights_expert = sample_weights[indices]
                
                # Train
                metrics = expert.train(
                    X_expert, y_binary_expert, y_reg_expert, weights_expert
                )
                
                # Update state after training (handles versioning, state transitions)
                expert.update_state_after_training()
                
                # Record per-sample profits for gating training
                expert_results[expert.expert_id] = self._calculate_per_sample_profits(
                    expert, X, y_reg_horizon
                )
                
                # Log state transition
                logger.info(
                    f"  {expert.expert_id}: state={expert.state.value}, "
                    f"profit={metrics.simulated_profit_pct:+.1%}, "
                    f"conf_mult={expert.confidence_multiplier:.2f}"
                )
            
            # 5. CROSS-EXPERT AWARENESS: Compute consensus features for training
            # After first-pass training, collect predictions from all trained experts
            logger.info("Computing cross-expert consensus features...")
            cross_expert_features = self._compute_cross_expert_training_features(
                experts_to_train, X, training_data
            )
            
            # Add consensus features to feature matrix for second pass
            if cross_expert_features is not None:
                X_enhanced = np.hstack([X, cross_expert_features])
                logger.info(f"Enhanced features: {X.shape[1]} -> {X_enhanced.shape[1]} (added cross-expert)")
                
                # Optional: Do a second training pass with cross-expert features
                # This allows experts to learn from each other's signals
                # (Commented out for now to avoid double training time - enable if needed)
                # self._second_pass_training(experts_to_train, X_enhanced, y_binary, y_reg, 
                #                           sample_weights, training_data, categories)
            
            # 6. Train gating network (only use active experts)
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
            # Use 24h horizon for discovery (primary trading horizon)
            new_experts = self.auto_discovery.discover(
                training_data, X, y_binary['24h'], y_reg['24h'], categories
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
            
            # Send Discord notification
            self._send_training_discord_notification(results)
            
            return results
            
        except Exception as e:
            logger.error(f"MoE training failed: {e}", exc_info=True)
            return None
            
        finally:
            self._is_training = False
    
    def _load_training_data(self, category_balanced: bool = True, max_per_category: int = 10000) -> List[Dict[str, Any]]:
        """Load training data from database with optional category balancing.

        Args:
            category_balanced: If True, limit samples per category to avoid
                              over-representation of dominant categories.
            max_per_category: Maximum samples per category when balancing.

        Returns:
            List of training examples.
        """
        if not os.path.exists(self.db_path):
            logger.error(f"Training database not found: {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if category_balanced:
                # Category-balanced sampling to avoid over-representation
                # of politics and sports in training
                cursor.execute("""
                    SELECT category, COUNT(*) as cnt
                    FROM training_examples
                    WHERE price_change_24h IS NOT NULL
                    GROUP BY category
                """)
                category_counts = {row[0]: row[1] for row in cursor.fetchall()}

                logger.info(f"Category distribution before balancing: {category_counts}")

                # Sample up to max_per_category from each category
                data = []
                for category in category_counts.keys():
                    cursor.execute("""
                        SELECT * FROM training_examples
                        WHERE price_change_24h IS NOT NULL
                          AND category = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (category, max_per_category))

                    rows = cursor.fetchall()
                    for row in rows:
                        d = dict(row)
                        if "features" in d and isinstance(d["features"], str):
                            import json
                            try:
                                d["features"] = json.loads(d["features"])
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.debug(f"Could not parse features JSON: {e}")
                                d["features"] = {}
                        data.append(d)

                # Also include examples with NULL category (legacy data)
                cursor.execute("""
                    SELECT * FROM training_examples
                    WHERE price_change_24h IS NOT NULL
                      AND (category IS NULL OR category = '')
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (max_per_category,))

                for row in cursor.fetchall():
                    d = dict(row)
                    if "features" in d and isinstance(d["features"], str):
                        import json
                        try:
                            d["features"] = json.loads(d["features"])
                        except (json.JSONDecodeError, TypeError):
                            d["features"] = {}
                    data.append(d)

                conn.close()

                # Log balanced distribution
                balanced_cats = {}
                for d in data:
                    cat = d.get("category") or "unknown"
                    balanced_cats[cat] = balanced_cats.get(cat, 0) + 1
                logger.info(f"Category distribution after balancing: {balanced_cats}")

                return data
            else:
                # Original behavior: no balancing
                cursor.execute("""
                    SELECT * FROM training_examples
                    WHERE price_change_24h IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 100000
                """)

                rows = cursor.fetchall()
                conn.close()

                data = []
                for row in rows:
                    d = dict(row)
                    if "features" in d and isinstance(d["features"], str):
                        import json
                        try:
                            d["features"] = json.loads(d["features"])
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.debug(f"Could not parse features JSON: {e}")
                            d["features"] = {}
                    data.append(d)

                return data

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return []
    
    def _prepare_data(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]], Optional[List[str]], Optional[List[str]]]:
        """Prepare feature matrix and labels for multiple horizons.
        
        Returns:
            Tuple of (X, y_binary_dict, y_reg_dict, timestamps, categories)
            where y_binary_dict and y_reg_dict have keys: '1h', '4h', '24h'
        """
        feature_cols = self._get_feature_cols()
        
        X_list = []
        y_binary_dict = {'1h': [], '4h': [], '24h': []}
        y_reg_dict = {'1h': [], '4h': [], '24h': []}
        timestamps = []
        categories = []
        
        for sample in data:
            # Get features
            features = sample.get("features", sample)
            if isinstance(features, str):
                import json
                try:
                    features = json.loads(features)
                except (json.JSONDecodeError, TypeError):
                    features = {}
            
            # Extract feature vector
            row = []
            for col in feature_cols:
                val = features.get(col, 0)
                try:
                    row.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    row.append(0.0)
            
            # Get labels for each horizon
            pc_1h = sample.get("label_price_change_1h") or sample.get("price_change_1h", 0)
            pc_4h = sample.get("label_price_change_4h") or sample.get("price_change_4h", 0)
            pc_24h = sample.get("label_price_change_24h") or sample.get("price_change_24h", 0)
            
            # Skip if no 24h label (minimum requirement)
            if pc_24h is None:
                continue
            
            # Use 24h as fallback if shorter horizons not available
            pc_1h = pc_1h if pc_1h is not None else pc_24h
            pc_4h = pc_4h if pc_4h is not None else pc_24h
            
            X_list.append(row)
            timestamps.append(sample.get("created_at", ""))
            # Handle None values explicitly - features may have category=None
            categories.append(features.get("category") or sample.get("category") or "other")
            
            # Calculate binary labels (is trade profitable?) for each horizon
            for horizon, pc in [('1h', pc_1h), ('4h', pc_4h), ('24h', pc_24h)]:
                is_profitable = abs(float(pc)) > (SPREAD_COST + MIN_PROFIT_THRESHOLD)
                y_binary_dict[horizon].append(1 if is_profitable else 0)
                y_reg_dict[horizon].append(float(pc))
        
        if not X_list:
            return None, None, None, None, None
        
        return (
            np.array(X_list),
            {k: np.array(v) for k, v in y_binary_dict.items()},
            {k: np.array(v) for k, v in y_reg_dict.items()},
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
            except (ValueError, AttributeError):
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
                except (json.JSONDecodeError, TypeError):
                    features = {}
            
            if expert.expert_type == "category":
                if category == expert.domain:
                    indices.append(i)
                elif expert.domain == "other" and category not in [
                    "sports", "politics_us", "politics_intl", "crypto",
                    "economics", "entertainment", "tech", "weather", 
                    "science", "legal"
                ]:
                    indices.append(i)
                    
            elif expert.expert_type == "risk":
                risk_level = self.pool.get_risk_level(features)
                if risk_level == expert.domain:
                    indices.append(i)
                    
            elif expert.expert_type == "time":
                time_horizon = self.pool.get_time_horizon(features)
                if time_horizon == expert.domain:
                    indices.append(i)
            
            elif expert.expert_type == "volume":
                volume_level = self.pool.get_volume_level(features)
                if volume_level == expert.domain:
                    indices.append(i)
                    
            elif expert.expert_type == "volatility":
                volatility_level = self.pool.get_volatility_level(features)
                if volatility_level == expert.domain:
                    indices.append(i)
                    
            elif expert.expert_type == "timing":
                timing_flag = self.pool.get_timing_flag(features)
                if timing_flag == expert.domain:
                    indices.append(i)
            
            elif expert.expert_type == "horizon":
                # Horizon experts use ALL data (they differ by label, not filtering)
                indices.append(i)
                    
            elif expert.expert_type == "dynamic":
                # Dynamic experts get a sample of all data for now
                indices.append(i)
        
        return np.array(indices)
    
    def _get_expert_horizon(self, expert: Expert) -> str:
        """Get the prediction horizon for an expert.
        
        Horizon experts use their domain directly (1h, 4h, 24h).
        All other experts default to 24h predictions.
        
        Returns:
            Horizon string: '1h', '4h', or '24h'
        """
        if expert.expert_type == "horizon":
            # Domain is '1h', '4h', or '24h'
            return expert.domain
        else:
            # All other experts use 24h predictions by default
            return "24h"
    
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
    
    def _compute_cross_expert_training_features(
        self,
        experts: List,
        X: np.ndarray,
        training_data: List[Dict],
    ) -> Optional[np.ndarray]:
        """Compute cross-expert consensus features for training data.
        
        After all experts are trained, collect their predictions on the
        training set and compute consensus features that can be used
        for enhanced training or as additional input features.
        
        Args:
            experts: List of trained experts
            X: Feature matrix (n_samples x n_features)
            training_data: Original training data dicts
            
        Returns:
            Array of shape (n_samples, n_cross_features) or None if no experts
        """
        trained_experts = [e for e in experts if e._model is not None]
        
        if len(trained_experts) < 2:
            logger.info("Not enough trained experts for cross-expert features")
            return None
        
        n_samples = len(X)
        
        # Collect predictions from all trained experts
        all_predictions = {}
        all_confidences = {}
        
        for expert in trained_experts:
            try:
                X_enhanced = expert._add_interaction_features(X)
                probs = expert._model.predict_proba(X_enhanced)
                
                # Prediction = probability of class 1 (profitable)
                predictions = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                confidences = np.max(probs, axis=1)
                
                all_predictions[expert.expert_id] = predictions
                all_confidences[expert.expert_id] = confidences
                
            except Exception as e:
                logger.debug(f"Error getting predictions from {expert.expert_id}: {e}")
                continue
        
        if len(all_predictions) < 2:
            return None
        
        # Convert to arrays
        pred_matrix = np.array(list(all_predictions.values()))  # (n_experts, n_samples)
        conf_matrix = np.array(list(all_confidences.values()))
        
        # Compute per-sample consensus features
        consensus_mean = np.mean(pred_matrix, axis=0)
        consensus_std = np.std(pred_matrix, axis=0)
        agreement_score = np.clip(1.0 - consensus_std * 2, 0, 1)
        bullish_ratio = np.mean(pred_matrix > 0.5, axis=0)
        mean_confidence = np.mean(conf_matrix, axis=0)
        
        # Top-3 consensus (by average confidence)
        if len(all_confidences) >= 3:
            avg_confs = {k: np.mean(v) for k, v in all_confidences.items()}
            top_3_ids = sorted(avg_confs.keys(), key=lambda k: avg_confs[k], reverse=True)[:3]
            top_3_preds = np.array([all_predictions[eid] for eid in top_3_ids])
            top_3_consensus = np.mean(top_3_preds, axis=0)
        else:
            top_3_consensus = consensus_mean
        
        # Stack into cross-expert feature matrix
        cross_features = np.column_stack([
            consensus_mean,
            consensus_std,
            agreement_score,
            bullish_ratio,
            mean_confidence,
            top_3_consensus,
        ])
        
        logger.info(
            f"Computed cross-expert features: {cross_features.shape[1]} features "
            f"from {len(all_predictions)} experts"
        )
        
        return cross_features
    
    def _send_training_discord_notification(self, results: Dict[str, Any]) -> None:
        """Send Discord notification with training results and GPT summary."""
        try:
            import asyncio
            from polyb0t.services.discord_notifier import get_discord_notifier
            
            notifier = get_discord_notifier()
            if not notifier.enabled:
                return
            
            # Get expert performance summary for context
            expert_perfs = []
            for expert in self.pool.experts.values():
                if expert.is_active:
                    profit = expert.metrics.simulated_profit_pct if expert.metrics else 0
                    expert_perfs.append((expert.expert_id, profit))
            
            # Sort by profit
            expert_perfs.sort(key=lambda x: x[1], reverse=True)
            
            # Add top experts to results for GPT context
            if expert_perfs:
                top_3 = expert_perfs[:3]
                results["top_experts"] = ", ".join([f"{e[0]} ({e[1]:+.1%})" for e in top_3])
            
            # Use the notify_training_complete method which includes GPT summarization
            async def send_notification():
                await notifier.notify_training_complete(
                    num_experts=results.get("n_experts_trained", 0),
                    training_time=results.get("training_time_seconds", 0),
                    active_count=results.get("n_active", 0),
                    training_data=results,  # Pass full data for GPT summary
                )
            
            # Send async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(send_notification())
                else:
                    loop.run_until_complete(send_notification())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send_notification())
                loop.close()
                
            logger.info("Discord training notification sent")
            
        except Exception as e:
            logger.debug(f"Failed to send Discord training notification: {e}")


# Singleton instance
_moe_trainer: Optional[MoETrainer] = None


def get_moe_trainer() -> MoETrainer:
    """Get or create the singleton MoE trainer."""
    global _moe_trainer
    if _moe_trainer is None:
        _moe_trainer = MoETrainer()
    return _moe_trainer
