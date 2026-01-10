"""AI Orchestrator - manages the entire AI training and prediction workflow.

This module ties together:
- Data collection
- Training
- Prediction
- Shutdown recovery
"""

import asyncio
import logging
import os
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Optional

from polyb0t.config import get_settings
from polyb0t.ml.continuous_collector import (
    ContinuousDataCollector,
    MarketSnapshot,
    get_data_collector,
)
from polyb0t.ml.ai_trainer import AITrainer, get_ai_trainer

logger = logging.getLogger(__name__)


class AIOrchestrator:
    """Orchestrates all AI operations."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.settings = get_settings()
        
        # Calculate max storage in bytes
        max_storage_bytes = int(self.settings.ai_max_storage_gb * 1024 * 1024 * 1024)
        
        # Initialize components
        self.collector = ContinuousDataCollector(
            db_path=self.settings.ai_training_db,
            max_storage_bytes=max_storage_bytes,
        )
        self.trainer = get_ai_trainer(
            model_dir=self.settings.ai_model_dir,
            min_training_examples=self.settings.ai_min_training_examples,
        )
        
        self._last_training_time: Optional[datetime] = None
        self._last_example_time: Optional[datetime] = None
        self._state_path = os.path.join(self.settings.ai_model_dir, "orchestrator_state.json")
        
        # Ensure directories exist
        os.makedirs(self.settings.ai_model_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.settings.ai_training_db) or "data", exist_ok=True)
        
        # Load state
        self._load_state()
        
        # Check for recovery needed
        self._check_and_recover()
        
    def _load_state(self) -> None:
        """Load orchestrator state from disk."""
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    state = json.load(f)
                    
                if state.get("last_training_time"):
                    self._last_training_time = datetime.fromisoformat(state["last_training_time"])
                if state.get("last_example_time"):
                    self._last_example_time = datetime.fromisoformat(state["last_example_time"])
                    
                logger.info(
                    f"Loaded AI orchestrator state: last_example_time={self._last_example_time}, "
                    f"last_training_time={self._last_training_time}"
                )
            except Exception as e:
                logger.warning(f"Failed to load orchestrator state: {e}")
        else:
            logger.info("No AI orchestrator state file found - starting fresh")
                
    def _save_state(self) -> None:
        """Save orchestrator state to disk."""
        state = {
            "last_training_time": self._last_training_time.isoformat() if self._last_training_time else None,
            "last_example_time": self._last_example_time.isoformat() if self._last_example_time else None,
            "saved_at": datetime.utcnow().isoformat(),
        }
        
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save orchestrator state: {e}")
            
    def _check_and_recover(self) -> None:
        """Check if recovery is needed after restart and perform it."""
        time_since_last = self.collector.get_time_since_last_collection()
        
        if time_since_last is None:
            logger.info("First run - no recovery needed")
            return
            
        if time_since_last > timedelta(minutes=30):
            logger.warning(
                f"Bot was down for {time_since_last}. Will catch up on missed data."
            )
            # The catch-up will happen naturally during the next data collection cycle
            
        # Check if training was due
        if self._last_training_time:
            hours_since_training = (datetime.utcnow() - self._last_training_time).total_seconds() / 3600
            if hours_since_training >= self.settings.ai_retrain_interval_hours:
                logger.info(f"Training overdue ({hours_since_training:.1f}h since last). Will retrain soon.")
                
    def is_ai_ready(self) -> bool:
        """Check if AI model is ready for trading.
        
        Returns:
            True if AI model is trained and ready.
        """
        return self.trainer.has_trained_model()
    
    def get_model_info(self) -> Optional[dict]:
        """Get information about the current AI model.
        
        Returns:
            Model info dict or None.
        """
        return self.trainer.get_model_info()
    
    def get_training_stats(self) -> dict:
        """Get training statistics.
        
        Returns:
            Dictionary of stats.
        """
        collector_stats = self.collector.get_stats()
        model_info = self.trainer.get_model_info()
        
        return {
            "collector": collector_stats,
            "model": model_info,
            "is_ready": self.is_ai_ready(),
            "is_training": self.trainer.is_training(),
            "last_training": self._last_training_time.isoformat() if self._last_training_time else None,
            "can_train": self.trainer.can_train(collector_stats.get("labeled_examples", 0)),
        }
    
    def should_train(self) -> bool:
        """Check if we should start training.
        
        Returns:
            True if training should start.
        """
        if self.trainer.is_training():
            return False
            
        # Check if enough examples
        labeled_examples = self.collector.get_labeled_examples()
        if not self.trainer.can_train(labeled_examples):
            return False
            
        # Check if enough time has passed since last training
        if self._last_training_time:
            hours_since = (datetime.utcnow() - self._last_training_time).total_seconds() / 3600
            if hours_since < self.settings.ai_retrain_interval_hours:
                return False
                
        return True
    
    def should_create_examples(self) -> bool:
        """Check if we should create new training examples.
        
        Returns:
            True if we should create examples.
        """
        if self._last_example_time is None:
            logger.info("should_create_examples: True (never created before)")
            return True
            
        minutes_since = (datetime.utcnow() - self._last_example_time).total_seconds() / 60
        interval = self.settings.ai_example_interval_minutes
        should_create = minutes_since >= interval
        
        if should_create:
            logger.info(f"should_create_examples: True ({minutes_since:.1f} min >= {interval} min interval)")
        
        return should_create
    
    def track_markets(self, markets: list[dict]) -> int:
        """Add markets to tracking for AI data collection.
        
        Args:
            markets: List of market dictionaries.
            
        Returns:
            Number of new markets added.
        """
        current_count = self.collector.get_tracked_market_count()
        max_markets = self.settings.ai_max_markets_to_track
        
        added = 0
        for market in markets:
            if current_count + added >= max_markets:
                break
                
            # Get token IDs from outcomes
            outcomes = market.get("outcomes", [])
            for outcome in outcomes:
                token_id = outcome.get("token_id")
                if token_id:
                    condition_id = market.get("condition_id", "")
                    if self.collector.add_market_to_track(token_id, condition_id):
                        added += 1
                        
        if added > 0:
            logger.info(f"Added {added} new markets to AI tracking (total: {current_count + added})")
            
        return added
    
    def collect_snapshot(
        self,
        token_id: str,
        market_id: str,
        price: float,
        bid: float,
        ask: float,
        orderbook_imbalance: float,
        volume_24h: float,
        liquidity: float,
        # Expanded features
        spread: float = 0,
        spread_pct: float = 0,
        mid_price: float = 0,
        volume_1h: float = 0,
        volume_6h: float = 0,
        liquidity_bid: float = 0,
        liquidity_ask: float = 0,
        bid_depth: float = 0,
        ask_depth: float = 0,
        bid_depth_5: float = 0,
        ask_depth_5: float = 0,
        bid_depth_10: float = 0,
        ask_depth_10: float = 0,
        bid_levels: int = 0,
        ask_levels: int = 0,
        best_bid_size: float = 0,
        best_ask_size: float = 0,
        bid_ask_size_ratio: float = 0,
        momentum_1h: float = 0,
        momentum_4h: float = 0,
        momentum_24h: float = 0,
        momentum_7d: float = 0,
        price_change_1h: float = 0,
        price_change_4h: float = 0,
        price_change_24h: float = 0,
        price_high_24h: float = 0,
        price_low_24h: float = 0,
        price_range_24h: float = 0,
        volatility_1h: float = 0,
        volatility_24h: float = 0,
        volatility_7d: float = 0,
        atr_24h: float = 0,
        trade_count_1h: int = 0,
        trade_count_24h: int = 0,
        avg_trade_size_1h: float = 0,
        avg_trade_size_24h: float = 0,
        buy_volume_1h: float = 0,
        sell_volume_1h: float = 0,
        buy_sell_ratio_1h: float = 0,
        large_trade_count_24h: int = 0,
        category: str = "",
        subcategory: str = "",
        market_slug: str = "",
        question_length: int = 0,
        description_length: int = 0,
        has_icon: bool = False,
        days_to_resolution: float = 30,
        hours_to_resolution: float = 720,
        market_age_days: float = 0,
        hour_of_day: int = 0,
        day_of_week: int = 0,
        is_weekend: bool = False,
        is_active: bool = True,
        is_closed: bool = False,
        total_yes_shares: float = 0,
        total_no_shares: float = 0,
        open_interest: float = 0,
        num_related_markets: int = 0,
        avg_related_price: float = 0,
        comment_count: int = 0,
        view_count: int = 0,
        unique_traders: int = 0,
        price_vs_volume_ratio: float = 0,
        liquidity_per_dollar_volume: float = 0,
        spread_adjusted_edge: float = 0,
    ) -> bool:
        """Collect a comprehensive market snapshot for training data.
        
        Args:
            All market data fields - expanded for richer training data.
            
        Returns:
            True if a training example was created, False otherwise.
        """
        # Compute spread if not provided
        if spread == 0 and price > 0:
            spread = ask - bid
        if spread_pct == 0 and price > 0:
            spread_pct = (ask - bid) / price
        if mid_price == 0:
            mid_price = price
        
        snapshot = MarketSnapshot(
            token_id=token_id,
            market_id=market_id,
            timestamp=datetime.utcnow(),
            price=price,
            bid=bid,
            ask=ask,
            spread=spread,
            spread_pct=spread_pct,
            mid_price=mid_price,
            volume_24h=volume_24h,
            volume_1h=volume_1h,
            volume_6h=volume_6h,
            liquidity=liquidity,
            liquidity_bid=liquidity_bid,
            liquidity_ask=liquidity_ask,
            orderbook_imbalance=orderbook_imbalance,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_depth_5=bid_depth_5,
            ask_depth_5=ask_depth_5,
            bid_depth_10=bid_depth_10,
            ask_depth_10=ask_depth_10,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
            bid_ask_size_ratio=bid_ask_size_ratio,
            momentum_1h=momentum_1h,
            momentum_4h=momentum_4h,
            momentum_24h=momentum_24h,
            momentum_7d=momentum_7d,
            price_change_1h=price_change_1h,
            price_change_4h=price_change_4h,
            price_change_24h=price_change_24h,
            price_high_24h=price_high_24h,
            price_low_24h=price_low_24h,
            price_range_24h=price_range_24h,
            volatility_1h=volatility_1h,
            volatility_24h=volatility_24h,
            volatility_7d=volatility_7d,
            atr_24h=atr_24h,
            trade_count_1h=trade_count_1h,
            trade_count_24h=trade_count_24h,
            avg_trade_size_1h=avg_trade_size_1h,
            avg_trade_size_24h=avg_trade_size_24h,
            buy_volume_1h=buy_volume_1h,
            sell_volume_1h=sell_volume_1h,
            buy_sell_ratio_1h=buy_sell_ratio_1h,
            large_trade_count_24h=large_trade_count_24h,
            category=category,
            subcategory=subcategory,
            market_slug=market_slug,
            question_length=question_length,
            description_length=description_length,
            has_icon=has_icon,
            days_to_resolution=days_to_resolution,
            hours_to_resolution=hours_to_resolution,
            market_age_days=market_age_days,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            is_active=is_active,
            is_closed=is_closed,
            total_yes_shares=total_yes_shares,
            total_no_shares=total_no_shares,
            open_interest=open_interest,
            num_related_markets=num_related_markets,
            avg_related_price=avg_related_price,
            comment_count=comment_count,
            view_count=view_count,
            unique_traders=unique_traders,
            price_vs_volume_ratio=price_vs_volume_ratio,
            liquidity_per_dollar_volume=liquidity_per_dollar_volume,
            spread_adjusted_edge=spread_adjusted_edge,
        )
        
        self.collector.record_snapshot(snapshot)
        
        # Create training example if it's time
        # Return True if example was created
        if self.should_create_examples():
            self.collector.create_training_example(snapshot)
            return True
        return False
            
    def finish_example_cycle(self, examples_created: bool = False) -> None:
        """Mark end of an example creation cycle.
        
        Args:
            examples_created: If True, update the example creation timestamp.
        """
        # Only update the example timestamp if we actually created examples
        # This prevents the timer from resetting every cycle
        if examples_created:
            self._last_example_time = datetime.utcnow()
            logger.info(f"Created training examples, next batch in {self.settings.ai_example_interval_minutes} minutes")
        
        self.collector.update_collection_time()
        self._save_state()
        
        # Label any unlabeled examples
        self.collector.label_examples()
    
    def mark_resolution(self, token_id: str, outcome: int) -> None:
        """Mark a market as resolved.
        
        Args:
            token_id: Token ID.
            outcome: 1 for Yes, 0 for No.
        """
        self.collector.mark_market_resolved(token_id, outcome)
        # Re-label examples with resolution data
        self.collector.label_examples()
    
    def run_training(self) -> bool:
        """Run the training process.
        
        Returns:
            True if new model was deployed.
        """
        if not self.should_train():
            return False
            
        logger.info("Starting AI training cycle...")
        
        # Get training data
        training_data = self.collector.get_training_data(only_labeled=True)
        
        if len(training_data) < self.settings.ai_min_training_examples:
            logger.info(
                f"Not enough training data: {len(training_data)} < {self.settings.ai_min_training_examples}"
            )
            return False
            
        # Train model
        result = self.trainer.train_model(training_data)
        
        self._last_training_time = datetime.utcnow()
        self._save_state()
        
        if result:
            logger.info(f"New AI model v{result.version} deployed with score {result.metrics.score():.3f}")
            return True
        else:
            logger.info("Training completed but model not deployed (not better than current)")
            return False
    
    def predict(self, features: dict) -> Optional[float]:
        """Make a prediction using the AI model.
        
        Args:
            features: Feature dictionary.
            
        Returns:
            Predicted price change, or None if no model.
        """
        if not self.is_ai_ready():
            return None
        return self.trainer.predict(features)
    
    def get_ai_signal(
        self,
        token_id: str,
        price: float,
        features: dict,
    ) -> Optional[dict]:
        """Get an AI trading signal.
        
        Args:
            token_id: Token ID.
            price: Current price.
            features: Feature dictionary.
            
        Returns:
            Signal dict with side, edge, confidence, or None.
        """
        prediction = self.predict(features)
        
        if prediction is None:
            return None
            
        # Convert prediction to signal
        # Prediction is expected price change
        edge = prediction
        
        # Only generate signal if edge is meaningful
        min_edge = self.settings.edge_threshold
        if abs(edge) < min_edge:
            return None
            
        # Determine side
        side = "BUY" if edge > 0 else "SELL"
        
        # Calculate confidence based on prediction magnitude
        # Higher prediction = higher confidence
        confidence = min(1.0, abs(edge) * 5)  # Scale to 0-1
        
        return {
            "token_id": token_id,
            "side": side,
            "edge": edge,
            "confidence": confidence,
            "source": "ai_model",
            "model_version": self.trainer.get_model_info().get("version") if self.trainer.get_model_info() else None,
        }


# Singleton instance
_orchestrator_instance: Optional[AIOrchestrator] = None


def get_ai_orchestrator() -> AIOrchestrator:
    """Get or create the singleton AI orchestrator.
    
    Returns:
        AIOrchestrator instance.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AIOrchestrator()
    return _orchestrator_instance


def check_ai_ready_or_exit() -> None:
    """Check if AI is ready; if not and in AI mode, exit with error.
    
    Raises:
        SystemExit if AI mode is on but no trained model.
    """
    settings = get_settings()
    
    if settings.strategy_mode != "ai":
        return  # Not in AI mode, no check needed
        
    orchestrator = get_ai_orchestrator()
    
    if not orchestrator.is_ai_ready():
        if settings.placing_orders:
            logger.error(
                "❌ AI MODE ENABLED but no trained model available!\n"
                "Cannot start in AI mode without a trained model.\n"
                "Either:\n"
                "  1. Set POLYBOT_STRATEGY_MODE=rules to use rules-based trading\n"
                "  2. Set POLYBOT_PLACING_ORDERS=false to collect training data first\n"
                "  3. Wait for AI model to be trained"
            )
            raise SystemExit(1)
        else:
            logger.warning(
                "AI mode enabled but no trained model. "
                "Bot will collect training data until model is ready."
            )
