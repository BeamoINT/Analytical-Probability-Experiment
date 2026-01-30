"""AI Orchestrator - manages the entire AI training and prediction workflow.

This module ties together:
- Data collection
- Training using Mixture of Experts (MoE only - legacy model deprecated)
- Prediction using ensemble of multiple experts
- Shutdown recovery
"""

import asyncio
import logging
import os
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple

from polyb0t.config import get_settings
from polyb0t.ml.continuous_collector import (
    ContinuousDataCollector,
    MarketSnapshot,
    get_data_collector,
)
from polyb0t.ml.ai_trainer import AITrainer, get_ai_trainer
from polyb0t.ml.category_tracker import get_category_tracker, MarketCategoryTracker

# MoE imports
from polyb0t.ml.moe.expert_pool import ExpertPool, get_expert_pool
from polyb0t.ml.moe.trainer import MoETrainer, get_moe_trainer

logger = logging.getLogger(__name__)


class AIOrchestrator:
    """Orchestrates all AI operations including Mixture of Experts."""
    
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
        
        # Category tracker for learning which market types to avoid
        self.category_tracker = get_category_tracker()
        
        # === MIXTURE OF EXPERTS ===
        # MoE is the primary prediction system
        self.expert_pool = get_expert_pool()
        self.moe_trainer = get_moe_trainer()
        self._use_moe = True  # Always use MoE for predictions
        
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
        
        logger.info(f"AI Orchestrator initialized with MoE ({len(self.expert_pool.get_active_experts())} experts)")
        
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
            True if MoE has active trained experts.
        """
        # Only check MoE - legacy model is deprecated
        active_experts = self.expert_pool.get_active_experts()
        trained_experts = [e for e in active_experts if e._model is not None]
        return len(trained_experts) > 0
    
    def get_model_info(self) -> Optional[dict]:
        """Get information about the current AI model.
        
        Returns:
            Model info dict or None.
        """
        return self.trainer.get_model_info()
    
    def get_training_stats(self) -> dict:
        """Get training statistics including MoE info.
        
        Returns:
            Dictionary of stats.
        """
        collector_stats = self.collector.get_stats()
        model_info = self.trainer.get_model_info()
        moe_stats = self.expert_pool.get_stats()
        
        return {
            "collector": collector_stats,
            "model": model_info,
            "moe": moe_stats,
            "is_ready": self.is_ai_ready(),
            "is_training": self.trainer.is_training(),
            "last_training": self._last_training_time.isoformat() if self._last_training_time else None,
            "can_train": self.trainer.can_train(collector_stats.get("labeled_examples", 0)),
            "use_moe": self._use_moe,
        }
    
    def get_moe_stats(self) -> dict:
        """Get Mixture of Experts statistics.
        
        Returns:
            Dictionary of MoE stats.
        """
        return self.expert_pool.get_stats()
    
    def should_train(self) -> bool:
        """Check if we should start training.

        In batch mode: never retrain automatically (training is done via CLI).
        In online mode: retrain periodically based on interval.

        Returns:
            True if training should start.
        """
        # Batch mode: no automatic retraining
        if self.settings.ai_training_mode == "batch":
            return False

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
        # Whale tracking features
        whale_activity_1h: int = 0,
        whale_net_direction_1h: float = 0,
        whale_activity_24h: int = 0,
        whale_net_direction_24h: float = 0,
        largest_trade_24h: float = 0,
        # Correlation features
        correlated_market_count: int = 0,
        correlated_avg_price: float = 0,
        correlated_momentum: float = 0,
        avg_correlation_strength: float = 0,
        # Microstructure features (V3)
        vpin: float = 0,
        order_flow_toxicity: float = 0,
        trade_impact_10usd: float = 0,
        trade_impact_100usd: float = 0,
        amihud_illiquidity: float = 0,
        # News/Sentiment features (V3)
        news_article_count: int = 0,
        news_recency_hours: float = 999.0,
        news_sentiment_score: float = 0,
        news_sentiment_confidence: float = 0,
        keyword_positive_count: int = 0,
        keyword_negative_count: int = 0,
        headline_confirmation: float = 0,
        headline_conf_confidence: float = 0,
        intelligent_confirmation: float = 0,
        intelligent_conf_confidence: float = 0,
        # Insider tracking features (V3)
        smart_wallet_buy_count_1h: int = 0,
        smart_wallet_sell_count_1h: int = 0,
        smart_wallet_net_direction_1h: float = 0,
        smart_wallet_volume_1h: float = 0,
        avg_buyer_reputation: float = 0.5,
        avg_seller_reputation: float = 0.5,
        smart_wallet_buy_count_24h: int = 0,
        smart_wallet_sell_count_24h: int = 0,
        smart_wallet_net_direction_24h: float = 0,
        unusual_activity_score: float = 0,
        # Optional: raw data for microstructure calculation
        recent_trades: list = None,
        orderbook: Any = None,
    ) -> bool:
        """Collect a comprehensive market snapshot for training data.
        
        Args:
            All market data fields - expanded for richer training data.
            
        Returns:
            True if a training example was created, False otherwise.
        """
        # Try to get whale features from whale tracker if not provided
        if whale_activity_1h == 0:
            try:
                from polyb0t.services.whale_tracker import get_whale_tracker
                whale_tracker = get_whale_tracker()
                whale_features = whale_tracker.get_whale_features(token_id)
                whale_activity_1h = whale_features.get("whale_activity_1h", 0)
                whale_net_direction_1h = whale_features.get("whale_net_direction_1h", 0.0)
                whale_activity_24h = whale_features.get("whale_activity_24h", 0)
                whale_net_direction_24h = whale_features.get("whale_net_direction_24h", 0.0)
                largest_trade_24h = whale_features.get("largest_trade_24h", 0.0)
            except Exception:
                pass  # Whale tracker not available
        
        # Try to get correlation features from correlation tracker if not provided
        if correlated_market_count == 0:
            try:
                from polyb0t.services.correlation_tracker import get_correlation_tracker
                corr_tracker = get_correlation_tracker()
                # Record price for correlation calculation
                corr_tracker.record_price(token_id, price)
                # Get correlation features
                corr_features = corr_tracker.get_correlation_features(token_id)
                correlated_market_count = corr_features.get("correlated_market_count", 0)
                correlated_avg_price = corr_features.get("correlated_avg_price", 0.0)
                correlated_momentum = corr_features.get("correlated_momentum", 0.0)
                avg_correlation_strength = corr_features.get("avg_correlation_strength", 0.0)
            except Exception:
                pass  # Correlation tracker not available

        # Try to get microstructure features if not provided
        if vpin == 0 and recent_trades:
            try:
                from polyb0t.ml.microstructure import (
                    get_microstructure_analyzer,
                    convert_trades_to_trade_data,
                    convert_orderbook_to_snapshot
                )
                analyzer = get_microstructure_analyzer()
                trade_data = convert_trades_to_trade_data(recent_trades)
                ob_snapshot = convert_orderbook_to_snapshot(orderbook) if orderbook else None
                micro_features = analyzer.get_microstructure_features(
                    trade_data, ob_snapshot, token_id
                )
                vpin = micro_features.get("vpin", 0)
                order_flow_toxicity = micro_features.get("order_flow_toxicity", 0)
                trade_impact_10usd = micro_features.get("trade_impact_10usd", 0)
                trade_impact_100usd = micro_features.get("trade_impact_100usd", 0)
                amihud_illiquidity = micro_features.get("amihud_illiquidity", 0)
            except Exception as e:
                logger.debug(f"Microstructure features not available: {e}")

        # Try to get sentiment features if not provided
        if news_article_count == 0 and market_slug:
            try:
                from polyb0t.ml.sentiment_features import get_sentiment_feature_engine
                sentiment_engine = get_sentiment_feature_engine()
                if sentiment_engine.is_available():
                    sentiment_features = sentiment_engine.get_features_dict(
                        market_id, market_slug, category
                    )
                    news_article_count = int(sentiment_features.get("news_article_count", 0))
                    news_recency_hours = sentiment_features.get("news_recency_hours", 999.0)
                    news_sentiment_score = sentiment_features.get("news_sentiment_score", 0)
                    news_sentiment_confidence = sentiment_features.get("news_sentiment_confidence", 0)
                    keyword_positive_count = int(sentiment_features.get("keyword_positive_count", 0))
                    keyword_negative_count = int(sentiment_features.get("keyword_negative_count", 0))
                    headline_confirmation = sentiment_features.get("headline_confirmation", 0)
                    headline_conf_confidence = sentiment_features.get("headline_conf_confidence", 0)
                    intelligent_confirmation = sentiment_features.get("intelligent_confirmation", 0)
                    intelligent_conf_confidence = sentiment_features.get("intelligent_conf_confidence", 0)
            except Exception as e:
                logger.debug(f"Sentiment features not available: {e}")

        # Try to get insider tracking features if not provided
        if smart_wallet_buy_count_1h == 0:
            try:
                from polyb0t.services.insider_tracker import get_insider_tracker
                insider_tracker = get_insider_tracker()
                insider_features = insider_tracker.get_features_dict(token_id)
                smart_wallet_buy_count_1h = int(insider_features.get("smart_wallet_buy_count_1h", 0))
                smart_wallet_sell_count_1h = int(insider_features.get("smart_wallet_sell_count_1h", 0))
                smart_wallet_net_direction_1h = insider_features.get("smart_wallet_net_direction_1h", 0)
                smart_wallet_volume_1h = insider_features.get("smart_wallet_volume_1h", 0)
                avg_buyer_reputation = insider_features.get("avg_buyer_reputation", 0.5)
                avg_seller_reputation = insider_features.get("avg_seller_reputation", 0.5)
                smart_wallet_buy_count_24h = int(insider_features.get("smart_wallet_buy_count_24h", 0))
                smart_wallet_sell_count_24h = int(insider_features.get("smart_wallet_sell_count_24h", 0))
                smart_wallet_net_direction_24h = insider_features.get("smart_wallet_net_direction_24h", 0)
                unusual_activity_score = insider_features.get("unusual_activity_score", 0)
            except Exception as e:
                logger.debug(f"Insider tracking features not available: {e}")

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
            # Whale features
            whale_activity_1h=whale_activity_1h,
            whale_net_direction_1h=whale_net_direction_1h,
            whale_activity_24h=whale_activity_24h,
            whale_net_direction_24h=whale_net_direction_24h,
            largest_trade_24h=largest_trade_24h,
            # Correlation features
            correlated_market_count=correlated_market_count,
            correlated_avg_price=correlated_avg_price,
            correlated_momentum=correlated_momentum,
            avg_correlation_strength=avg_correlation_strength,
            # Microstructure features (V3)
            vpin=vpin,
            order_flow_toxicity=order_flow_toxicity,
            trade_impact_10usd=trade_impact_10usd,
            trade_impact_100usd=trade_impact_100usd,
            amihud_illiquidity=amihud_illiquidity,
            # News/Sentiment features (V3)
            news_article_count=news_article_count,
            news_recency_hours=news_recency_hours,
            news_sentiment_score=news_sentiment_score,
            news_sentiment_confidence=news_sentiment_confidence,
            keyword_positive_count=keyword_positive_count,
            keyword_negative_count=keyword_negative_count,
            headline_confirmation=headline_confirmation,
            headline_conf_confidence=headline_conf_confidence,
            intelligent_confirmation=intelligent_confirmation,
            intelligent_conf_confidence=intelligent_conf_confidence,
            # Insider tracking features (V3)
            smart_wallet_buy_count_1h=smart_wallet_buy_count_1h,
            smart_wallet_sell_count_1h=smart_wallet_sell_count_1h,
            smart_wallet_net_direction_1h=smart_wallet_net_direction_1h,
            smart_wallet_volume_1h=smart_wallet_volume_1h,
            avg_buyer_reputation=avg_buyer_reputation,
            avg_seller_reputation=avg_seller_reputation,
            smart_wallet_buy_count_24h=smart_wallet_buy_count_24h,
            smart_wallet_sell_count_24h=smart_wallet_sell_count_24h,
            smart_wallet_net_direction_24h=smart_wallet_net_direction_24h,
            unusual_activity_score=unusual_activity_score,
        )

        self.collector.record_snapshot(snapshot)
        
        # Create training example if it's time
        # Return True if example was created
        if self.should_create_examples():
            # === PREDICTION SIMULATION FOR CATEGORY LEARNING ===
            # Make a prediction using the current model (if available)
            # This allows us to learn which categories we're good/bad at
            # even before we start trading
            predicted_change = None

            # Use category from snapshot if available (passed from market data)
            # Fall back to re-categorizing from title/slug
            market_category = category if category else None
            market_title = market_slug if market_slug else None

            # If no category yet, try to categorize from title
            if not market_category and market_title:
                try:
                    market_category, _ = self.category_tracker.categorize_market(
                        market_id=market_id,
                        title=market_title,
                    )
                except Exception as e:
                    logger.debug(f"Categorization failed: {e}")
            
            # Make prediction if AI is ready
            if self.is_ai_ready():
                try:
                    # Get prediction from current model
                    features = snapshot.to_dict()
                    result = self.predict(features)
                    
                    # Handle both old (float) and new (tuple) return types
                    if result is not None:
                        if isinstance(result, tuple):
                            predicted_change, confidence = result
                        else:
                            predicted_change = result
                except Exception as e:
                    logger.debug(f"Prediction simulation failed: {e}")
            
            self.collector.create_training_example(
                snapshot=snapshot,
                predicted_change=predicted_change,
                category=market_category,
                market_title=market_title,
            )
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
        
        # Evaluate predictions for category learning
        self._evaluate_predictions_for_categories()
    
    def _evaluate_predictions_for_categories(self) -> None:
        """Evaluate simulated predictions and record results for category learning.

        This runs after labeling to check which predictions were right/wrong
        and update category statistics.
        """
        try:
            # Use collector's connection method for WAL mode and proper timeout
            conn = self.collector._get_connection()
            cursor = conn.cursor()
            
            # Find examples with predictions that haven't been evaluated yet
            # and have 24h labels (so we know the outcome)
            cursor.execute("""
                SELECT example_id, token_id, market_id, predicted_change, 
                       price_change_24h, category, market_title
                FROM training_examples
                WHERE predicted_change IS NOT NULL
                  AND prediction_evaluated = 0
                  AND price_change_24h IS NOT NULL
            """)
            
            rows = cursor.fetchall()
            
            if not rows:
                return
            
            evaluated = 0
            for row in rows:
                example_id, token_id, market_id, predicted_change, actual_change, category, title = row
                
                # If no category stored, try to categorize now
                if not category and title:
                    category, _ = self.category_tracker.categorize_market(
                        market_id=market_id,
                        title=title,
                    )
                
                if category:
                    # Record the prediction outcome
                    self.category_tracker.record_prediction(
                        market_id=market_id,
                        token_id=token_id,
                        category=category,
                        predicted_change=predicted_change,
                        actual_change=actual_change,
                    )
                
                # Mark as evaluated
                cursor.execute(
                    "UPDATE training_examples SET prediction_evaluated = 1 WHERE example_id = ?",
                    (example_id,)
                )
                evaluated += 1
            
            conn.commit()
            conn.close()
            
            if evaluated > 0:
                logger.info(f"Evaluated {evaluated} simulated predictions for category learning")
                
        except Exception as e:
            logger.warning(f"Failed to evaluate predictions for categories: {e}")
    
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
        """Run the training process for both MoE and legacy systems.
        
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
        
        moe_result = None
        legacy_result = None
        
        # === MIXTURE OF EXPERTS TRAINING ===
        # This is the primary training system
        try:
            logger.info("=" * 60)
            logger.info("TRAINING MIXTURE OF EXPERTS")
            logger.info("=" * 60)
            
            moe_result = self.moe_trainer.train()
            
            if moe_result and moe_result.get("success"):
                logger.info(
                    f"MoE training complete: "
                    f"{moe_result.get('n_experts_trained', 0)} experts, "
                    f"{moe_result.get('training_time_seconds', 0):.1f}s"
                )
        except Exception as e:
            logger.error(f"MoE training failed: {e}", exc_info=True)
        
        # Legacy model training removed - MoE is the only training system
        
        self._last_training_time = datetime.utcnow()
        self._save_state()
        
        # === CATEGORY RE-EVALUATION ===
        try:
            category_summary = self.category_tracker.get_performance_summary()
            avoided = category_summary.get("avoided_categories", [])
            if avoided:
                logger.info(f"Re-evaluating {len(avoided)} avoided categories...")
                for category in avoided:
                    result = self.category_tracker.trigger_reevaluation(category)
                    if result.get("status") == "un-avoided":
                        logger.info(
                            f"Category '{category}' UN-AVOIDED: "
                            f"profitable_acc={result.get('profitable_accuracy', 0):.1%}"
                        )
        except Exception as e:
            logger.warning(f"Category re-evaluation failed: {e}")
        
        return moe_result is not None
    
    def predict(self, features: dict) -> Optional[Tuple[float, float]]:
        """Make a prediction using the MoE system with expert mixing.
        
        Uses the MetaController to combine multiple experts for better
        predictions based on learned optimal mixtures.
        
        Args:
            features: Feature dictionary.
            
        Returns:
            Tuple of (predicted_change, confidence), or None if no model.
        """
        if not self.is_ai_ready():
            return None
        
        # === USE MIXTURE OF EXPERTS WITH META-CONTROLLER ===
        if self._use_moe:
            # Try mixture prediction first (combines multiple experts)
            from polyb0t.ml.moe.meta_controller import get_meta_controller

            # In dry-run mode, include ALL trainable experts (not just active)
            # to gather performance data for training feedback
            include_inactive = getattr(self.settings, "dry_run", False)

            meta = get_meta_controller(self.expert_pool)
            result = meta.predict_with_mixture(features, include_inactive=include_inactive)

            if result is not None:
                prediction, confidence, metadata = result

                # Store metadata for later analysis
                self._last_prediction_metadata = metadata

                # Convert binary prediction to edge-like value
                # 1.0 = strongly profitable, 0.0 = not profitable
                # Map to: positive = bullish, near-zero = neutral
                if prediction > 0.5:
                    # Profitable prediction - determine direction from momentum
                    momentum = features.get("momentum_24h", 0)
                    if momentum >= 0:
                        edge = (prediction - 0.5) * 2 * confidence  # 0 to 1
                    else:
                        edge = -(prediction - 0.5) * 2 * confidence  # -1 to 0
                else:
                    edge = 0  # Not profitable, no signal

                return (edge, confidence)

            # Fallback to standard pool predict if mixture fails
            result = self.expert_pool.predict(features, include_inactive=include_inactive)
            if result is not None:
                prediction, confidence, best_expert = result
                self._last_prediction_metadata = {"best_expert": best_expert, "include_inactive": include_inactive}

                if prediction > 0.5:
                    momentum = features.get("momentum_24h", 0)
                    if momentum >= 0:
                        edge = (prediction - 0.5) * 2 * confidence
                    else:
                        edge = -(prediction - 0.5) * 2 * confidence
                else:
                    edge = 0

                return (edge, confidence)
        
        # No MoE prediction available
        return None
    
    def get_ai_signal(
        self,
        token_id: str,
        market_id: str,
        market_title: str,
        price: float,
        features: dict,
    ) -> Optional[dict]:
        """Get an AI trading signal using Mixture of Experts.

        Args:
            token_id: Token ID.
            market_id: Market/condition ID.
            market_title: Market title for categorization.
            price: Current price.
            features: Feature dictionary.

        Returns:
            Signal dict with side, edge, confidence, expert info, or None.
        """
        # === MIXTURE OF EXPERTS PREDICTION ===
        best_expert_id = None
        
        if self._use_moe:
            moe_result = self.expert_pool.predict(features)
            if moe_result is not None:
                prediction, model_confidence, best_expert_id = moe_result
                
                # Convert binary prediction to edge
                if prediction > 0.5:
                    momentum = features.get("momentum_24h", 0)
                    if momentum >= 0:
                        edge = (prediction - 0.5) * 2 * model_confidence
                    else:
                        edge = -(prediction - 0.5) * 2 * model_confidence
                else:
                    edge = 0
            else:
                # Fall back to legacy
                result = self.predict(features)
                if result is None:
                    return None
                edge, model_confidence = result if isinstance(result, tuple) else (result, 0.5)
        else:
            result = self.predict(features)
            if result is None:
                return None
            edge, model_confidence = result if isinstance(result, tuple) else (result, 0.5)

        # === CATEGORY TRACKING ===
        category, category_confidence = self.category_tracker.categorize_market(
            market_id=market_id,
            title=market_title,
            description=features.get("description", ""),
        )
        
        # Check if this category should be avoided
        if self.category_tracker.should_avoid_category(category):
            logger.debug(f"Skipping {market_id[:12]} - category '{category}' is avoided")
            return None
        
        # Get category confidence multiplier
        category_multiplier = self.category_tracker.get_confidence_multiplier(category)

        # === CONFIDENCE THRESHOLD ===
        CONFIDENCE_THRESHOLD = 0.60
        if model_confidence < CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Skipping {market_id[:12]} - low confidence "
                f"({model_confidence:.1%} < {CONFIDENCE_THRESHOLD:.0%})"
            )
            return None

        # Only generate signal if edge is meaningful
        min_edge = self.settings.edge_threshold
        if abs(edge) < min_edge:
            return None
            
        # Determine side
        side = "BUY" if edge > 0 else "SELL"
        
        # Apply category confidence adjustment
        adjusted_confidence = model_confidence * category_multiplier
        
        # If adjusted confidence is too low, skip
        if adjusted_confidence < 0.3:
            logger.debug(
                f"Skipping {market_id[:12]} - low category confidence "
                f"(base={model_confidence:.2f}, mult={category_multiplier:.2f})"
            )
            return None
        
        # Get model version info
        model_info = self.trainer.get_model_info()
        
        return {
            "token_id": token_id,
            "market_id": market_id,
            "side": side,
            "edge": edge,
            "confidence": adjusted_confidence,
            "base_confidence": model_confidence,
            "category": category,
            "category_multiplier": category_multiplier,
            "source": "moe" if self._use_moe and best_expert_id else "ai_model",
            "best_expert": best_expert_id,
            "model_version": model_info.get("version") if model_info else None,
        }
    
    def record_prediction_outcome(
        self,
        market_id: str,
        token_id: str,
        category: str,
        predicted_change: float,
        actual_change: float,
    ) -> None:
        """Record a prediction outcome for category learning.
        
        This should be called when we have the actual price change data
        (e.g., 24 hours after the prediction was made).
        
        Args:
            market_id: Market identifier.
            token_id: Token identifier.
            category: Market category.
            predicted_change: What the model predicted.
            actual_change: What actually happened.
        """
        self.category_tracker.record_prediction(
            market_id=market_id,
            token_id=token_id,
            category=category,
            predicted_change=predicted_change,
            actual_change=actual_change,
        )
    
    def get_category_performance(self) -> dict:
        """Get category performance summary.
        
        Returns:
            Summary of category performance.
        """
        return self.category_tracker.get_performance_summary()


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
    """Check if AI is ready. Does NOT exit - allows data collection mode.
    
    This function now just logs status. The bot can run without a trained
    model to collect training data.
    """
    orchestrator = get_ai_orchestrator()
    
    if orchestrator.is_ai_ready():
        moe_stats = orchestrator.get_moe_stats()
        active_experts = moe_stats.get("active_experts", 0)
        logger.info(f"AI ready with {active_experts} active experts (MoE)")
    else:
        logger.info(
            "AI model not ready - bot will collect training data. "
            "Trading will begin once experts are trained."
        )
