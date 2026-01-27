"""Unified Trade Outcome Recorder.

Centralized service that records trade outcomes to ALL learning systems,
ensuring consistent feedback across the entire AI system.

When a position closes, this recorder notifies:
1. DataCollector - for training data labeling
2. MetaController - for mixture learning
3. MarketEdge - for edge calculation
4. ExpertPool - for confidence multiplier updates
5. TradePostmortem - for detailed analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Complete record of a trade outcome."""
    
    # Trade identification
    intent_id: str
    token_id: str
    market_id: str
    market_title: str = ""
    
    # Trade details
    side: str = "BUY"  # BUY or SELL
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    
    # Outcome
    profit_usd: float = 0.0
    profit_pct: float = 0.0
    hold_time_hours: float = 0.0
    
    # AI metadata
    mixture_id: Optional[str] = None
    expert_predictions: Dict[str, float] = field(default_factory=dict)
    expert_weights: Dict[str, float] = field(default_factory=dict)
    prediction_confidence: float = 0.0
    
    # Timing
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    
    # Category for learning
    category: str = ""
    
    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable."""
        return self.profit_pct > 0


class TradeOutcomeRecorder:
    """Unified outcome recording for all learning systems.
    
    Ensures that when a trade closes, all components of the AI system
    receive consistent feedback for learning and improvement.
    """
    
    def __init__(self):
        """Initialize the outcome recorder."""
        # Lazy imports to avoid circular dependencies
        self._data_collector = None
        self._meta_controller = None
        self._market_edge = None
        self._expert_pool = None

        # Track recorded outcomes for debugging
        self._recent_outcomes: List[TradeOutcome] = []
        self._max_recent = 100

        logger.info("TradeOutcomeRecorder initialized")
    
    def _get_data_collector(self):
        """Lazy load data collector."""
        if self._data_collector is None:
            try:
                from polyb0t.ml.continuous_collector import get_data_collector
                self._data_collector = get_data_collector()
            except Exception as e:
                logger.warning(f"Could not load data collector: {e}")
        return self._data_collector
    
    def _get_meta_controller(self):
        """Lazy load meta controller."""
        if self._meta_controller is None:
            try:
                from polyb0t.ml.moe.meta_controller import get_meta_controller
                from polyb0t.ml.moe.expert_pool import get_expert_pool
                pool = get_expert_pool()
                self._meta_controller = get_meta_controller(pool)
            except Exception as e:
                logger.warning(f"Could not load meta controller: {e}")
        return self._meta_controller
    
    def _get_market_edge(self):
        """Lazy load market edge engine."""
        if self._market_edge is None:
            try:
                from polyb0t.models.market_edge import get_market_edge_engine
                self._market_edge = get_market_edge_engine()
            except Exception as e:
                logger.warning(f"Could not load market edge: {e}")
        return self._market_edge
    
    def _get_expert_pool(self):
        """Lazy load expert pool."""
        if self._expert_pool is None:
            try:
                from polyb0t.ml.moe.expert_pool import get_expert_pool
                self._expert_pool = get_expert_pool()
            except Exception as e:
                logger.warning(f"Could not load expert pool: {e}")
        return self._expert_pool

    async def record_outcome(self, outcome: TradeOutcome) -> Dict[str, bool]:
        """Record a trade outcome to all learning systems.
        
        Args:
            outcome: Complete trade outcome record.
            
        Returns:
            Dict of system name -> success status.
        """
        results = {}
        
        logger.info(
            f"Recording outcome: {outcome.market_title[:30]}... "
            f"P&L: {outcome.profit_pct:+.2f}% (${outcome.profit_usd:+.2f})"
        )
        
        # 1. Record to DataCollector (training data)
        try:
            collector = self._get_data_collector()
            if collector:
                collector.record_trade_outcome(
                    token_id=outcome.token_id,
                    profit_pct=outcome.profit_pct,
                    was_traded=True,
                )
                results["data_collector"] = True
        except Exception as e:
            logger.error(f"Failed to record to data collector: {e}")
            results["data_collector"] = False
        
        # 2. Record to MetaController (mixture learning)
        try:
            meta = self._get_meta_controller()
            if meta and outcome.mixture_id:
                meta.record_trade_outcome(
                    mixture_id=outcome.mixture_id,
                    profit_pct=outcome.profit_pct,
                    category=outcome.category,
                )
                results["meta_controller"] = True
        except Exception as e:
            logger.error(f"Failed to record to meta controller: {e}")
            results["meta_controller"] = False
        
        # 3. Record to MarketEdge (edge calculation learning)
        try:
            edge = self._get_market_edge()
            if edge:
                edge.update_from_trade(
                    token_id=outcome.token_id,
                    entry_price=outcome.entry_price,
                    exit_price=outcome.exit_price,
                    pnl_pct=outcome.profit_pct,
                    hold_time_hours=outcome.hold_time_hours,
                    signal_strength=outcome.prediction_confidence,
                )
                results["market_edge"] = True
        except Exception as e:
            logger.error(f"Failed to record to market edge: {e}")
            results["market_edge"] = False
        
        # 4. Update ExpertPool confidence multipliers
        try:
            pool = self._get_expert_pool()
            if pool and outcome.expert_predictions:
                for expert_id, prediction in outcome.expert_predictions.items():
                    expert = pool.experts.get(expert_id)
                    if expert:
                        # Prediction was correct if:
                        # - Predicted profitable (>0.5) and actually profitable
                        # - Predicted unprofitable (<0.5) and actually unprofitable
                        predicted_profitable = prediction > 0.5
                        was_correct = predicted_profitable == outcome.is_profitable

                        # Adjust confidence multiplier
                        adjustment = 0.02 if was_correct else -0.02
                        expert.confidence_multiplier = max(0.3, min(1.0,
                            expert.confidence_multiplier + adjustment
                        ))

                results["expert_pool"] = True
        except Exception as e:
            logger.error(f"Failed to update expert pool: {e}")
            results["expert_pool"] = False

        # Store for debugging
        self._recent_outcomes.append(outcome)
        if len(self._recent_outcomes) > self._max_recent:
            self._recent_outcomes = self._recent_outcomes[-self._max_recent:]
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Outcome recorded to {success_count}/{len(results)} systems")
        
        return results
    
    def record_outcome_sync(self, outcome: TradeOutcome) -> Dict[str, bool]:
        """Synchronous version of record_outcome.
        
        For use when not in an async context.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a task if already in async context
                future = asyncio.ensure_future(self.record_outcome(outcome))
                return {"scheduled": True}
            else:
                return loop.run_until_complete(self.record_outcome(outcome))
        except RuntimeError:
            # No event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.record_outcome(outcome))
            finally:
                loop.close()
    
    def get_recent_outcomes(self, limit: int = 20) -> List[TradeOutcome]:
        """Get recent recorded outcomes.
        
        Args:
            limit: Maximum outcomes to return.
            
        Returns:
            List of recent TradeOutcome objects.
        """
        return self._recent_outcomes[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of recent outcomes."""
        if not self._recent_outcomes:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_profit_pct": 0,
                "total_profit_pct": 0,
            }
        
        wins = sum(1 for o in self._recent_outcomes if o.is_profitable)
        total_profit = sum(o.profit_pct for o in self._recent_outcomes)
        avg_profit = total_profit / len(self._recent_outcomes)
        
        return {
            "total_trades": len(self._recent_outcomes),
            "win_rate": wins / len(self._recent_outcomes),
            "avg_profit_pct": avg_profit,
            "total_profit_pct": total_profit,
            "total_profit_usd": sum(o.profit_usd for o in self._recent_outcomes),
        }


# Singleton instance
_outcome_recorder: Optional[TradeOutcomeRecorder] = None


def get_outcome_recorder() -> TradeOutcomeRecorder:
    """Get or create the outcome recorder singleton."""
    global _outcome_recorder
    if _outcome_recorder is None:
        _outcome_recorder = TradeOutcomeRecorder()
    return _outcome_recorder
