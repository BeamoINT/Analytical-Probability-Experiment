"""Backtest strategy interfaces and implementations.

Provides:
- BacktestStrategy protocol for custom strategies
- MoEBacktestStrategy for testing the MoE system
- Simple baseline strategies for comparison
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Trading signal from a strategy."""
    
    side: str  # BUY or SELL
    size: float  # Position size in USD
    confidence: float  # Signal confidence (0-1)
    
    # Optional metadata
    expert_id: str = ""
    reason: str = ""
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None


class BacktestStrategy(Protocol):
    """Protocol for backtest strategies."""
    
    @property
    def name(self) -> str:
        """Strategy name."""
        ...
    
    def generate_signal(self, snapshot: Dict[str, Any]) -> Optional[Signal]:
        """Generate a trading signal from market snapshot.
        
        Args:
            snapshot: Market data snapshot with features.
            
        Returns:
            Signal to trade, or None to skip.
        """
        ...
    
    def should_exit(
        self,
        snapshot: Dict[str, Any],
        position: Any,
    ) -> bool:
        """Check if an existing position should be closed.
        
        Args:
            snapshot: Current market data.
            position: Current position details.
            
        Returns:
            True if position should be closed.
        """
        ...


class MoEBacktestStrategy:
    """Strategy that uses the MoE system for backtesting.
    
    This allows backtesting the actual production AI system
    against historical data.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        position_size_pct: float = 0.05,
        max_position_size: float = 1000.0,
    ):
        """Initialize MoE backtest strategy.
        
        Args:
            min_confidence: Minimum confidence to trade.
            position_size_pct: Position size as % of available capital.
            max_position_size: Maximum position in USD.
        """
        self.min_confidence = min_confidence
        self.position_size_pct = position_size_pct
        self.max_position_size = max_position_size
        
        # Lazy load MoE components
        self._meta_controller = None
        self._expert_pool = None
    
    @property
    def name(self) -> str:
        return "MoE_Strategy"
    
    def _get_meta_controller(self):
        """Lazy load meta controller."""
        if self._meta_controller is None:
            try:
                from polyb0t.ml.moe.meta_controller import get_meta_controller
                from polyb0t.ml.moe.expert_pool import get_expert_pool
                self._expert_pool = get_expert_pool()
                self._meta_controller = get_meta_controller(self._expert_pool)
            except Exception as e:
                logger.warning(f"Could not load MoE: {e}")
        return self._meta_controller
    
    def generate_signal(self, snapshot: Dict[str, Any]) -> Optional[Signal]:
        """Generate signal using MoE system."""
        meta = self._get_meta_controller()
        if meta is None:
            return None
        
        # Convert snapshot to features dict
        features = snapshot.get("features", snapshot)
        if isinstance(features, str):
            import json
            try:
                features = json.loads(features)
            except:
                features = snapshot
        
        try:
            prediction, confidence, metadata = meta.predict_with_mixture(features)
            
            if confidence < self.min_confidence:
                return None
            
            # Determine side based on prediction
            # prediction > 0.5 = bullish, < 0.5 = bearish
            if prediction > 0.55:
                side = "BUY"
            elif prediction < 0.45:
                side = "SELL"
            else:
                return None  # No clear signal
            
            # Calculate position size
            price = features.get("price", 1.0)
            size = min(self.max_position_size, 
                      self.max_position_size * confidence * self.position_size_pct * 20)
            
            # Get primary expert from metadata
            expert_id = metadata.get("primary_expert", "")
            
            return Signal(
                side=side,
                size=size,
                confidence=confidence,
                expert_id=expert_id,
                reason=f"MoE prediction: {prediction:.3f}",
            )
            
        except Exception as e:
            logger.debug(f"MoE prediction error: {e}")
            return None
    
    def should_exit(
        self,
        snapshot: Dict[str, Any],
        position: Any,
    ) -> bool:
        """Check exit conditions."""
        features = snapshot.get("features", snapshot)
        if isinstance(features, str):
            import json
            try:
                features = json.loads(features)
            except:
                features = snapshot
        
        price = features.get("price", position.current_price)
        
        # Calculate unrealized P&L
        if position.side == "BUY":
            pnl_pct = (price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - price) / position.entry_price
        
        # Exit on significant profit or loss
        if pnl_pct > 0.10:  # +10% take profit
            return True
        if pnl_pct < -0.05:  # -5% stop loss
            return True
        
        # Exit if position is old (24+ hours)
        if hasattr(position, 'entry_time') and position.entry_time:
            from datetime import datetime, timedelta
            ts = snapshot.get("timestamp") or snapshot.get("created_at")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    if ts - position.entry_time > timedelta(hours=48):
                        return True
                except:
                    pass
        
        return False


class MomentumStrategy:
    """Simple momentum-based strategy for comparison."""
    
    def __init__(
        self,
        momentum_threshold: float = 0.05,
        position_size: float = 500.0,
    ):
        self.momentum_threshold = momentum_threshold
        self.position_size = position_size
    
    @property
    def name(self) -> str:
        return "Momentum_Strategy"
    
    def generate_signal(self, snapshot: Dict[str, Any]) -> Optional[Signal]:
        """Generate signal based on 24h momentum."""
        features = snapshot.get("features", snapshot)
        if isinstance(features, str):
            import json
            try:
                features = json.loads(features)
            except:
                features = snapshot
        
        momentum = features.get("momentum_24h", 0)
        
        if momentum > self.momentum_threshold:
            return Signal(
                side="BUY",
                size=self.position_size,
                confidence=min(1.0, abs(momentum) / 0.1),
                reason=f"Positive momentum: {momentum:.2%}",
            )
        elif momentum < -self.momentum_threshold:
            return Signal(
                side="SELL",
                size=self.position_size,
                confidence=min(1.0, abs(momentum) / 0.1),
                reason=f"Negative momentum: {momentum:.2%}",
            )
        
        return None
    
    def should_exit(self, snapshot: Dict[str, Any], position: Any) -> bool:
        """Exit when momentum reverses."""
        features = snapshot.get("features", snapshot)
        if isinstance(features, str):
            import json
            try:
                features = json.loads(features)
            except:
                features = snapshot
        
        momentum = features.get("momentum_24h", 0)
        
        if position.side == "BUY" and momentum < 0:
            return True
        if position.side == "SELL" and momentum > 0:
            return True
        
        return False


class MeanReversionStrategy:
    """Mean reversion strategy for comparison."""
    
    def __init__(
        self,
        oversold_threshold: float = 0.20,
        overbought_threshold: float = 0.80,
        position_size: float = 500.0,
    ):
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.position_size = position_size
    
    @property
    def name(self) -> str:
        return "MeanReversion_Strategy"
    
    def generate_signal(self, snapshot: Dict[str, Any]) -> Optional[Signal]:
        """Buy oversold, sell overbought."""
        features = snapshot.get("features", snapshot)
        if isinstance(features, str):
            import json
            try:
                features = json.loads(features)
            except:
                features = snapshot
        
        price = features.get("price", 0.5)
        
        if price < self.oversold_threshold:
            return Signal(
                side="BUY",
                size=self.position_size,
                confidence=1.0 - price / self.oversold_threshold,
                reason=f"Oversold at {price:.2%}",
            )
        elif price > self.overbought_threshold:
            return Signal(
                side="SELL",
                size=self.position_size,
                confidence=(price - self.overbought_threshold) / (1 - self.overbought_threshold),
                reason=f"Overbought at {price:.2%}",
            )
        
        return None
    
    def should_exit(self, snapshot: Dict[str, Any], position: Any) -> bool:
        """Exit when price returns toward mean."""
        features = snapshot.get("features", snapshot)
        if isinstance(features, str):
            import json
            try:
                features = json.loads(features)
            except:
                features = snapshot
        
        price = features.get("price", 0.5)
        
        # Exit when price returns to middle range
        if 0.40 < price < 0.60:
            return True
        
        return False
