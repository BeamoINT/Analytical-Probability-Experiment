"""Whale tracking service for detecting large market-moving trades.

Whale criteria:
- Trade size >= 5% of market's total volume
- OR Trade size >= $25,000

Tracks:
- Whale activity count per market
- Net whale direction (buying vs selling pressure)
- Largest trade sizes
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)

# Whale detection thresholds
WHALE_VOLUME_PCT = 0.05  # 5% of market volume
WHALE_USD_THRESHOLD = 25000.0  # $25,000


@dataclass
class WhaleEvent:
    """A detected whale trade."""
    
    asset_id: str
    market_id: str
    price: float
    size: float
    side: str  # BUY or SELL
    timestamp: datetime
    
    # Whale classification
    value_usd: float
    pct_of_volume: Optional[float] = None
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side.upper() == "BUY"
    
    @property
    def direction(self) -> int:
        """Get direction: +1 for buy, -1 for sell."""
        return 1 if self.is_buy else -1


@dataclass
class MarketWhaleMetrics:
    """Aggregated whale metrics for a single market."""
    
    market_id: str
    asset_id: str
    
    # Time-windowed metrics (last hour)
    whale_count_1h: int = 0
    whale_buy_volume_1h: float = 0.0
    whale_sell_volume_1h: float = 0.0
    
    # Time-windowed metrics (last 24 hours)
    whale_count_24h: int = 0
    whale_buy_volume_24h: float = 0.0
    whale_sell_volume_24h: float = 0.0
    
    # All-time largest trade
    largest_trade_usd: float = 0.0
    largest_trade_timestamp: Optional[datetime] = None
    
    # Last update
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def net_whale_direction_1h(self) -> float:
        """Net whale buying pressure in last hour (-1 to +1).
        
        +1 = all whale volume is buying
        -1 = all whale volume is selling
        0 = balanced or no activity
        """
        total = self.whale_buy_volume_1h + self.whale_sell_volume_1h
        if total == 0:
            return 0.0
        return (self.whale_buy_volume_1h - self.whale_sell_volume_1h) / total
    
    @property
    def net_whale_direction_24h(self) -> float:
        """Net whale buying pressure in last 24 hours (-1 to +1)."""
        total = self.whale_buy_volume_24h + self.whale_sell_volume_24h
        if total == 0:
            return 0.0
        return (self.whale_buy_volume_24h - self.whale_sell_volume_24h) / total


class WhaleTracker:
    """Service to track whale trades across markets.
    
    Integrates with WebSocket to receive real-time trade events and
    detect whale activity based on trade size thresholds.
    """
    
    def __init__(self):
        """Initialize whale tracker."""
        self.settings = get_settings()
        
        # Market volume cache (used to calculate % of volume)
        # market_id -> total 24h volume in USD
        self._market_volumes: Dict[str, float] = {}
        
        # Recent whale events (for time-windowed analysis)
        # asset_id -> list of WhaleEvent
        self._whale_events: Dict[str, List[WhaleEvent]] = defaultdict(list)
        
        # Aggregated metrics per market
        # asset_id -> MarketWhaleMetrics
        self._metrics: Dict[str, MarketWhaleMetrics] = {}
        
        # Maximum events to keep per market
        self._max_events_per_market = 500
        
        # Callbacks for whale detection
        self._on_whale_callbacks: List[Callable[[WhaleEvent], None]] = []
        
        logger.info(
            f"Whale tracker initialized (thresholds: {WHALE_VOLUME_PCT*100}% of volume OR ${WHALE_USD_THRESHOLD:,.0f})"
        )
    
    def set_market_volume(self, market_id: str, volume_24h_usd: float) -> None:
        """Update the 24h volume for a market.
        
        Args:
            market_id: Market condition ID.
            volume_24h_usd: Total 24h volume in USD.
        """
        self._market_volumes[market_id] = volume_24h_usd
    
    def process_trade(
        self,
        asset_id: str,
        market_id: str,
        price: float,
        size: float,
        side: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[WhaleEvent]:
        """Process a trade and check if it's a whale trade.
        
        Args:
            asset_id: Token ID.
            market_id: Market condition ID.
            price: Trade price.
            size: Trade size (number of shares).
            side: BUY or SELL.
            timestamp: Trade timestamp.
            
        Returns:
            WhaleEvent if this is a whale trade, None otherwise.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Calculate trade value
        value_usd = price * size
        
        # Get market volume for percentage calculation
        market_volume = self._market_volumes.get(market_id, 0)
        pct_of_volume = None
        
        if market_volume > 0:
            pct_of_volume = value_usd / market_volume
        
        # Check whale criteria
        is_whale = False
        
        if value_usd >= WHALE_USD_THRESHOLD:
            is_whale = True
            logger.debug(f"Whale detected (${value_usd:,.0f} >= ${WHALE_USD_THRESHOLD:,.0f})")
        elif pct_of_volume and pct_of_volume >= WHALE_VOLUME_PCT:
            is_whale = True
            logger.debug(f"Whale detected ({pct_of_volume*100:.1f}% >= {WHALE_VOLUME_PCT*100}% of volume)")
        
        if not is_whale:
            return None
        
        # Create whale event
        event = WhaleEvent(
            asset_id=asset_id,
            market_id=market_id,
            price=price,
            size=size,
            side=side.upper(),
            timestamp=timestamp,
            value_usd=value_usd,
            pct_of_volume=pct_of_volume,
        )
        
        # Store event
        self._whale_events[asset_id].append(event)
        
        # Trim old events
        if len(self._whale_events[asset_id]) > self._max_events_per_market:
            self._whale_events[asset_id] = self._whale_events[asset_id][-self._max_events_per_market:]
        
        # Update metrics
        self._update_metrics(asset_id, market_id)
        
        # Log whale detection
        logger.info(
            f"WHALE: {side} ${value_usd:,.0f} ({size:,.0f} shares @ ${price:.3f}) "
            f"on {asset_id[:16]}... "
            f"[{pct_of_volume*100:.1f}% of volume]" if pct_of_volume else ""
        )
        
        # Notify callbacks
        for callback in self._on_whale_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Whale callback error: {e}")
        
        return event
    
    def _update_metrics(self, asset_id: str, market_id: str) -> None:
        """Update aggregated metrics for a market."""
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        twenty_four_hours_ago = now - timedelta(hours=24)
        
        events = self._whale_events.get(asset_id, [])
        
        # Calculate time-windowed metrics
        count_1h = 0
        buy_vol_1h = 0.0
        sell_vol_1h = 0.0
        count_24h = 0
        buy_vol_24h = 0.0
        sell_vol_24h = 0.0
        largest_trade = 0.0
        largest_timestamp = None
        
        for event in events:
            # 24h window
            if event.timestamp >= twenty_four_hours_ago:
                count_24h += 1
                if event.is_buy:
                    buy_vol_24h += event.value_usd
                else:
                    sell_vol_24h += event.value_usd
                
                # 1h window
                if event.timestamp >= one_hour_ago:
                    count_1h += 1
                    if event.is_buy:
                        buy_vol_1h += event.value_usd
                    else:
                        sell_vol_1h += event.value_usd
            
            # Track largest trade
            if event.value_usd > largest_trade:
                largest_trade = event.value_usd
                largest_timestamp = event.timestamp
        
        # Update metrics object
        self._metrics[asset_id] = MarketWhaleMetrics(
            market_id=market_id,
            asset_id=asset_id,
            whale_count_1h=count_1h,
            whale_buy_volume_1h=buy_vol_1h,
            whale_sell_volume_1h=sell_vol_1h,
            whale_count_24h=count_24h,
            whale_buy_volume_24h=buy_vol_24h,
            whale_sell_volume_24h=sell_vol_24h,
            largest_trade_usd=largest_trade,
            largest_trade_timestamp=largest_timestamp,
            last_update=now,
        )
    
    def get_metrics(self, asset_id: str) -> Optional[MarketWhaleMetrics]:
        """Get whale metrics for a market.
        
        Args:
            asset_id: Token ID.
            
        Returns:
            MarketWhaleMetrics or None if no data.
        """
        return self._metrics.get(asset_id)
    
    def get_whale_features(self, asset_id: str) -> Dict[str, float]:
        """Get whale features for ML training.
        
        Args:
            asset_id: Token ID.
            
        Returns:
            Dict of feature name -> value.
        """
        metrics = self._metrics.get(asset_id)
        
        if not metrics:
            return {
                "whale_activity_1h": 0,
                "whale_net_direction_1h": 0.0,
                "whale_activity_24h": 0,
                "whale_net_direction_24h": 0.0,
                "largest_trade_24h": 0.0,
            }
        
        return {
            "whale_activity_1h": metrics.whale_count_1h,
            "whale_net_direction_1h": metrics.net_whale_direction_1h,
            "whale_activity_24h": metrics.whale_count_24h,
            "whale_net_direction_24h": metrics.net_whale_direction_24h,
            "largest_trade_24h": metrics.largest_trade_usd,
        }
    
    def get_recent_whales(
        self,
        asset_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[WhaleEvent]:
        """Get recent whale events.
        
        Args:
            asset_id: Optional filter by asset.
            limit: Maximum events to return.
            
        Returns:
            List of recent whale events.
        """
        if asset_id:
            events = self._whale_events.get(asset_id, [])
        else:
            # Combine all events
            events = []
            for asset_events in self._whale_events.values():
                events.extend(asset_events)
            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return events[:limit]
    
    def on_whale(self, callback: Callable[[WhaleEvent], None]) -> None:
        """Register callback for whale detection events."""
        self._on_whale_callbacks.append(callback)
    
    def connect_to_websocket(self, ws_client) -> None:
        """Connect to WebSocket client to receive trade events.
        
        Args:
            ws_client: PolymarketWebSocket instance.
        """
        def on_trade(trade):
            """Handle trade from WebSocket."""
            self.process_trade(
                asset_id=trade.asset_id,
                market_id=trade.market,
                price=trade.price,
                size=trade.size,
                side=trade.side,
                timestamp=trade.timestamp,
            )
        
        ws_client.on_trade(on_trade)
        logger.info("Whale tracker connected to WebSocket")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get whale tracker statistics."""
        total_whales = sum(len(events) for events in self._whale_events.values())
        markets_with_whales = len(self._whale_events)
        
        # Calculate total whale volume
        total_buy_vol = sum(m.whale_buy_volume_24h for m in self._metrics.values())
        total_sell_vol = sum(m.whale_sell_volume_24h for m in self._metrics.values())
        
        return {
            "total_whale_events": total_whales,
            "markets_with_whales": markets_with_whales,
            "total_buy_volume_24h": total_buy_vol,
            "total_sell_volume_24h": total_sell_vol,
            "net_whale_direction": (
                (total_buy_vol - total_sell_vol) / (total_buy_vol + total_sell_vol)
                if (total_buy_vol + total_sell_vol) > 0 else 0
            ),
        }


# Singleton instance
_whale_tracker: Optional[WhaleTracker] = None


def get_whale_tracker() -> WhaleTracker:
    """Get or create the whale tracker singleton."""
    global _whale_tracker
    if _whale_tracker is None:
        _whale_tracker = WhaleTracker()
    return _whale_tracker
