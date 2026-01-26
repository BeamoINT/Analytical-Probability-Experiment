"""Market correlation tracker for identifying related markets.

Tracks price correlations between related markets (same event/category)
to identify momentum signals and divergences.

Features tracked:
- Number of correlated markets
- Average price of related outcomes
- Momentum agreement across related markets
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)

# Minimum correlation for markets to be considered related
MIN_CORRELATION_THRESHOLD = 0.3

# Time window for correlation calculation
CORRELATION_WINDOW_HOURS = 24


@dataclass
class MarketPricePoint:
    """A price observation for correlation calculation."""
    timestamp: datetime
    price: float


@dataclass
class MarketCorrelationInfo:
    """Correlation information for a market."""
    
    market_id: str
    asset_id: str
    
    # Related markets
    correlated_market_count: int = 0
    related_market_ids: List[str] = field(default_factory=list)
    
    # Price correlations
    correlated_avg_price: float = 0.0  # Avg price of related outcomes
    correlated_momentum: float = 0.0  # -1 to +1, agreement with related markets
    
    # Correlation strength (average r^2 with related markets)
    avg_correlation_strength: float = 0.0
    
    # Category/group info
    category: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Last update
    last_update: datetime = field(default_factory=datetime.utcnow)


class CorrelationTracker:
    """Service to track correlations between related markets.
    
    Groups markets by category/tags and calculates price correlations
    to identify momentum signals across related markets.
    """
    
    def __init__(self):
        """Initialize correlation tracker."""
        self.settings = get_settings()
        
        # Price history for correlation calculation
        # asset_id -> list of MarketPricePoint
        self._price_history: Dict[str, List[MarketPricePoint]] = defaultdict(list)
        
        # Market groupings
        # category/tag -> set of asset_ids
        self._category_markets: Dict[str, Set[str]] = defaultdict(set)
        self._tag_markets: Dict[str, Set[str]] = defaultdict(set)
        
        # Market metadata
        # asset_id -> (market_id, category, tags)
        self._market_metadata: Dict[str, Tuple[str, str, List[str]]] = {}
        
        # Cached correlations
        # asset_id -> MarketCorrelationInfo
        self._correlations: Dict[str, MarketCorrelationInfo] = {}
        
        # Max price history points per market
        self._max_history_points = 500
        
        logger.info("Correlation tracker initialized")
    
    def register_market(
        self,
        asset_id: str,
        market_id: str,
        category: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a market for correlation tracking.
        
        Args:
            asset_id: Token ID.
            market_id: Market condition ID.
            category: Market category (e.g., "politics_us", "sports").
            tags: List of tags for grouping.
        """
        tags = tags or []
        
        # Store metadata
        self._market_metadata[asset_id] = (market_id, category, tags)
        
        # Add to category group
        if category:
            self._category_markets[category].add(asset_id)
        
        # Add to tag groups
        for tag in tags:
            self._tag_markets[tag].add(asset_id)
    
    def record_price(
        self,
        asset_id: str,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a price observation for correlation calculation.
        
        Args:
            asset_id: Token ID.
            price: Current price.
            timestamp: Observation timestamp.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        point = MarketPricePoint(timestamp=timestamp, price=price)
        self._price_history[asset_id].append(point)
        
        # Trim old history
        if len(self._price_history[asset_id]) > self._max_history_points:
            self._price_history[asset_id] = self._price_history[asset_id][-self._max_history_points:]
    
    def get_related_markets(self, asset_id: str) -> List[str]:
        """Get list of related markets for an asset.
        
        Args:
            asset_id: Token ID.
            
        Returns:
            List of related asset IDs.
        """
        related = set()
        
        metadata = self._market_metadata.get(asset_id)
        if not metadata:
            return []
        
        _, category, tags = metadata
        
        # Get markets in same category
        if category:
            related.update(self._category_markets.get(category, set()))
        
        # Get markets with same tags
        for tag in tags:
            related.update(self._tag_markets.get(tag, set()))
        
        # Remove self
        related.discard(asset_id)
        
        return list(related)
    
    def calculate_correlation(
        self,
        asset_id_1: str,
        asset_id_2: str,
        window_hours: int = CORRELATION_WINDOW_HOURS,
    ) -> Optional[float]:
        """Calculate price correlation between two assets.
        
        Args:
            asset_id_1: First asset ID.
            asset_id_2: Second asset ID.
            window_hours: Time window for correlation.
            
        Returns:
            Pearson correlation coefficient (-1 to +1), or None if insufficient data.
        """
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        
        # Get price histories within window
        prices_1 = [
            p for p in self._price_history.get(asset_id_1, [])
            if p.timestamp >= cutoff
        ]
        prices_2 = [
            p for p in self._price_history.get(asset_id_2, [])
            if p.timestamp >= cutoff
        ]
        
        if len(prices_1) < 5 or len(prices_2) < 5:
            return None
        
        # Align by timestamp (simple nearest-neighbor matching)
        aligned_1 = []
        aligned_2 = []
        
        for p1 in prices_1:
            # Find closest price_2 point
            closest = min(
                prices_2,
                key=lambda p2: abs((p2.timestamp - p1.timestamp).total_seconds()),
            )
            # Only include if within 30 minutes
            if abs((closest.timestamp - p1.timestamp).total_seconds()) < 1800:
                aligned_1.append(p1.price)
                aligned_2.append(closest.price)
        
        if len(aligned_1) < 5:
            return None
        
        # Calculate Pearson correlation
        n = len(aligned_1)
        sum_1 = sum(aligned_1)
        sum_2 = sum(aligned_2)
        sum_sq_1 = sum(x**2 for x in aligned_1)
        sum_sq_2 = sum(x**2 for x in aligned_2)
        sum_prod = sum(x * y for x, y in zip(aligned_1, aligned_2))
        
        numerator = n * sum_prod - sum_1 * sum_2
        denominator = ((n * sum_sq_1 - sum_1**2) * (n * sum_sq_2 - sum_2**2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def update_correlations(self, asset_id: str) -> MarketCorrelationInfo:
        """Update correlation metrics for a market.
        
        Args:
            asset_id: Token ID.
            
        Returns:
            Updated MarketCorrelationInfo.
        """
        metadata = self._market_metadata.get(asset_id)
        if not metadata:
            return MarketCorrelationInfo(market_id="", asset_id=asset_id)
        
        market_id, category, tags = metadata
        
        # Get related markets
        related = self.get_related_markets(asset_id)
        
        # Calculate correlations with related markets
        correlations = []
        related_prices = []
        related_momentums = []
        
        my_prices = self._price_history.get(asset_id, [])
        my_momentum = self._calculate_momentum(my_prices)
        
        for related_id in related:
            corr = self.calculate_correlation(asset_id, related_id)
            if corr is not None and abs(corr) >= MIN_CORRELATION_THRESHOLD:
                correlations.append(corr)
                
                # Get current price of related market
                related_history = self._price_history.get(related_id, [])
                if related_history:
                    related_prices.append(related_history[-1].price)
                    related_momentums.append(self._calculate_momentum(related_history))
        
        # Calculate metrics
        correlated_count = len(correlations)
        avg_corr_strength = sum(abs(c) for c in correlations) / len(correlations) if correlations else 0
        avg_related_price = sum(related_prices) / len(related_prices) if related_prices else 0
        
        # Momentum agreement: do related markets agree with our momentum?
        momentum_agreement = 0.0
        if related_momentums and my_momentum != 0:
            # Count how many related markets have same momentum direction
            same_direction = sum(
                1 for rm in related_momentums
                if (rm > 0 and my_momentum > 0) or (rm < 0 and my_momentum < 0)
            )
            momentum_agreement = (same_direction / len(related_momentums)) * 2 - 1  # Scale to -1 to +1
        
        info = MarketCorrelationInfo(
            market_id=market_id,
            asset_id=asset_id,
            correlated_market_count=correlated_count,
            related_market_ids=related[:10],  # Top 10 related
            correlated_avg_price=avg_related_price,
            correlated_momentum=momentum_agreement,
            avg_correlation_strength=avg_corr_strength,
            category=category,
            tags=tags,
            last_update=datetime.utcnow(),
        )
        
        self._correlations[asset_id] = info
        return info
    
    def _calculate_momentum(self, prices: List[MarketPricePoint]) -> float:
        """Calculate simple momentum from price history.
        
        Returns:
            Price change over recent history, normalized.
        """
        if len(prices) < 2:
            return 0.0
        
        # Compare current price to average of older prices
        recent = prices[-1].price
        older = sum(p.price for p in prices[:-10]) / max(1, len(prices) - 10) if len(prices) > 10 else prices[0].price
        
        if older == 0:
            return 0.0
        
        return (recent - older) / older
    
    def get_correlation_info(self, asset_id: str) -> Optional[MarketCorrelationInfo]:
        """Get cached correlation info for a market.
        
        Args:
            asset_id: Token ID.
            
        Returns:
            MarketCorrelationInfo or None.
        """
        return self._correlations.get(asset_id)
    
    def get_correlation_features(self, asset_id: str) -> Dict[str, float]:
        """Get correlation features for ML training.
        
        Args:
            asset_id: Token ID.
            
        Returns:
            Dict of feature name -> value.
        """
        info = self._correlations.get(asset_id)
        
        if not info:
            return {
                "correlated_market_count": 0,
                "correlated_avg_price": 0.0,
                "correlated_momentum": 0.0,
                "avg_correlation_strength": 0.0,
            }
        
        return {
            "correlated_market_count": info.correlated_market_count,
            "correlated_avg_price": info.correlated_avg_price,
            "correlated_momentum": info.correlated_momentum,
            "avg_correlation_strength": info.avg_correlation_strength,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get correlation tracker statistics."""
        total_markets = len(self._market_metadata)
        total_correlations = len(self._correlations)
        total_price_points = sum(len(h) for h in self._price_history.values())
        
        avg_related = 0
        if self._correlations:
            avg_related = sum(
                c.correlated_market_count for c in self._correlations.values()
            ) / len(self._correlations)
        
        return {
            "tracked_markets": total_markets,
            "correlations_calculated": total_correlations,
            "total_price_points": total_price_points,
            "categories": list(self._category_markets.keys()),
            "avg_related_markets": avg_related,
        }


# Singleton instance
_correlation_tracker: Optional[CorrelationTracker] = None


def get_correlation_tracker() -> CorrelationTracker:
    """Get or create the correlation tracker singleton."""
    global _correlation_tracker
    if _correlation_tracker is None:
        _correlation_tracker = CorrelationTracker()
    return _correlation_tracker
