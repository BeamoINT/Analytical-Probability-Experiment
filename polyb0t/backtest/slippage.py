"""Slippage and execution simulation models.

Provides realistic execution simulation including:
- Price impact based on order size
- Spread-based slippage
- Orderbook walk simulation
- Fill probability estimation
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderbookLevel:
    """Single orderbook level."""
    price: float
    size: float


@dataclass
class ExecutionEstimate:
    """Estimated execution details."""
    
    expected_price: float
    slippage_pct: float
    slippage_usd: float
    fill_probability: float
    price_impact: float
    
    # Breakdown
    spread_cost: float
    market_impact: float


class SlippageModel:
    """Model for estimating execution slippage.
    
    Combines multiple slippage sources:
    1. Spread cost (crossing the bid-ask spread)
    2. Market impact (price moves against you)
    3. Orderbook depth (large orders walk the book)
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.001,  # 0.1% base slippage
        impact_factor: float = 0.0001,      # Impact per $1000 traded
        min_slippage_pct: float = 0.0005,   # Minimum 0.05%
        max_slippage_pct: float = 0.02,     # Maximum 2%
    ):
        """Initialize the slippage model.
        
        Args:
            base_slippage_pct: Base slippage for small orders.
            impact_factor: Additional slippage per $1000 traded.
            min_slippage_pct: Minimum slippage floor.
            max_slippage_pct: Maximum slippage cap.
        """
        self.base_slippage_pct = base_slippage_pct
        self.impact_factor = impact_factor
        self.min_slippage_pct = min_slippage_pct
        self.max_slippage_pct = max_slippage_pct
    
    def estimate_slippage(
        self,
        size: float,
        price: float,
        side: str,
        spread: float = 0.02,
        liquidity: Optional[float] = None,
        orderbook: Optional[Dict[str, List[OrderbookLevel]]] = None,
    ) -> float:
        """Estimate slippage for an order.
        
        Args:
            size: Order size in USD.
            price: Current price.
            side: 'BUY' or 'SELL'.
            spread: Current bid-ask spread (as decimal).
            liquidity: Available liquidity (optional).
            orderbook: Full orderbook if available.
            
        Returns:
            Estimated slippage in price units.
        """
        if orderbook:
            return self._estimate_from_orderbook(size, side, orderbook)
        
        # Base slippage: half the spread (crossing to other side)
        spread_cost = spread / 2
        
        # Market impact: increases with order size
        notional = size
        impact = self.impact_factor * (notional / 1000)
        
        # Liquidity adjustment
        if liquidity and liquidity > 0:
            # Less liquid = more slippage
            liquidity_factor = min(2.0, 10000 / max(liquidity, 1000))
            impact *= liquidity_factor
        
        # Total slippage percentage
        total_slippage_pct = spread_cost + impact
        
        # Apply bounds
        total_slippage_pct = max(self.min_slippage_pct, 
                                min(total_slippage_pct, self.max_slippage_pct))
        
        # Convert to price units
        slippage = price * total_slippage_pct
        
        return slippage
    
    def _estimate_from_orderbook(
        self,
        size: float,
        side: str,
        orderbook: Dict[str, List[OrderbookLevel]],
    ) -> float:
        """Estimate slippage by walking the orderbook.
        
        Simulates filling an order by consuming liquidity at each
        price level until the order is filled.
        """
        if side == "BUY":
            levels = orderbook.get("asks", [])
        else:
            levels = orderbook.get("bids", [])
        
        if not levels:
            return self.base_slippage_pct * 100  # Fallback
        
        remaining_size = size
        total_cost = 0.0
        
        for level in levels:
            if remaining_size <= 0:
                break
            
            fill_at_level = min(remaining_size, level.size * level.price)
            total_cost += fill_at_level
            remaining_size -= fill_at_level
        
        if remaining_size > 0:
            # Order too large for book - apply max slippage
            return levels[-1].price * self.max_slippage_pct
        
        # Calculate average fill price
        avg_fill_price = total_cost / size if size > 0 else levels[0].price
        
        # Slippage is difference from best price
        best_price = levels[0].price
        slippage = abs(avg_fill_price - best_price)
        
        return slippage
    
    def estimate_fill_probability(
        self,
        limit_price: float,
        current_price: float,
        side: str,
        spread: float = 0.02,
        volatility: float = 0.01,
    ) -> float:
        """Estimate probability of limit order fill.
        
        Based on distance from current price and volatility.
        
        Args:
            limit_price: Limit order price.
            current_price: Current market price.
            side: 'BUY' or 'SELL'.
            spread: Current spread.
            volatility: Recent price volatility.
            
        Returns:
            Probability of fill (0 to 1).
        """
        if side == "BUY":
            # Buy limit: needs price to fall to limit
            distance = current_price - limit_price
        else:
            # Sell limit: needs price to rise to limit
            distance = limit_price - current_price
        
        if distance <= 0:
            # Already at or past limit price
            return 1.0
        
        # Distance as multiple of volatility
        if volatility > 0:
            vol_distance = distance / (current_price * volatility)
        else:
            vol_distance = distance / (current_price * 0.01)
        
        # Probability decreases with distance
        # Using exponential decay
        probability = math.exp(-vol_distance)
        
        return min(1.0, max(0.0, probability))
    
    def get_execution_estimate(
        self,
        size: float,
        price: float,
        side: str,
        spread: float = 0.02,
        liquidity: Optional[float] = None,
        orderbook: Optional[Dict] = None,
    ) -> ExecutionEstimate:
        """Get detailed execution estimate.
        
        Returns comprehensive execution analysis including
        expected price, slippage breakdown, and fill probability.
        """
        slippage = self.estimate_slippage(
            size=size,
            price=price,
            side=side,
            spread=spread,
            liquidity=liquidity,
            orderbook=orderbook,
        )
        
        # Calculate expected execution price
        if side == "BUY":
            expected_price = price + slippage
        else:
            expected_price = price - slippage
        
        slippage_pct = slippage / price if price > 0 else 0
        slippage_usd = slippage * size / price if price > 0 else 0
        
        # Breakdown
        spread_cost = spread / 2
        market_impact = slippage_pct - spread_cost
        
        # Price impact for others (how much we move the market)
        price_impact = market_impact * 2  # Roughly 2x our slippage
        
        # Fill probability for market order is always 1.0
        fill_probability = 1.0
        
        return ExecutionEstimate(
            expected_price=expected_price,
            slippage_pct=slippage_pct,
            slippage_usd=slippage_usd,
            fill_probability=fill_probability,
            price_impact=price_impact,
            spread_cost=spread_cost,
            market_impact=market_impact,
        )
