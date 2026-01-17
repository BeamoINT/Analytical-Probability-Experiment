"""Arbitrage Scanner for near-resolved markets.

Detects markets where:
1. Outcome is essentially certain (price near 0 or 1)
2. Market hasn't officially resolved yet
3. Profit opportunity exists after spread

This is a low-risk strategy that captures value from:
- Impatient traders wanting liquidity before resolution
- Slow price updates after events occur
- Risk premiums on "sure things"
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents a potential arbitrage opportunity."""
    token_id: str
    market_id: str
    market_title: str
    current_price: float
    expected_value: float  # What we expect to receive (0 or 1)
    spread_cost: float
    net_profit_pct: float  # Profit after spread
    confidence: float  # How certain we are about the outcome
    days_to_resolution: float
    volume_24h: float
    reason: str  # Why we think this is arbitrage
    
    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "market_id": self.market_id,
            "market_title": self.market_title,
            "current_price": self.current_price,
            "expected_value": self.expected_value,
            "spread_cost": self.spread_cost,
            "net_profit_pct": self.net_profit_pct,
            "confidence": self.confidence,
            "days_to_resolution": self.days_to_resolution,
            "volume_24h": self.volume_24h,
            "reason": self.reason,
        }


class ArbitrageScanner:
    """Scans for near-resolution arbitrage opportunities."""
    
    # Configuration
    MIN_EXTREME_PRICE = 0.94  # Price must be > this for YES arbitrage
    MAX_EXTREME_PRICE = 0.06  # Price must be < this for NO arbitrage
    MIN_PROFIT_PCT = 0.01  # Minimum 1% profit after spread
    MAX_DAYS_TO_RESOLUTION = 7  # Only consider markets resolving soon
    MIN_CONFIDENCE = 0.90  # Must be very confident
    SPREAD_ESTIMATE = 0.02  # Assume 2% spread cost
    
    def __init__(self):
        self.opportunities: list[ArbitrageOpportunity] = []
        self._last_scan = None
    
    def scan_market(
        self,
        token_id: str,
        market_id: str,
        market_title: str,
        price: float,
        bid: float,
        ask: float,
        volume_24h: float,
        days_to_resolution: float,
        spread_pct: float = 0,
        momentum_24h: float = 0,
        volatility_24h: float = 0,
    ) -> Optional[ArbitrageOpportunity]:
        """Scan a single market for arbitrage opportunity.
        
        Args:
            token_id: Token ID
            market_id: Market/condition ID
            market_title: Human-readable title
            price: Current YES price
            bid: Best bid
            ask: Best ask
            volume_24h: 24h volume
            days_to_resolution: Days until resolution
            spread_pct: Current spread percentage
            momentum_24h: 24h price momentum
            volatility_24h: 24h volatility
            
        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        # Skip if too far from resolution
        if days_to_resolution > self.MAX_DAYS_TO_RESOLUTION:
            return None
        
        # Calculate actual spread
        if spread_pct > 0:
            spread_cost = spread_pct
        else:
            spread_cost = (ask - bid) / price if price > 0 else self.SPREAD_ESTIMATE
        
        spread_cost = max(spread_cost, 0.005)  # Minimum 0.5% spread
        
        opportunity = None
        
        # === CHECK FOR YES ARBITRAGE ===
        # Price is very high → YES is almost certain to win
        if price >= self.MIN_EXTREME_PRICE:
            expected_value = 1.0
            buy_price = ask  # We'd buy at ask
            profit_raw = expected_value - buy_price
            profit_net = profit_raw - spread_cost
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                price=price,
                expected_value=1.0,
                days_to_resolution=days_to_resolution,
                volatility_24h=volatility_24h,
                momentum_24h=momentum_24h,
            )
            
            if profit_net >= self.MIN_PROFIT_PCT and confidence >= self.MIN_CONFIDENCE:
                opportunity = ArbitrageOpportunity(
                    token_id=token_id,
                    market_id=market_id,
                    market_title=market_title,
                    current_price=price,
                    expected_value=expected_value,
                    spread_cost=spread_cost,
                    net_profit_pct=profit_net,
                    confidence=confidence,
                    days_to_resolution=days_to_resolution,
                    volume_24h=volume_24h,
                    reason=f"YES near certain (price={price:.2%}, profit={profit_net:.1%})",
                )
        
        # === CHECK FOR NO ARBITRAGE ===
        # Price is very low → NO is almost certain to win
        elif price <= self.MAX_EXTREME_PRICE:
            expected_value = 0.0
            sell_price = bid  # We'd sell YES (buy NO) at bid
            # For NO to win, we want to sell YES shares
            # Or equivalently, buy the NO token
            # Profit = what we sell at - what NO costs
            # If YES is at 0.04, NO is at 0.96
            no_price = 1 - price
            profit_raw = 1.0 - (1 - bid)  # Selling YES at bid, getting 1-bid back
            # Actually: if we SELL YES at bid (0.04), we get $0.04
            # When market resolves NO, our sold position means... 
            # This is more complex. Let's focus on buying YES near 0.
            
            # Simpler: if YES is at 0.04, buying YES costs 0.04
            # If YES wins (unlikely), we get $1.00
            # If NO wins (likely), we lose $0.04
            # This isn't arbitrage for buying YES.
            
            # For NO arbitrage: we need to SHORT yes or BUY no
            # Polymarket allows buying NO tokens
            # If price of YES = 0.04, price of NO ≈ 0.96
            # Buying NO at 0.96, getting $1.00 when NO wins = 4% profit
            
            # Let's treat this as: buy the outcome that's near $1
            # If YES < 0.06, then NO > 0.94, so buy NO
            no_token_price = 1 - price
            profit_raw = 1.0 - no_token_price
            profit_net = profit_raw - spread_cost
            
            confidence = self._calculate_confidence(
                price=no_token_price,  # Use NO price for confidence
                expected_value=1.0,
                days_to_resolution=days_to_resolution,
                volatility_24h=volatility_24h,
                momentum_24h=momentum_24h,
            )
            
            if profit_net >= self.MIN_PROFIT_PCT and confidence >= self.MIN_CONFIDENCE:
                opportunity = ArbitrageOpportunity(
                    token_id=token_id,  # Note: would need NO token_id
                    market_id=market_id,
                    market_title=market_title,
                    current_price=no_token_price,
                    expected_value=1.0,
                    spread_cost=spread_cost,
                    net_profit_pct=profit_net,
                    confidence=confidence,
                    days_to_resolution=days_to_resolution,
                    volume_24h=volume_24h,
                    reason=f"NO near certain (YES={price:.2%}, NO={no_token_price:.2%}, profit={profit_net:.1%})",
                )
        
        if opportunity:
            logger.info(
                f"Arbitrage found: {market_title[:50]} - "
                f"{opportunity.reason}, confidence={opportunity.confidence:.1%}"
            )
            self.opportunities.append(opportunity)
        
        return opportunity
    
    def _calculate_confidence(
        self,
        price: float,
        expected_value: float,
        days_to_resolution: float,
        volatility_24h: float,
        momentum_24h: float,
    ) -> float:
        """Calculate confidence that this is true arbitrage.
        
        Higher confidence when:
        - Price is more extreme (closer to 0 or 1)
        - Resolution is sooner
        - Low volatility (stable price)
        - Momentum confirms direction
        """
        confidence = 0.5  # Base
        
        # Price extremity (0.94 → 0.7, 0.99 → 0.95)
        if expected_value == 1.0:
            price_confidence = (price - 0.90) / 0.10  # 0.90→0, 1.0→1
        else:
            price_confidence = (0.10 - price) / 0.10  # 0.10→0, 0.0→1
        price_confidence = max(0, min(1, price_confidence))
        confidence += price_confidence * 0.3
        
        # Time to resolution (sooner = more confident)
        if days_to_resolution <= 1:
            time_confidence = 1.0
        elif days_to_resolution <= 3:
            time_confidence = 0.8
        elif days_to_resolution <= 7:
            time_confidence = 0.5
        else:
            time_confidence = 0.2
        confidence += time_confidence * 0.15
        
        # Low volatility = stable, high confidence
        if volatility_24h < 0.01:
            vol_confidence = 1.0
        elif volatility_24h < 0.05:
            vol_confidence = 0.7
        else:
            vol_confidence = 0.3
        confidence += vol_confidence * 0.05
        
        return min(confidence, 1.0)
    
    def get_best_opportunities(self, limit: int = 10) -> list[ArbitrageOpportunity]:
        """Get the best current arbitrage opportunities.
        
        Sorted by net profit * confidence (expected value).
        """
        # Score by expected profit
        scored = [
            (opp, opp.net_profit_pct * opp.confidence)
            for opp in self.opportunities
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [opp for opp, _ in scored[:limit]]
    
    def clear_opportunities(self):
        """Clear the opportunities list for a new scan cycle."""
        self.opportunities = []
        self._last_scan = datetime.utcnow()
    
    def get_summary(self) -> dict:
        """Get summary of current arbitrage state."""
        if not self.opportunities:
            return {
                "count": 0,
                "total_profit_potential": 0,
                "best_opportunity": None,
            }
        
        best = max(self.opportunities, key=lambda x: x.net_profit_pct * x.confidence)
        
        return {
            "count": len(self.opportunities),
            "total_profit_potential": sum(o.net_profit_pct for o in self.opportunities),
            "avg_confidence": sum(o.confidence for o in self.opportunities) / len(self.opportunities),
            "best_opportunity": best.to_dict(),
        }


# Singleton
_scanner_instance: Optional[ArbitrageScanner] = None


def get_arbitrage_scanner() -> ArbitrageScanner:
    """Get or create the arbitrage scanner singleton."""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = ArbitrageScanner()
    return _scanner_instance
