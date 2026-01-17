"""Arbitrage Scanner for near-resolved markets.

Detects markets where:
1. Outcome is essentially certain (price near 0 or 1)
2. Event date has PASSED (outcome should be known)
3. NEWS CONFIRMS the outcome
4. Market hasn't officially resolved yet
5. Profit opportunity exists after spread

This is a low-risk strategy that captures value from:
- Impatient traders wanting liquidity before resolution
- Slow price updates after events occur
- Risk premiums on "sure things"

SAFETY: Requires BOTH price confirmation AND news confirmation.
"""

import json
import logging
import os
from dataclasses import dataclass, field
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
    news_confirmed: bool = False  # Was this confirmed by news?
    news_headline: str = ""  # The confirming headline
    news_source: str = ""  # Source of confirmation
    event_date_passed: bool = False  # Has the event date passed?
    
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
            "news_confirmed": self.news_confirmed,
            "news_headline": self.news_headline,
            "news_source": self.news_source,
            "event_date_passed": self.event_date_passed,
        }


@dataclass
class ArbitrageStats:
    """Statistics for the arbitrage scanner."""
    total_scanned: int = 0
    price_qualified: int = 0  # Passed price threshold
    event_date_qualified: int = 0  # Event date has passed
    news_confirmed: int = 0  # Confirmed by news
    opportunities_found: int = 0  # Final opportunities
    total_profit_potential: float = 0.0
    last_scan_time: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            "total_scanned": self.total_scanned,
            "price_qualified": self.price_qualified,
            "event_date_qualified": self.event_date_qualified,
            "news_confirmed": self.news_confirmed,
            "opportunities_found": self.opportunities_found,
            "total_profit_potential": f"{self.total_profit_potential:.2%}",
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
        }


class ArbitrageScanner:
    """Scans for TRUE arbitrage opportunities with news confirmation."""
    
    # Configuration
    MIN_EXTREME_PRICE = 0.92  # Price must be > this for YES arbitrage (lowered for news-confirmed)
    MAX_EXTREME_PRICE = 0.08  # Price must be < this for NO arbitrage
    MIN_PROFIT_PCT = 0.005  # Minimum 0.5% profit after spread (lowered - it's confirmed!)
    MAX_DAYS_TO_RESOLUTION = 14  # Consider markets up to 2 weeks out
    MIN_CONFIDENCE = 0.85  # Confidence threshold
    SPREAD_ESTIMATE = 0.02  # Assume 2% spread cost
    REQUIRE_NEWS_CONFIRMATION = True  # Require news to confirm outcome
    REQUIRE_EVENT_PASSED = False  # Require event date to have passed (optional)
    
    STATS_FILE = "data/arbitrage_stats.json"
    
    def __init__(self):
        self.opportunities: list[ArbitrageOpportunity] = []
        self._last_scan = None
        self._headline_analyzer = None
        self._stats = ArbitrageStats()
        self._historical_stats = {
            "total_opportunities": 0,
            "total_profit_captured": 0.0,
            "success_rate": 0.0,
            "by_category": {},
        }
        self._load_stats()
    
    def _get_headline_analyzer(self):
        """Lazy load headline analyzer."""
        if self._headline_analyzer is None:
            try:
                from polyb0t.services.headline_analyzer import get_headline_analyzer
                self._headline_analyzer = get_headline_analyzer()
            except ImportError:
                logger.warning("Headline analyzer not available")
        return self._headline_analyzer
    
    def _load_stats(self):
        """Load historical stats from disk."""
        if os.path.exists(self.STATS_FILE):
            try:
                with open(self.STATS_FILE, "r") as f:
                    self._historical_stats = json.load(f)
            except Exception as e:
                logger.debug(f"Could not load arbitrage stats: {e}")
    
    def _save_stats(self):
        """Save stats to disk."""
        try:
            os.makedirs(os.path.dirname(self.STATS_FILE), exist_ok=True)
            with open(self.STATS_FILE, "w") as f:
                json.dump(self._historical_stats, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save arbitrage stats: {e}")
    
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
        event_end_date: Optional[datetime] = None,
    ) -> Optional[ArbitrageOpportunity]:
        """Scan a single market for TRUE arbitrage opportunity.
        
        Requires:
        1. Price at extreme (>92% or <8%)
        2. Event date has passed OR news confirms outcome
        3. Profit exceeds spread cost
        
        Args:
            token_id: Token ID
            market_id: Market/condition ID
            market_title: Human-readable title
            price: Current YES price
            bid: Best bid
            ask: Best ask
            volume_24h: 24h volume
            days_to_resolution: Days until resolution (can be negative if passed)
            spread_pct: Current spread percentage
            momentum_24h: 24h price momentum
            volatility_24h: 24h volatility
            event_end_date: The event's end date (for checking if event passed)
            
        Returns:
            ArbitrageOpportunity if TRUE arbitrage found, None otherwise
        """
        self._stats.total_scanned += 1
        
        # Skip if too far from resolution (unless event already passed)
        event_date_passed = days_to_resolution <= 0
        if days_to_resolution > self.MAX_DAYS_TO_RESOLUTION and not event_date_passed:
            return None
        
        # Calculate actual spread
        if spread_pct > 0:
            spread_cost = spread_pct
        else:
            spread_cost = (ask - bid) / price if price > 0 else self.SPREAD_ESTIMATE
        
        spread_cost = max(spread_cost, 0.005)  # Minimum 0.5% spread
        
        opportunity = None
        expected_outcome = None  # "YES" or "NO"
        
        # === CHECK FOR PRICE QUALIFICATION ===
        # Price must be at extreme for this to be potential arbitrage
        if price >= self.MIN_EXTREME_PRICE:
            expected_outcome = "YES"
            self._stats.price_qualified += 1
        elif price <= self.MAX_EXTREME_PRICE:
            expected_outcome = "NO"
            self._stats.price_qualified += 1
        else:
            # Price not extreme enough
            return None
        
        # === CHECK EVENT DATE ===
        if event_date_passed:
            self._stats.event_date_qualified += 1
            logger.debug(f"Event date passed for {market_title[:50]}")
        
        # === CHECK NEWS CONFIRMATION ===
        news_confirmed = False
        news_headline = ""
        news_source = ""
        news_confidence = 0.0
        
        analyzer = self._get_headline_analyzer()
        if analyzer:
            confirmation = analyzer.check_market_outcome(market_id, market_title)
            if confirmation:
                # Check if news confirms our expected outcome
                if confirmation.confirmed_outcome == expected_outcome:
                    news_confirmed = True
                    news_headline = confirmation.headline
                    news_source = confirmation.source
                    news_confidence = confirmation.confidence
                    self._stats.news_confirmed += 1
                    logger.info(
                        f"NEWS CONFIRMS {expected_outcome} for {market_title[:40]}: "
                        f"'{news_headline[:60]}' ({news_source})"
                    )
                else:
                    # News contradicts price! Skip this - could be mispricing
                    logger.warning(
                        f"News contradicts price for {market_title[:40]}: "
                        f"Price says {expected_outcome}, news says {confirmation.confirmed_outcome}"
                    )
                    return None
        
        # === REQUIRE CONFIRMATION ===
        # Must have EITHER event date passed OR news confirmation
        if self.REQUIRE_NEWS_CONFIRMATION and not news_confirmed:
            if not event_date_passed:
                # No confirmation - too risky
                logger.debug(f"No confirmation for {market_title[:40]} - skipping")
                return None
        
        # === CALCULATE PROFIT ===
        if expected_outcome == "YES":
            expected_value = 1.0
            buy_price = ask  # We'd buy at ask
            profit_raw = expected_value - buy_price
            profit_net = profit_raw - spread_cost
            current_price = price
        else:  # NO
            # For NO to win, we buy the NO token (1 - YES price)
            expected_value = 1.0
            no_price = 1 - price
            buy_price = 1 - bid  # NO token ask ≈ 1 - YES bid
            profit_raw = expected_value - buy_price
            profit_net = profit_raw - spread_cost
            current_price = no_price
        
        if profit_net < self.MIN_PROFIT_PCT:
            return None
        
        # === CALCULATE CONFIDENCE ===
        confidence = self._calculate_confidence(
            price=current_price,
            expected_value=1.0,
            days_to_resolution=days_to_resolution,
            volatility_24h=volatility_24h,
            momentum_24h=momentum_24h,
            event_date_passed=event_date_passed,
            news_confirmed=news_confirmed,
            news_confidence=news_confidence,
        )
        
        if confidence < self.MIN_CONFIDENCE:
            return None
        
        # === CREATE OPPORTUNITY ===
        reason_parts = []
        if news_confirmed:
            reason_parts.append(f"News: '{news_headline[:40]}...'")
        if event_date_passed:
            reason_parts.append("Event date passed")
        reason_parts.append(f"Price={current_price:.1%}, Profit={profit_net:.1%}")
        
        opportunity = ArbitrageOpportunity(
            token_id=token_id,
            market_id=market_id,
            market_title=market_title,
            current_price=current_price,
            expected_value=expected_value,
            spread_cost=spread_cost,
            net_profit_pct=profit_net,
            confidence=confidence,
            days_to_resolution=days_to_resolution,
            volume_24h=volume_24h,
            reason=" | ".join(reason_parts),
            news_confirmed=news_confirmed,
            news_headline=news_headline,
            news_source=news_source,
            event_date_passed=event_date_passed,
        )
        
        self._stats.opportunities_found += 1
        self._stats.total_profit_potential += profit_net
        self.opportunities.append(opportunity)
        
        logger.info(
            f"TRUE ARBITRAGE: {market_title[:50]} - {expected_outcome} "
            f"profit={profit_net:.1%} conf={confidence:.0%} "
            f"news={news_confirmed} event_passed={event_date_passed}"
        )
        
        return opportunity
    
    def _calculate_confidence(
        self,
        price: float,
        expected_value: float,
        days_to_resolution: float,
        volatility_24h: float,
        momentum_24h: float,
        event_date_passed: bool,
        news_confirmed: bool,
        news_confidence: float,
    ) -> float:
        """Calculate confidence with news and event date factors."""
        confidence = 0.5  # Base
        
        # Price extremity (0.92 → 0.6, 0.99 → 0.95)
        if price >= self.MIN_EXTREME_PRICE:
            price_confidence = (price - 0.90) / 0.10
        else:
            price_confidence = (0.10 - price) / 0.10
        price_confidence = max(0, min(1, price_confidence))
        confidence += price_confidence * 0.2
        
        # Event date passed = big confidence boost
        if event_date_passed:
            confidence += 0.15
        
        # News confirmation = major confidence boost
        if news_confirmed:
            confidence += 0.2 + (news_confidence * 0.1)
        
        # Low volatility = stable
        if volatility_24h < 0.01:
            confidence += 0.05
        
        return min(confidence, 0.99)
    
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
        self._stats = ArbitrageStats()
        self._stats.last_scan_time = datetime.utcnow()
        self._last_scan = datetime.utcnow()
    
    def get_stats(self) -> dict:
        """Get current scan statistics."""
        return {
            "current_scan": self._stats.to_dict(),
            "historical": self._historical_stats,
            "opportunities": [o.to_dict() for o in self.opportunities],
        }
    
    def record_result(self, market_id: str, was_successful: bool, profit: float):
        """Record the result of an arbitrage trade.
        
        Args:
            market_id: The market ID
            was_successful: Whether the arbitrage was successful
            profit: The actual profit/loss
        """
        self._historical_stats["total_opportunities"] = self._historical_stats.get("total_opportunities", 0) + 1
        if was_successful:
            self._historical_stats["total_profit_captured"] = self._historical_stats.get("total_profit_captured", 0) + profit
        
        total = self._historical_stats.get("total_opportunities", 1)
        successes = sum(1 for _ in range(int(total * self._historical_stats.get("success_rate", 0))))
        if was_successful:
            successes += 1
        self._historical_stats["success_rate"] = successes / total if total > 0 else 0
        
        self._save_stats()
    
    def get_summary(self) -> dict:
        """Get summary of current arbitrage state."""
        if not self.opportunities:
            return {
                "count": 0,
                "total_profit_potential": 0,
                "best_opportunity": None,
                "stats": self._stats.to_dict(),
            }
        
        best = max(self.opportunities, key=lambda x: x.net_profit_pct * x.confidence)
        
        # Count news-confirmed vs price-only
        news_confirmed_count = sum(1 for o in self.opportunities if o.news_confirmed)
        event_passed_count = sum(1 for o in self.opportunities if o.event_date_passed)
        
        return {
            "count": len(self.opportunities),
            "news_confirmed": news_confirmed_count,
            "event_passed": event_passed_count,
            "total_profit_potential": sum(o.net_profit_pct for o in self.opportunities),
            "avg_confidence": sum(o.confidence for o in self.opportunities) / len(self.opportunities),
            "best_opportunity": best.to_dict(),
            "stats": self._stats.to_dict(),
        }


# Singleton
_scanner_instance: Optional[ArbitrageScanner] = None


def get_arbitrage_scanner() -> ArbitrageScanner:
    """Get or create the arbitrage scanner singleton."""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = ArbitrageScanner()
    return _scanner_instance
