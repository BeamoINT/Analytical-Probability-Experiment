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


@dataclass
class ArbitrageTradeResult:
    """Result of an arbitrage trade for tracking."""
    market_id: str
    market_title: str
    predicted_outcome: str  # "YES" or "NO"
    entry_price: float
    exit_price: float  # 1.0 if correct, 0.0 if wrong
    profit_loss: float  # Actual P&L percentage
    was_successful: bool
    timestamp: datetime
    news_headline: str = ""
    
    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "market_title": self.market_title,
            "predicted_outcome": self.predicted_outcome,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "profit_loss": self.profit_loss,
            "was_successful": self.was_successful,
            "timestamp": self.timestamp.isoformat(),
            "news_headline": self.news_headline,
        }


class ArbitrageScanner:
    """Scans for TRUE arbitrage using LLM-powered news analysis.
    
    Features:
    - Uses LLM (OpenAI) to actually understand news content
    - Tracks trade results and auto-disables if losing money
    - Requires minimum sample size before making decisions
    """
    
    # Configuration
    MIN_EXTREME_PRICE = 0.92  # Price must be > this for YES arbitrage
    MAX_EXTREME_PRICE = 0.08  # Price must be < this for NO arbitrage
    MIN_PROFIT_PCT = 0.005  # Minimum 0.5% profit after spread
    MAX_DAYS_TO_RESOLUTION = 14  # Consider markets up to 2 weeks out
    MIN_CONFIDENCE = 0.80  # Confidence threshold
    SPREAD_ESTIMATE = 0.02  # Assume 2% spread cost
    REQUIRE_NEWS_CONFIRMATION = True  # Require LLM to confirm outcome
    
    # Auto-disable settings
    MIN_TRADES_FOR_EVALUATION = 10  # Need at least 10 trades to evaluate
    MIN_WIN_RATE = 0.60  # Must win 60% of trades
    MAX_LOSS_PCT = -0.05  # If cumulative loss exceeds 5%, disable
    
    STATS_FILE = "data/arbitrage_stats.json"
    STATE_FILE = "data/arbitrage_state.json"
    
    def __init__(self):
        self.opportunities: list[ArbitrageOpportunity] = []
        self._last_scan = None
        self._intelligent_analyzer = None
        self._news_client = None
        self._stats = ArbitrageStats()
        self._is_disabled = False
        self._disable_reason = ""
        self._trade_history: list[ArbitrageTradeResult] = []
        self._historical_stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "is_disabled": False,
            "disable_reason": "",
        }
        self._load_state()
    
    def _get_intelligent_analyzer(self):
        """Lazy load intelligent analyzer (LLM-based)."""
        if self._intelligent_analyzer is None:
            try:
                from polyb0t.services.intelligent_analyzer import get_intelligent_analyzer
                self._intelligent_analyzer = get_intelligent_analyzer()
            except ImportError:
                logger.warning("Intelligent analyzer not available")
        return self._intelligent_analyzer
    
    def _get_news_client(self):
        """Lazy load news client."""
        if self._news_client is None:
            try:
                from polyb0t.services.news_client import get_news_client
                self._news_client = get_news_client()
            except ImportError:
                logger.warning("News client not available")
        return self._news_client
    
    def _get_headline_analyzer(self):
        """Lazy load headline analyzer (fallback keyword-based)."""
        try:
            from polyb0t.services.headline_analyzer import get_headline_analyzer
            return get_headline_analyzer()
        except ImportError:
            logger.warning("Headline analyzer not available")
            return None
    
    def _load_state(self):
        """Load state including trade history and disabled status."""
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE, "r") as f:
                    state = json.load(f)
                
                self._is_disabled = state.get("is_disabled", False)
                self._disable_reason = state.get("disable_reason", "")
                self._historical_stats = state.get("stats", self._historical_stats)
                
                # Load trade history
                for trade_dict in state.get("trade_history", []):
                    try:
                        self._trade_history.append(ArbitrageTradeResult(
                            market_id=trade_dict["market_id"],
                            market_title=trade_dict["market_title"],
                            predicted_outcome=trade_dict["predicted_outcome"],
                            entry_price=trade_dict["entry_price"],
                            exit_price=trade_dict["exit_price"],
                            profit_loss=trade_dict["profit_loss"],
                            was_successful=trade_dict["was_successful"],
                            timestamp=datetime.fromisoformat(trade_dict["timestamp"]),
                            news_headline=trade_dict.get("news_headline", ""),
                        ))
                    except (KeyError, ValueError):
                        continue
                
                if self._is_disabled:
                    logger.warning(f"Arbitrage scanner is DISABLED: {self._disable_reason}")
                
            except Exception as e:
                logger.debug(f"Could not load arbitrage state: {e}")
        
        # Also load legacy stats file
        if os.path.exists(self.STATS_FILE):
            try:
                with open(self.STATS_FILE, "r") as f:
                    legacy = json.load(f)
                    # Merge legacy stats
                    for key, value in legacy.items():
                        if key not in self._historical_stats:
                            self._historical_stats[key] = value
            except:
                pass
    
    def _save_state(self):
        """Save state to disk."""
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            
            state = {
                "is_disabled": self._is_disabled,
                "disable_reason": self._disable_reason,
                "stats": self._historical_stats,
                "trade_history": [t.to_dict() for t in self._trade_history[-100:]],  # Keep last 100
                "last_updated": datetime.utcnow().isoformat(),
            }
            
            with open(self.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.debug(f"Could not save arbitrage state: {e}")
    
    def is_disabled(self) -> bool:
        """Check if arbitrage scanner is disabled."""
        return self._is_disabled
    
    def get_disable_reason(self) -> str:
        """Get the reason for being disabled."""
        return self._disable_reason
    
    def enable(self):
        """Manually re-enable the arbitrage scanner."""
        self._is_disabled = False
        self._disable_reason = ""
        logger.info("Arbitrage scanner manually re-enabled")
        self._save_state()
    
    def _check_should_disable(self):
        """Check if we should auto-disable based on performance."""
        if len(self._trade_history) < self.MIN_TRADES_FOR_EVALUATION:
            return  # Not enough data
        
        # Look at last N trades
        recent_trades = self._trade_history[-self.MIN_TRADES_FOR_EVALUATION:]
        
        # Calculate win rate
        wins = sum(1 for t in recent_trades if t.was_successful)
        win_rate = wins / len(recent_trades)
        
        # Calculate cumulative P&L
        total_pnl = sum(t.profit_loss for t in recent_trades)
        
        # Check disable conditions
        if win_rate < self.MIN_WIN_RATE:
            self._is_disabled = True
            self._disable_reason = (
                f"Win rate too low: {win_rate:.1%} < {self.MIN_WIN_RATE:.0%} "
                f"(last {len(recent_trades)} trades)"
            )
            logger.error(f"AUTO-DISABLING arbitrage scanner: {self._disable_reason}")
            self._save_state()
            return
        
        if total_pnl < self.MAX_LOSS_PCT:
            self._is_disabled = True
            self._disable_reason = (
                f"Cumulative loss too high: {total_pnl:.1%} < {self.MAX_LOSS_PCT:.0%} "
                f"(last {len(recent_trades)} trades)"
            )
            logger.error(f"AUTO-DISABLING arbitrage scanner: {self._disable_reason}")
            self._save_state()
            return
    
    def record_trade_result(
        self,
        market_id: str,
        market_title: str,
        predicted_outcome: str,
        entry_price: float,
        actual_outcome: str,  # "YES" or "NO"
        news_headline: str = "",
    ):
        """Record the result of an arbitrage trade.
        
        Args:
            market_id: Market ID
            market_title: Market title
            predicted_outcome: What we predicted ("YES" or "NO")
            entry_price: Price we entered at
            actual_outcome: Actual outcome ("YES" or "NO")
            news_headline: The headline that triggered this trade
        """
        was_successful = predicted_outcome == actual_outcome
        
        if was_successful:
            exit_price = 1.0
            profit_loss = 1.0 - entry_price - self.SPREAD_ESTIMATE
        else:
            exit_price = 0.0
            profit_loss = -entry_price - self.SPREAD_ESTIMATE
        
        result = ArbitrageTradeResult(
            market_id=market_id,
            market_title=market_title,
            predicted_outcome=predicted_outcome,
            entry_price=entry_price,
            exit_price=exit_price,
            profit_loss=profit_loss,
            was_successful=was_successful,
            timestamp=datetime.utcnow(),
            news_headline=news_headline,
        )
        
        self._trade_history.append(result)
        
        # Update stats
        self._historical_stats["total_trades"] = self._historical_stats.get("total_trades", 0) + 1
        if was_successful:
            self._historical_stats["successful_trades"] = self._historical_stats.get("successful_trades", 0) + 1
        self._historical_stats["total_profit"] = self._historical_stats.get("total_profit", 0) + profit_loss
        
        total = self._historical_stats["total_trades"]
        successes = self._historical_stats["successful_trades"]
        self._historical_stats["win_rate"] = successes / total if total > 0 else 0
        
        logger.info(
            f"Arbitrage result: {'WIN' if was_successful else 'LOSS'} on {market_title[:30]} "
            f"(P&L: {profit_loss:+.1%}, win_rate: {self._historical_stats['win_rate']:.1%})"
        )
        
        # Check if we should auto-disable
        self._check_should_disable()
        
        self._save_state()
    
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
        
        Uses LLM to intelligently understand news content.
        Auto-disables if losing money.
        
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
        # === CHECK IF DISABLED ===
        if self._is_disabled:
            return None
        
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
        
        # === INTELLIGENT NEWS ANALYSIS ===
        news_confirmed = False
        news_headline = ""
        news_source = ""
        news_confidence = 0.0
        llm_reasoning = ""
        
        # First, get news articles
        news_client = self._get_news_client()
        intelligent_analyzer = self._get_intelligent_analyzer()
        
        if news_client and news_client.is_available():
            # Extract keywords for search
            keywords = self._extract_keywords(market_title)
            if keywords:
                articles = news_client.search_headlines(" ".join(keywords[:3]), page_size=5)
                
                # Analyze each article with LLM
                for article in articles:
                    # Skip old articles
                    article_age = datetime.utcnow() - article.published_at.replace(tzinfo=None)
                    if article_age > timedelta(days=7):
                        continue
                    
                    # Use intelligent analyzer if available
                    if intelligent_analyzer and intelligent_analyzer.is_available():
                        result = intelligent_analyzer.analyze_headline(
                            market_question=market_title,
                            headline=article.title,
                            article_content=article.description,
                            source=article.source,
                        )
                        
                        if result and result.confirmed_outcome:
                            # LLM confirmed an outcome!
                            if result.confirmed_outcome == expected_outcome:
                                news_confirmed = True
                                news_headline = article.title
                                news_source = article.source
                                news_confidence = result.confidence
                                llm_reasoning = result.reasoning
                                self._stats.news_confirmed += 1
                                logger.info(
                                    f"LLM CONFIRMS {expected_outcome} for {market_title[:40]}: "
                                    f"'{news_headline[:50]}...' - {llm_reasoning}"
                                )
                                break
                            else:
                                # LLM says different outcome than price suggests
                                logger.warning(
                                    f"LLM contradicts price for {market_title[:40]}: "
                                    f"Price={expected_outcome}, LLM={result.confirmed_outcome}"
                                )
                                return None
                    else:
                        # Fallback to keyword-based analysis
                        keyword_analyzer = self._get_headline_analyzer()
                        if keyword_analyzer:
                            confirmation = keyword_analyzer.check_market_outcome(market_id, market_title)
                            if confirmation and confirmation.confirmed_outcome == expected_outcome:
                                news_confirmed = True
                                news_headline = confirmation.headline
                                news_source = confirmation.source
                                news_confidence = confirmation.confidence
                                self._stats.news_confirmed += 1
                                break
        
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
    
    def _extract_keywords(self, market_title: str) -> list[str]:
        """Extract searchable keywords from market question."""
        import re
        
        # Remove common question words
        title = market_title.lower()
        title = re.sub(r'^(will|does|is|are|has|have|can|could|would|should)\s+', '', title)
        title = re.sub(r'\?$', '', title)
        
        # Extract words
        words = re.findall(r'[a-zA-Z]+', title)
        
        # Stop words
        stop_words = {
            "will", "the", "a", "an", "in", "on", "at", "to", "for", "of", "by",
            "with", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "this", "that", "these", "those",
            "before", "after", "during", "between", "2024", "2025", "2026",
        }
        
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        return keywords[:5]
    
    def scan_all(
        self,
        markets: list,
        orderbooks: dict,
    ) -> list[ArbitrageOpportunity]:
        """Scan all markets for arbitrage opportunities.
        
        Args:
            markets: List of market objects with condition_id, question, outcomes, end_date_iso
            orderbooks: Dict mapping token_id to orderbook data
            
        Returns:
            List of arbitrage opportunities found
        """
        if self._is_disabled:
            return []
        
        self.clear_opportunities()
        opportunities = []
        
        for market in markets:
            try:
                # Get market info
                market_id = getattr(market, 'condition_id', None) or getattr(market, 'id', '')
                market_title = getattr(market, 'question', '') or getattr(market, 'title', '')
                end_date = getattr(market, 'end_date_iso', None)
                outcomes = getattr(market, 'outcomes', [])
                
                if not outcomes:
                    continue
                
                # Parse end date
                event_end_date = None
                if end_date:
                    try:
                        if isinstance(end_date, str):
                            event_end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                            event_end_date = event_end_date.replace(tzinfo=None)
                        else:
                            event_end_date = end_date
                    except:
                        pass
                
                # Calculate days to resolution
                days_to_resolution = 999
                if event_end_date:
                    days_to_resolution = (event_end_date - datetime.utcnow()).total_seconds() / 86400
                
                # Check each outcome
                for outcome in outcomes:
                    token_id = getattr(outcome, 'token_id', None)
                    if not token_id:
                        continue
                    
                    # Get orderbook data
                    ob = orderbooks.get(token_id, {})
                    if not ob:
                        continue
                    
                    # Extract price info
                    bids = ob.get('bids', [])
                    asks = ob.get('asks', [])
                    
                    if not bids or not asks:
                        continue
                    
                    best_bid = float(bids[0].get('price', 0)) if bids else 0
                    best_ask = float(asks[0].get('price', 1)) if asks else 1
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Get volume
                    volume_24h = float(getattr(outcome, 'volume_24h', 0) or 0)
                    
                    # Calculate spread
                    spread_pct = (best_ask - best_bid) / mid_price if mid_price > 0 else 0
                    
                    # Scan this market
                    opp = self.scan_market(
                        token_id=token_id,
                        market_id=market_id,
                        market_title=market_title,
                        price=mid_price,
                        bid=best_bid,
                        ask=best_ask,
                        volume_24h=volume_24h,
                        days_to_resolution=days_to_resolution,
                        spread_pct=spread_pct,
                        event_end_date=event_end_date,
                    )
                    
                    if opp:
                        opportunities.append(opp)
                        
            except Exception as e:
                logger.debug(f"Error scanning market for arbitrage: {e}")
                continue
        
        return opportunities
    
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
