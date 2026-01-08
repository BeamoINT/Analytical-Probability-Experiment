"""Market Microstructure Analysis - Advanced signals for profitable trading.

This module provides:
1. Order Book Imbalance - Predicts short-term price direction
2. Momentum Detection - Identifies trends, avoids falling knives
3. Volume Analysis - Detects unusual activity preceding big moves
4. Smart Entry Timing - Avoids chasing, waits for pullbacks
5. Spread Analysis - Identifies liquid vs illiquid markets
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureSignal:
    """Aggregated microstructure signal for a market."""
    
    token_id: str
    
    # Order Book Analysis
    order_book_imbalance: float  # -1 (heavy asks) to +1 (heavy bids)
    bid_depth_usd: float  # Total USD on bid side
    ask_depth_usd: float  # Total USD on ask side
    spread_pct: float  # Bid-ask spread as percentage
    
    # Momentum Analysis
    price_momentum_1h: float  # Price change in last hour (%)
    price_momentum_24h: float  # Price change in last 24h (%)
    momentum_score: float  # -1 (strong down) to +1 (strong up)
    is_falling_knife: bool  # Rapid decline - avoid buying
    is_pump: bool  # Rapid increase - avoid chasing
    
    # Volume Analysis
    volume_24h: float
    volume_ratio: float  # Current volume vs average (>1 = unusual)
    is_volume_spike: bool  # Unusual activity detected
    
    # Entry Timing
    entry_score: float  # -1 (bad entry) to +1 (good entry)
    should_wait: bool  # True if should wait for better entry
    wait_reason: str | None
    
    # Overall Signal
    composite_score: float  # -1 (strong sell) to +1 (strong buy)
    confidence: float  # 0 to 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "order_book_imbalance": self.order_book_imbalance,
            "bid_depth_usd": self.bid_depth_usd,
            "ask_depth_usd": self.ask_depth_usd,
            "spread_pct": self.spread_pct,
            "price_momentum_1h": self.price_momentum_1h,
            "price_momentum_24h": self.price_momentum_24h,
            "momentum_score": self.momentum_score,
            "is_falling_knife": self.is_falling_knife,
            "is_pump": self.is_pump,
            "volume_24h": self.volume_24h,
            "volume_ratio": self.volume_ratio,
            "is_volume_spike": self.is_volume_spike,
            "entry_score": self.entry_score,
            "should_wait": self.should_wait,
            "wait_reason": self.wait_reason,
            "composite_score": self.composite_score,
            "confidence": self.confidence,
        }


class MicrostructureAnalyzer:
    """Analyzes market microstructure for trading signals.
    
    Key Insights Used:
    
    1. ORDER BOOK IMBALANCE
       - Heavy bid side (more buyers waiting) = price likely to rise
       - Heavy ask side (more sellers waiting) = price likely to fall
       - Imbalance ratio > 2:1 is significant
    
    2. MOMENTUM
       - Positive momentum = trend continuation likely
       - Negative momentum = avoid buying (falling knife)
       - But extreme momentum = mean reversion likely
    
    3. VOLUME SPIKES
       - Unusual volume often precedes big moves
       - Get in early when volume spikes but price hasn't moved yet
       - High volume + no price move = accumulation phase
    
    4. SPREAD ANALYSIS
       - Wide spread = illiquid, avoid or size down
       - Tight spread = liquid, can trade larger
    
    5. ENTRY TIMING
       - Don't chase pumps (price just went up a lot)
       - Don't catch falling knives (price dropping rapidly)
       - Best entries: pullback after uptrend, bounce after oversold
    """
    
    # Thresholds (can be made configurable)
    IMBALANCE_SIGNIFICANT = 0.3  # 30% imbalance is meaningful
    MOMENTUM_STRONG = 0.10  # 10% move is strong
    FALLING_KNIFE_THRESHOLD = -0.15  # -15% in 24h = falling knife
    PUMP_THRESHOLD = 0.20  # +20% in 24h = pump (don't chase)
    VOLUME_SPIKE_RATIO = 2.0  # 2x normal volume = spike
    SPREAD_WIDE = 0.05  # 5% spread = too wide
    PULLBACK_THRESHOLD = 0.05  # 5% pullback from high = good entry
    
    def __init__(self) -> None:
        """Initialize microstructure analyzer."""
        self._price_history: dict[str, list[tuple[float, float]]] = {}  # token -> [(timestamp, price)]
        self._volume_history: dict[str, list[float]] = {}  # token -> [volumes]
    
    def analyze(
        self,
        token_id: str,
        orderbook: dict[str, Any] | None,
        current_price: float,
        price_history: list[dict[str, Any]] | None = None,
        volume_24h: float = 0.0,
        avg_volume: float = 0.0,
    ) -> MicrostructureSignal:
        """Analyze market microstructure and generate trading signal.
        
        Args:
            token_id: Token identifier.
            orderbook: Order book data with bids/asks.
            current_price: Current market price.
            price_history: Historical price data [(timestamp, price), ...].
            volume_24h: 24-hour trading volume.
            avg_volume: Average daily volume.
            
        Returns:
            MicrostructureSignal with analysis results.
        """
        # === ORDER BOOK ANALYSIS ===
        bid_depth, ask_depth, imbalance, spread_pct = self._analyze_orderbook(
            orderbook, current_price
        )
        
        # === MOMENTUM ANALYSIS ===
        momentum_1h, momentum_24h, momentum_score, is_falling_knife, is_pump = (
            self._analyze_momentum(price_history, current_price)
        )
        
        # === VOLUME ANALYSIS ===
        volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
        is_volume_spike = volume_ratio >= self.VOLUME_SPIKE_RATIO
        
        # === ENTRY TIMING ===
        entry_score, should_wait, wait_reason = self._analyze_entry_timing(
            momentum_score, is_falling_knife, is_pump, spread_pct, imbalance
        )
        
        # === COMPOSITE SCORE ===
        composite_score, confidence = self._compute_composite_score(
            imbalance, momentum_score, entry_score, is_volume_spike, spread_pct
        )
        
        return MicrostructureSignal(
            token_id=token_id,
            order_book_imbalance=imbalance,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth,
            spread_pct=spread_pct,
            price_momentum_1h=momentum_1h,
            price_momentum_24h=momentum_24h,
            momentum_score=momentum_score,
            is_falling_knife=is_falling_knife,
            is_pump=is_pump,
            volume_24h=volume_24h,
            volume_ratio=volume_ratio,
            is_volume_spike=is_volume_spike,
            entry_score=entry_score,
            should_wait=should_wait,
            wait_reason=wait_reason,
            composite_score=composite_score,
            confidence=confidence,
        )
    
    def _analyze_orderbook(
        self,
        orderbook: dict[str, Any] | None,
        current_price: float,
    ) -> tuple[float, float, float, float]:
        """Analyze order book for imbalance and spread.
        
        Returns:
            (bid_depth_usd, ask_depth_usd, imbalance, spread_pct)
        """
        if not orderbook:
            return 0.0, 0.0, 0.0, 0.05  # Default 5% spread if no data
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # Calculate depth (sum of price * size for top N levels)
        # Focus on top 5 levels as they're most relevant
        bid_depth = sum(
            float(b.get("price", 0)) * float(b.get("size", 0))
            for b in bids[:5]
        )
        ask_depth = sum(
            float(a.get("price", 0)) * float(a.get("size", 0))
            for a in asks[:5]
        )
        
        # Calculate imbalance: (bid - ask) / (bid + ask)
        # +1 = all bids, -1 = all asks, 0 = balanced
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            imbalance = (bid_depth - ask_depth) / total_depth
        else:
            imbalance = 0.0
        
        # Calculate spread
        best_bid = float(bids[0].get("price", 0)) if bids else current_price * 0.95
        best_ask = float(asks[0].get("price", 0)) if asks else current_price * 1.05
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 else current_price
        spread_pct = (best_ask - best_bid) / mid_price if mid_price > 0 else 0.05
        
        return bid_depth, ask_depth, imbalance, spread_pct
    
    def _analyze_momentum(
        self,
        price_history: list[dict[str, Any]] | None,
        current_price: float,
    ) -> tuple[float, float, float, bool, bool]:
        """Analyze price momentum.
        
        Returns:
            (momentum_1h, momentum_24h, momentum_score, is_falling_knife, is_pump)
        """
        if not price_history or len(price_history) < 2:
            return 0.0, 0.0, 0.0, False, False
        
        try:
            # Extract prices - handle different formats
            prices = []
            for p in price_history:
                if isinstance(p, dict):
                    price = p.get("price") or p.get("close") or p.get("p")
                    if price is not None:
                        prices.append(float(price))
                elif isinstance(p, (int, float)):
                    prices.append(float(p))
            
            if len(prices) < 2:
                return 0.0, 0.0, 0.0, False, False
            
            # Calculate momentum over different periods
            # Assuming data is sorted oldest to newest
            oldest_price = prices[0]
            newest_price = prices[-1]
            
            # 24h momentum (full history)
            momentum_24h = (newest_price - oldest_price) / oldest_price if oldest_price > 0 else 0
            
            # 1h momentum (last ~1/24 of data or last 4 points if we have hourly)
            recent_idx = max(0, len(prices) - max(4, len(prices) // 24))
            recent_price = prices[recent_idx]
            momentum_1h = (newest_price - recent_price) / recent_price if recent_price > 0 else 0
            
            # Compute momentum score (-1 to +1)
            # Use a combination of short and long-term momentum
            momentum_score = 0.4 * self._normalize(momentum_1h, 0.1) + 0.6 * self._normalize(momentum_24h, 0.2)
            momentum_score = max(-1.0, min(1.0, momentum_score))
            
            # Detect falling knife (rapid decline)
            is_falling_knife = momentum_24h <= self.FALLING_KNIFE_THRESHOLD
            
            # Detect pump (rapid increase - don't chase)
            is_pump = momentum_24h >= self.PUMP_THRESHOLD
            
            return momentum_1h, momentum_24h, momentum_score, is_falling_knife, is_pump
            
        except Exception as e:
            logger.warning(f"Error analyzing momentum: {e}")
            return 0.0, 0.0, 0.0, False, False
    
    def _analyze_entry_timing(
        self,
        momentum_score: float,
        is_falling_knife: bool,
        is_pump: bool,
        spread_pct: float,
        imbalance: float,
    ) -> tuple[float, bool, str | None]:
        """Analyze whether now is a good time to enter.
        
        Returns:
            (entry_score, should_wait, wait_reason)
        """
        entry_score = 0.5  # Start neutral
        should_wait = False
        wait_reason = None
        
        # === AVOID FALLING KNIVES ===
        if is_falling_knife:
            entry_score -= 0.5
            should_wait = True
            wait_reason = "Falling knife: price dropping rapidly, wait for stabilization"
        
        # === AVOID CHASING PUMPS ===
        if is_pump:
            entry_score -= 0.4
            should_wait = True
            wait_reason = "Price pumped recently, wait for pullback"
        
        # === WIDE SPREAD = BAD ENTRY ===
        if spread_pct > self.SPREAD_WIDE:
            entry_score -= 0.3
            if not should_wait:
                should_wait = True
                wait_reason = f"Wide spread ({spread_pct:.1%}), liquidity too low"
        
        # === ORDER BOOK IMBALANCE SIGNALS ===
        # Strong bid imbalance = good for buying
        if imbalance > self.IMBALANCE_SIGNIFICANT:
            entry_score += 0.3 * imbalance
        # Strong ask imbalance = bad for buying
        elif imbalance < -self.IMBALANCE_SIGNIFICANT:
            entry_score -= 0.3 * abs(imbalance)
            if not should_wait:
                should_wait = True
                wait_reason = "Heavy sell pressure in orderbook"
        
        # === MOMENTUM ALIGNMENT ===
        # Good momentum = good entry
        if momentum_score > 0.3:
            entry_score += 0.2
        # Bad momentum but not extreme = wait for reversal
        elif -0.3 < momentum_score < -0.1:
            entry_score -= 0.1
        
        # Normalize to -1 to +1
        entry_score = max(-1.0, min(1.0, entry_score))
        
        return entry_score, should_wait, wait_reason
    
    def _compute_composite_score(
        self,
        imbalance: float,
        momentum_score: float,
        entry_score: float,
        is_volume_spike: bool,
        spread_pct: float,
    ) -> tuple[float, float]:
        """Compute overall composite trading score.
        
        Returns:
            (composite_score, confidence)
        """
        # Weights for different factors
        WEIGHT_IMBALANCE = 0.25
        WEIGHT_MOMENTUM = 0.30
        WEIGHT_ENTRY = 0.35
        WEIGHT_VOLUME = 0.10
        
        # Base composite score
        composite = (
            WEIGHT_IMBALANCE * imbalance +
            WEIGHT_MOMENTUM * momentum_score +
            WEIGHT_ENTRY * entry_score +
            WEIGHT_VOLUME * (0.3 if is_volume_spike else 0.0)
        )
        
        # Normalize
        composite = max(-1.0, min(1.0, composite))
        
        # Calculate confidence based on data quality
        confidence = 1.0
        
        # Lower confidence if spread is wide (illiquid market)
        if spread_pct > 0.03:
            confidence *= 0.8
        if spread_pct > 0.05:
            confidence *= 0.7
        
        # Higher confidence if signals agree
        signals_agree = (
            (imbalance > 0 and momentum_score > 0 and entry_score > 0) or
            (imbalance < 0 and momentum_score < 0 and entry_score < 0)
        )
        if signals_agree:
            confidence = min(1.0, confidence * 1.2)
        
        return composite, confidence
    
    @staticmethod
    def _normalize(value: float, scale: float) -> float:
        """Normalize value to roughly -1 to +1 range."""
        return value / scale if scale > 0 else 0.0


class CorrelatedMarketAnalyzer:
    """Detects arbitrage opportunities in correlated markets.
    
    Key Insight: In binary markets (YES/NO), prices should sum to ~1.0
    If YES is 0.60 and NO is 0.50, that's a 10 cent arbitrage opportunity.
    
    Also detects related markets that should move together.
    """
    
    def __init__(self) -> None:
        """Initialize correlated market analyzer."""
        pass
    
    def find_binary_arbitrage(
        self,
        yes_price: float,
        no_price: float,
        fees_pct: float = 0.02,  # 2% total fees
    ) -> dict[str, Any]:
        """Check for arbitrage in binary YES/NO markets.
        
        Args:
            yes_price: Price of YES token.
            no_price: Price of NO token.
            fees_pct: Total fees as percentage.
            
        Returns:
            Arbitrage opportunity details.
        """
        # In a binary market, YES + NO should = 1.0
        total = yes_price + no_price
        
        # Account for fees
        min_total_for_profit = 1.0 + fees_pct
        max_total_for_profit = 1.0 - fees_pct
        
        result = {
            "yes_price": yes_price,
            "no_price": no_price,
            "total": total,
            "is_arbitrage": False,
            "type": None,
            "expected_profit_pct": 0.0,
            "action": None,
        }
        
        if total > min_total_for_profit:
            # Prices too high - sell both
            # If YES=0.60, NO=0.50, total=1.10
            # Sell $1 of each, pay $1.10, receive $1 on resolution = -$0.10
            # Wait, this is wrong. Let me reconsider.
            # Actually: Buy $1 of YES and $1 of NO costs $1.10
            # On resolution, one pays out $1, the other $0
            # Net: -$0.10. So no arbitrage here.
            # The arbitrage is when total < 1.0
            pass
        
        if total < max_total_for_profit:
            # Prices too low - BUY BOTH
            # If YES=0.40, NO=0.50, total=0.90
            # Buy $1 of YES for $0.40, buy $1 of NO for $0.50 = $0.90 cost
            # On resolution, one pays $1, receive $1 for $0.90 spent = $0.10 profit
            profit_pct = (1.0 - total) / total * 100
            result["is_arbitrage"] = True
            result["type"] = "buy_both"
            result["expected_profit_pct"] = profit_pct
            result["action"] = f"Buy both YES and NO: guaranteed ${1.0 - total:.4f} profit per $1 resolved"
        
        return result
    
    def find_related_market_divergence(
        self,
        market_a_price: float,
        market_b_price: float,
        expected_correlation: float = 1.0,  # 1.0 = should move together
        correlation_threshold: float = 0.1,  # 10% divergence is significant
    ) -> dict[str, Any]:
        """Find divergence between markets that should be correlated.
        
        Example: "Trump wins 2024" and "Republican wins 2024" should correlate.
        
        Args:
            market_a_price: Price of market A.
            market_b_price: Price of market B.
            expected_correlation: Expected correlation (1.0 = same, -1.0 = inverse).
            correlation_threshold: Divergence threshold for signal.
            
        Returns:
            Divergence analysis.
        """
        if expected_correlation > 0:
            # Markets should move together
            divergence = abs(market_a_price - market_b_price)
        else:
            # Markets should be inverse (A + B ≈ 1)
            divergence = abs(market_a_price + market_b_price - 1.0)
        
        is_significant = divergence > correlation_threshold
        
        return {
            "market_a_price": market_a_price,
            "market_b_price": market_b_price,
            "divergence": divergence,
            "is_significant": is_significant,
            "expected_correlation": expected_correlation,
            "action": self._suggest_convergence_trade(
                market_a_price, market_b_price, expected_correlation
            ) if is_significant else None,
        }
    
    def _suggest_convergence_trade(
        self,
        price_a: float,
        price_b: float,
        correlation: float,
    ) -> str:
        """Suggest trade to profit from expected convergence."""
        if correlation > 0:
            # Should be equal - buy cheaper, sell pricier
            if price_a < price_b:
                return f"Buy A (cheaper at {price_a:.2f}), markets should converge"
            else:
                return f"Buy B (cheaper at {price_b:.2f}), markets should converge"
        else:
            # Should be inverse
            total = price_a + price_b
            if total > 1.0:
                return "Both overpriced relative to inverse correlation"
            else:
                return "Both underpriced relative to inverse correlation"


class KellyCriterionSizer:
    """Optimal position sizing using Kelly Criterion.
    
    The Kelly Criterion determines the optimal bet size to maximize
    long-term growth while avoiding ruin.
    
    Formula: f* = (bp - q) / b
    Where:
        f* = fraction of bankroll to bet
        b = odds received (payout / stake)
        p = probability of winning
        q = probability of losing (1 - p)
    
    For prediction markets where you buy at price P:
        b = (1 - P) / P  (if you win, you get $1 for $P spent)
        p = your estimated true probability
        
    IMPORTANT: Full Kelly is aggressive. Most use fractional Kelly (25-50%)
    to reduce volatility and account for estimation errors.
    """
    
    def __init__(self, kelly_fraction: float = 0.25) -> None:
        """Initialize Kelly sizer.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly).
        """
        self.kelly_fraction = kelly_fraction
    
    def calculate_position_size(
        self,
        bankroll: float,
        market_price: float,
        estimated_probability: float,
        max_position_pct: float = 0.10,  # Never bet more than 10% of bankroll
        min_edge: float = 0.02,  # Minimum 2% edge to bet
    ) -> dict[str, Any]:
        """Calculate optimal position size using Kelly Criterion.
        
        Args:
            bankroll: Total bankroll available.
            market_price: Current market price (what you'd pay).
            estimated_probability: Your estimate of true probability.
            max_position_pct: Maximum position as % of bankroll.
            min_edge: Minimum edge required to bet.
            
        Returns:
            Position sizing recommendation.
        """
        # Validate inputs
        if market_price <= 0 or market_price >= 1:
            return {
                "should_bet": False,
                "reason": "Invalid market price",
                "recommended_size": 0,
            }
        
        if estimated_probability <= 0 or estimated_probability >= 1:
            return {
                "should_bet": False,
                "reason": "Invalid probability estimate",
                "recommended_size": 0,
            }
        
        # Calculate edge
        edge = estimated_probability - market_price
        
        if edge < min_edge:
            return {
                "should_bet": False,
                "reason": f"Insufficient edge ({edge:.2%} < {min_edge:.2%})",
                "edge": edge,
                "recommended_size": 0,
            }
        
        # Kelly formula for binary outcomes
        # b = payout odds = (1 - P) / P
        # f* = (b * p - q) / b = (b * p - (1 - p)) / b
        # Simplified: f* = p - q/b = p - (1-p)*P/(1-P) = p - P(1-p)/(1-P)
        
        # Alternative formulation for prediction markets:
        # f* = (estimated_prob / market_price) - 1
        # But capped and adjusted for our formula
        
        b = (1 - market_price) / market_price  # Odds
        p = estimated_probability
        q = 1 - p
        
        if b <= 0:
            return {
                "should_bet": False,
                "reason": "Invalid odds",
                "recommended_size": 0,
            }
        
        # Full Kelly
        full_kelly = (b * p - q) / b
        
        # Can be negative (meaning don't bet, or bet other side)
        if full_kelly <= 0:
            return {
                "should_bet": False,
                "reason": "Kelly suggests no bet or opposite side",
                "full_kelly": full_kelly,
                "recommended_size": 0,
            }
        
        # Apply fractional Kelly
        fractional_kelly = full_kelly * self.kelly_fraction
        
        # Apply maximum cap
        capped_kelly = min(fractional_kelly, max_position_pct)
        
        # Calculate dollar amount
        recommended_size = bankroll * capped_kelly
        
        # Calculate expected value
        expected_return = p * (1 / market_price - 1) - q
        
        return {
            "should_bet": True,
            "edge": edge,
            "edge_pct": edge * 100,
            "full_kelly_fraction": full_kelly,
            "fractional_kelly": fractional_kelly,
            "capped_kelly": capped_kelly,
            "recommended_size_usd": recommended_size,
            "recommended_pct": capped_kelly * 100,
            "expected_return_pct": expected_return * 100,
            "market_price": market_price,
            "estimated_probability": estimated_probability,
            "kelly_fraction_used": self.kelly_fraction,
        }
    
    def adjust_for_correlation(
        self,
        base_size: float,
        num_correlated_positions: int,
        correlation_factor: float = 0.5,
    ) -> float:
        """Reduce position size when holding correlated positions.
        
        If you have 3 positions that are all "Trump wins" type bets,
        they're correlated and you should size down.
        
        Args:
            base_size: Original recommended size.
            num_correlated_positions: Number of correlated positions held.
            correlation_factor: How correlated they are (0-1).
            
        Returns:
            Adjusted position size.
        """
        if num_correlated_positions <= 1:
            return base_size
        
        # Reduce size based on correlation
        # More correlated positions = smaller individual sizes
        reduction = 1 / (1 + correlation_factor * (num_correlated_positions - 1))
        
        return base_size * reduction

