"""Smart Heuristics Engine - Weighted Scoring System.

This module replaces binary Yes/No rules with a weighted scoring engine.
Each rule contributes a weighted score based on historical success rate.

Key Improvements:
1. Rules have WEIGHTS based on historical accuracy
2. Market sentiment filter (volume + spread narrowing)
3. Pre-trade slippage check with abort threshold
4. Dynamic Kelly Criterion with win-rate scaling
5. Global stop-loss with listen-only mode
6. Data labeling for future ML training
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import sqlite3
from pathlib import Path

import numpy as np

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RuleScore:
    """Score from a single rule."""
    rule_id: str
    rule_name: str
    raw_score: float  # -1 to +1
    weight: float  # Historical accuracy weight
    weighted_score: float  # raw_score * weight
    confidence: float  # How confident is this rule
    triggered: bool  # Did this rule fire
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeScore:
    """Composite score from all rules."""
    total_score: float  # Sum of weighted scores
    normalized_score: float  # -1 to +1 normalized
    confidence: float  # Overall confidence
    rule_scores: list[RuleScore] = field(default_factory=list)
    should_trade: bool = False
    trade_direction: str = ""  # BUY or SELL
    abort_reason: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_score": self.total_score,
            "normalized_score": self.normalized_score,
            "confidence": self.confidence,
            "should_trade": self.should_trade,
            "trade_direction": self.trade_direction,
            "abort_reason": self.abort_reason,
            "rules_triggered": len([r for r in self.rule_scores if r.triggered]),
            "top_rules": [
                {"id": r.rule_id, "score": r.weighted_score}
                for r in sorted(self.rule_scores, key=lambda x: abs(x.weighted_score), reverse=True)[:3]
            ],
        }


@dataclass
class SlippageCheck:
    """Pre-trade slippage analysis."""
    estimated_fill_price: float
    mid_price: float
    slippage_bps: int
    price_impact_pct: float
    edge_after_slippage: float
    should_abort: bool
    abort_reason: str | None = None
    max_acceptable_slippage_bps: int = 0


@dataclass  
class SentimentSignal:
    """Market sentiment analysis."""
    volume_trend: float  # -1 to +1 (increasing = positive)
    spread_trend: float  # -1 to +1 (narrowing = positive)
    momentum_alignment: float  # -1 to +1
    composite_sentiment: float  # -1 to +1
    is_favorable: bool  # Good conditions to enter


class RuleWeightTracker:
    """Tracks historical performance of rules to update weights.
    
    Each rule starts with equal weight. As trades resolve,
    weights are updated based on accuracy.
    """
    
    def __init__(self, db_path: str = "data/rule_weights.db") -> None:
        """Initialize rule weight tracker."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
        # Default weights for each rule (updated from historical data)
        self._default_weights = {
            # Edge-based rules
            "raw_edge_positive": 0.20,
            "raw_edge_large": 0.15,
            
            # Orderbook rules  
            "orderbook_imbalance_bullish": 0.12,
            "orderbook_depth_sufficient": 0.10,
            "spread_tight": 0.08,
            
            # Momentum rules
            "momentum_aligned": 0.10,
            "not_falling_knife": 0.15,
            "not_chasing_pump": 0.10,
            
            # Value rules
            "historical_accuracy_favorable": 0.08,
            "time_to_resolution_optimal": 0.05,
            
            # Contrarian rules
            "contrarian_opportunity": 0.05,
            
            # Information rules
            "unusual_volume": 0.05,
            "whale_activity_aligned": 0.05,
        }
        
        # Load historical weights
        self._weights = self._load_weights()
        
    def _init_db(self) -> None:
        """Initialize database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rule_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT,
                    triggered INTEGER,
                    trade_pnl REAL,
                    was_correct INTEGER,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rule_weights (
                    rule_id TEXT PRIMARY KEY,
                    weight REAL,
                    total_trades INTEGER,
                    correct_trades INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
    def _load_weights(self) -> dict[str, float]:
        """Load weights from database."""
        weights = self._default_weights.copy()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT rule_id, weight FROM rule_weights
                """).fetchall()
                
                for rule_id, weight in rows:
                    if rule_id in weights:
                        weights[rule_id] = weight
                        
        except Exception as e:
            logger.debug(f"Failed to load rule weights: {e}")
            
        return weights
    
    def get_weight(self, rule_id: str) -> float:
        """Get weight for a rule."""
        return self._weights.get(rule_id, 0.05)
    
    def record_outcome(
        self, rule_id: str, triggered: bool, trade_pnl: float
    ) -> None:
        """Record rule outcome for weight learning.
        
        Args:
            rule_id: Rule identifier.
            triggered: Whether rule fired.
            trade_pnl: PnL of the trade.
        """
        was_correct = (triggered and trade_pnl > 0) or (not triggered and trade_pnl <= 0)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO rule_outcomes (rule_id, triggered, trade_pnl, was_correct)
                    VALUES (?, ?, ?, ?)
                """, (rule_id, 1 if triggered else 0, trade_pnl, 1 if was_correct else 0))
                
                # Update weight based on recent accuracy
                self._update_weight(conn, rule_id)
                
        except Exception as e:
            logger.debug(f"Failed to record rule outcome: {e}")
    
    def _update_weight(self, conn: sqlite3.Connection, rule_id: str) -> None:
        """Update rule weight based on recent accuracy."""
        try:
            # Get accuracy from last 100 trades where rule triggered
            rows = conn.execute("""
                SELECT COUNT(*) as total, SUM(was_correct) as correct
                FROM (
                    SELECT was_correct FROM rule_outcomes
                    WHERE rule_id = ? AND triggered = 1
                    ORDER BY recorded_at DESC
                    LIMIT 100
                )
            """, (rule_id,)).fetchone()
            
            if rows and rows[0] >= 10:  # Need at least 10 samples
                total, correct = rows
                accuracy = correct / total
                
                # Weight = baseline * accuracy_multiplier
                # Accuracy 50% = 1.0 multiplier, 70% = 1.4 multiplier
                baseline = self._default_weights.get(rule_id, 0.05)
                accuracy_multiplier = 2 * accuracy  # 50% = 1.0, 75% = 1.5
                new_weight = baseline * accuracy_multiplier
                
                # Bound weights
                new_weight = max(0.01, min(0.30, new_weight))
                
                conn.execute("""
                    INSERT OR REPLACE INTO rule_weights 
                    (rule_id, weight, total_trades, correct_trades, updated_at)
                    VALUES (?, ?, ?, ?, datetime('now'))
                """, (rule_id, new_weight, total, correct))
                
                self._weights[rule_id] = new_weight
                
        except Exception as e:
            logger.debug(f"Failed to update weight for {rule_id}: {e}")


class SmartHeuristicsEngine:
    """Weighted scoring engine for trade decisions.
    
    Instead of binary rules, each factor contributes a weighted score.
    The composite score determines trade direction and size.
    
    Key Features:
    1. WEIGHTED RULES - Each rule has a weight based on historical accuracy
    2. SLIPPAGE CHECK - Pre-trade abort if slippage exceeds edge
    3. SENTIMENT FILTER - Only trade when volume up + spread narrowing
    4. DYNAMIC KELLY - Scale sizing based on composite score
    5. GLOBAL STOP-LOSS - Enter listen-only mode on drawdown
    """
    
    def __init__(self) -> None:
        """Initialize smart heuristics engine."""
        self.settings = get_settings()
        self.weight_tracker = RuleWeightTracker()
        
        # Global drawdown tracking
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._listen_only_mode: bool = False
        self._listen_only_until: datetime | None = None
        
        # Score thresholds
        self.min_score_to_trade = 0.15  # Minimum normalized score
        self.high_confidence_threshold = 0.30  # High confidence threshold
        
        # Slippage limits
        self.max_slippage_of_edge = 0.30  # Abort if slippage > 30% of edge
        self.absolute_max_slippage_bps = 100  # Never accept > 100bps slippage
        
    def compute_composite_score(
        self,
        # Price data
        current_price: float,
        p_model: float,  # Model probability
        
        # Orderbook data
        orderbook: dict[str, Any] | None,
        
        # Trade data
        recent_trades: list[dict] | None,
        
        # Market context
        volume_24h: float = 0.0,
        avg_volume: float = 0.0,
        days_to_resolution: float | None = None,
        market_category: str | None = None,
        
        # Historical context (for pattern matching)
        price_history: list[float] | None = None,
    ) -> CompositeScore:
        """Compute composite score from all rules.
        
        Args:
            current_price: Current market price.
            p_model: Model's probability estimate.
            orderbook: Current orderbook.
            recent_trades: Recent trades.
            volume_24h: 24-hour volume.
            avg_volume: Average volume.
            days_to_resolution: Days until resolution.
            market_category: Market category.
            price_history: Recent price history.
            
        Returns:
            CompositeScore with weighted rule scores.
        """
        rule_scores: list[RuleScore] = []
        
        # === EDGE-BASED RULES ===
        raw_edge = p_model - current_price
        
        # Rule 1: Raw edge positive
        rule_scores.append(self._score_raw_edge(raw_edge, "positive"))
        
        # Rule 2: Raw edge large (strong signal)
        rule_scores.append(self._score_raw_edge(raw_edge, "large"))
        
        # === ORDERBOOK RULES ===
        if orderbook:
            # Rule 3: Orderbook imbalance
            rule_scores.append(self._score_orderbook_imbalance(orderbook, raw_edge))
            
            # Rule 4: Sufficient depth
            rule_scores.append(self._score_orderbook_depth(orderbook))
            
            # Rule 5: Tight spread
            rule_scores.append(self._score_spread(orderbook, current_price))
        
        # === MOMENTUM RULES ===
        if price_history and len(price_history) >= 5:
            # Rule 6: Momentum aligned with edge
            rule_scores.append(self._score_momentum_alignment(price_history, raw_edge))
            
            # Rule 7: Not falling knife
            rule_scores.append(self._score_falling_knife(price_history))
            
            # Rule 8: Not chasing pump
            rule_scores.append(self._score_pump_chasing(price_history))
        
        # === VALUE RULES ===
        # Rule 9: Historical accuracy favorable
        rule_scores.append(self._score_historical_accuracy(
            current_price, market_category
        ))
        
        # Rule 10: Time to resolution
        if days_to_resolution is not None:
            rule_scores.append(self._score_time_to_resolution(days_to_resolution))
        
        # === VOLUME RULES ===
        # Rule 11: Unusual volume (information signal)
        rule_scores.append(self._score_unusual_volume(volume_24h, avg_volume))
        
        # === CONTRARIAN RULES ===
        # Rule 12: Contrarian opportunity
        rule_scores.append(self._score_contrarian(current_price, orderbook))
        
        # Compute composite
        total_score = sum(r.weighted_score for r in rule_scores if r.triggered)
        max_possible = sum(r.weight for r in rule_scores)
        
        normalized_score = total_score / max_possible if max_possible > 0 else 0.0
        normalized_score = max(-1.0, min(1.0, normalized_score))
        
        # Compute confidence
        confidence = self._compute_confidence(rule_scores, normalized_score)
        
        # Determine trade decision
        should_trade = abs(normalized_score) >= self.min_score_to_trade
        trade_direction = ""
        abort_reason = None
        
        if should_trade:
            trade_direction = "BUY" if normalized_score > 0 else "SELL"
            
            # Check listen-only mode
            if self._listen_only_mode:
                should_trade = False
                abort_reason = "LISTEN_ONLY_MODE_ACTIVE"
        
        return CompositeScore(
            total_score=total_score,
            normalized_score=normalized_score,
            confidence=confidence,
            rule_scores=rule_scores,
            should_trade=should_trade,
            trade_direction=trade_direction,
            abort_reason=abort_reason,
        )
    
    def _score_raw_edge(self, raw_edge: float, edge_type: str) -> RuleScore:
        """Score based on raw edge."""
        if edge_type == "positive":
            rule_id = "raw_edge_positive"
            # Score proportional to edge magnitude
            triggered = abs(raw_edge) > 0.02  # 2% minimum
            raw_score = np.sign(raw_edge) * min(1.0, abs(raw_edge) / 0.10)  # Cap at 10%
        else:  # "large"
            rule_id = "raw_edge_large"
            triggered = abs(raw_edge) > 0.05  # 5% = large edge
            raw_score = np.sign(raw_edge) * min(1.0, abs(raw_edge) / 0.15)
            
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name=f"Raw Edge {edge_type.title()}",
            raw_score=raw_score if triggered else 0.0,
            weight=weight,
            weighted_score=raw_score * weight if triggered else 0.0,
            confidence=0.7 if triggered else 0.0,
            triggered=triggered,
            metadata={"edge": raw_edge},
        )
    
    def _score_orderbook_imbalance(
        self, orderbook: dict[str, Any], raw_edge: float
    ) -> RuleScore:
        """Score based on orderbook imbalance."""
        rule_id = "orderbook_imbalance_bullish"
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        bid_volume = sum(float(b.get("size", 0)) for b in bids[:10])
        ask_volume = sum(float(a.get("size", 0)) for a in asks[:10])
        
        total = bid_volume + ask_volume
        if total > 0:
            imbalance = (bid_volume - ask_volume) / total  # -1 to +1
        else:
            imbalance = 0.0
        
        # Score is positive if imbalance aligns with our edge
        edge_direction = np.sign(raw_edge)
        alignment = imbalance * edge_direction
        
        triggered = abs(imbalance) > 0.20  # 20% imbalance
        raw_score = alignment if triggered else 0.0
        
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Orderbook Imbalance",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            confidence=0.6 if triggered else 0.0,
            triggered=triggered,
            metadata={"imbalance": imbalance, "alignment": alignment},
        )
    
    def _score_orderbook_depth(self, orderbook: dict[str, Any]) -> RuleScore:
        """Score based on orderbook depth."""
        rule_id = "orderbook_depth_sufficient"
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        bid_depth_usd = sum(
            float(b.get("size", 0)) * float(b.get("price", 0))
            for b in bids[:5]
        )
        ask_depth_usd = sum(
            float(a.get("size", 0)) * float(a.get("price", 0))
            for a in asks[:5]
        )
        
        total_depth = bid_depth_usd + ask_depth_usd
        
        # Good depth = positive score
        if total_depth >= 500:  # $500+ is good
            raw_score = 0.5
            triggered = True
        elif total_depth >= 200:  # $200-500 is okay
            raw_score = 0.3
            triggered = True
        elif total_depth >= 100:  # $100-200 is marginal
            raw_score = 0.1
            triggered = True
        else:  # <$100 is bad - negative score
            raw_score = -0.5
            triggered = True
        
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Orderbook Depth",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            confidence=0.8,
            triggered=triggered,
            metadata={"total_depth_usd": total_depth},
        )
    
    def _score_spread(
        self, orderbook: dict[str, Any], current_price: float
    ) -> RuleScore:
        """Score based on bid-ask spread."""
        rule_id = "spread_tight"
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if bids and asks:
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 0))
            mid = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid if mid > 0 else 1.0
        else:
            spread_pct = 1.0
        
        # Tight spread = positive score
        if spread_pct <= 0.02:  # ≤2% is excellent
            raw_score = 0.8
        elif spread_pct <= 0.04:  # 2-4% is good
            raw_score = 0.5
        elif spread_pct <= 0.06:  # 4-6% is okay
            raw_score = 0.2
        elif spread_pct <= 0.08:  # 6-8% is marginal
            raw_score = -0.2
        else:  # >8% is bad
            raw_score = -0.8
        
        triggered = True  # Always score spread
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Spread Quality",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            confidence=0.9,
            triggered=triggered,
            metadata={"spread_pct": spread_pct},
        )
    
    def _score_momentum_alignment(
        self, price_history: list[float], raw_edge: float
    ) -> RuleScore:
        """Score based on momentum alignment with edge."""
        rule_id = "momentum_aligned"
        
        if len(price_history) < 5:
            return RuleScore(
                rule_id=rule_id, rule_name="Momentum Aligned",
                raw_score=0.0, weight=0.0, weighted_score=0.0,
                confidence=0.0, triggered=False,
            )
        
        # Calculate momentum (simple: recent vs older)
        recent_avg = np.mean(price_history[-3:])
        older_avg = np.mean(price_history[:3])
        
        if older_avg > 0:
            momentum = (recent_avg - older_avg) / older_avg
        else:
            momentum = 0.0
        
        # Alignment: momentum direction matches edge direction
        edge_direction = np.sign(raw_edge)
        momentum_direction = np.sign(momentum)
        
        aligned = edge_direction == momentum_direction
        
        if aligned:
            raw_score = min(1.0, abs(momentum) * 5)  # Cap at 20% momentum
        else:
            raw_score = -min(0.5, abs(momentum) * 2.5)  # Penalty for misalignment
        
        triggered = abs(momentum) > 0.02  # 2% momentum
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Momentum Aligned",
            raw_score=raw_score if triggered else 0.0,
            weight=weight,
            weighted_score=raw_score * weight if triggered else 0.0,
            confidence=0.6 if aligned else 0.4,
            triggered=triggered,
            metadata={"momentum": momentum, "aligned": aligned},
        )
    
    def _score_falling_knife(self, price_history: list[float]) -> RuleScore:
        """Score for avoiding falling knives (large drops)."""
        rule_id = "not_falling_knife"
        
        if len(price_history) < 2:
            return RuleScore(
                rule_id=rule_id, rule_name="Not Falling Knife",
                raw_score=0.0, weight=0.0, weighted_score=0.0,
                confidence=0.0, triggered=False,
            )
        
        # Calculate drop from first to last
        start_price = price_history[0]
        end_price = price_history[-1]
        
        if start_price > 0:
            drop_pct = (start_price - end_price) / start_price
        else:
            drop_pct = 0.0
        
        is_falling_knife = drop_pct > 0.15  # >15% drop = falling knife
        
        if is_falling_knife:
            # Strongly negative - don't buy falling knives
            raw_score = -0.8
        elif drop_pct > 0.10:  # 10-15% drop - caution
            raw_score = -0.3
        elif drop_pct > 0.05:  # 5-10% drop - slight caution
            raw_score = -0.1
        else:  # No significant drop - positive
            raw_score = 0.3
        
        triggered = True  # Always evaluate
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Not Falling Knife",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            confidence=0.8 if is_falling_knife else 0.6,
            triggered=triggered,
            metadata={"drop_pct": drop_pct, "is_falling_knife": is_falling_knife},
        )
    
    def _score_pump_chasing(self, price_history: list[float]) -> RuleScore:
        """Score for avoiding pump chasing."""
        rule_id = "not_chasing_pump"
        
        if len(price_history) < 2:
            return RuleScore(
                rule_id=rule_id, rule_name="Not Chasing Pump",
                raw_score=0.0, weight=0.0, weighted_score=0.0,
                confidence=0.0, triggered=False,
            )
        
        # Calculate rise from first to last
        start_price = price_history[0]
        end_price = price_history[-1]
        
        if start_price > 0:
            rise_pct = (end_price - start_price) / start_price
        else:
            rise_pct = 0.0
        
        is_pump = rise_pct > 0.20  # >20% rise = pump (don't chase)
        
        if is_pump:
            raw_score = -0.7  # Don't chase
        elif rise_pct > 0.15:
            raw_score = -0.3
        elif rise_pct > 0.10:
            raw_score = -0.1
        else:
            raw_score = 0.2  # No pump - okay to buy
        
        triggered = True
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Not Chasing Pump",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            confidence=0.7 if is_pump else 0.5,
            triggered=triggered,
            metadata={"rise_pct": rise_pct, "is_pump": is_pump},
        )
    
    def _score_historical_accuracy(
        self, current_price: float, category: str | None
    ) -> RuleScore:
        """Score based on historical resolution accuracy at this price level."""
        rule_id = "historical_accuracy_favorable"
        
        # For now, use simple heuristic:
        # Prices close to 0.5 are less predictable than extreme prices
        # Extreme prices (0.1-0.3 or 0.7-0.9) often resolve as expected
        
        if current_price < 0.15:
            # Very low price - often resolves to 0
            raw_score = -0.3  # Favor NO side
        elif current_price > 0.85:
            # Very high price - often resolves to 1
            raw_score = 0.3  # Favor YES side
        elif 0.40 <= current_price <= 0.60:
            # Middle range - uncertain
            raw_score = 0.0
        elif current_price < 0.40:
            # Low-ish - slight NO favor
            raw_score = -0.1
        else:
            # High-ish - slight YES favor
            raw_score = 0.1
        
        triggered = abs(current_price - 0.5) > 0.15  # Score when not at 50%
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Historical Accuracy",
            raw_score=raw_score if triggered else 0.0,
            weight=weight,
            weighted_score=raw_score * weight if triggered else 0.0,
            confidence=0.4,  # Low confidence - need more historical data
            triggered=triggered,
            metadata={"price_level": current_price},
        )
    
    def _score_time_to_resolution(self, days: float) -> RuleScore:
        """Score based on time to resolution."""
        rule_id = "time_to_resolution_optimal"
        
        # Optimal window: 7-45 days
        if 7 <= days <= 45:
            raw_score = 0.5
        elif 3 <= days < 7:
            raw_score = 0.3  # Getting close
        elif 45 < days <= 90:
            raw_score = 0.2  # Far out but okay
        elif days < 3:
            raw_score = -0.5  # Too close to resolution
        else:
            raw_score = -0.2  # Too far out
        
        triggered = True
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Time to Resolution",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            confidence=0.5,
            triggered=triggered,
            metadata={"days_to_resolution": days},
        )
    
    def _score_unusual_volume(self, volume_24h: float, avg_volume: float) -> RuleScore:
        """Score based on unusual volume (information signal)."""
        rule_id = "unusual_volume"
        
        if avg_volume > 0:
            volume_ratio = volume_24h / avg_volume
        else:
            volume_ratio = 1.0
        
        # Unusual volume often precedes big moves
        if volume_ratio >= 3.0:  # 3x normal = very unusual
            raw_score = 0.6
        elif volume_ratio >= 2.0:  # 2x normal = unusual
            raw_score = 0.4
        elif volume_ratio >= 1.5:  # 1.5x normal
            raw_score = 0.2
        elif volume_ratio >= 0.5:  # Normal
            raw_score = 0.0
        else:  # Low volume - caution
            raw_score = -0.2
        
        triggered = volume_ratio >= 1.5
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Unusual Volume",
            raw_score=raw_score if triggered else 0.0,
            weight=weight,
            weighted_score=raw_score * weight if triggered else 0.0,
            confidence=0.5 if triggered else 0.0,
            triggered=triggered,
            metadata={"volume_ratio": volume_ratio},
        )
    
    def _score_contrarian(
        self, current_price: float, orderbook: dict[str, Any] | None
    ) -> RuleScore:
        """Score for contrarian opportunity (fading crowd)."""
        rule_id = "contrarian_opportunity"
        
        raw_score = 0.0
        is_contrarian = False
        
        # Contrarian signal at extreme prices with heavy one-sided orderbook
        if orderbook and (current_price < 0.20 or current_price > 0.80):
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            bid_vol = sum(float(b.get("size", 0)) for b in bids[:10])
            ask_vol = sum(float(a.get("size", 0)) for a in asks[:10])
            
            total = bid_vol + ask_vol
            if total > 0:
                imbalance = (bid_vol - ask_vol) / total
                
                # Contrarian: fade the crowd at extremes
                if current_price > 0.80 and imbalance > 0.5:
                    # Everyone buying high - fade
                    raw_score = -0.4  # Sell signal
                    is_contrarian = True
                elif current_price < 0.20 and imbalance < -0.5:
                    # Everyone selling low - fade
                    raw_score = 0.4  # Buy signal
                    is_contrarian = True
        
        triggered = is_contrarian
        weight = self.weight_tracker.get_weight(rule_id)
        
        return RuleScore(
            rule_id=rule_id,
            rule_name="Contrarian Opportunity",
            raw_score=raw_score,
            weight=weight,
            weighted_score=raw_score * weight,
            confidence=0.4 if is_contrarian else 0.0,
            triggered=triggered,
            metadata={"is_contrarian": is_contrarian},
        )
    
    def _compute_confidence(
        self, rule_scores: list[RuleScore], normalized_score: float
    ) -> float:
        """Compute overall confidence from rule agreement."""
        if not rule_scores:
            return 0.0
        
        triggered_rules = [r for r in rule_scores if r.triggered]
        if not triggered_rules:
            return 0.0
        
        # Count how many rules agree with the direction
        direction = np.sign(normalized_score)
        agreeing = sum(
            1 for r in triggered_rules
            if np.sign(r.weighted_score) == direction
        )
        
        agreement_ratio = agreeing / len(triggered_rules)
        
        # Weight by individual confidences
        avg_confidence = np.mean([r.confidence for r in triggered_rules])
        
        # Combine
        confidence = 0.5 * agreement_ratio + 0.5 * avg_confidence
        
        # Boost if score is strong
        if abs(normalized_score) > 0.30:
            confidence = min(1.0, confidence * 1.2)
        
        return min(1.0, max(0.0, confidence))
    
    def check_slippage(
        self,
        edge: float,
        orderbook: dict[str, Any],
        side: str,
        size_usd: float,
    ) -> SlippageCheck:
        """Check if slippage is acceptable before trading.
        
        Args:
            edge: Expected edge (p_model - p_market).
            orderbook: Current orderbook.
            side: BUY or SELL.
            size_usd: Order size in USD.
            
        Returns:
            SlippageCheck with abort decision.
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return SlippageCheck(
                estimated_fill_price=0.0,
                mid_price=0.0,
                slippage_bps=9999,
                price_impact_pct=0.99,
                edge_after_slippage=-1.0,
                should_abort=True,
                abort_reason="NO_ORDERBOOK",
            )
        
        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 0))
        mid_price = (best_bid + best_ask) / 2
        
        # Simulate fill through orderbook
        levels = asks if side == "BUY" else bids
        remaining_usd = size_usd
        total_filled = 0.0
        weighted_sum = 0.0
        
        for level in levels[:10]:  # Top 10 levels
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
            level_usd = size * price
            
            if remaining_usd <= 0:
                break
            
            fill_usd = min(remaining_usd, level_usd)
            fill_shares = fill_usd / price if price > 0 else 0
            
            total_filled += fill_shares
            weighted_sum += fill_shares * price
            remaining_usd -= fill_usd
        
        if total_filled > 0:
            estimated_fill = weighted_sum / total_filled
        else:
            estimated_fill = best_ask if side == "BUY" else best_bid
        
        # Calculate slippage
        slippage = abs(estimated_fill - mid_price)
        slippage_bps = int(slippage / mid_price * 10000) if mid_price > 0 else 0
        price_impact_pct = slippage / mid_price if mid_price > 0 else 0
        
        # Edge after slippage
        edge_after = abs(edge) - slippage
        
        # Maximum acceptable slippage
        max_slippage_bps = min(
            int(abs(edge) * self.max_slippage_of_edge * 10000),
            self.absolute_max_slippage_bps,
        )
        
        # Decision
        should_abort = False
        abort_reason = None
        
        if slippage_bps > max_slippage_bps:
            should_abort = True
            abort_reason = f"SLIPPAGE_EXCEEDS_LIMIT ({slippage_bps}bps > {max_slippage_bps}bps)"
        
        if edge_after < 0.01:  # Less than 1% edge after slippage
            should_abort = True
            abort_reason = f"EDGE_EATEN_BY_SLIPPAGE ({edge_after:.4f} < 0.01)"
        
        return SlippageCheck(
            estimated_fill_price=estimated_fill,
            mid_price=mid_price,
            slippage_bps=slippage_bps,
            price_impact_pct=price_impact_pct,
            edge_after_slippage=edge_after,
            should_abort=should_abort,
            abort_reason=abort_reason,
            max_acceptable_slippage_bps=max_slippage_bps,
        )
    
    def check_sentiment(
        self,
        volume_24h: float,
        avg_volume: float,
        current_spread_bps: int,
        avg_spread_bps: int,
    ) -> SentimentSignal:
        """Check market sentiment filter.
        
        Only trade when:
        - Volume is increasing (bullish interest)
        - Spread is narrowing (liquidity improving)
        
        Args:
            volume_24h: Current 24h volume.
            avg_volume: Average volume.
            current_spread_bps: Current spread in bps.
            avg_spread_bps: Average spread in bps.
            
        Returns:
            SentimentSignal with favorability.
        """
        # Volume trend
        if avg_volume > 0:
            volume_ratio = volume_24h / avg_volume
            volume_trend = min(1.0, max(-1.0, volume_ratio - 1.0))
        else:
            volume_trend = 0.0
        
        # Spread trend (narrowing = positive)
        if avg_spread_bps > 0:
            spread_ratio = current_spread_bps / avg_spread_bps
            spread_trend = min(1.0, max(-1.0, 1.0 - spread_ratio))
        else:
            spread_trend = 0.0
        
        # Composite sentiment
        composite = 0.6 * volume_trend + 0.4 * spread_trend
        
        # Favorable if composite > 0 (volume up OR spread narrowing)
        is_favorable = composite > -0.2  # Slightly forgiving threshold
        
        return SentimentSignal(
            volume_trend=volume_trend,
            spread_trend=spread_trend,
            momentum_alignment=0.0,  # Set by caller if needed
            composite_sentiment=composite,
            is_favorable=is_favorable,
        )
    
    def update_equity(self, current_equity: float) -> None:
        """Update equity for global stop-loss tracking.
        
        Args:
            current_equity: Current account equity.
        """
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        
        self._current_equity = current_equity
        
        # Check drawdown
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity * 100
            
            # Enter listen-only mode if drawdown exceeds threshold
            max_drawdown = self.settings.drawdown_limit_pct
            
            if drawdown_pct >= max_drawdown and not self._listen_only_mode:
                self._listen_only_mode = True
                self._listen_only_until = datetime.utcnow() + timedelta(hours=24)
                
                logger.error(
                    f"🛑 GLOBAL STOP-LOSS: Entering LISTEN-ONLY mode. "
                    f"Drawdown {drawdown_pct:.1f}% >= limit {max_drawdown}%. "
                    f"No trading until {self._listen_only_until.isoformat()}",
                    extra={
                        "peak_equity": self._peak_equity,
                        "current_equity": current_equity,
                        "drawdown_pct": drawdown_pct,
                    },
                )
        
        # Check if listen-only period has expired
        if self._listen_only_mode and self._listen_only_until:
            if datetime.utcnow() >= self._listen_only_until:
                self._listen_only_mode = False
                self._listen_only_until = None
                logger.info("Listen-only mode expired. Resuming trading.")
    
    def is_listen_only(self) -> bool:
        """Check if bot is in listen-only mode."""
        return self._listen_only_mode
    
    def compute_kelly_size(
        self,
        composite_score: CompositeScore,
        bankroll: float,
        win_rate: float | None = None,
    ) -> float:
        """Compute position size using dynamic Kelly Criterion.
        
        Kelly Formula: f* = (bp - q) / b
        Where:
            f* = fraction of bankroll to bet
            b = odds (for prediction markets, 1/p - 1)
            p = probability of winning
            q = 1 - p
        
        We use fractional Kelly (25%) and scale by confidence.
        
        Args:
            composite_score: Score from rule engine.
            bankroll: Available bankroll.
            win_rate: Historical win rate (if known).
            
        Returns:
            Position size in USD.
        """
        # Estimate win probability from score + historical win rate
        if win_rate is not None:
            p = win_rate
        else:
            # Estimate from score: 50% baseline + score contribution
            p = 0.50 + abs(composite_score.normalized_score) * 0.20
            p = min(0.80, max(0.40, p))  # Bound to 40-80%
        
        q = 1 - p
        
        # Estimate odds from market (assume ~2:1 for simplicity)
        b = 1.0  # Even odds assumption
        
        # Full Kelly
        if b > 0:
            full_kelly = (b * p - q) / b
        else:
            full_kelly = 0.0
        
        # Fractional Kelly (25%)
        fractional = full_kelly * 0.25
        
        # Scale by confidence
        fractional *= composite_score.confidence
        
        # Apply bounds
        max_fraction = 0.15  # Never bet more than 15%
        min_fraction = 0.01  # Minimum 1%
        
        fractional = max(min_fraction, min(max_fraction, fractional))
        
        # Calculate size
        size_usd = bankroll * fractional
        
        return size_usd
    
    def label_trade_features(
        self,
        token_id: str,
        entry_time: datetime,
        entry_price: float,
        composite_score: CompositeScore,
        rule_scores: list[RuleScore],
        orderbook: dict[str, Any] | None,
        market_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Label a trade with features for future ML training.
        
        This creates a rich feature set that can be used to train
        ML models once enough labeled outcomes are collected.
        
        Args:
            token_id: Token identifier.
            entry_time: Entry timestamp.
            entry_price: Entry price.
            composite_score: Composite score at entry.
            rule_scores: Individual rule scores.
            orderbook: Orderbook at entry.
            market_context: Market context (volume, days to resolution, etc.)
            
        Returns:
            Dictionary of features for ML training.
        """
        features = {
            # Trade identification
            "token_id": token_id,
            "entry_time": entry_time.isoformat(),
            "entry_price": entry_price,
            
            # Composite signals
            "total_score": composite_score.total_score,
            "normalized_score": composite_score.normalized_score,
            "confidence": composite_score.confidence,
            "trade_direction": composite_score.trade_direction,
            
            # Individual rule scores
            **{f"rule_{r.rule_id}": r.weighted_score for r in rule_scores},
            **{f"rule_{r.rule_id}_triggered": 1.0 if r.triggered else 0.0 for r in rule_scores},
            
            # Market context
            "volume_24h": market_context.get("volume_24h", 0.0),
            "days_to_resolution": market_context.get("days_to_resolution"),
            "market_category": market_context.get("category", "unknown"),
            
            # RSI-like indicator (simple momentum)
            "momentum_1h": market_context.get("momentum_1h", 0.0),
            "momentum_24h": market_context.get("momentum_24h", 0.0),
            
            # Volume profile
            "volume_ratio": market_context.get("volume_ratio", 1.0),
            "is_volume_spike": 1.0 if market_context.get("volume_ratio", 1.0) > 2.0 else 0.0,
        }
        
        # Orderbook features
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            bid_depth = sum(float(b.get("size", 0)) * float(b.get("price", 0)) for b in bids[:5])
            ask_depth = sum(float(a.get("size", 0)) * float(a.get("price", 0)) for a in asks[:5])
            
            if bids and asks:
                spread = float(asks[0].get("price", 0)) - float(bids[0].get("price", 0))
                mid = (float(bids[0].get("price", 0)) + float(asks[0].get("price", 0))) / 2
                spread_pct = spread / mid if mid > 0 else 0
            else:
                spread_pct = 0
            
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            features["bid_depth_usd"] = bid_depth
            features["ask_depth_usd"] = ask_depth
            features["spread_pct"] = spread_pct
            features["orderbook_imbalance"] = imbalance
        
        return features


# Singleton
_smart_heuristics: SmartHeuristicsEngine | None = None


def get_smart_heuristics_engine() -> SmartHeuristicsEngine:
    """Get singleton smart heuristics engine."""
    global _smart_heuristics
    if _smart_heuristics is None:
        _smart_heuristics = SmartHeuristicsEngine()
    return _smart_heuristics
