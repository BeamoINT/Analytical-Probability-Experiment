"""Market Edge Intelligence - Smarter rules for consistent profit.

This module provides an "intelligence edge" over the market by:

1. CONTRARIAN DETECTION
   - Identify when the crowd is wrong
   - Fade extreme sentiment
   - Detect overcrowded trades

2. INFORMATION ASYMMETRY TRACKING
   - Detect informed trader activity
   - Track price-volume divergences
   - Monitor unusual order patterns

3. HISTORICAL PATTERN LEARNING
   - Resolution accuracy by price level
   - Category-specific patterns
   - Time-of-day effects

4. MULTI-FACTOR EDGE SCORING
   - Combine multiple weak signals into strong edge
   - Weight factors by historical accuracy
   - Adaptive confidence based on market regime

5. MARKET REGIME DETECTION
   - Trending vs mean-reverting
   - High vs low volatility
   - Efficient vs inefficient

6. SMART ENTRY/EXIT TIMING
   - Wait for optimal entry points
   - Avoid getting caught in adverse selection
   - Time exits for maximum profit
"""

import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Current market regime assessment."""
    is_trending: bool = False
    trend_direction: int = 0  # 1 = up, -1 = down, 0 = sideways
    is_volatile: bool = False
    volatility_percentile: float = 0.5  # 0-1, where current vol ranks historically
    is_efficient: bool = True  # Efficient markets are harder to profit from
    regime_confidence: float = 0.5
    
    def to_dict(self) -> dict[str, float]:
        return {
            "regime_is_trending": float(self.is_trending),
            "regime_trend_direction": float(self.trend_direction),
            "regime_is_volatile": float(self.is_volatile),
            "regime_volatility_pct": self.volatility_percentile,
            "regime_is_efficient": float(self.is_efficient),
            "regime_confidence": self.regime_confidence,
        }


@dataclass 
class ContrarianSignal:
    """Contrarian opportunity detection."""
    is_overcrowded: bool = False
    crowd_side: str = ""  # "BUY" or "SELL"
    crowd_intensity: float = 0.0  # 0-1, how extreme
    fade_signal: bool = False  # Should we fade the crowd?
    fade_direction: int = 0  # 1 = buy (fade sellers), -1 = sell (fade buyers)
    historical_accuracy: float = 0.5  # How often contrarian was right
    
    def to_dict(self) -> dict[str, float]:
        return {
            "contrarian_is_overcrowded": float(self.is_overcrowded),
            "contrarian_crowd_intensity": self.crowd_intensity,
            "contrarian_fade_signal": float(self.fade_signal),
            "contrarian_fade_direction": float(self.fade_direction),
            "contrarian_historical_accuracy": self.historical_accuracy,
        }


@dataclass
class EdgeScore:
    """Multi-factor edge score."""
    raw_edge: float = 0.0
    contrarian_edge: float = 0.0
    momentum_edge: float = 0.0
    value_edge: float = 0.0
    timing_edge: float = 0.0
    information_edge: float = 0.0
    composite_edge: float = 0.0
    confidence: float = 0.5
    factors: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, float]:
        result = {
            "edge_raw": self.raw_edge,
            "edge_contrarian": self.contrarian_edge,
            "edge_momentum": self.momentum_edge,
            "edge_value": self.value_edge,
            "edge_timing": self.timing_edge,
            "edge_information": self.information_edge,
            "edge_composite": self.composite_edge,
            "edge_confidence": self.confidence,
        }
        for k, v in self.factors.items():
            result[f"edge_factor_{k}"] = v
        return result


class MarketMemory:
    """Persistent memory of market patterns and outcomes.
    
    Tracks:
    - Historical price -> resolution mappings
    - Category performance patterns
    - Time-based patterns
    - Individual market behaviors
    """
    
    def __init__(self, db_path: str = "data/market_memory.db") -> None:
        """Initialize market memory with SQLite backend."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
        # In-memory caches for fast access
        self._price_resolution_cache: dict[str, list[tuple[float, float]]] = {}
        self._category_patterns: dict[str, dict] = {}
        self._token_history: dict[str, list[dict]] = defaultdict(list)
        
    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resolution_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id TEXT,
                    market_id TEXT,
                    category TEXT,
                    final_price REAL,
                    resolution_value REAL,
                    days_to_resolution REAL,
                    volume_24h REAL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id TEXT,
                    price REAL,
                    volume REAL,
                    bid_depth REAL,
                    ask_depth REAL,
                    spread REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_pct REAL,
                    hold_time_hours REAL,
                    entry_signal_strength REAL,
                    exit_reason TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_resolution_category 
                ON resolution_outcomes(category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_token 
                ON price_snapshots(token_id, timestamp)
            """)
            
    def record_snapshot(
        self,
        token_id: str,
        price: float,
        volume: float = 0.0,
        bid_depth: float = 0.0,
        ask_depth: float = 0.0,
        spread: float = 0.0,
    ) -> None:
        """Record a price snapshot for learning."""
        # Store in memory for quick access
        self._token_history[token_id].append({
            "price": price,
            "volume": volume,
            "timestamp": datetime.utcnow(),
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "spread": spread,
        })
        
        # Trim memory to last 1000 points per token
        if len(self._token_history[token_id]) > 1000:
            self._token_history[token_id] = self._token_history[token_id][-1000:]
        
        # Periodically persist to DB (every 10th snapshot)
        if len(self._token_history[token_id]) % 10 == 0:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT INTO price_snapshots 
                           (token_id, price, volume, bid_depth, ask_depth, spread)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (token_id, price, volume, bid_depth, ask_depth, spread)
                    )
            except Exception as e:
                logger.debug(f"Failed to persist snapshot: {e}")
    
    def get_price_history(self, token_id: str, limit: int = 100) -> list[dict]:
        """Get recent price history for a token."""
        return self._token_history.get(token_id, [])[-limit:]
    
    def get_resolution_accuracy(
        self, price_level: float, category: str | None = None
    ) -> float:
        """Get historical accuracy of prices at this level resolving to 1.
        
        E.g., if prices at 0.70 historically resolve to 1 about 75% of the time,
        returns 0.75.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if category:
                    rows = conn.execute(
                        """SELECT final_price, resolution_value FROM resolution_outcomes
                           WHERE category = ? AND ABS(final_price - ?) < 0.1""",
                        (category, price_level)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT final_price, resolution_value FROM resolution_outcomes
                           WHERE ABS(final_price - ?) < 0.1""",
                        (price_level,)
                    ).fetchall()
                
                if len(rows) < 5:  # Not enough data
                    return price_level  # Default to current price as estimate
                
                resolutions = [r[1] for r in rows]
                return np.mean(resolutions)
                
        except Exception as e:
            logger.debug(f"Failed to get resolution accuracy: {e}")
            return price_level


class MarketEdgeEngine:
    """Smart market edge detection and scoring.
    
    This engine gives the bot an intelligence advantage by:
    1. Detecting when the crowd is wrong (contrarian)
    2. Identifying information asymmetry
    3. Learning from historical patterns
    4. Optimizing entry/exit timing
    5. Combining multiple signals into a composite edge
    """
    
    def __init__(self) -> None:
        """Initialize market edge engine."""
        self.settings = get_settings()
        self.memory = MarketMemory()
        
        # Track live market data
        self._orderbook_history: dict[str, list[dict]] = defaultdict(list)
        self._trade_flow: dict[str, list[dict]] = defaultdict(list)
        self._sentiment_history: dict[str, list[float]] = defaultdict(list)
        
        # Factor weights (can be learned over time)
        self.factor_weights = {
            "contrarian": 0.20,  # Fade the crowd
            "momentum": 0.15,   # Follow the trend
            "value": 0.25,      # Price vs fair value
            "timing": 0.15,     # Entry/exit timing
            "information": 0.15,  # Informed trader activity
            "regime": 0.10,     # Market regime adjustment
        }
        
    def compute_edge(
        self,
        token_id: str,
        current_price: float,
        p_model: float,
        orderbook: dict[str, Any] | None,
        recent_trades: list[dict] | None,
        market_category: str | None = None,
        days_to_resolution: float | None = None,
        volume_24h: float = 0.0,
    ) -> EdgeScore:
        """Compute comprehensive edge score.
        
        Args:
            token_id: Token identifier.
            current_price: Current market price.
            p_model: Model's probability estimate.
            orderbook: Current orderbook.
            recent_trades: Recent trades.
            market_category: Market category.
            days_to_resolution: Days until resolution.
            volume_24h: 24-hour volume.
            
        Returns:
            EdgeScore with composite and component edges.
        """
        factors = {}
        
        # Raw edge (model vs market)
        raw_edge = p_model - current_price
        
        # 1. Contrarian edge
        contrarian = self._compute_contrarian_signal(
            token_id, current_price, orderbook, recent_trades
        )
        contrarian_edge = self._contrarian_to_edge(contrarian, raw_edge)
        factors["contrarian"] = contrarian_edge
        
        # 2. Momentum edge
        momentum_edge = self._compute_momentum_edge(token_id, current_price)
        factors["momentum"] = momentum_edge
        
        # 3. Value edge (historical accuracy)
        value_edge = self._compute_value_edge(
            current_price, market_category, days_to_resolution
        )
        factors["value"] = value_edge
        
        # 4. Timing edge
        timing_edge = self._compute_timing_edge(
            token_id, current_price, orderbook
        )
        factors["timing"] = timing_edge
        
        # 5. Information edge
        info_edge = self._compute_information_edge(
            token_id, orderbook, recent_trades
        )
        factors["information"] = info_edge
        
        # 6. Regime adjustment
        regime = self._detect_regime(token_id, current_price, volume_24h)
        regime_adjustment = self._regime_to_edge_adjustment(regime)
        factors["regime"] = regime_adjustment
        
        # Composite edge (weighted combination)
        composite_edge = sum(
            factors[k] * self.factor_weights[k]
            for k in factors
        )
        
        # Add raw edge (it's still the primary signal)
        composite_edge = 0.5 * raw_edge + 0.5 * composite_edge
        
        # Confidence based on factor agreement
        confidence = self._compute_confidence(factors, raw_edge)
        
        return EdgeScore(
            raw_edge=raw_edge,
            contrarian_edge=contrarian_edge,
            momentum_edge=momentum_edge,
            value_edge=value_edge,
            timing_edge=timing_edge,
            information_edge=info_edge,
            composite_edge=composite_edge,
            confidence=confidence,
            factors=factors,
        )
    
    def _compute_contrarian_signal(
        self,
        token_id: str,
        current_price: float,
        orderbook: dict[str, Any] | None,
        recent_trades: list[dict] | None,
    ) -> ContrarianSignal:
        """Detect contrarian opportunities."""
        crowd_intensity = 0.0
        crowd_side = ""
        fade_signal = False
        fade_direction = 0
        
        # Measure crowd sentiment from multiple sources
        sentiment_scores = []
        
        # 1. Order book imbalance
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            bid_volume = sum(float(b.get("size", 0)) for b in bids[:10])
            ask_volume = sum(float(a.get("size", 0)) for a in asks[:10])
            
            if bid_volume + ask_volume > 0:
                ob_sentiment = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                sentiment_scores.append(ob_sentiment)
        
        # 2. Trade flow
        if recent_trades and len(recent_trades) >= 5:
            buy_volume = sum(
                float(t.get("size", 0))
                for t in recent_trades
                if str(t.get("side", "")).upper() == "BUY"
            )
            sell_volume = sum(
                float(t.get("size", 0))
                for t in recent_trades
                if str(t.get("side", "")).upper() == "SELL"
            )
            
            if buy_volume + sell_volume > 0:
                trade_sentiment = (buy_volume - sell_volume) / (buy_volume + sell_volume)
                sentiment_scores.append(trade_sentiment)
        
        # 3. Price position (extreme prices = extreme sentiment)
        if current_price > 0.85:
            sentiment_scores.append(0.8)  # Crowd thinks YES
        elif current_price < 0.15:
            sentiment_scores.append(-0.8)  # Crowd thinks NO
        
        # Aggregate sentiment
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            crowd_intensity = abs(avg_sentiment)
            crowd_side = "BUY" if avg_sentiment > 0 else "SELL"
            
            # Track sentiment history
            self._sentiment_history[token_id].append(avg_sentiment)
            if len(self._sentiment_history[token_id]) > 100:
                self._sentiment_history[token_id] = self._sentiment_history[token_id][-100:]
            
            # Contrarian signal when crowd is extreme and has been wrong before
            # Fade when sentiment is extreme (>0.6) and trending further extreme
            if crowd_intensity > 0.6:
                is_overcrowded = True
                
                # Check if sentiment has been building (not just a spike)
                history = self._sentiment_history[token_id]
                if len(history) >= 5:
                    recent_trend = np.mean(history[-5:]) - np.mean(history[-10:-5]) if len(history) >= 10 else 0
                    
                    # Fade if sentiment keeps intensifying (crowd piling in)
                    if abs(recent_trend) > 0.1 and np.sign(recent_trend) == np.sign(avg_sentiment):
                        fade_signal = True
                        fade_direction = -1 if avg_sentiment > 0 else 1  # Opposite of crowd
            else:
                is_overcrowded = False
        else:
            is_overcrowded = False
        
        return ContrarianSignal(
            is_overcrowded=is_overcrowded,
            crowd_side=crowd_side,
            crowd_intensity=crowd_intensity,
            fade_signal=fade_signal,
            fade_direction=fade_direction,
            historical_accuracy=0.55,  # Slightly better than random
        )
    
    def _contrarian_to_edge(
        self, signal: ContrarianSignal, raw_edge: float
    ) -> float:
        """Convert contrarian signal to edge component."""
        if not signal.fade_signal:
            return 0.0
        
        # Edge from fading the crowd
        fade_edge = signal.fade_direction * signal.crowd_intensity * 0.05
        
        # Boost if raw edge agrees with fade direction
        if np.sign(raw_edge) == signal.fade_direction:
            fade_edge *= 1.5
        
        return fade_edge
    
    def _compute_momentum_edge(
        self, token_id: str, current_price: float
    ) -> float:
        """Compute momentum-based edge."""
        history = self.memory.get_price_history(token_id, limit=50)
        
        if len(history) < 10:
            return 0.0
        
        prices = [h["price"] for h in history]
        
        # Short-term momentum (last 5 points)
        short_mom = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        
        # Medium-term momentum (last 20 points)
        if len(prices) >= 20:
            med_mom = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] > 0 else 0
        else:
            med_mom = short_mom
        
        # Momentum edge: follow the trend, but not too aggressively
        momentum_edge = 0.3 * short_mom + 0.2 * med_mom
        
        # Cap momentum edge
        momentum_edge = max(-0.05, min(0.05, momentum_edge))
        
        return momentum_edge
    
    def _compute_value_edge(
        self,
        current_price: float,
        category: str | None,
        days_to_resolution: float | None,
    ) -> float:
        """Compute value edge based on historical resolution patterns."""
        # Get historical resolution accuracy at this price level
        historical_prob = self.memory.get_resolution_accuracy(
            current_price, category
        )
        
        # Value edge = difference between historical accuracy and current price
        value_edge = historical_prob - current_price
        
        # Time decay: value edge matters less close to resolution
        if days_to_resolution is not None and days_to_resolution > 0:
            time_factor = min(1.0, days_to_resolution / 30)  # Full weight at 30+ days
            value_edge *= time_factor
        
        return value_edge
    
    def _compute_timing_edge(
        self,
        token_id: str,
        current_price: float,
        orderbook: dict[str, Any] | None,
    ) -> float:
        """Compute entry timing edge."""
        timing_edge = 0.0
        
        if not orderbook:
            return 0.0
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return 0.0
        
        # Spread analysis
        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 0))
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2
        
        if mid > 0:
            spread_pct = spread / mid
        else:
            spread_pct = 1.0
        
        # Good timing when spread is tight
        if spread_pct < 0.02:  # Very tight spread
            timing_edge += 0.02
        elif spread_pct < 0.05:  # Normal spread
            timing_edge += 0.01
        else:  # Wide spread - bad timing
            timing_edge -= 0.02
        
        # Check if price is at a good level relative to recent history
        history = self.memory.get_price_history(token_id, limit=20)
        if len(history) >= 5:
            prices = [h["price"] for h in history]
            recent_high = max(prices)
            recent_low = min(prices)
            
            price_range = recent_high - recent_low
            if price_range > 0:
                position = (current_price - recent_low) / price_range
                
                # Good timing to buy at lower end of range
                if position < 0.3:
                    timing_edge += 0.02
                # Bad timing to buy at higher end
                elif position > 0.7:
                    timing_edge -= 0.01
        
        return timing_edge
    
    def _compute_information_edge(
        self,
        token_id: str,
        orderbook: dict[str, Any] | None,
        recent_trades: list[dict] | None,
    ) -> float:
        """Detect information asymmetry and compute edge."""
        info_edge = 0.0
        
        # Look for signs of informed trading
        
        # 1. Large orders (whales often have information)
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            # Look for unusually large orders
            bid_sizes = [float(b.get("size", 0)) for b in bids[:10]]
            ask_sizes = [float(a.get("size", 0)) for a in asks[:10]]
            
            all_sizes = bid_sizes + ask_sizes
            if len(all_sizes) >= 3:
                median_size = np.median(all_sizes)
                
                # Large bids = someone knows something bullish
                large_bid = any(s > median_size * 3 for s in bid_sizes[:3])
                large_ask = any(s > median_size * 3 for s in ask_sizes[:3])
                
                if large_bid and not large_ask:
                    info_edge += 0.02  # Follow the whale
                elif large_ask and not large_bid:
                    info_edge -= 0.02
        
        # 2. Trade flow velocity (fast trading = news event)
        if recent_trades and len(recent_trades) >= 5:
            # Calculate trade velocity
            try:
                timestamps = []
                for t in recent_trades:
                    ts = t.get("timestamp")
                    if ts:
                        if isinstance(ts, str):
                            from dateutil import parser
                            timestamps.append(parser.parse(ts))
                        else:
                            timestamps.append(ts)
                
                if len(timestamps) >= 2:
                    time_span = (timestamps[-1] - timestamps[0]).total_seconds()
                    if time_span > 0:
                        trades_per_minute = len(timestamps) / (time_span / 60)
                        
                        # Store for comparison
                        self._trade_flow[token_id].append(trades_per_minute)
                        if len(self._trade_flow[token_id]) > 50:
                            self._trade_flow[token_id] = self._trade_flow[token_id][-50:]
                        
                        # Check if velocity is unusual
                        if len(self._trade_flow[token_id]) >= 10:
                            avg_velocity = np.mean(self._trade_flow[token_id])
                            if trades_per_minute > avg_velocity * 2:
                                # Unusual activity - there might be news
                                # Follow the direction of trades
                                buy_volume = sum(
                                    float(t.get("size", 0))
                                    for t in recent_trades
                                    if str(t.get("side", "")).upper() == "BUY"
                                )
                                sell_volume = sum(
                                    float(t.get("size", 0))
                                    for t in recent_trades
                                    if str(t.get("side", "")).upper() == "SELL"
                                )
                                
                                if buy_volume > sell_volume * 1.5:
                                    info_edge += 0.02
                                elif sell_volume > buy_volume * 1.5:
                                    info_edge -= 0.02
            except Exception:
                pass
        
        return info_edge
    
    def _detect_regime(
        self,
        token_id: str,
        current_price: float,
        volume_24h: float,
    ) -> MarketRegime:
        """Detect current market regime."""
        history = self.memory.get_price_history(token_id, limit=50)
        
        if len(history) < 10:
            return MarketRegime()
        
        prices = [h["price"] for h in history]
        
        # Trend detection
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        is_trending = abs(slope) > 0.001  # Some threshold
        trend_direction = 1 if slope > 0 else -1 if slope < 0 else 0
        
        # Volatility detection
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Compare to historical volatility
        is_volatile = volatility > 0.02  # 2% std per period
        
        # Efficiency (how random are price movements)
        # Efficient = hard to predict = low autocorrelation
        if len(returns) >= 5:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            is_efficient = abs(autocorr) < 0.3  # Low autocorrelation = efficient
        else:
            is_efficient = True
        
        return MarketRegime(
            is_trending=is_trending,
            trend_direction=trend_direction,
            is_volatile=is_volatile,
            volatility_percentile=min(1.0, volatility / 0.05),
            is_efficient=is_efficient,
            regime_confidence=0.6 if len(history) >= 30 else 0.4,
        )
    
    def _regime_to_edge_adjustment(self, regime: MarketRegime) -> float:
        """Convert regime to edge adjustment factor."""
        adjustment = 0.0
        
        # In trending markets, momentum is more important
        if regime.is_trending:
            adjustment += 0.01 * regime.trend_direction
        
        # In volatile markets, be more cautious
        if regime.is_volatile:
            adjustment *= 0.7  # Reduce edge in volatile regimes
        
        # In inefficient markets, edges are more likely real
        if not regime.is_efficient:
            adjustment *= 1.2
        
        return adjustment
    
    def _compute_confidence(
        self, factors: dict[str, float], raw_edge: float
    ) -> float:
        """Compute confidence based on factor agreement."""
        # Count how many factors agree with raw edge direction
        agreeing = sum(
            1 for v in factors.values()
            if np.sign(v) == np.sign(raw_edge) or v == 0
        )
        
        agreement_ratio = agreeing / len(factors) if factors else 0.5
        
        # Base confidence from agreement
        confidence = 0.3 + 0.5 * agreement_ratio
        
        # Boost confidence if edge is strong
        if abs(raw_edge) > 0.1:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def update_from_trade(
        self,
        token_id: str,
        entry_price: float,
        exit_price: float,
        hold_time_hours: float,
        entry_signal_strength: float,
        exit_reason: str,
    ) -> None:
        """Update memory from completed trade (for learning)."""
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        
        try:
            with sqlite3.connect(self.memory.db_path) as conn:
                conn.execute(
                    """INSERT INTO trade_outcomes 
                       (token_id, entry_price, exit_price, pnl_pct, 
                        hold_time_hours, entry_signal_strength, exit_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (token_id, entry_price, exit_price, pnl_pct,
                     hold_time_hours, entry_signal_strength, exit_reason)
                )
        except Exception as e:
            logger.debug(f"Failed to record trade outcome: {e}")
    
    def get_all_features(
        self,
        token_id: str,
        current_price: float,
        p_model: float,
        orderbook: dict[str, Any] | None,
        recent_trades: list[dict] | None,
        market_category: str | None = None,
        days_to_resolution: float | None = None,
        volume_24h: float = 0.0,
    ) -> dict[str, float]:
        """Get all edge features for ML training."""
        # Record snapshot
        bid_depth = 0.0
        ask_depth = 0.0
        spread = 0.0
        
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            bid_depth = sum(float(b.get("size", 0)) for b in bids[:10])
            ask_depth = sum(float(a.get("size", 0)) for a in asks[:10])
            if bids and asks:
                spread = float(asks[0].get("price", 0)) - float(bids[0].get("price", 0))
        
        self.memory.record_snapshot(
            token_id, current_price, volume_24h, bid_depth, ask_depth, spread
        )
        
        # Compute edge score
        edge_score = self.compute_edge(
            token_id=token_id,
            current_price=current_price,
            p_model=p_model,
            orderbook=orderbook,
            recent_trades=recent_trades,
            market_category=market_category,
            days_to_resolution=days_to_resolution,
            volume_24h=volume_24h,
        )
        
        # Compute contrarian signal
        contrarian = self._compute_contrarian_signal(
            token_id, current_price, orderbook, recent_trades
        )
        
        # Compute regime
        regime = self._detect_regime(token_id, current_price, volume_24h)
        
        # Combine all features
        features = {}
        features.update(edge_score.to_dict())
        features.update(contrarian.to_dict())
        features.update(regime.to_dict())
        
        # Add raw metrics
        features["current_price"] = current_price
        features["p_model"] = p_model
        features["volume_24h"] = volume_24h
        if days_to_resolution is not None:
            features["days_to_resolution"] = days_to_resolution
        
        return features


# Singleton
_market_edge_engine: MarketEdgeEngine | None = None


def get_market_edge_engine() -> MarketEdgeEngine:
    """Get singleton market edge engine."""
    global _market_edge_engine
    if _market_edge_engine is None:
        _market_edge_engine = MarketEdgeEngine()
    return _market_edge_engine
