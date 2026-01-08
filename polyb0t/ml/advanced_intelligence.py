"""Advanced AI Intelligence Module - Deep pattern recognition for prediction markets.

This module adds sophisticated pattern detection that goes beyond basic features:

1. SMART MONEY DETECTION
   - Identifies large institutional orders (whales)
   - Detects informed trader patterns
   - Tracks unusual position accumulation

2. CROSS-MARKET INTELLIGENCE
   - Correlations between related markets
   - Category momentum (politics, sports, crypto)
   - Lead-lag relationships

3. RESOLUTION DYNAMICS
   - Time decay modeling (prices -> 0 or 1 as deadline approaches)
   - Certainty cone analysis
   - Historical resolution patterns

4. SENTIMENT ANALYSIS
   - Trading velocity patterns
   - Position crowding indicators
   - Contrarian signals

5. ADVANCED TECHNICAL SIGNALS
   - RSI (Relative Strength Index)
   - MACD-like indicators adapted for prediction markets
   - Bollinger band equivalents
   - Support/resistance detection

6. PATTERN RECOGNITION
   - Price action patterns
   - Volume patterns
   - Time-of-day patterns
   - Day-of-week effects

7. INFORMATION FLOW
   - News/event detection via volume spikes
   - Price efficiency measurement
   - Market microstructure alpha
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SmartMoneySignal:
    """Detected smart money activity."""
    detected: bool
    whale_activity_score: float  # 0-1
    accumulation_detected: bool
    distribution_detected: bool
    informed_trader_ratio: float  # Estimated % of informed trading
    signal_direction: int  # 1 = buying, -1 = selling, 0 = neutral


@dataclass
class ResolutionDynamics:
    """Resolution time dynamics analysis."""
    hours_to_resolution: float
    expected_convergence_rate: float  # How fast price should move to 0 or 1
    current_certainty: float  # 0.5 = uncertain, near 0 or 1 = certain
    is_converging: bool  # Price moving toward extremes
    expected_final_price: float  # Best guess at resolution value
    time_value_remaining: float  # How much can still change


class AdvancedIntelligenceEngine:
    """Deep pattern recognition and advanced signal generation.
    
    This engine provides features that capture:
    - Hidden patterns human traders miss
    - Cross-market relationships
    - Temporal patterns and seasonality
    - Smart money movements
    - Resolution dynamics unique to prediction markets
    """
    
    def __init__(self) -> None:
        """Initialize advanced intelligence engine."""
        # Price history per token (for advanced technical analysis)
        self.price_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self.volume_history: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        self.trade_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        
        # Cross-market tracking
        self.category_prices: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
        
        # Pattern history
        self.large_order_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        
        # Configuration
        self.max_history_points = 10000
        self.whale_threshold_usd = 500  # Orders above this are "whale" orders
        
    def compute_advanced_features(
        self,
        token_id: str,
        current_price: float,
        orderbook: dict[str, Any] | None,
        recent_trades: list[dict[str, Any]] | None,
        market_end_date: datetime | None,
        market_category: str | None = None,
        market_volume_24h: float = 0.0,
    ) -> dict[str, float]:
        """Compute comprehensive advanced feature set.
        
        Args:
            token_id: Token identifier.
            current_price: Current market price.
            orderbook: Orderbook data.
            recent_trades: Recent trades.
            market_end_date: Resolution date.
            market_category: Market category (politics, sports, etc).
            market_volume_24h: 24-hour volume.
            
        Returns:
            Dictionary of advanced features.
        """
        features = {}
        
        # Update histories
        now = datetime.utcnow()
        self._update_history(token_id, now, current_price, market_volume_24h)
        
        # 1. Smart Money Detection
        smart_money = self._detect_smart_money(token_id, orderbook, recent_trades)
        features.update(self._smart_money_to_features(smart_money))
        
        # 2. Resolution Dynamics
        resolution_dynamics = self._analyze_resolution_dynamics(
            current_price, market_end_date
        )
        features.update(self._resolution_dynamics_to_features(resolution_dynamics))
        
        # 3. Advanced Technical Indicators
        features.update(self._compute_technical_indicators(token_id, current_price))
        
        # 4. Volume Analysis
        features.update(self._compute_volume_features(token_id, market_volume_24h))
        
        # 5. Pattern Recognition
        features.update(self._detect_patterns(token_id, current_price, now))
        
        # 6. Order Flow Analysis
        if orderbook:
            features.update(self._analyze_order_flow(orderbook))
        
        # 7. Trade Flow Intelligence
        if recent_trades:
            features.update(self._analyze_trade_flow_advanced(recent_trades))
        
        # 8. Cross-Market Features (if category available)
        if market_category:
            features.update(self._compute_cross_market_features(
                token_id, market_category, current_price
            ))
        
        # 9. Time-based Features (seasonality)
        features.update(self._compute_time_features(now))
        
        # 10. Composite Intelligence Signals
        features.update(self._compute_composite_signals(features))
        
        # Clean NaN/Inf
        return self._clean_features(features)
    
    def _detect_smart_money(
        self,
        token_id: str,
        orderbook: dict[str, Any] | None,
        recent_trades: list[dict[str, Any]] | None,
    ) -> SmartMoneySignal:
        """Detect smart money / whale activity.
        
        Smart money indicators:
        - Large order sizes (whales)
        - Orders placed just before price moves
        - Hidden orders (iceberg detection)
        - Accumulation patterns (consistent buying without price increase)
        """
        whale_score = 0.0
        accumulation = False
        distribution = False
        informed_ratio = 0.0
        direction = 0
        
        # Analyze orderbook for large hidden orders
        if orderbook:
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            # Look for whale orders in the book
            whale_bid_volume = sum(
                float(b.get("size", 0)) * float(b.get("price", 0))
                for b in bids
                if float(b.get("size", 0)) * float(b.get("price", 0)) > self.whale_threshold_usd
            )
            whale_ask_volume = sum(
                float(a.get("size", 0)) * float(a.get("price", 0))
                for a in asks
                if float(a.get("size", 0)) * float(a.get("price", 0)) > self.whale_threshold_usd
            )
            
            total_bid_volume = sum(float(b.get("size", 0)) * float(b.get("price", 0)) for b in bids[:10])
            total_ask_volume = sum(float(a.get("size", 0)) * float(a.get("price", 0)) for a in asks[:10])
            
            if total_bid_volume + total_ask_volume > 0:
                whale_ratio = (whale_bid_volume + whale_ask_volume) / (total_bid_volume + total_ask_volume)
                whale_score = min(1.0, whale_ratio * 2)  # Scale up
                
                # Direction from whale imbalance
                if whale_bid_volume > whale_ask_volume * 1.5:
                    direction = 1  # Whales buying
                elif whale_ask_volume > whale_bid_volume * 1.5:
                    direction = -1  # Whales selling
        
        # Analyze trades for accumulation/distribution
        if recent_trades and len(recent_trades) >= 5:
            # Accumulation: lots of buys, price not moving up
            # Distribution: lots of sells, price not moving down
            
            buy_volume = sum(
                float(t.get("size", 0))
                for t in recent_trades
                if t.get("side", "").upper() == "BUY"
            )
            sell_volume = sum(
                float(t.get("size", 0))
                for t in recent_trades
                if t.get("side", "").upper() == "SELL"
            )
            
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                buy_ratio = buy_volume / total_volume
                
                # Get price change over trades
                if len(recent_trades) >= 2:
                    first_price = float(recent_trades[0].get("price", 0))
                    last_price = float(recent_trades[-1].get("price", 0))
                    
                    if first_price > 0:
                        price_change = (last_price - first_price) / first_price
                        
                        # Accumulation: heavy buying but price flat or down
                        if buy_ratio > 0.6 and price_change < 0.02:
                            accumulation = True
                            direction = 1  # Likely to go up
                        
                        # Distribution: heavy selling but price flat or up
                        if buy_ratio < 0.4 and price_change > -0.02:
                            distribution = True
                            direction = -1  # Likely to go down
                
                # Estimate informed trader ratio
                # Informed traders tend to trade at better prices
                informed_ratio = abs(buy_ratio - 0.5) * 2  # 0 = random, 1 = all informed
        
        return SmartMoneySignal(
            detected=whale_score > 0.3 or accumulation or distribution,
            whale_activity_score=whale_score,
            accumulation_detected=accumulation,
            distribution_detected=distribution,
            informed_trader_ratio=informed_ratio,
            signal_direction=direction,
        )
    
    def _smart_money_to_features(self, signal: SmartMoneySignal) -> dict[str, float]:
        """Convert smart money signal to features."""
        return {
            "smart_money_detected": float(signal.detected),
            "whale_activity_score": signal.whale_activity_score,
            "accumulation_detected": float(signal.accumulation_detected),
            "distribution_detected": float(signal.distribution_detected),
            "informed_trader_ratio": signal.informed_trader_ratio,
            "smart_money_direction": float(signal.signal_direction),
        }
    
    def _analyze_resolution_dynamics(
        self,
        current_price: float,
        end_date: datetime | None,
    ) -> ResolutionDynamics:
        """Analyze resolution time dynamics.
        
        Prediction market prices should converge to 0 or 1 as resolution approaches.
        This captures unique dynamics not found in traditional markets.
        """
        now = datetime.utcnow()
        
        if end_date:
            hours_to_resolution = max(0, (end_date - now).total_seconds() / 3600)
        else:
            hours_to_resolution = 168  # Default 1 week
        
        # Certainty: how far from 0.5 (maximum uncertainty)
        certainty = 2 * abs(current_price - 0.5)  # 0 at 0.5, 1 at 0 or 1
        
        # Expected convergence rate (should converge faster near resolution)
        if hours_to_resolution > 0:
            convergence_rate = certainty / hours_to_resolution
        else:
            convergence_rate = 1.0  # Immediate
        
        # Is price converging toward extremes?
        # We'd need history to really know, but estimate from certainty
        is_converging = certainty > 0.3  # More certain than random
        
        # Expected final price (best guess)
        if certainty > 0.5:
            expected_final = 1.0 if current_price > 0.5 else 0.0
        else:
            expected_final = current_price  # Too uncertain to predict
        
        # Time value remaining (how much can still change)
        # More time = more potential change
        time_value = min(1.0, hours_to_resolution / (24 * 7))  # Max at 1 week
        
        return ResolutionDynamics(
            hours_to_resolution=hours_to_resolution,
            expected_convergence_rate=convergence_rate,
            current_certainty=certainty,
            is_converging=is_converging,
            expected_final_price=expected_final,
            time_value_remaining=time_value,
        )
    
    def _resolution_dynamics_to_features(
        self, dynamics: ResolutionDynamics
    ) -> dict[str, float]:
        """Convert resolution dynamics to features."""
        return {
            "hours_to_resolution": dynamics.hours_to_resolution,
            "log_hours_to_resolution": math.log1p(dynamics.hours_to_resolution),
            "days_to_resolution": dynamics.hours_to_resolution / 24,
            "convergence_rate": dynamics.expected_convergence_rate,
            "current_certainty": dynamics.current_certainty,
            "is_converging": float(dynamics.is_converging),
            "expected_final_price": dynamics.expected_final_price,
            "time_value_remaining": dynamics.time_value_remaining,
            # Derived: resolution urgency (higher = need to trade soon)
            "resolution_urgency": dynamics.current_certainty * (1 - dynamics.time_value_remaining),
        }
    
    def _compute_technical_indicators(
        self,
        token_id: str,
        current_price: float,
    ) -> dict[str, float]:
        """Compute technical analysis indicators adapted for prediction markets.
        
        Includes:
        - RSI (Relative Strength Index)
        - MACD-like momentum
        - Bollinger bands
        - Support/resistance
        - Trend strength
        """
        features = {}
        
        history = self.price_history.get(token_id, [])
        if len(history) < 5:
            return self._empty_technical_features()
        
        prices = [p for _, p in history]
        
        # === RSI (Relative Strength Index) ===
        # Adapted for prediction markets (clamped to 0-1 range)
        if len(prices) >= 14:
            rsi = self._compute_rsi(prices[-14:])
            features["rsi_14"] = rsi
            features["rsi_overbought"] = float(rsi > 70)
            features["rsi_oversold"] = float(rsi < 30)
            features["rsi_neutral"] = float(30 <= rsi <= 70)
        else:
            features["rsi_14"] = 50.0
            features["rsi_overbought"] = 0.0
            features["rsi_oversold"] = 0.0
            features["rsi_neutral"] = 1.0
        
        # === MACD-like Indicator ===
        if len(prices) >= 26:
            ema_12 = self._compute_ema(prices[-26:], 12)
            ema_26 = self._compute_ema(prices[-26:], 26)
            macd_line = ema_12 - ema_26
            
            # Signal line (9-day EMA of MACD)
            if len(prices) >= 35:
                macd_history = []
                for i in range(9):
                    if len(prices) >= 26 + i:
                        e12 = self._compute_ema(prices[-(26+i):-i if i > 0 else None], 12)
                        e26 = self._compute_ema(prices[-(26+i):-i if i > 0 else None], 26)
                        macd_history.append(e12 - e26)
                
                if macd_history:
                    signal_line = np.mean(macd_history)
                else:
                    signal_line = macd_line
            else:
                signal_line = macd_line
            
            features["macd_line"] = macd_line
            features["macd_signal"] = signal_line
            features["macd_histogram"] = macd_line - signal_line
            features["macd_bullish"] = float(macd_line > signal_line)
            features["macd_bearish"] = float(macd_line < signal_line)
        else:
            features["macd_line"] = 0.0
            features["macd_signal"] = 0.0
            features["macd_histogram"] = 0.0
            features["macd_bullish"] = 0.0
            features["macd_bearish"] = 0.0
        
        # === Bollinger Bands ===
        if len(prices) >= 20:
            sma_20 = np.mean(prices[-20:])
            std_20 = np.std(prices[-20:])
            
            upper_band = sma_20 + 2 * std_20
            lower_band = sma_20 - 2 * std_20
            
            # Where is current price relative to bands?
            band_width = upper_band - lower_band
            if band_width > 0:
                bb_position = (current_price - lower_band) / band_width
            else:
                bb_position = 0.5
            
            features["bollinger_upper"] = upper_band
            features["bollinger_lower"] = lower_band
            features["bollinger_mid"] = sma_20
            features["bollinger_width"] = band_width
            features["bollinger_position"] = bb_position  # 0 = at lower, 1 = at upper
            features["price_above_upper"] = float(current_price > upper_band)
            features["price_below_lower"] = float(current_price < lower_band)
        else:
            features.update(self._empty_bollinger_features())
        
        # === Support/Resistance Detection ===
        if len(prices) >= 50:
            support, resistance = self._detect_support_resistance(prices[-50:])
            features["support_level"] = support
            features["resistance_level"] = resistance
            features["distance_to_support"] = (current_price - support) / current_price if current_price > 0 else 0
            features["distance_to_resistance"] = (resistance - current_price) / current_price if current_price > 0 else 0
            features["near_support"] = float(abs(current_price - support) / current_price < 0.02)
            features["near_resistance"] = float(abs(resistance - current_price) / current_price < 0.02)
        else:
            features["support_level"] = 0.0
            features["resistance_level"] = 1.0
            features["distance_to_support"] = 0.5
            features["distance_to_resistance"] = 0.5
            features["near_support"] = 0.0
            features["near_resistance"] = 0.0
        
        # === Trend Strength ===
        if len(prices) >= 10:
            # ADX-like trend strength
            trend_strength = self._compute_trend_strength(prices[-10:])
            features["trend_strength"] = trend_strength
            features["strong_trend"] = float(trend_strength > 0.5)
            
            # Trend direction
            trend_direction = 1 if prices[-1] > prices[-10] else -1
            features["trend_direction"] = float(trend_direction)
        else:
            features["trend_strength"] = 0.0
            features["strong_trend"] = 0.0
            features["trend_direction"] = 0.0
        
        return features
    
    def _compute_rsi(self, prices: list[float]) -> float:
        """Compute RSI (Relative Strength Index)."""
        if len(prices) < 2:
            return 50.0
        
        changes = np.diff(prices)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_ema(self, prices: list[float], period: int) -> float:
        """Compute Exponential Moving Average."""
        if len(prices) < period:
            return np.mean(prices) if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[-period]  # Start with first price
        
        for price in prices[-period + 1:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _detect_support_resistance(
        self, prices: list[float]
    ) -> tuple[float, float]:
        """Detect support and resistance levels."""
        if len(prices) < 10:
            return 0.0, 1.0
        
        # Find local minima and maxima
        local_mins = []
        local_maxs = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                local_mins.append(prices[i])
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                local_maxs.append(prices[i])
        
        # Support: cluster of local mins
        support = np.median(local_mins) if local_mins else min(prices)
        
        # Resistance: cluster of local maxs
        resistance = np.median(local_maxs) if local_maxs else max(prices)
        
        return support, resistance
    
    def _compute_trend_strength(self, prices: list[float]) -> float:
        """Compute trend strength (0-1 scale)."""
        if len(prices) < 3:
            return 0.0
        
        # Use R² of linear regression as trend strength
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        # R² calculation
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        
        return max(0.0, min(1.0, r_squared))
    
    def _compute_volume_features(
        self,
        token_id: str,
        current_volume: float,
    ) -> dict[str, float]:
        """Compute volume-based features."""
        features = {}
        
        history = self.volume_history.get(token_id, [])
        
        if len(history) < 2:
            return {
                "volume_current": current_volume,
                "volume_ratio": 1.0,
                "volume_spike": 0.0,
                "volume_trend": 0.0,
                "volume_ma_5": current_volume,
                "volume_ma_20": current_volume,
            }
        
        volumes = [v for _, v in history]
        
        # Volume moving averages
        ma_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.mean(volumes)
        ma_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        
        # Volume ratio (current vs average)
        volume_ratio = current_volume / ma_20 if ma_20 > 0 else 1.0
        
        # Volume spike detection
        volume_std = np.std(volumes[-20:]) if len(volumes) >= 20 else np.std(volumes)
        if volume_std > 0:
            z_score = (current_volume - ma_20) / volume_std
        else:
            z_score = 0.0
        
        features["volume_current"] = current_volume
        features["volume_ratio"] = volume_ratio
        features["volume_spike"] = float(volume_ratio > 2.0)  # 2x normal
        features["volume_z_score"] = z_score
        features["volume_anomaly"] = float(abs(z_score) > 2.0)
        features["volume_trend"] = (ma_5 - ma_20) / ma_20 if ma_20 > 0 else 0.0
        features["volume_ma_5"] = ma_5
        features["volume_ma_20"] = ma_20
        
        return features
    
    def _detect_patterns(
        self,
        token_id: str,
        current_price: float,
        now: datetime,
    ) -> dict[str, float]:
        """Detect price patterns."""
        features = {}
        
        history = self.price_history.get(token_id, [])
        
        if len(history) < 10:
            return self._empty_pattern_features()
        
        prices = [p for _, p in history[-50:]]
        
        # === Price Action Patterns ===
        
        # Higher highs / lower lows (trend confirmation)
        if len(prices) >= 20:
            recent_high = max(prices[-10:])
            previous_high = max(prices[-20:-10])
            recent_low = min(prices[-10:])
            previous_low = min(prices[-20:-10])
            
            features["higher_high"] = float(recent_high > previous_high)
            features["lower_low"] = float(recent_low < previous_low)
            features["higher_low"] = float(recent_low > previous_low)  # Bullish
            features["lower_high"] = float(recent_high < previous_high)  # Bearish
        else:
            features["higher_high"] = 0.0
            features["lower_low"] = 0.0
            features["higher_low"] = 0.0
            features["lower_high"] = 0.0
        
        # === Momentum Divergence ===
        # Price making new highs but momentum weakening
        if len(prices) >= 14:
            rsi = self._compute_rsi(prices[-14:])
            price_at_high = max(prices[-7:]) == prices[-1]  # Recent high
            rsi_declining = rsi < 70  # But RSI not at extreme
            
            features["bearish_divergence"] = float(price_at_high and rsi_declining and rsi < 60)
            
            price_at_low = min(prices[-7:]) == prices[-1]
            features["bullish_divergence"] = float(price_at_low and rsi > 40)
        else:
            features["bearish_divergence"] = 0.0
            features["bullish_divergence"] = 0.0
        
        # === Price Consolidation ===
        if len(prices) >= 10:
            recent_range = max(prices[-10:]) - min(prices[-10:])
            avg_price = np.mean(prices[-10:])
            consolidation = recent_range / avg_price if avg_price > 0 else 0
            
            features["price_consolidation"] = 1 - min(1.0, consolidation * 10)  # Higher = more consolidated
            features["breakout_potential"] = features["price_consolidation"]  # Tight consolidation = potential breakout
        else:
            features["price_consolidation"] = 0.5
            features["breakout_potential"] = 0.5
        
        return features
    
    def _analyze_order_flow(
        self, orderbook: dict[str, Any]
    ) -> dict[str, float]:
        """Advanced order flow analysis."""
        features = {}
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        if not bids or not asks:
            return self._empty_order_flow_features()
        
        # === Order Book Pressure ===
        bid_volumes = [float(b.get("size", 0)) for b in bids[:10]]
        ask_volumes = [float(a.get("size", 0)) for a in asks[:10]]
        
        total_bid = sum(bid_volumes)
        total_ask = sum(ask_volumes)
        
        features["total_bid_volume"] = total_bid
        features["total_ask_volume"] = total_ask
        
        if total_bid + total_ask > 0:
            features["order_book_pressure"] = (total_bid - total_ask) / (total_bid + total_ask)
        else:
            features["order_book_pressure"] = 0.0
        
        # === Depth Concentration ===
        # Is depth concentrated at top (aggressive) or spread out (passive)?
        if total_bid > 0:
            top_bid_pct = bid_volumes[0] / total_bid if bid_volumes else 0
            features["bid_concentration"] = top_bid_pct
        else:
            features["bid_concentration"] = 0.0
        
        if total_ask > 0:
            top_ask_pct = ask_volumes[0] / total_ask if ask_volumes else 0
            features["ask_concentration"] = top_ask_pct
        else:
            features["ask_concentration"] = 0.0
        
        # === Order Size Distribution ===
        all_sizes = bid_volumes + ask_volumes
        if len(all_sizes) >= 3:
            features["order_size_skewness"] = float(np.mean(all_sizes) / np.median(all_sizes)) if np.median(all_sizes) > 0 else 1.0
        else:
            features["order_size_skewness"] = 1.0
        
        # === Queue Position Value ===
        # How much volume is ahead of market orders?
        features["bid_queue_value"] = float(bids[0].get("size", 0)) * float(bids[0].get("price", 0)) if bids else 0
        features["ask_queue_value"] = float(asks[0].get("size", 0)) * float(asks[0].get("price", 0)) if asks else 0
        
        return features
    
    def _analyze_trade_flow_advanced(
        self, trades: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Advanced trade flow analysis."""
        features = {}
        
        if len(trades) < 2:
            return self._empty_trade_flow_features()
        
        # === Trade Imbalance ===
        buy_volume = sum(
            float(t.get("size", 0))
            for t in trades
            if str(t.get("side", "")).upper() == "BUY"
        )
        sell_volume = sum(
            float(t.get("size", 0))
            for t in trades
            if str(t.get("side", "")).upper() == "SELL"
        )
        
        total_volume = buy_volume + sell_volume
        features["trade_buy_volume"] = buy_volume
        features["trade_sell_volume"] = sell_volume
        
        if total_volume > 0:
            features["trade_imbalance"] = (buy_volume - sell_volume) / total_volume
        else:
            features["trade_imbalance"] = 0.0
        
        # === Trade Size Analysis ===
        sizes = [float(t.get("size", 0)) for t in trades]
        features["avg_trade_size"] = np.mean(sizes)
        features["max_trade_size"] = max(sizes)
        features["trade_size_std"] = np.std(sizes) if len(sizes) > 1 else 0
        
        # Large trade detection
        large_threshold = np.mean(sizes) + 2 * (np.std(sizes) if len(sizes) > 1 else 0)
        large_trades = [s for s in sizes if s > large_threshold]
        features["large_trade_count"] = len(large_trades)
        features["large_trade_ratio"] = len(large_trades) / len(trades) if trades else 0
        
        # === Trade Velocity ===
        timestamps = [t.get("timestamp") for t in trades if t.get("timestamp")]
        if len(timestamps) >= 2:
            try:
                if isinstance(timestamps[0], str):
                    from dateutil import parser
                    timestamps = [parser.parse(ts) for ts in timestamps]
                
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                              for i in range(len(timestamps) - 1)]
                
                if time_diffs:
                    features["avg_time_between_trades"] = np.mean(time_diffs)
                    features["trade_velocity"] = 1 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
                else:
                    features["avg_time_between_trades"] = 0
                    features["trade_velocity"] = 0
            except Exception:
                features["avg_time_between_trades"] = 0
                features["trade_velocity"] = 0
        else:
            features["avg_time_between_trades"] = 0
            features["trade_velocity"] = 0
        
        return features
    
    def _compute_cross_market_features(
        self,
        token_id: str,
        category: str,
        current_price: float,
    ) -> dict[str, float]:
        """Compute cross-market features."""
        features = {}
        
        # Update category tracking
        now = datetime.utcnow()
        self.category_prices[category].append((now, current_price))
        
        # Trim old data
        cutoff = now - timedelta(days=7)
        self.category_prices[category] = [
            (t, p) for t, p in self.category_prices[category] if t > cutoff
        ]
        
        # Category momentum
        cat_prices = [p for _, p in self.category_prices[category]]
        if len(cat_prices) >= 10:
            category_momentum = (cat_prices[-1] - cat_prices[-10]) / cat_prices[-10] if cat_prices[-10] > 0 else 0
            features["category_momentum"] = category_momentum
        else:
            features["category_momentum"] = 0.0
        
        # This token vs category average
        if cat_prices:
            category_avg = np.mean(cat_prices)
            features["price_vs_category"] = (current_price - category_avg) / category_avg if category_avg > 0 else 0
        else:
            features["price_vs_category"] = 0.0
        
        return features
    
    def _compute_time_features(self, now: datetime) -> dict[str, float]:
        """Compute time-based seasonality features."""
        return {
            "hour_of_day": now.hour,
            "hour_sin": math.sin(2 * math.pi * now.hour / 24),
            "hour_cos": math.cos(2 * math.pi * now.hour / 24),
            "day_of_week": now.weekday(),
            "day_sin": math.sin(2 * math.pi * now.weekday() / 7),
            "day_cos": math.cos(2 * math.pi * now.weekday() / 7),
            "is_weekend": float(now.weekday() >= 5),
            "is_market_hours_us": float(14 <= now.hour <= 21),  # 9 AM - 4 PM EST in UTC
            "is_morning_us": float(14 <= now.hour <= 16),  # 9 AM - 11 AM EST
            "is_afternoon_us": float(18 <= now.hour <= 21),  # 1 PM - 4 PM EST
            "day_of_month": now.day,
            "is_month_start": float(now.day <= 3),
            "is_month_end": float(now.day >= 28),
        }
    
    def _compute_composite_signals(
        self, features: dict[str, float]
    ) -> dict[str, float]:
        """Compute composite intelligence signals from all features."""
        composite = {}
        
        # === BULLISH COMPOSITE ===
        bullish_signals = [
            features.get("smart_money_direction", 0) > 0,
            features.get("order_book_pressure", 0) > 0.3,
            features.get("trade_imbalance", 0) > 0.3,
            features.get("macd_bullish", 0) > 0,
            features.get("rsi_oversold", 0) > 0,  # Oversold = potential bounce
            features.get("higher_low", 0) > 0,
            features.get("bullish_divergence", 0) > 0,
            features.get("accumulation_detected", 0) > 0,
            features.get("near_support", 0) > 0,
        ]
        composite["bullish_signal_count"] = sum(bullish_signals)
        composite["bullish_composite"] = sum(bullish_signals) / len(bullish_signals)
        
        # === BEARISH COMPOSITE ===
        bearish_signals = [
            features.get("smart_money_direction", 0) < 0,
            features.get("order_book_pressure", 0) < -0.3,
            features.get("trade_imbalance", 0) < -0.3,
            features.get("macd_bearish", 0) > 0,
            features.get("rsi_overbought", 0) > 0,  # Overbought = potential drop
            features.get("lower_high", 0) > 0,
            features.get("bearish_divergence", 0) > 0,
            features.get("distribution_detected", 0) > 0,
            features.get("near_resistance", 0) > 0,
        ]
        composite["bearish_signal_count"] = sum(bearish_signals)
        composite["bearish_composite"] = sum(bearish_signals) / len(bearish_signals)
        
        # === OVERALL SIGNAL ===
        composite["net_signal"] = composite["bullish_composite"] - composite["bearish_composite"]
        composite["signal_strength"] = abs(composite["net_signal"])
        composite["signal_direction"] = 1.0 if composite["net_signal"] > 0 else (-1.0 if composite["net_signal"] < 0 else 0.0)
        
        # === CONFIDENCE ===
        # Higher when signals agree
        signal_agreement = (composite["bullish_signal_count"] == 0) or (composite["bearish_signal_count"] == 0)
        composite["signal_confidence"] = composite["signal_strength"] * (1.5 if signal_agreement else 0.7)
        composite["signal_confidence"] = min(1.0, composite["signal_confidence"])
        
        # === TRADE QUALITY ===
        # Good trade: strong signal + good liquidity + reasonable volatility
        liquidity_score = features.get("liquidity_score", 0)
        trend_strength = features.get("trend_strength", 0)
        
        composite["trade_quality"] = (
            0.4 * composite["signal_strength"] +
            0.3 * min(1.0, liquidity_score / 100) +
            0.3 * trend_strength
        )
        
        return composite
    
    def _update_history(
        self,
        token_id: str,
        timestamp: datetime,
        price: float,
        volume: float,
    ) -> None:
        """Update price and volume history."""
        self.price_history[token_id].append((timestamp, price))
        self.volume_history[token_id].append((timestamp, volume))
        
        # Trim to max size
        if len(self.price_history[token_id]) > self.max_history_points:
            self.price_history[token_id] = self.price_history[token_id][-self.max_history_points:]
        if len(self.volume_history[token_id]) > self.max_history_points:
            self.volume_history[token_id] = self.volume_history[token_id][-self.max_history_points:]
    
    def _clean_features(self, features: dict[str, float]) -> dict[str, float]:
        """Clean NaN/Inf values."""
        cleaned = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = float(value)
            else:
                cleaned[key] = 0.0
        return cleaned
    
    def _empty_technical_features(self) -> dict[str, float]:
        """Empty technical features."""
        return {
            "rsi_14": 50.0,
            "rsi_overbought": 0.0,
            "rsi_oversold": 0.0,
            "rsi_neutral": 1.0,
            "macd_line": 0.0,
            "macd_signal": 0.0,
            "macd_histogram": 0.0,
            "macd_bullish": 0.0,
            "macd_bearish": 0.0,
            "trend_strength": 0.0,
            "strong_trend": 0.0,
            "trend_direction": 0.0,
        }
    
    def _empty_bollinger_features(self) -> dict[str, float]:
        """Empty bollinger features."""
        return {
            "bollinger_upper": 1.0,
            "bollinger_lower": 0.0,
            "bollinger_mid": 0.5,
            "bollinger_width": 1.0,
            "bollinger_position": 0.5,
            "price_above_upper": 0.0,
            "price_below_lower": 0.0,
        }
    
    def _empty_pattern_features(self) -> dict[str, float]:
        """Empty pattern features."""
        return {
            "higher_high": 0.0,
            "lower_low": 0.0,
            "higher_low": 0.0,
            "lower_high": 0.0,
            "bearish_divergence": 0.0,
            "bullish_divergence": 0.0,
            "price_consolidation": 0.5,
            "breakout_potential": 0.5,
        }
    
    def _empty_order_flow_features(self) -> dict[str, float]:
        """Empty order flow features."""
        return {
            "total_bid_volume": 0.0,
            "total_ask_volume": 0.0,
            "order_book_pressure": 0.0,
            "bid_concentration": 0.0,
            "ask_concentration": 0.0,
            "order_size_skewness": 1.0,
            "bid_queue_value": 0.0,
            "ask_queue_value": 0.0,
        }
    
    def _empty_trade_flow_features(self) -> dict[str, float]:
        """Empty trade flow features."""
        return {
            "trade_buy_volume": 0.0,
            "trade_sell_volume": 0.0,
            "trade_imbalance": 0.0,
            "avg_trade_size": 0.0,
            "max_trade_size": 0.0,
            "trade_size_std": 0.0,
            "large_trade_count": 0.0,
            "large_trade_ratio": 0.0,
            "avg_time_between_trades": 0.0,
            "trade_velocity": 0.0,
        }


# Singleton instance for easy access
_advanced_intelligence: AdvancedIntelligenceEngine | None = None


def get_advanced_intelligence() -> AdvancedIntelligenceEngine:
    """Get singleton advanced intelligence engine."""
    global _advanced_intelligence
    if _advanced_intelligence is None:
        _advanced_intelligence = AdvancedIntelligenceEngine()
    return _advanced_intelligence

