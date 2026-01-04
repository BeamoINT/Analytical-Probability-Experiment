"""Advanced feature engineering for ML-powered trading intelligence.

This module computes sophisticated features from order book microstructure,
time series patterns, and market dynamics to enable predictive modeling.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from polyb0t.data.models import Market, OrderBook, Trade

logger = logging.getLogger(__name__)


class AdvancedFeatureEngine:
    """Computes sophisticated features for ML prediction.
    
    Features include:
    - Order book microstructure (imbalance, depth slope, toxicity)
    - Time series patterns (momentum, volatility, autocorrelation)
    - Market quality metrics (spread dynamics, liquidity)
    - Derived signals (regime detection, relative value)
    """

    def __init__(self) -> None:
        """Initialize feature engine."""
        # Historical price cache for time series features
        self.price_history: Dict[str, List[tuple[datetime, float]]] = {}
        self.max_history_size = 10000  # Keep last 10,000 prices per token (2+ years)
        
        # Volume history
        self.volume_history: Dict[str, List[tuple[datetime, float]]] = {}
        
    def compute_features(
        self,
        market: Market,
        outcome_idx: int,
        orderbook: OrderBook | None,
        recent_trades: List[Trade],
        current_price: float,
    ) -> Dict[str, float]:
        """Compute comprehensive feature set.
        
        Args:
            market: Market data.
            outcome_idx: Outcome index.
            orderbook: Current orderbook snapshot.
            recent_trades: Recent trades list.
            current_price: Current market price (mid or last).
            
        Returns:
            Dictionary of feature name -> value.
        """
        features = {}
        
        outcome = market.outcomes[outcome_idx]
        token_id = outcome.token_id
        
        # Update price history
        self._update_price_history(token_id, current_price)
        
        # 1. Basic features
        features.update(self._compute_basic_features(
            market, outcome_idx, current_price
        ))
        
        # 2. Order book microstructure features
        if orderbook:
            features.update(self._compute_orderbook_features(orderbook))
        
        # 3. Trade flow features
        if recent_trades:
            features.update(self._compute_trade_flow_features(recent_trades))
        
        # 4. Time series features
        features.update(self._compute_timeseries_features(token_id))
        
        # 5. Market quality features
        if orderbook:
            features.update(self._compute_market_quality_features(
                orderbook, current_price
            ))
        
        # 6. Derived/engineered features
        features.update(self._compute_derived_features(features))
        
        # Handle NaN/Inf
        features = self._clean_features(features)
        
        return features
    
    def _compute_basic_features(
        self,
        market: Market,
        outcome_idx: int,
        current_price: float,
    ) -> Dict[str, float]:
        """Compute basic market features."""
        outcome = market.outcomes[outcome_idx]
        
        # Time-based features
        now = datetime.utcnow()
        if market.end_date:
            end_date = market.end_date
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=now.tzinfo)
            time_to_resolution_hours = max(0, (end_date - now).total_seconds() / 3600)
        else:
            time_to_resolution_hours = 168  # Default 1 week
        
        return {
            'p_market': current_price,
            'time_to_resolution_hours': time_to_resolution_hours,
            'time_to_resolution_days': time_to_resolution_hours / 24,
            'log_time_to_resolution': np.log1p(time_to_resolution_hours),
            'market_volume': float(market.volume or 0),
            'market_liquidity': float(market.liquidity or 0),
            'log_volume': np.log1p(float(market.volume or 0)),
            'log_liquidity': np.log1p(float(market.liquidity or 0)),
            'hour_of_day': now.hour,
            'day_of_week': now.weekday(),
            'is_weekend': float(now.weekday() >= 5),
        }
    
    def _compute_orderbook_features(
        self,
        orderbook: OrderBook,
    ) -> Dict[str, float]:
        """Compute order book microstructure features."""
        features = {}
        
        if not orderbook.bids or not orderbook.asks:
            return self._empty_orderbook_features()
        
        # Best bid/ask
        best_bid = orderbook.bids[0].price
        best_ask = orderbook.asks[0].price
        mid = (best_bid + best_ask) / 2
        
        # Spread
        spread = best_ask - best_bid
        spread_pct = spread / mid if mid > 0 else 0
        
        features['best_bid'] = best_bid
        features['best_ask'] = best_ask
        features['mid_price'] = mid
        features['spread'] = spread
        features['spread_pct'] = spread_pct
        features['spread_bps'] = spread_pct * 10000
        
        # Depth at top levels (1-5)
        bid_sizes = [b.size for b in orderbook.bids[:5]]
        ask_sizes = [a.size for a in orderbook.asks[:5]]
        
        bid_depth_total = sum(bid_sizes)
        ask_depth_total = sum(ask_sizes)
        total_depth = bid_depth_total + ask_depth_total
        
        features['bid_depth_total'] = bid_depth_total
        features['ask_depth_total'] = ask_depth_total
        features['total_depth'] = total_depth
        features['log_total_depth'] = np.log1p(total_depth)
        
        # Depth imbalance (key microstructure signal)
        if total_depth > 0:
            features['depth_imbalance'] = (bid_depth_total - ask_depth_total) / total_depth
        else:
            features['depth_imbalance'] = 0.0
        
        # Depth at specific levels
        for i in range(min(5, len(bid_sizes))):
            features[f'bid_depth_L{i+1}'] = bid_sizes[i]
        for i in range(min(5, len(ask_sizes))):
            features[f'ask_depth_L{i+1}'] = ask_sizes[i]
        
        # Depth slope (measures how quickly depth falls off)
        if len(orderbook.bids) >= 3:
            bid_prices = [b.price for b in orderbook.bids[:3]]
            bid_sizes_top3 = [b.size for b in orderbook.bids[:3]]
            if len(set(bid_prices)) > 1:  # Need variation for slope
                features['bid_depth_slope'] = np.polyfit(bid_prices, bid_sizes_top3, 1)[0]
            else:
                features['bid_depth_slope'] = 0.0
        else:
            features['bid_depth_slope'] = 0.0
            
        if len(orderbook.asks) >= 3:
            ask_prices = [a.price for a in orderbook.asks[:3]]
            ask_sizes_top3 = [a.size for a in orderbook.asks[:3]]
            if len(set(ask_prices)) > 1:
                features['ask_depth_slope'] = np.polyfit(ask_prices, ask_sizes_top3, 1)[0]
            else:
                features['ask_depth_slope'] = 0.0
        else:
            features['ask_depth_slope'] = 0.0
        
        # Weighted mid price (depth-weighted)
        if total_depth > 0:
            weighted_mid = (best_bid * bid_depth_total + best_ask * ask_depth_total) / total_depth
            features['weighted_mid'] = weighted_mid
            features['mid_vs_weighted'] = mid - weighted_mid
        else:
            features['weighted_mid'] = mid
            features['mid_vs_weighted'] = 0.0
        
        # Price levels
        if len(orderbook.bids) > 1:
            features['bid_price_range'] = orderbook.bids[0].price - orderbook.bids[-1].price
        else:
            features['bid_price_range'] = 0.0
            
        if len(orderbook.asks) > 1:
            features['ask_price_range'] = orderbook.asks[-1].price - orderbook.asks[0].price
        else:
            features['ask_price_range'] = 0.0
        
        return features
    
    def _compute_trade_flow_features(
        self,
        recent_trades: List[Trade],
    ) -> Dict[str, float]:
        """Compute trade flow and order toxicity features."""
        features = {}
        
        if not recent_trades:
            return self._empty_trade_features()
        
        # Basic trade statistics
        features['num_trades'] = len(recent_trades)
        features['log_num_trades'] = np.log1p(len(recent_trades))
        
        # Price statistics
        prices = [t.price for t in recent_trades]
        features['trade_price_mean'] = np.mean(prices)
        features['trade_price_std'] = np.std(prices) if len(prices) > 1 else 0.0
        features['trade_price_min'] = np.min(prices)
        features['trade_price_max'] = np.max(prices)
        features['trade_price_range'] = features['trade_price_max'] - features['trade_price_min']
        
        # Volume statistics
        sizes = [t.size for t in recent_trades]
        features['trade_volume_total'] = sum(sizes)
        features['trade_volume_mean'] = np.mean(sizes)
        features['trade_volume_std'] = np.std(sizes) if len(sizes) > 1 else 0.0
        features['log_trade_volume'] = np.log1p(sum(sizes))
        
        # Trade direction/aggressiveness (if available from side field)
        buy_trades = [t for t in recent_trades if hasattr(t, 'side') and t.side == 'BUY']
        if len(recent_trades) > 0:
            features['buy_ratio'] = len(buy_trades) / len(recent_trades)
            features['sell_ratio'] = 1 - features['buy_ratio']
        else:
            features['buy_ratio'] = 0.5
            features['sell_ratio'] = 0.5
        
        # Order flow imbalance (net buying pressure)
        features['order_flow_imbalance'] = features['buy_ratio'] - features['sell_ratio']
        
        # Trade intensity (trades per unit time)
        if len(recent_trades) > 1:
            time_span_seconds = (recent_trades[-1].timestamp - recent_trades[0].timestamp).total_seconds()
            if time_span_seconds > 0:
                features['trade_intensity_per_minute'] = len(recent_trades) / (time_span_seconds / 60)
            else:
                features['trade_intensity_per_minute'] = 0.0
        else:
            features['trade_intensity_per_minute'] = 0.0
        
        # Recent price momentum from trades
        if len(recent_trades) >= 2:
            recent_return = (recent_trades[-1].price - recent_trades[0].price) / recent_trades[0].price
            features['recent_trade_return'] = recent_return
        else:
            features['recent_trade_return'] = 0.0
        
        return features
    
    def _compute_timeseries_features(
        self,
        token_id: str,
    ) -> Dict[str, float]:
        """Compute time series features from price history."""
        features = {}
        
        if token_id not in self.price_history or len(self.price_history[token_id]) < 2:
            return self._empty_timeseries_features()
        
        history = self.price_history[token_id]
        prices = [p for _, p in history]
        timestamps = [t for t, _ in history]
        
        current_price = prices[-1]
        
        # Returns at multiple horizons
        horizons = {
            '5m': 5 * 60,
            '15m': 15 * 60,
            '1h': 3600,
            '4h': 4 * 3600,
            '24h': 24 * 3600,
        }
        
        now = timestamps[-1]
        for name, seconds in horizons.items():
            cutoff = now - timedelta(seconds=seconds)
            historical_prices = [(t, p) for t, p in history if t <= cutoff]
            
            if historical_prices:
                old_price = historical_prices[-1][1]
                ret = (current_price - old_price) / old_price if old_price > 0 else 0.0
                features[f'return_{name}'] = ret
            else:
                features[f'return_{name}'] = 0.0
        
        # Volatility (rolling standard deviation of returns)
        if len(prices) >= 10:
            returns = np.diff(prices) / prices[:-1]
            features['volatility_raw'] = np.std(returns)
            features['volatility_annualized'] = np.std(returns) * np.sqrt(365 * 24)  # Hourly data
        else:
            features['volatility_raw'] = 0.0
            features['volatility_annualized'] = 0.0
        
        # Momentum indicators
        if len(prices) >= 20:
            # Simple moving averages
            sma_5 = np.mean(prices[-5:])
            sma_20 = np.mean(prices[-20:])
            
            features['sma_5'] = sma_5
            features['sma_20'] = sma_20
            features['price_vs_sma5'] = (current_price - sma_5) / sma_5 if sma_5 > 0 else 0.0
            features['price_vs_sma20'] = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0.0
            features['sma5_vs_sma20'] = (sma_5 - sma_20) / sma_20 if sma_20 > 0 else 0.0
        else:
            features['sma_5'] = current_price
            features['sma_20'] = current_price
            features['price_vs_sma5'] = 0.0
            features['price_vs_sma20'] = 0.0
            features['sma5_vs_sma20'] = 0.0
        
        # Autocorrelation (mean reversion signal)
        if len(prices) >= 30:
            returns = np.diff(prices[-30:]) / prices[-30:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                # Lag-1 autocorrelation
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                features['autocorr_lag1'] = autocorr if not np.isnan(autocorr) else 0.0
            else:
                features['autocorr_lag1'] = 0.0
        else:
            features['autocorr_lag1'] = 0.0
        
        # Hurst exponent (trend vs mean-reversion)
        if len(prices) >= 50:
            features['hurst_exponent'] = self._compute_hurst(prices[-50:])
        else:
            features['hurst_exponent'] = 0.5  # Neutral
        
        # Price extremes
        if len(prices) >= 20:
            window_prices = prices[-20:]
            features['price_percentile_20'] = stats.percentileofscore(window_prices, current_price) / 100.0
        else:
            features['price_percentile_20'] = 0.5
        
        return features
    
    def _compute_market_quality_features(
        self,
        orderbook: OrderBook,
        current_price: float,
    ) -> Dict[str, float]:
        """Compute market quality and liquidity features."""
        features = {}
        
        if not orderbook.bids or not orderbook.asks:
            return {}
        
        # Effective spread (cost of immediate execution)
        # Would need trade data to compute properly, using bid-ask as proxy
        mid = (orderbook.bids[0].price + orderbook.asks[0].price) / 2
        spread = orderbook.asks[0].price - orderbook.bids[0].price
        
        features['effective_spread_pct'] = spread / mid if mid > 0 else 0.0
        
        # Price impact (how much price moves for given size)
        # Estimate: if you buy $10 worth, what's the avg price?
        test_sizes = [10, 50, 100]
        for size_usd in test_sizes:
            remaining = size_usd
            total_cost = 0.0
            for ask in orderbook.asks[:10]:
                if remaining <= 0:
                    break
                depth_usd = ask.size * ask.price
                consumed = min(remaining, depth_usd)
                total_cost += consumed
                remaining -= consumed
            
            if size_usd - remaining > 0:
                avg_price = total_cost / (size_usd - remaining)
                price_impact = (avg_price - mid) / mid if mid > 0 else 0.0
                features[f'price_impact_{size_usd}usd'] = price_impact
            else:
                features[f'price_impact_{size_usd}usd'] = 0.0
        
        # Liquidity score (composite measure)
        total_depth = sum(b.size for b in orderbook.bids[:5]) + sum(a.size for a in orderbook.asks[:5])
        spread_penalty = max(0, 1 - spread / mid) if mid > 0 else 0
        features['liquidity_score'] = np.log1p(total_depth) * spread_penalty
        
        return features
    
    def _compute_derived_features(
        self,
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute derived/engineered features from base features."""
        derived = {}
        
        # Regime detection
        volatility = features.get('volatility_raw', 0)
        spread_pct = features.get('spread_pct', 0)
        
        # Classify regime
        if volatility > 0.03 and spread_pct > 0.05:
            regime = 2  # Volatile
        elif volatility < 0.01 and spread_pct < 0.02:
            regime = 0  # Calm
        else:
            regime = 1  # Normal
        
        derived['regime'] = float(regime)
        derived['is_volatile_regime'] = float(regime == 2)
        derived['is_calm_regime'] = float(regime == 0)
        
        # Combined signals
        depth_imbalance = features.get('depth_imbalance', 0)
        order_flow_imbalance = features.get('order_flow_imbalance', 0)
        
        # If both depth and flow point same direction, stronger signal
        derived['flow_depth_agreement'] = depth_imbalance * order_flow_imbalance
        derived['flow_depth_combined'] = (depth_imbalance + order_flow_imbalance) / 2
        
        # Momentum strength
        return_1h = features.get('return_1h', 0)
        return_24h = features.get('return_24h', 0)
        if return_1h * return_24h > 0:  # Same direction
            derived['momentum_strength'] = abs(return_1h) + abs(return_24h)
        else:
            derived['momentum_strength'] = 0.0
        
        # Mean reversion signal
        price_vs_sma5 = features.get('price_vs_sma5', 0)
        autocorr = features.get('autocorr_lag1', 0)
        derived['mean_reversion_signal'] = -price_vs_sma5 * (1 - autocorr)  # Revert if far from SMA and negative autocorr
        
        # Quality-adjusted edge (combine multiple signals)
        liquidity_score = features.get('liquidity_score', 0)
        spread_pct = features.get('spread_pct', 1)
        if spread_pct > 0:
            derived['quality_factor'] = liquidity_score / spread_pct
        else:
            derived['quality_factor'] = 0.0
        
        return derived
    
    def _compute_hurst(self, prices: List[float]) -> float:
        """Compute Hurst exponent (H).
        
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(prices) < 20:
            return 0.5
        
        try:
            prices_arr = np.array(prices)
            lags = range(2, min(20, len(prices) // 2))
            
            # Compute RS statistic
            tau = []
            rs_values = []
            
            for lag in lags:
                # Standard deviation
                std = np.std(prices_arr[:lag])
                if std == 0:
                    continue
                
                # Range
                mean = np.mean(prices_arr[:lag])
                cumsum = np.cumsum(prices_arr[:lag] - mean)
                R = np.max(cumsum) - np.min(cumsum)
                
                # RS ratio
                rs = R / std if std > 0 else 0
                
                tau.append(lag)
                rs_values.append(rs)
            
            if len(tau) < 2:
                return 0.5
            
            # Hurst = slope of log(RS) vs log(lag)
            log_tau = np.log(tau)
            log_rs = np.log([r for r in rs_values if r > 0])
            
            if len(log_rs) < 2:
                return 0.5
            
            hurst = np.polyfit(log_tau[:len(log_rs)], log_rs, 1)[0]
            
            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))
            
        except Exception:
            return 0.5
    
    def _update_price_history(self, token_id: str, price: float) -> None:
        """Update price history for time series features."""
        if token_id not in self.price_history:
            self.price_history[token_id] = []
        
        now = datetime.utcnow()
        self.price_history[token_id].append((now, price))
        
        # Trim old history
        if len(self.price_history[token_id]) > self.max_history_size:
            self.price_history[token_id] = self.price_history[token_id][-self.max_history_size:]
    
    def _clean_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Replace NaN/Inf with safe defaults."""
        cleaned = {}
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                cleaned[key] = 0.0
            else:
                cleaned[key] = float(value)
        return cleaned
    
    def _empty_orderbook_features(self) -> Dict[str, float]:
        """Default features when orderbook missing."""
        return {
            'best_bid': 0.5,
            'best_ask': 0.5,
            'mid_price': 0.5,
            'spread': 0.0,
            'spread_pct': 0.0,
            'spread_bps': 0.0,
            'bid_depth_total': 0.0,
            'ask_depth_total': 0.0,
            'total_depth': 0.0,
            'log_total_depth': 0.0,
            'depth_imbalance': 0.0,
            'bid_depth_slope': 0.0,
            'ask_depth_slope': 0.0,
            'weighted_mid': 0.5,
            'mid_vs_weighted': 0.0,
        }
    
    def _empty_trade_features(self) -> Dict[str, float]:
        """Default features when trades missing."""
        return {
            'num_trades': 0.0,
            'log_num_trades': 0.0,
            'trade_price_mean': 0.0,
            'trade_price_std': 0.0,
            'trade_volume_total': 0.0,
            'trade_volume_mean': 0.0,
            'log_trade_volume': 0.0,
            'buy_ratio': 0.5,
            'sell_ratio': 0.5,
            'order_flow_imbalance': 0.0,
            'trade_intensity_per_minute': 0.0,
            'recent_trade_return': 0.0,
        }
    
    def _empty_timeseries_features(self) -> Dict[str, float]:
        """Default features when history missing."""
        return {
            'return_5m': 0.0,
            'return_15m': 0.0,
            'return_1h': 0.0,
            'return_4h': 0.0,
            'return_24h': 0.0,
            'volatility_raw': 0.0,
            'volatility_annualized': 0.0,
            'sma_5': 0.5,
            'sma_20': 0.5,
            'price_vs_sma5': 0.0,
            'price_vs_sma20': 0.0,
            'sma5_vs_sma20': 0.0,
            'autocorr_lag1': 0.0,
            'hurst_exponent': 0.5,
            'price_percentile_20': 0.5,
        }

