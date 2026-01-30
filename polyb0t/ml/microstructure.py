"""Market microstructure analysis for ML features.

This module computes advanced market microstructure features:
- VPIN (Volume-synchronized Probability of Informed Trading)
- Order flow toxicity (Kyle's lambda)
- Trade impact modeling
- Quote-level microstructure metrics

These features help detect informed trading and improve prediction accuracy.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeData:
    """Simplified trade data for microstructure analysis."""
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    timestamp: datetime
    value_usd: float = 0.0

    def __post_init__(self):
        if self.value_usd == 0.0:
            self.value_usd = self.price * self.size


@dataclass
class OrderBookSnapshot:
    """Simplified orderbook snapshot."""
    best_bid: float
    best_ask: float
    bid_depth: float  # Total bid liquidity
    ask_depth: float  # Total ask liquidity
    bid_levels: List[Tuple[float, float]] = None  # [(price, size), ...]
    ask_levels: List[Tuple[float, float]] = None
    timestamp: datetime = None


class MicrostructureAnalyzer:
    """Analyzes market microstructure for ML features.

    Implements academic measures of informed trading and market quality:
    - VPIN: Easley, Lopez de Prado, O'Hara (2012)
    - Kyle's Lambda: Kyle (1985) price impact measure
    - Amihud Illiquidity: Amihud (2002) illiquidity ratio
    """

    def __init__(
        self,
        vpin_bucket_volume: float = 1000.0,  # USD per bucket
        vpin_num_buckets: int = 50,  # Rolling window size
        toxicity_window_trades: int = 100,  # Trades for toxicity calc
    ):
        """Initialize the analyzer.

        Args:
            vpin_bucket_volume: Volume in USD per VPIN bucket
            vpin_num_buckets: Number of buckets in VPIN rolling window
            toxicity_window_trades: Number of trades for toxicity calculation
        """
        self.vpin_bucket_volume = vpin_bucket_volume
        self.vpin_num_buckets = vpin_num_buckets
        self.toxicity_window_trades = toxicity_window_trades

        # State for incremental VPIN calculation
        self._vpin_buckets: Dict[str, List[float]] = {}  # token_id -> [imbalances]
        self._current_bucket: Dict[str, Dict] = {}  # token_id -> {buy_vol, sell_vol}

    def compute_vpin(
        self,
        trades: List[TradeData],
        token_id: str = "default"
    ) -> float:
        """Compute Volume-synchronized Probability of Informed Trading.

        VPIN measures the probability that trades are coming from informed
        traders. Higher VPIN indicates more toxic order flow.

        Implementation:
        1. Bucket trades by volume (not time)
        2. For each bucket, compute |buy_volume - sell_volume| / total_volume
        3. VPIN = average of bucket imbalances over rolling window

        Args:
            trades: List of trades sorted by timestamp
            token_id: Token identifier for state tracking

        Returns:
            VPIN score between 0 and 1 (higher = more informed trading)
        """
        if not trades:
            return 0.0

        # Initialize state if needed
        if token_id not in self._vpin_buckets:
            self._vpin_buckets[token_id] = []
            self._current_bucket[token_id] = {"buy_vol": 0.0, "sell_vol": 0.0}

        buckets = self._vpin_buckets[token_id]
        current = self._current_bucket[token_id]

        # Process trades into volume buckets
        for trade in trades:
            volume_usd = trade.value_usd

            if trade.side == "BUY":
                current["buy_vol"] += volume_usd
            else:
                current["sell_vol"] += volume_usd

            total_bucket_volume = current["buy_vol"] + current["sell_vol"]

            # Check if bucket is full
            if total_bucket_volume >= self.vpin_bucket_volume:
                # Compute bucket imbalance
                if total_bucket_volume > 0:
                    imbalance = abs(current["buy_vol"] - current["sell_vol"]) / total_bucket_volume
                else:
                    imbalance = 0.0

                buckets.append(imbalance)

                # Reset current bucket
                current["buy_vol"] = 0.0
                current["sell_vol"] = 0.0

                # Maintain rolling window
                if len(buckets) > self.vpin_num_buckets:
                    buckets.pop(0)

        # Compute VPIN as average of bucket imbalances
        if not buckets:
            return 0.0

        vpin = sum(buckets) / len(buckets)
        return min(1.0, max(0.0, vpin))

    def compute_order_flow_toxicity(
        self,
        trades: List[TradeData],
        orderbook: Optional[OrderBookSnapshot] = None
    ) -> float:
        """Compute order flow toxicity using Kyle's lambda.

        Kyle's lambda measures the price impact per unit of order flow.
        Higher values indicate more adverse selection (toxic flow).

        λ = ΔPrice / SignedVolume

        We normalize this to a 0-1 scale based on typical values.

        Args:
            trades: List of recent trades
            orderbook: Optional orderbook for spread-based adjustment

        Returns:
            Normalized toxicity score between 0 and 1
        """
        if len(trades) < 2:
            return 0.0

        # Compute signed order flow and price changes
        signed_volumes = []
        price_changes = []

        for i in range(1, len(trades)):
            prev_trade = trades[i - 1]
            curr_trade = trades[i]

            # Signed volume (positive for buys, negative for sells)
            sign = 1.0 if curr_trade.side == "BUY" else -1.0
            signed_vol = sign * curr_trade.value_usd
            signed_volumes.append(signed_vol)

            # Price change
            if prev_trade.price > 0:
                price_change = (curr_trade.price - prev_trade.price) / prev_trade.price
                price_changes.append(price_change)

        if not signed_volumes or not price_changes:
            return 0.0

        # Compute Kyle's lambda using regression
        # λ = Cov(ΔP, V) / Var(V)
        signed_volumes = np.array(signed_volumes[:len(price_changes)])
        price_changes = np.array(price_changes)

        var_v = np.var(signed_volumes)
        if var_v < 1e-10:
            return 0.0

        cov_pv = np.cov(price_changes, signed_volumes)[0, 1]
        kyle_lambda = cov_pv / var_v

        # Normalize to 0-1 scale
        # Typical lambda values are in the range 0 to 0.001
        # We use a sigmoid-like transformation
        normalized = 2.0 / (1.0 + math.exp(-kyle_lambda * 10000)) - 1.0

        return min(1.0, max(0.0, normalized))

    def compute_amihud_illiquidity(
        self,
        trades: List[TradeData],
        window_hours: float = 24.0
    ) -> float:
        """Compute Amihud illiquidity ratio.

        Amihud = Average(|Return| / DollarVolume)

        Higher values indicate less liquid markets where trades have
        larger price impact.

        Args:
            trades: List of trades
            window_hours: Time window for calculation

        Returns:
            Amihud illiquidity ratio (higher = less liquid)
        """
        if len(trades) < 2:
            return 0.0

        # Filter trades by time window
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        filtered_trades = [t for t in trades if t.timestamp >= cutoff]

        if len(filtered_trades) < 2:
            return 0.0

        illiquidity_values = []

        for i in range(1, len(filtered_trades)):
            prev_trade = filtered_trades[i - 1]
            curr_trade = filtered_trades[i]

            if prev_trade.price > 0 and curr_trade.value_usd > 0:
                abs_return = abs(curr_trade.price - prev_trade.price) / prev_trade.price
                illiquidity = abs_return / curr_trade.value_usd
                illiquidity_values.append(illiquidity)

        if not illiquidity_values:
            return 0.0

        return np.mean(illiquidity_values)

    def compute_trade_impact(
        self,
        trades: List[TradeData],
        trade_size_usd: float = 100.0
    ) -> float:
        """Estimate price impact for a given trade size.

        Uses historical trades to estimate how much price would move
        for a trade of the given size.

        Args:
            trades: List of historical trades
            trade_size_usd: Size of trade to estimate impact for

        Returns:
            Estimated price impact as a percentage (0.01 = 1%)
        """
        if len(trades) < 10:
            return 0.0

        # Group trades by size bucket and compute average price change
        impacts = []

        for i in range(1, len(trades)):
            prev_trade = trades[i - 1]
            curr_trade = trades[i]

            if prev_trade.price > 0:
                price_change = abs(curr_trade.price - prev_trade.price) / prev_trade.price
                # Weight by proximity to target size
                size_ratio = curr_trade.value_usd / trade_size_usd if trade_size_usd > 0 else 1.0
                weight = 1.0 / (1.0 + abs(1.0 - size_ratio))
                impacts.append((price_change, weight))

        if not impacts:
            return 0.0

        # Weighted average impact
        total_weight = sum(w for _, w in impacts)
        if total_weight < 1e-10:
            return 0.0

        weighted_impact = sum(impact * weight for impact, weight in impacts) / total_weight

        # Scale by trade size ratio (larger trades have more impact)
        avg_trade_size = np.mean([t.value_usd for t in trades])
        if avg_trade_size > 0:
            size_multiplier = math.sqrt(trade_size_usd / avg_trade_size)
            weighted_impact *= size_multiplier

        return weighted_impact

    def compute_spread_metrics(
        self,
        orderbook: OrderBookSnapshot
    ) -> Dict[str, float]:
        """Compute spread-based microstructure metrics.

        Args:
            orderbook: Current orderbook snapshot

        Returns:
            Dictionary of spread metrics
        """
        if not orderbook or orderbook.best_bid <= 0 or orderbook.best_ask <= 0:
            return {
                "quoted_spread": 0.0,
                "relative_spread": 0.0,
                "effective_spread_est": 0.0,
                "depth_weighted_spread": 0.0,
            }

        mid_price = (orderbook.best_bid + orderbook.best_ask) / 2
        quoted_spread = orderbook.best_ask - orderbook.best_bid
        relative_spread = quoted_spread / mid_price if mid_price > 0 else 0.0

        # Estimate effective spread (typically about half quoted spread)
        effective_spread_est = relative_spread * 0.5

        # Depth-weighted spread
        total_depth = orderbook.bid_depth + orderbook.ask_depth
        if total_depth > 0:
            # Weight spread by depth imbalance
            depth_ratio = orderbook.bid_depth / total_depth
            # Spread widens when depth is imbalanced
            imbalance_factor = 1.0 + abs(depth_ratio - 0.5) * 2
            depth_weighted_spread = relative_spread * imbalance_factor
        else:
            depth_weighted_spread = relative_spread

        return {
            "quoted_spread": quoted_spread,
            "relative_spread": relative_spread,
            "effective_spread_est": effective_spread_est,
            "depth_weighted_spread": depth_weighted_spread,
        }

    def get_microstructure_features(
        self,
        trades: List[TradeData],
        orderbook: Optional[OrderBookSnapshot] = None,
        token_id: str = "default"
    ) -> Dict[str, float]:
        """Compute all microstructure features for ML training.

        This is the main entry point for feature extraction.

        Args:
            trades: List of recent trades
            orderbook: Current orderbook snapshot
            token_id: Token identifier for state tracking

        Returns:
            Dictionary of microstructure features
        """
        features = {
            "vpin": 0.0,
            "order_flow_toxicity": 0.0,
            "trade_impact_10usd": 0.0,
            "trade_impact_100usd": 0.0,
            "amihud_illiquidity": 0.0,
        }

        try:
            # VPIN
            features["vpin"] = self.compute_vpin(trades, token_id)

            # Order flow toxicity
            features["order_flow_toxicity"] = self.compute_order_flow_toxicity(
                trades[-self.toxicity_window_trades:] if len(trades) > self.toxicity_window_trades else trades,
                orderbook
            )

            # Trade impact at different sizes
            features["trade_impact_10usd"] = self.compute_trade_impact(trades, 10.0)
            features["trade_impact_100usd"] = self.compute_trade_impact(trades, 100.0)

            # Amihud illiquidity
            features["amihud_illiquidity"] = self.compute_amihud_illiquidity(trades, 24.0)

            # Add spread metrics if orderbook available
            if orderbook:
                spread_metrics = self.compute_spread_metrics(orderbook)
                features.update(spread_metrics)

        except Exception as e:
            logger.warning(f"Error computing microstructure features: {e}")

        return features

    def reset_state(self, token_id: Optional[str] = None):
        """Reset internal state.

        Args:
            token_id: Specific token to reset, or None for all
        """
        if token_id:
            self._vpin_buckets.pop(token_id, None)
            self._current_bucket.pop(token_id, None)
        else:
            self._vpin_buckets.clear()
            self._current_bucket.clear()


# Singleton instance
_microstructure_analyzer: Optional[MicrostructureAnalyzer] = None


def get_microstructure_analyzer() -> MicrostructureAnalyzer:
    """Get the singleton MicrostructureAnalyzer instance."""
    global _microstructure_analyzer
    if _microstructure_analyzer is None:
        _microstructure_analyzer = MicrostructureAnalyzer()
    return _microstructure_analyzer


def convert_trades_to_trade_data(trades: List[Any]) -> List[TradeData]:
    """Convert Trade objects to TradeData for analysis.

    Args:
        trades: List of Trade objects from data models

    Returns:
        List of TradeData objects
    """
    result = []
    for trade in trades:
        try:
            result.append(TradeData(
                price=float(trade.price),
                size=float(trade.size),
                side=str(trade.side).upper(),
                timestamp=trade.timestamp if hasattr(trade, 'timestamp') else datetime.utcnow(),
                value_usd=float(trade.price) * float(trade.size)
            ))
        except Exception as e:
            logger.debug(f"Skipping trade conversion: {e}")
            continue
    return result


def convert_orderbook_to_snapshot(orderbook: Any) -> Optional[OrderBookSnapshot]:
    """Convert OrderBook object to OrderBookSnapshot.

    Args:
        orderbook: OrderBook object from data models

    Returns:
        OrderBookSnapshot or None
    """
    if not orderbook:
        return None

    try:
        bids = orderbook.bids if hasattr(orderbook, 'bids') else []
        asks = orderbook.asks if hasattr(orderbook, 'asks') else []

        best_bid = float(bids[0].price) if bids else 0.0
        best_ask = float(asks[0].price) if asks else 0.0

        bid_depth = sum(float(level.size) for level in bids) if bids else 0.0
        ask_depth = sum(float(level.size) for level in asks) if asks else 0.0

        return OrderBookSnapshot(
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_levels=[(float(l.price), float(l.size)) for l in bids] if bids else [],
            ask_levels=[(float(l.price), float(l.size)) for l in asks] if asks else [],
            timestamp=orderbook.timestamp if hasattr(orderbook, 'timestamp') else datetime.utcnow()
        )
    except Exception as e:
        logger.debug(f"Error converting orderbook: {e}")
        return None
