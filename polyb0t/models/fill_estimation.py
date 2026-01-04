"""Expected fill price estimation using orderbook depth."""

import logging
from dataclasses import dataclass
from typing import Any

from polyb0t.config import get_settings
from polyb0t.data.models import OrderBook

logger = logging.getLogger(__name__)


@dataclass
class FillEstimate:
    """Estimated fill price and execution quality."""

    expected_price: float  # Expected fill price
    avg_price: float  # Volume-weighted average price
    price_impact_pct: float  # Price impact as percentage
    levels_consumed: int  # Number of orderbook levels consumed
    total_depth: float  # Total depth used
    slippage_bps: int  # Slippage in basis points
    is_feasible: bool  # Whether order can be filled
    metadata: dict[str, Any]  # Additional diagnostics


class FillPriceEstimator:
    """Estimate realistic fill prices using orderbook data."""

    def __init__(self) -> None:
        """Initialize fill price estimator."""
        self.settings = get_settings()
        # Conservative assumptions
        self.taker_fee_bps = int(self.settings.fee_bps)  # Default 20 bps
        self.min_depth_usd = 10.0  # Minimum depth to consider level
        self.max_levels_to_consume = 5  # Conservative limit

    def estimate_fill(
        self,
        orderbook: OrderBook,
        side: str,
        size_usd: float,
    ) -> FillEstimate | None:
        """Estimate fill price for a taker order.

        Args:
            orderbook: Current orderbook snapshot.
            side: "BUY" or "SELL".
            size_usd: Order size in USD notional.

        Returns:
            FillEstimate or None if cannot estimate.
        """
        if not orderbook or size_usd <= 0:
            return None

        # Choose correct side of book
        if side == "BUY":
            levels = orderbook.asks  # Buy from asks
        else:
            levels = orderbook.bids  # Sell to bids

        if not levels:
            return None

        # Walk through orderbook levels
        # In Polymarket: to buy $X of outcome at price p, you need X/p shares
        # Total cost = sum(shares * price) = X for each filled portion
        remaining_size_usd = size_usd
        total_shares = 0.0  # Total shares we'll acquire
        weighted_price_sum = 0.0  # For VWAP calculation
        levels_used = 0

        for level in levels[: self.max_levels_to_consume]:
            if remaining_size_usd <= 0:
                break

            price = level.price
            if price <= 0 or price > 1:
                continue  # Invalid price

            size_shares = level.size  # Shares available at this level
            depth_usd = size_shares * price  # USD value at this level

            # Skip levels with insufficient depth
            if depth_usd < self.min_depth_usd:
                continue

            # How much can we consume from this level?
            consumed_usd = min(remaining_size_usd, depth_usd)
            consumed_shares = consumed_usd / price
            
            total_shares += consumed_shares
            weighted_price_sum += consumed_shares * price
            remaining_size_usd -= consumed_usd
            levels_used += 1

        # Check if we got a complete fill
        if remaining_size_usd > size_usd * 0.1:  # Allow 10% shortfall
            # Not enough liquidity
            return FillEstimate(
                expected_price=0.0,
                avg_price=0.0,
                price_impact_pct=0.0,
                levels_consumed=0,
                total_depth=0.0,
                slippage_bps=0,
                is_feasible=False,
                metadata={"reason": "insufficient_liquidity", "remaining_size": remaining_size_usd},
            )

        # Calculate volume-weighted average price (probability)
        if total_shares <= 0:
            return None

        avg_price = weighted_price_sum / total_shares

        # Add taker fees
        # For BUY: you pay avg_price + fee (increases effective price)
        # For SELL: you receive avg_price - fee (decreases effective price)
        # Fee is on notional, so effective price shift is small
        fee_rate = self.taker_fee_bps / 10000.0
        if side == "BUY":
            # Buying: effective price is higher due to fees
            expected_price = min(1.0, avg_price * (1 + fee_rate))
        else:
            # Selling: effective price is lower due to fees
            expected_price = max(0.0, avg_price * (1 - fee_rate))

        # Calculate price impact vs mid
        best_price = levels[0].price
        price_impact_pct = abs(expected_price - best_price) / best_price if best_price > 0 else 0.0

        # Slippage in basis points
        slippage_bps = int(price_impact_pct * 10000)

        return FillEstimate(
            expected_price=expected_price,
            avg_price=avg_price,
            price_impact_pct=price_impact_pct,
            levels_consumed=levels_used,
            total_depth=total_shares,
            slippage_bps=slippage_bps,
            is_feasible=True,
            metadata={
                "filled_size_usd": size_usd - remaining_size_usd,
                "total_shares": total_shares,
                "best_price": best_price,
                "taker_fee_bps": self.taker_fee_bps,
            },
        )

    def compute_net_edge(
        self,
        p_model: float,
        p_market_mid: float,
        orderbook: OrderBook,
        side: str,
        size_usd: float,
    ) -> tuple[float, float, FillEstimate | None]:
        """Compute edge_raw (mid-based) and edge_net (fill-based).

        Args:
            p_model: Model probability estimate.
            p_market_mid: Market mid-price probability.
            orderbook: Current orderbook.
            side: "BUY" or "SELL".
            size_usd: Proposed order size.

        Returns:
            Tuple of (edge_raw, edge_net, fill_estimate).
        """
        # Raw edge (naive mid-price)
        if side == "BUY":
            edge_raw = p_model - p_market_mid
        else:
            edge_raw = p_market_mid - p_model

        # Estimate fill price
        fill_est = self.estimate_fill(orderbook, side, size_usd)
        
        if not fill_est or not fill_est.is_feasible:
            # Cannot fill, edge_net is undefined
            return edge_raw, float('-inf'), fill_est

        # Expected fill probability
        p_fill = fill_est.expected_price

        # Net edge (realistic)
        if side == "BUY":
            edge_net = p_model - p_fill
        else:
            edge_net = p_fill - p_model

        return edge_raw, edge_net, fill_est

