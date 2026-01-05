"""Risk-aware position sizing using Kelly-inspired approach."""

import logging
from dataclasses import dataclass

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Position sizing calculation result."""

    size_usd_raw: float  # Initial calculated size
    size_usd_final: float  # Final size after all caps
    sizing_reason: str  # Explanation of sizing logic
    kelly_fraction: float  # Effective Kelly fraction used
    edge_contribution: float  # Portion of size from edge
    metadata: dict[str, float]  # Additional metrics


class PositionSizer:
    """Compute position sizes using risk-aware Kelly-inspired approach."""

    def __init__(self) -> None:
        """Initialize position sizer with settings."""
        self.settings = get_settings()
        
        # Kelly parameters (conservative)
        self.base_kelly_fraction = 0.25  # Base Kelly fraction (1/4 Kelly)
        self.min_kelly_fraction = 0.05  # Minimum Kelly
        self.max_kelly_fraction = 0.50  # Maximum Kelly (half Kelly max)
        
        # Dynamic position sizing limits
        self.min_pct_per_trade = 0.05  # Min 5% of available cash per trade
        self.max_pct_per_trade = 0.45  # Max 45% of available cash per trade
        self.max_pct_total_exposure = 0.90  # Hard cap: max 90% total portfolio exposure (not a target)
        
        # Edge-based scaling for dynamic sizing
        self.edge_scale_min = 0.045  # 4.5% edge = minimum sizing (5%)
        self.edge_scale_max = 0.10   # 10% edge = maximum sizing (45%)
        
    def compute_size(
        self,
        edge_net: float,
        confidence: float,
        available_usdc: float,
        reserved_usdc: float,
    ) -> SizingResult:
        """Compute position size using Kelly-inspired approach.

        Args:
            edge_net: Net edge after fees/slippage (expected value).
            confidence: Signal confidence (0-1).
            available_usdc: Available USDC balance.
            reserved_usdc: Already reserved USDC (open intents/orders).

        Returns:
            SizingResult with calculated size and reasoning.
        """
        if available_usdc <= 0:
            return SizingResult(
                size_usd_raw=0.0,
                size_usd_final=0.0,
                sizing_reason="no_available_balance",
                kelly_fraction=0.0,
                edge_contribution=0.0,
                metadata={"available_usdc": available_usdc},
            )

        # Total bankroll (available + reserved)
        total_bankroll = available_usdc + reserved_usdc

        # DYNAMIC SIZING: Scale position size between 5-45% based on edge quality
        # Edge scaling: 4.5% edge → 5% position, 10% edge → 45% position
        edge_abs = abs(edge_net)
        
        if edge_abs <= self.edge_scale_min:
            # At minimum edge threshold (4.5%), use minimum size (5%)
            edge_scale = 0.0
        elif edge_abs >= self.edge_scale_max:
            # At or above 10% edge, use maximum size (45%)
            edge_scale = 1.0
        else:
            # Linear interpolation between min and max
            edge_range = self.edge_scale_max - self.edge_scale_min
            edge_scale = (edge_abs - self.edge_scale_min) / edge_range
        
        # Calculate target percentage: interpolate between min (5%) and max (45%)
        pct_range = self.max_pct_per_trade - self.min_pct_per_trade
        target_pct = self.min_pct_per_trade + (edge_scale * pct_range)
        
        # Apply confidence scaling (high confidence = closer to target, low confidence = reduced)
        final_pct = target_pct * confidence
        
        # Calculate dollar amount
        sized_amount = available_usdc * final_pct
        
        # Cap at absolute maximum (45%)
        max_per_trade = available_usdc * self.max_pct_per_trade
        after_per_trade_cap = min(sized_amount, max_per_trade)
        
        logger.debug(
            f"Dynamic sizing: edge={edge_abs:.4f} ({edge_abs*100:.1f}%), "
            f"edge_scale={edge_scale:.3f}, target_pct={target_pct*100:.1f}%, "
            f"confidence={confidence:.2f}, final_pct={final_pct*100:.1f}%, "
            f"sized=${sized_amount:.2f}, max=${max_per_trade:.2f}"
        )

        # Cap 2: Ensure we don't exceed available (already handled by Cap 1)
        after_available_cap = after_per_trade_cap

        # Cap 3: Total exposure limit
        max_total_exposure = total_bankroll * self.max_pct_total_exposure
        remaining_exposure_capacity = max(0, max_total_exposure - reserved_usdc)
        after_exposure_cap = min(after_available_cap, remaining_exposure_capacity)

        # Cap 4: Configured absolute limits
        min_order = float(self.settings.min_order_usd)
        max_order = float(self.settings.max_order_usd)
        size_final = max(0, min(after_exposure_cap, max_order))
        
        actual_pct = (size_final / available_usdc * 100) if available_usdc > 0 else 0
        logger.info(
            f"Position size: ${size_final:.2f} ({actual_pct:.1f}% of ${available_usdc:.2f} available) "
            f"[edge={edge_abs*100:.1f}%, confidence={confidence:.0%}]"
        )

        # Determine primary reason for final size
        reason = self._determine_sizing_reason(
            kelly_size=sized_amount,
            after_per_trade_cap=after_per_trade_cap,
            after_available_cap=after_available_cap,
            after_exposure_cap=after_exposure_cap,
            size_final=size_final,
            min_order=min_order,
            max_order=max_order,
        )

        return SizingResult(
            size_usd_raw=sized_amount,
            size_usd_final=size_final,
            sizing_reason=reason,
            kelly_fraction=final_pct,  # Actual percentage used (0.05 to 0.45)
            edge_contribution=abs(edge_net),
            metadata={
                "total_bankroll": total_bankroll,
                "available_usdc": available_usdc,
                "reserved_usdc": reserved_usdc,
                "edge_abs": edge_abs,
                "edge_scale": edge_scale,
                "target_pct": target_pct,
                "confidence": confidence,
                "final_pct": final_pct,
                "sized_amount": sized_amount,
                "after_per_trade_cap": after_per_trade_cap,
                "after_available_cap": after_available_cap,
                "after_exposure_cap": after_exposure_cap,
                "remaining_exposure_capacity": remaining_exposure_capacity,
                "actual_pct": actual_pct,
            },
        )

    def _compute_kelly_fraction(self, edge_net: float, confidence: float) -> float:
        """Compute Kelly fraction based on edge and confidence.

        Args:
            edge_net: Net edge.
            confidence: Confidence score.

        Returns:
            Kelly fraction to use.
        """
        # Scale Kelly fraction with edge magnitude
        # More edge = more aggressive (but capped)
        edge_abs = abs(edge_net)
        
        if edge_abs <= self.edge_scale_min:
            # Minimum edge: use minimum Kelly
            kelly = self.min_kelly_fraction
        elif edge_abs >= self.edge_scale_max:
            # Maximum edge: use maximum Kelly
            kelly = self.max_kelly_fraction
        else:
            # Linear interpolation
            progress = (edge_abs - self.edge_scale_min) / (
                self.edge_scale_max - self.edge_scale_min
            )
            kelly = self.min_kelly_fraction + progress * (
                self.max_kelly_fraction - self.min_kelly_fraction
            )

        # Further reduce by confidence
        # Low confidence = more conservative
        kelly_adjusted = kelly * (0.5 + 0.5 * confidence)

        return max(self.min_kelly_fraction, min(kelly_adjusted, self.max_kelly_fraction))

    def _determine_sizing_reason(
        self,
        kelly_size: float,
        after_per_trade_cap: float,
        after_available_cap: float,
        after_exposure_cap: float,
        size_final: float,
        min_order: float,
        max_order: float,
    ) -> str:
        """Determine primary reason for final size."""
        # Check which cap was the binding constraint
        if size_final < min_order:
            return "below_min_order"
        if size_final >= max_order:
            return "capped_at_max_order"
        
        # Work backwards through caps
        if size_final == after_exposure_cap and after_exposure_cap < after_available_cap:
            return "capped_by_total_exposure"
        if size_final == after_available_cap and after_available_cap < after_per_trade_cap:
            return "capped_by_available_balance"
        if size_final == after_per_trade_cap and after_per_trade_cap < kelly_size:
            return "capped_by_max_pct_per_trade"
        
        return "kelly_based"

