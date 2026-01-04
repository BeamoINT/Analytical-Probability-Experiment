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
        
        # Bankroll allocation limits
        self.max_pct_per_trade = 0.15  # Max 15% of bankroll per trade
        self.max_pct_total_exposure = 0.40  # Max 40% total exposure
        
        # Edge-based scaling
        self.edge_scale_min = 0.02  # Minimum edge for min sizing
        self.edge_scale_max = 0.10  # Edge for max sizing
        
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

        # Kelly fraction scales with edge (conservative)
        kelly_frac = self._compute_kelly_fraction(edge_net, confidence)

        # Kelly suggests: size = bankroll * kelly_fraction * edge
        # For binary outcome with edge e and confidence c:
        # kelly = (p_win * b - p_lose) / b
        # Simplified: kelly ≈ edge (for small edges)
        kelly_size = total_bankroll * kelly_frac * abs(edge_net)

        # Scale with confidence
        confidence_adjusted = kelly_size * confidence

        # Cap 1: Max percentage of bankroll per trade
        max_per_trade = total_bankroll * self.max_pct_per_trade
        after_per_trade_cap = min(confidence_adjusted, max_per_trade)

        # Cap 2: Max percentage of available (don't over-commit available)
        max_available_commit = available_usdc * 0.95  # Keep 5% cushion
        after_available_cap = min(after_per_trade_cap, max_available_commit)

        # Cap 3: Total exposure limit
        max_total_exposure = total_bankroll * self.max_pct_total_exposure
        remaining_exposure_capacity = max(0, max_total_exposure - reserved_usdc)
        after_exposure_cap = min(after_available_cap, remaining_exposure_capacity)

        # Cap 4: Configured absolute limits
        min_order = float(self.settings.min_order_usd)
        max_order = float(self.settings.max_order_usd)
        size_final = max(0, min(after_exposure_cap, max_order))

        # Determine primary reason for final size
        reason = self._determine_sizing_reason(
            kelly_size=kelly_size,
            after_per_trade_cap=after_per_trade_cap,
            after_available_cap=after_available_cap,
            after_exposure_cap=after_exposure_cap,
            size_final=size_final,
            min_order=min_order,
            max_order=max_order,
        )

        return SizingResult(
            size_usd_raw=kelly_size,
            size_usd_final=size_final,
            sizing_reason=reason,
            kelly_fraction=kelly_frac,
            edge_contribution=abs(edge_net),
            metadata={
                "total_bankroll": total_bankroll,
                "available_usdc": available_usdc,
                "reserved_usdc": reserved_usdc,
                "confidence_adjusted": confidence_adjusted,
                "after_per_trade_cap": after_per_trade_cap,
                "after_available_cap": after_available_cap,
                "after_exposure_cap": after_exposure_cap,
                "remaining_exposure_capacity": remaining_exposure_capacity,
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

