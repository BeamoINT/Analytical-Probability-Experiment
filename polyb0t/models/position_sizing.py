"""Risk-aware position sizing using percentage-based approach.

Position sizing is 100% percentage-based with NO dollar caps:
- Max 15% of available balance per trade
- Size scales with edge quality and prediction confidence
- Expert confidence multiplier affects final sizing (0.3x - 1.0x)
"""

import logging
from dataclasses import dataclass

from polyb0t.config import get_settings

logger = logging.getLogger(__name__)


# Maximum position size as percentage of available balance
MAX_POSITION_PCT = 0.15  # 15% max per trade


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
    """Compute position sizes using percentage-based approach.
    
    NO dollar caps - only percentage-based limits:
    - Max 15% of available balance per trade
    - Size scales with edge, prediction confidence, and expert confidence
    """

    def __init__(self) -> None:
        """Initialize position sizer with settings."""
        self.settings = get_settings()
        
        # Percentage-based limits (NO dollar caps)
        self.min_pct_per_trade = 0.02  # Min 2% of available cash per trade
        self.max_pct_per_trade = MAX_POSITION_PCT  # Max 15% per trade
        
        # Total exposure comes from settings
        self.max_pct_total_exposure = float(self.settings.max_total_exposure_pct) / 100.0
        
        # Edge-based scaling for dynamic sizing
        self.edge_scale_min = 0.02   # 2% edge = minimum sizing
        self.edge_scale_max = 0.08   # 8% edge = maximum sizing
        
    def compute_size(
        self,
        edge_net: float,
        confidence: float,
        available_usdc: float,
        reserved_usdc: float,
        expert_confidence_multiplier: float = 1.0,
    ) -> SizingResult:
        """Compute position size using percentage-based approach.

        Size formula: final_pct = base_pct * prediction_confidence * expert_confidence_multiplier
        Maximum: 15% of available balance per trade
        NO dollar caps - purely percentage-based.

        Args:
            edge_net: Net edge after fees/slippage (expected value).
            confidence: Prediction confidence (0-1).
            available_usdc: Available USDC balance.
            reserved_usdc: Already reserved USDC (open intents/orders).
            expert_confidence_multiplier: Expert-specific multiplier (0.3 to 1.0).

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

        # Clamp expert confidence multiplier to valid range (0.3 to 1.0)
        expert_mult = max(0.3, min(1.0, expert_confidence_multiplier))

        # PERCENTAGE-BASED SIZING: Scale between 2-15% based on edge quality
        edge_abs = abs(edge_net)
        
        if edge_abs <= self.edge_scale_min:
            # At minimum edge threshold, use minimum size
            edge_scale = 0.0
        elif edge_abs >= self.edge_scale_max:
            # At or above max edge, use maximum size
            edge_scale = 1.0
        else:
            # Linear interpolation between min and max
            edge_range = self.edge_scale_max - self.edge_scale_min
            edge_scale = (edge_abs - self.edge_scale_min) / edge_range
        
        # Calculate BASE percentage: interpolate between min (2%) and max (15%)
        pct_range = self.max_pct_per_trade - self.min_pct_per_trade
        base_pct = self.min_pct_per_trade + (edge_scale * pct_range)
        
        # Apply the sizing formula:
        # final_pct = base_pct * prediction_confidence * expert_confidence_multiplier
        final_pct = base_pct * confidence * expert_mult
        
        # Ensure we don't exceed max (15%)
        final_pct = min(final_pct, self.max_pct_per_trade)
        
        # Calculate dollar amount
        sized_amount = available_usdc * final_pct
        
        # Cap at absolute maximum (15%)
        max_per_trade = available_usdc * self.max_pct_per_trade
        after_per_trade_cap = min(sized_amount, max_per_trade)
        
        logger.debug(
            f"Percentage sizing: edge={edge_abs:.4f} ({edge_abs*100:.1f}%), "
            f"base_pct={base_pct*100:.1f}%, confidence={confidence:.2f}, "
            f"expert_mult={expert_mult:.2f}, final_pct={final_pct*100:.1f}%, "
            f"size=${sized_amount:.2f}"
        )

        # Cap: Ensure we don't exceed available
        after_available_cap = after_per_trade_cap

        # Cap: Total exposure limit
        max_total_exposure = total_bankroll * self.max_pct_total_exposure
        remaining_exposure_capacity = max(0, max_total_exposure - reserved_usdc)
        after_exposure_cap = min(after_available_cap, remaining_exposure_capacity)

        # Final size (NO DOLLAR CAPS - purely percentage-based)
        size_final = max(0, after_exposure_cap)
        
        # Minimum order check (use settings but this is the only dollar-based check)
        min_order = float(self.settings.min_order_usd)
        if size_final < min_order and size_final > 0:
            # Too small to be worth it
            size_final = 0
            reason = "below_min_order"
        else:
            reason = self._determine_sizing_reason(
                kelly_size=sized_amount,
                after_per_trade_cap=after_per_trade_cap,
                after_available_cap=after_available_cap,
                after_exposure_cap=after_exposure_cap,
                size_final=size_final,
            )
        
        actual_pct = (size_final / available_usdc * 100) if available_usdc > 0 else 0
        market_pct = (size_final / total_bankroll * 100) if total_bankroll > 0 else 0
        logger.info(
            f"Position size: ${size_final:.2f} ({actual_pct:.1f}% of available, {market_pct:.1f}% of portfolio) "
            f"[edge={edge_abs*100:.1f}%, conf={confidence:.2f}, expert_mult={expert_mult:.2f}]"
        )

        return SizingResult(
            size_usd_raw=sized_amount,
            size_usd_final=size_final,
            sizing_reason=reason,
            kelly_fraction=final_pct,  # Actual percentage used
            edge_contribution=abs(edge_net),
            metadata={
                "total_bankroll": total_bankroll,
                "available_usdc": available_usdc,
                "reserved_usdc": reserved_usdc,
                "edge_abs": edge_abs,
                "edge_scale": edge_scale,
                "base_pct": base_pct,
                "confidence": confidence,
                "expert_confidence_multiplier": expert_mult,
                "final_pct": final_pct,
                "sized_amount": sized_amount,
                "after_per_trade_cap": after_per_trade_cap,
                "after_available_cap": after_available_cap,
                "after_exposure_cap": after_exposure_cap,
                "remaining_exposure_capacity": remaining_exposure_capacity,
                "actual_pct": actual_pct,
            },
        )

    def _determine_sizing_reason(
        self,
        kelly_size: float,
        after_per_trade_cap: float,
        after_available_cap: float,
        after_exposure_cap: float,
        size_final: float,
    ) -> str:
        """Determine primary reason for final size."""
        # Work backwards through caps
        if size_final == after_exposure_cap and after_exposure_cap < after_available_cap:
            return "capped_by_total_exposure"
        if size_final == after_available_cap and after_available_cap < after_per_trade_cap:
            return "capped_by_available_balance"
        if size_final == after_per_trade_cap and after_per_trade_cap < kelly_size:
            return "capped_by_max_pct_per_trade"
        
        return "percentage_based"

