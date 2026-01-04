"""Risk management and position sizing."""

import logging
from typing import Any

from polyb0t.config import get_settings
from polyb0t.models.strategy_baseline import TradingSignal

logger = logging.getLogger(__name__)


class RiskCheckResult:
    """Result of risk check."""

    def __init__(
        self,
        approved: bool,
        reason: str | None = None,
        max_position_size: float | None = None,
    ) -> None:
        """Initialize risk check result.

        Args:
            approved: Whether position is approved.
            reason: Reason for rejection if not approved.
            max_position_size: Maximum allowed position size if approved.
        """
        self.approved = approved
        self.reason = reason
        self.max_position_size = max_position_size


class RiskManager:
    """Enforce portfolio risk limits and position sizing."""

    def __init__(self) -> None:
        """Initialize risk manager."""
        self.settings = get_settings()
        self.peak_equity = self.settings.paper_bankroll
        self.is_trading_halted = False

    def check_position(
        self,
        signal: TradingSignal,
        current_cash: float,
        current_positions: dict[str, Any],
        portfolio_exposure: float,
    ) -> RiskCheckResult:
        """Check if new position passes risk constraints.

        Args:
            signal: Trading signal to evaluate.
            current_cash: Available cash balance.
            current_positions: Dict of token_id -> position info.
            portfolio_exposure: Total current exposure value.

        Returns:
            RiskCheckResult with approval and sizing.
        """
        # Check if trading is halted
        if self.is_trading_halted:
            return RiskCheckResult(
                approved=False,
                reason="Trading halted due to drawdown limit",
            )

        # Check if already have position in this token
        if signal.token_id in current_positions:
            existing = current_positions[signal.token_id]
            if existing["side"] == signal.side:
                return RiskCheckResult(
                    approved=False,
                    reason=f"Already have {signal.side} position in {signal.token_id}",
                )

        # Calculate position size
        position_size = self._calculate_position_size(
            signal,
            current_cash,
            portfolio_exposure,
        )

        if position_size <= 0:
            return RiskCheckResult(
                approved=False,
                reason="Calculated position size is zero or negative",
            )

        # Check exposure limits
        max_total_exposure = (
            self.settings.paper_bankroll * self.settings.max_total_exposure_pct / 100
        )

        if portfolio_exposure + position_size > max_total_exposure:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"Would exceed max total exposure: "
                    f"{portfolio_exposure + position_size:.2f} > {max_total_exposure:.2f}"
                ),
            )

        # Check category exposure (if applicable)
        # This is a simplified check; more sophisticated would track per-category
        category_exposure = self._get_category_exposure(
            signal.market_id,
            current_positions,
        )
        max_category_exposure = (
            self.settings.paper_bankroll * self.settings.max_per_category_exposure_pct / 100
        )

        if category_exposure + position_size > max_category_exposure:
            return RiskCheckResult(
                approved=False,
                reason=(
                    f"Would exceed max category exposure: "
                    f"{category_exposure + position_size:.2f} > {max_category_exposure:.2f}"
                ),
            )

        return RiskCheckResult(
            approved=True,
            max_position_size=position_size,
        )

    def _calculate_position_size(
        self,
        signal: TradingSignal,
        current_cash: float,
        portfolio_exposure: float,
    ) -> float:
        """Calculate position size using capped fractional approach.

        Args:
            signal: Trading signal.
            current_cash: Available cash.
            portfolio_exposure: Current total exposure.

        Returns:
            Position size in dollars.
        """
        # Use fixed percentage of bankroll (simpler than Kelly for MVP)
        max_position_pct = self.settings.max_position_pct / 100
        bankroll = self.settings.paper_bankroll

        # Base position size
        base_size = bankroll * max_position_pct

        # Adjust by confidence
        adjusted_size = base_size * signal.confidence

        # Ensure we have enough cash
        available_size = min(adjusted_size, current_cash)

        # Round to reasonable precision
        return round(available_size, 2)

    def _get_category_exposure(
        self,
        market_id: str,
        current_positions: dict[str, Any],
    ) -> float:
        """Calculate current exposure in same category/market.

        Args:
            market_id: Market condition ID.
            current_positions: Current positions dict.

        Returns:
            Total exposure in this category.
        """
        # Simplified: sum exposure for positions in same market
        exposure = 0.0
        for pos in current_positions.values():
            if pos.get("market_id") == market_id:
                exposure += pos.get("exposure", 0.0)

        return exposure

    def check_drawdown(self, current_equity: float) -> bool:
        """Check if drawdown limit exceeded.

        Args:
            current_equity: Current portfolio equity.

        Returns:
            True if drawdown limit exceeded (should halt trading).
        """
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate drawdown
        drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100

        if drawdown_pct >= self.settings.drawdown_limit_pct:
            logger.error(
                f"Drawdown limit exceeded: {drawdown_pct:.2f}% >= "
                f"{self.settings.drawdown_limit_pct}%"
            )
            self.is_trading_halted = True
            return True

        return False

    def reset_halt(self) -> None:
        """Reset trading halt (for manual intervention)."""
        self.is_trading_halted = False
        logger.info("Trading halt manually reset")

