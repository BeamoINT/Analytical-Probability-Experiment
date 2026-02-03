"""Risk management and position sizing."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from polyb0t.config import get_settings
from polyb0t.models.strategy_baseline import TradingSignal

logger = logging.getLogger(__name__)

# File to persist peak equity across restarts
PEAK_EQUITY_FILE = Path("data/peak_equity.json")


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
        self.is_trading_halted = False
        self.halt_reason: str | None = None
        self.halted_at: datetime | None = None

        # Load persisted peak equity or use paper bankroll as default
        self._peak_equity_file = PEAK_EQUITY_FILE
        saved_peak = self._load_peak_equity()

        if self.settings.mode == "live":
            # For live mode, start with a conservative peak (will be updated from on-chain)
            # If we have a saved peak, use that; otherwise start at 0 (will be set on first update)
            self.peak_equity = saved_peak if saved_peak > 0 else 0.0
            logger.info(f"Live mode: loaded peak equity ${self.peak_equity:.2f}")
        else:
            # Paper mode: use paper bankroll
            self.peak_equity = self.settings.paper_bankroll

    def _load_peak_equity(self) -> float:
        """Load peak equity from persistent storage."""
        try:
            if self._peak_equity_file.exists():
                with open(self._peak_equity_file, "r") as f:
                    data = json.load(f)
                    return float(data.get("peak_equity", 0.0))
        except Exception as e:
            logger.warning(f"Could not load peak equity: {e}")
        return 0.0

    def _save_peak_equity(self) -> None:
        """Save peak equity to persistent storage."""
        try:
            self._peak_equity_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._peak_equity_file, "w") as f:
                json.dump({
                    "peak_equity": self.peak_equity,
                    "updated_at": datetime.utcnow().isoformat(),
                }, f)
        except Exception as e:
            logger.warning(f"Could not save peak equity: {e}")

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
        if current_equity <= 0:
            return False

        # Update peak equity (only if positive and higher than current peak)
        if current_equity > self.peak_equity:
            old_peak = self.peak_equity
            self.peak_equity = current_equity
            self._save_peak_equity()
            if old_peak > 0:
                logger.info(f"New peak equity: ${self.peak_equity:.2f} (was ${old_peak:.2f})")

        # If peak is 0 (first run in live mode), initialize it
        if self.peak_equity == 0:
            self.peak_equity = current_equity
            self._save_peak_equity()
            logger.info(f"Initialized peak equity to ${self.peak_equity:.2f}")
            return False

        # Calculate drawdown
        drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100

        if drawdown_pct >= self.settings.drawdown_limit_pct:
            if not self.is_trading_halted:
                logger.error(
                    f"CIRCUIT BREAKER TRIGGERED: Drawdown {drawdown_pct:.2f}% >= limit {self.settings.drawdown_limit_pct}%",
                    extra={
                        "current_equity": current_equity,
                        "peak_equity": self.peak_equity,
                        "drawdown_pct": drawdown_pct,
                        "limit_pct": self.settings.drawdown_limit_pct,
                    }
                )
                self.is_trading_halted = True
                self.halt_reason = f"Drawdown {drawdown_pct:.2f}% exceeded limit {self.settings.drawdown_limit_pct}%"
                self.halted_at = datetime.utcnow()
            return True

        return False

    def get_drawdown_pct(self, current_equity: float) -> float:
        """Get current drawdown percentage.

        Args:
            current_equity: Current portfolio equity.

        Returns:
            Drawdown as percentage (0-100).
        """
        if self.peak_equity <= 0 or current_equity <= 0:
            return 0.0
        return max(0.0, ((self.peak_equity - current_equity) / self.peak_equity) * 100)

    def reset_halt(self, new_peak: float | None = None) -> None:
        """Reset trading halt (for manual intervention).

        Args:
            new_peak: Optional new peak equity to set (useful if resetting after deposit).
        """
        self.is_trading_halted = False
        self.halt_reason = None
        self.halted_at = None

        if new_peak is not None and new_peak > 0:
            self.peak_equity = new_peak
            self._save_peak_equity()
            logger.info(f"Trading halt reset with new peak equity: ${new_peak:.2f}")
        else:
            logger.info("Trading halt manually reset")

    def get_halt_status(self) -> dict[str, Any]:
        """Get current halt status.

        Returns:
            Dict with halt status information.
        """
        return {
            "is_halted": self.is_trading_halted,
            "reason": self.halt_reason,
            "halted_at": self.halted_at.isoformat() if self.halted_at else None,
            "peak_equity": self.peak_equity,
        }

