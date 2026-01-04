"""Kill switch system for automated safety halts."""

import logging
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque

from sqlalchemy.orm import Session

from polyb0t.config import get_settings
from polyb0t.data.storage import KillSwitchEventDB

logger = logging.getLogger(__name__)


class KillSwitchType:
    """Kill switch types."""

    DRAWDOWN = "DRAWDOWN"
    API_ERROR_RATE = "API_ERROR_RATE"
    STALE_DATA = "STALE_DATA"
    SPREAD_ANOMALY = "SPREAD_ANOMALY"
    DAILY_LOSS = "DAILY_LOSS"
    MANUAL = "MANUAL"


class KillSwitchManager:
    """Manages kill switches and safety halts."""

    def __init__(self, db_session: Session) -> None:
        """Initialize kill switch manager.

        Args:
            db_session: Database session.
        """
        self.db_session = db_session
        self.settings = get_settings()

        # Track recent API calls for error rate calculation
        self.api_calls: Deque[tuple[datetime, bool]] = deque(
            maxlen=100
        )  # (timestamp, success)

        # Track last data timestamp
        self.last_data_timestamp: datetime | None = None

        # Track daily starting equity for daily loss check
        self.daily_start_equity: float | None = None
        self.daily_start_date: datetime | None = None

        # Active kill switches
        self.active_switches: dict[str, str] = {}  # switch_type -> event_id

    def check_all_switches(
        self,
        current_equity: float,
        peak_equity: float,
        current_spreads: dict[str, float],
        cycle_id: str,
    ) -> list[str]:
        """Check all kill switches.

        Args:
            current_equity: Current portfolio equity.
            peak_equity: Peak equity (for drawdown).
            current_spreads: Dict of token_id -> spread.
            cycle_id: Current cycle ID.

        Returns:
            List of triggered switch types.
        """
        triggered = []

        # Check drawdown
        if self._check_drawdown(current_equity, peak_equity, cycle_id):
            triggered.append(KillSwitchType.DRAWDOWN)

        # Check API error rate
        if self._check_api_error_rate(cycle_id):
            triggered.append(KillSwitchType.API_ERROR_RATE)

        # Check stale data
        if self._check_stale_data(cycle_id):
            triggered.append(KillSwitchType.STALE_DATA)

        # Check spread anomalies
        if self._check_spread_anomaly(current_spreads, cycle_id):
            triggered.append(KillSwitchType.SPREAD_ANOMALY)

        # Check daily loss
        if self._check_daily_loss(current_equity, cycle_id):
            triggered.append(KillSwitchType.DAILY_LOSS)

        return triggered

    def _check_drawdown(
        self, current_equity: float, peak_equity: float, cycle_id: str
    ) -> bool:
        """Check drawdown limit.

        Args:
            current_equity: Current equity.
            peak_equity: Peak equity.
            cycle_id: Current cycle ID.

        Returns:
            True if kill switch triggered.
        """
        if peak_equity <= 0:
            return False

        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100

        if drawdown_pct >= self.settings.drawdown_limit_pct:
            if KillSwitchType.DRAWDOWN not in self.active_switches:
                self._trigger_switch(
                    KillSwitchType.DRAWDOWN,
                    trigger_value=drawdown_pct,
                    threshold_value=self.settings.drawdown_limit_pct,
                    description=f"Drawdown {drawdown_pct:.2f}% >= limit {self.settings.drawdown_limit_pct}%",
                    cycle_id=cycle_id,
                )
            return True

        return False

    def _check_api_error_rate(self, cycle_id: str) -> bool:
        """Check API error rate.

        Args:
            cycle_id: Current cycle ID.

        Returns:
            True if kill switch triggered.
        """
        if len(self.api_calls) < 10:  # Need at least 10 calls
            return False

        # Calculate error rate over last 5 minutes
        five_min_ago = datetime.utcnow() - timedelta(minutes=5)
        recent_calls = [call for call in self.api_calls if call[0] >= five_min_ago]

        if len(recent_calls) < 5:
            return False

        errors = sum(1 for _, success in recent_calls if not success)
        error_rate = (errors / len(recent_calls)) * 100

        if error_rate >= self.settings.max_api_error_rate_pct:
            if KillSwitchType.API_ERROR_RATE not in self.active_switches:
                self._trigger_switch(
                    KillSwitchType.API_ERROR_RATE,
                    trigger_value=error_rate,
                    threshold_value=self.settings.max_api_error_rate_pct,
                    description=f"API error rate {error_rate:.1f}% >= limit {self.settings.max_api_error_rate_pct}%",
                    cycle_id=cycle_id,
                )
            return True

        return False

    def _check_stale_data(self, cycle_id: str) -> bool:
        """Check for stale data.

        Args:
            cycle_id: Current cycle ID.

        Returns:
            True if kill switch triggered.
        """
        if self.last_data_timestamp is None:
            return False

        age_seconds = (datetime.utcnow() - self.last_data_timestamp).total_seconds()

        if age_seconds >= self.settings.max_stale_data_seconds:
            if KillSwitchType.STALE_DATA not in self.active_switches:
                self._trigger_switch(
                    KillSwitchType.STALE_DATA,
                    trigger_value=age_seconds,
                    threshold_value=float(self.settings.max_stale_data_seconds),
                    description=f"Data age {age_seconds:.0f}s >= limit {self.settings.max_stale_data_seconds}s",
                    cycle_id=cycle_id,
                )
            return True

        return False

    def _check_spread_anomaly(
        self, current_spreads: dict[str, float], cycle_id: str
    ) -> bool:
        """Check for spread anomalies.

        Args:
            current_spreads: Current spreads by token.
            cycle_id: Current cycle ID.

        Returns:
            True if kill switch triggered.
        """
        if not current_spreads:
            return False

        # Check if any spread exceeds max_spread * multiplier
        max_normal_spread = self.settings.max_spread * self.settings.max_spread_multiplier

        anomalies = {
            token_id: spread
            for token_id, spread in current_spreads.items()
            if spread > max_normal_spread
        }

        if anomalies:
            if KillSwitchType.SPREAD_ANOMALY not in self.active_switches:
                max_spread = max(anomalies.values())
                self._trigger_switch(
                    KillSwitchType.SPREAD_ANOMALY,
                    trigger_value=max_spread,
                    threshold_value=max_normal_spread,
                    description=f"Spread {max_spread:.4f} > normal {max_normal_spread:.4f} ({len(anomalies)} markets)",
                    cycle_id=cycle_id,
                )
            return True

        return False

    def _check_daily_loss(self, current_equity: float, cycle_id: str) -> bool:
        """Check daily loss limit.

        Args:
            current_equity: Current equity.
            cycle_id: Current cycle ID.

        Returns:
            True if kill switch triggered.
        """
        # Reset daily tracking if new day
        today = datetime.utcnow().date()
        if self.daily_start_date is None or self.daily_start_date.date() != today:
            self.daily_start_equity = current_equity
            self.daily_start_date = datetime.utcnow()
            return False

        if self.daily_start_equity is None or self.daily_start_equity <= 0:
            return False

        daily_loss_pct = (
            (self.daily_start_equity - current_equity) / self.daily_start_equity * 100
        )

        if daily_loss_pct >= self.settings.max_daily_loss_pct:
            if KillSwitchType.DAILY_LOSS not in self.active_switches:
                self._trigger_switch(
                    KillSwitchType.DAILY_LOSS,
                    trigger_value=daily_loss_pct,
                    threshold_value=self.settings.max_daily_loss_pct,
                    description=f"Daily loss {daily_loss_pct:.2f}% >= limit {self.settings.max_daily_loss_pct}%",
                    cycle_id=cycle_id,
                )
            return True

        return False

    def _trigger_switch(
        self,
        switch_type: str,
        trigger_value: float,
        threshold_value: float,
        description: str,
        cycle_id: str,
    ) -> None:
        """Trigger a kill switch.

        Args:
            switch_type: Type of kill switch.
            trigger_value: Value that triggered switch.
            threshold_value: Threshold value.
            description: Human-readable description.
            cycle_id: Current cycle ID.
        """
        event_id = str(uuid.uuid4())

        # Persist to database
        db_event = KillSwitchEventDB(
            event_id=event_id,
            cycle_id=cycle_id,
            timestamp=datetime.utcnow(),
            switch_type=switch_type,
            trigger_value=trigger_value,
            threshold_value=threshold_value,
            description=description,
            is_active=True,
        )
        self.db_session.add(db_event)
        self.db_session.commit()

        # Track in memory
        self.active_switches[switch_type] = event_id

        logger.error(
            f"KILL SWITCH TRIGGERED: {switch_type}",
            extra={
                "event_id": event_id,
                "switch_type": switch_type,
                "trigger_value": trigger_value,
                "threshold_value": threshold_value,
                "description": description,
            },
        )

    def record_api_call(self, success: bool) -> None:
        """Record API call result for error rate tracking.

        Args:
            success: Whether call succeeded.
        """
        self.api_calls.append((datetime.utcnow(), success))

    def update_data_timestamp(self) -> None:
        """Update last data timestamp."""
        self.last_data_timestamp = datetime.utcnow()

    def is_any_active(self) -> bool:
        """Check if any kill switches are active.

        Returns:
            True if any switch is active.
        """
        return len(self.active_switches) > 0

    def get_active_switches(self) -> dict[str, str]:
        """Get active kill switches.

        Returns:
            Dict of switch_type -> event_id.
        """
        return self.active_switches.copy()

    def clear_switch(self, switch_type: str) -> bool:
        """Clear a kill switch (manual override).

        Args:
            switch_type: Type of switch to clear.

        Returns:
            True if cleared successfully.
        """
        if switch_type not in self.active_switches:
            return False

        event_id = self.active_switches[switch_type]

        # Update database
        db_event = (
            self.db_session.query(KillSwitchEventDB).filter_by(event_id=event_id).first()
        )
        if db_event:
            db_event.is_active = False
            db_event.cleared_at = datetime.utcnow()
            self.db_session.commit()

        # Remove from active tracking
        del self.active_switches[switch_type]

        logger.warning(
            f"Kill switch cleared: {switch_type}",
            extra={"switch_type": switch_type, "event_id": event_id},
        )

        return True

    def clear_all_switches(self) -> int:
        """Clear all active kill switches (manual override).

        Returns:
            Number of switches cleared.
        """
        count = 0
        for switch_type in list(self.active_switches.keys()):
            if self.clear_switch(switch_type):
                count += 1

        return count

