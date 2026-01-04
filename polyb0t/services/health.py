"""Health check and status monitoring."""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus:
    """Application health status."""

    def __init__(self) -> None:
        """Initialize health status."""
        self.started_at = datetime.utcnow()
        self.last_cycle_at: datetime | None = None
        self.last_cycle_duration: float = 0.0
        self.total_cycles = 0
        self.is_healthy = True
        self.error_message: str | None = None

    def update_cycle(self, duration: float) -> None:
        """Update cycle metrics.

        Args:
            duration: Cycle duration in seconds.
        """
        self.last_cycle_at = datetime.utcnow()
        self.last_cycle_duration = duration
        self.total_cycles += 1

    def mark_unhealthy(self, error: str) -> None:
        """Mark as unhealthy.

        Args:
            error: Error message.
        """
        self.is_healthy = False
        self.error_message = error
        logger.error(f"Health check failed: {error}")

    def mark_healthy(self) -> None:
        """Mark as healthy."""
        self.is_healthy = True
        self.error_message = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Health status as dict.
        """
        uptime = (datetime.utcnow() - self.started_at).total_seconds()

        return {
            "is_healthy": self.is_healthy,
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": uptime,
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "last_cycle_duration": self.last_cycle_duration,
            "total_cycles": self.total_cycles,
            "error_message": self.error_message,
        }


# Global health status instance
_health_status = HealthStatus()


def get_health_status() -> HealthStatus:
    """Get global health status instance.

    Returns:
        HealthStatus instance.
    """
    return _health_status

