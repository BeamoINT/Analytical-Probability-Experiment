"""Executor interface for human-in-the-loop intent execution.

This module defines a small interface used by live mode to execute ONLY user-approved
intents. It is intentionally minimal to keep safety boundaries obvious.
"""

from __future__ import annotations

from typing import Any, Protocol

from polyb0t.execution.intents import TradeIntent


class Executor(Protocol):
    """Minimal executor interface for processing approved intents."""

    def process_approved_intents(self, cycle_id: str) -> dict[str, Any]:
        """Process all approved intents awaiting execution and return summary."""

    def execute_intent(self, intent: TradeIntent, cycle_id: str) -> dict[str, Any]:
        """Execute a single approved intent and return result."""


