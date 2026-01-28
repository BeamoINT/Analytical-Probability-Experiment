"""Dry-Run Stats Collector.

Collects and saves performance statistics during dry-run trading.
Stats are collected periodically (default: every 6 hours) and saved
to a JSON file for monitoring model performance over time.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DryRunStatsCollector:
    """Collects performance stats during dry-run trading.

    This collector tracks:
    - Signals generated and their outcomes
    - Hypothetical P&L from dry-run trades
    - Expert usage and performance
    - Category-level performance

    Stats are appended to a JSON file periodically for long-term analysis.
    """

    def __init__(
        self,
        output_path: str = "data/dryrun_stats.json",
        max_history: int = 1000,
    ):
        """Initialize the stats collector.

        Args:
            output_path: Path to save stats JSON file.
            max_history: Maximum number of stat entries to keep in file.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history

        # In-memory stats for current period
        self._current_period_stats = self._init_period_stats()
        self._period_start = datetime.now(timezone.utc)

    def _init_period_stats(self) -> dict[str, Any]:
        """Initialize a new stats period.

        Returns:
            Empty stats dictionary for a new period.
        """
        return {
            "signals_generated": 0,
            "signals_bullish": 0,
            "signals_bearish": 0,
            "intents_created": 0,
            "intents_by_action": {"BUY": 0, "SELL": 0},
            "hypothetical_trades": 0,
            "hypothetical_wins": 0,
            "hypothetical_losses": 0,
            "hypothetical_pnl_usd": 0.0,
            "hypothetical_pnl_pct": 0.0,
            "expert_usage": {},
            "category_stats": {},
            "confidence_distribution": {
                "high": 0,    # >0.7
                "medium": 0,  # 0.5-0.7
                "low": 0,     # <0.5
            },
        }

    def record_signal(
        self,
        signal: dict[str, Any],
        expert_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        """Record a generated signal.

        Args:
            signal: Signal dictionary with prediction info.
            expert_id: ID of the expert that generated the signal.
            category: Market category.
        """
        self._current_period_stats["signals_generated"] += 1

        # Track direction
        prediction = signal.get("prediction", 0.5)
        if prediction > 0.5:
            self._current_period_stats["signals_bullish"] += 1
        else:
            self._current_period_stats["signals_bearish"] += 1

        # Track confidence
        confidence = signal.get("confidence", 0.5)
        if confidence > 0.7:
            self._current_period_stats["confidence_distribution"]["high"] += 1
        elif confidence > 0.5:
            self._current_period_stats["confidence_distribution"]["medium"] += 1
        else:
            self._current_period_stats["confidence_distribution"]["low"] += 1

        # Track expert usage
        if expert_id:
            usage = self._current_period_stats["expert_usage"]
            usage[expert_id] = usage.get(expert_id, 0) + 1

        # Track category
        if category:
            cat_stats = self._current_period_stats["category_stats"]
            if category not in cat_stats:
                cat_stats[category] = {
                    "signals": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl_pct": 0.0,
                }
            cat_stats[category]["signals"] += 1

    def record_intent(
        self,
        intent: dict[str, Any],
    ) -> None:
        """Record a created trade intent.

        Args:
            intent: Intent dictionary with trade info.
        """
        self._current_period_stats["intents_created"] += 1

        action = intent.get("action", "BUY")
        by_action = self._current_period_stats["intents_by_action"]
        by_action[action] = by_action.get(action, 0) + 1

    def record_hypothetical_trade(
        self,
        pnl_pct: float,
        category: Optional[str] = None,
    ) -> None:
        """Record a hypothetical trade result (for dry-run evaluation).

        Args:
            pnl_pct: Profit/loss as percentage (e.g., 0.05 for 5% profit).
            category: Market category.
        """
        self._current_period_stats["hypothetical_trades"] += 1
        self._current_period_stats["hypothetical_pnl_pct"] += pnl_pct

        if pnl_pct > 0:
            self._current_period_stats["hypothetical_wins"] += 1
        else:
            self._current_period_stats["hypothetical_losses"] += 1

        # Track by category
        if category:
            cat_stats = self._current_period_stats["category_stats"]
            if category not in cat_stats:
                cat_stats[category] = {
                    "signals": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl_pct": 0.0,
                }
            cat_stats[category]["pnl_pct"] += pnl_pct
            if pnl_pct > 0:
                cat_stats[category]["wins"] += 1
            else:
                cat_stats[category]["losses"] += 1

    def collect_stats(self) -> dict[str, Any]:
        """Collect current period stats and prepare for saving.

        Returns:
            Dictionary with complete stats for this period.
        """
        now = datetime.now(timezone.utc)
        period_hours = (now - self._period_start).total_seconds() / 3600

        stats = {
            "timestamp": now.isoformat(),
            "period_start": self._period_start.isoformat(),
            "period_hours": round(period_hours, 2),
            **self._current_period_stats,
        }

        # Calculate derived metrics
        trades = stats["hypothetical_trades"]
        if trades > 0:
            stats["win_rate"] = stats["hypothetical_wins"] / trades
            stats["avg_pnl_per_trade_pct"] = stats["hypothetical_pnl_pct"] / trades
        else:
            stats["win_rate"] = 0.0
            stats["avg_pnl_per_trade_pct"] = 0.0

        signals = stats["signals_generated"]
        if signals > 0:
            stats["trade_rate"] = trades / signals  # What % of signals became trades
        else:
            stats["trade_rate"] = 0.0

        return stats

    def save_stats(self) -> None:
        """Save current stats to JSON file and reset for next period."""
        stats = self.collect_stats()

        # Load existing history
        history = []
        if self.output_path.exists():
            try:
                with open(self.output_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        history = data
                    elif isinstance(data, dict) and "history" in data:
                        history = data["history"]
            except Exception as e:
                logger.warning(f"Could not load existing stats: {e}")

        # Append new stats
        history.append(stats)

        # Trim to max history
        if len(history) > self.max_history:
            history = history[-self.max_history:]

        # Save
        output = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_periods": len(history),
            "history": history,
        }

        try:
            with open(self.output_path, "w") as f:
                json.dump(output, f, indent=2)

            logger.info(
                f"Saved dry-run stats: {stats['signals_generated']} signals, "
                f"{stats['hypothetical_trades']} trades, "
                f"{stats['hypothetical_pnl_pct']:.2%} P&L"
            )

        except Exception as e:
            logger.error(f"Failed to save dry-run stats: {e}")

        # Reset for next period
        self._current_period_stats = self._init_period_stats()
        self._period_start = datetime.now(timezone.utc)

    def get_summary(self, periods: int = 10) -> dict[str, Any]:
        """Get summary of recent stats periods.

        Args:
            periods: Number of recent periods to summarize.

        Returns:
            Summary dictionary with aggregated metrics.
        """
        if not self.output_path.exists():
            return {"error": "No stats file exists"}

        try:
            with open(self.output_path) as f:
                data = json.load(f)
        except Exception as e:
            return {"error": f"Could not load stats: {e}"}

        history = data.get("history", [])
        if not history:
            return {"error": "No history available"}

        # Get recent periods
        recent = history[-periods:]

        # Aggregate
        total_signals = sum(p.get("signals_generated", 0) for p in recent)
        total_trades = sum(p.get("hypothetical_trades", 0) for p in recent)
        total_wins = sum(p.get("hypothetical_wins", 0) for p in recent)
        total_pnl = sum(p.get("hypothetical_pnl_pct", 0) for p in recent)
        total_hours = sum(p.get("period_hours", 0) for p in recent)

        return {
            "periods_analyzed": len(recent),
            "total_hours": round(total_hours, 1),
            "total_signals": total_signals,
            "total_trades": total_trades,
            "total_wins": total_wins,
            "win_rate": total_wins / max(1, total_trades),
            "total_pnl_pct": total_pnl,
            "avg_pnl_per_trade_pct": total_pnl / max(1, total_trades),
            "trades_per_day": total_trades / max(1, total_hours / 24),
        }


# Singleton instance
_dryrun_stats_collector: Optional[DryRunStatsCollector] = None


def get_dryrun_stats_collector(
    output_path: str = "data/dryrun_stats.json",
) -> DryRunStatsCollector:
    """Get or create the dry-run stats collector singleton.

    Args:
        output_path: Path to save stats file.

    Returns:
        DryRunStatsCollector instance.
    """
    global _dryrun_stats_collector

    if _dryrun_stats_collector is None:
        _dryrun_stats_collector = DryRunStatsCollector(output_path=output_path)

    return _dryrun_stats_collector
