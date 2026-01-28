"""Training History Tracker - Records all model training runs for analysis.

This module maintains a persistent record of every training cycle, including:
- Expert performance metrics (profit, win rate, Sharpe, etc.)
- Model comparison results (new vs old)
- Deployment decisions (deployed, rejected, reason)
- Validation metrics over time

This data enables:
- Historical performance analysis
- Debugging model regressions
- Tracking improvement trends
- Identifying patterns in model performance
"""

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingRunRecord:
    """Record of a single training run for an expert."""

    run_id: str
    expert_id: str
    timestamp: str
    version_id: int

    # Training configuration
    training_examples: int
    n_features: int
    training_mode: str  # 'batch' or 'online'

    # Performance metrics (new model)
    simulated_profit_pct: float
    simulated_num_trades: int
    simulated_win_rate: float
    simulated_profit_factor: float
    simulated_max_drawdown: float
    simulated_sharpe: float

    # Validation metrics
    nn_val_acc: float
    xgb_val_acc: float
    lgb_val_acc: float
    ensemble_val_acc: float

    # Comparison with previous model
    previous_version_id: Optional[int]
    previous_profit_pct: Optional[float]
    improvement_pct: Optional[float]

    # Deployment decision
    deployed: bool
    deployment_reason: str  # 'improvement', 'first_model', 'rejected_worse', 'rejected_threshold'

    # Expert state
    expert_state: str  # 'probation', 'active', 'deprecated', etc.

    # Additional context
    category: Optional[str] = None
    training_time_seconds: float = 0.0
    notes: Optional[str] = None


class TrainingHistoryTracker:
    """Tracks and persists all training runs for analysis."""

    def __init__(self, db_path: str = "data/training_history.db"):
        """Initialize the tracker.

        Args:
            db_path: Path to SQLite database for persistence.
        """
        self.db_path = db_path
        self._ensure_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_db(self) -> None:
        """Create database tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Training runs table - one row per expert per training cycle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                expert_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                version_id INTEGER NOT NULL,

                -- Training config
                training_examples INTEGER NOT NULL,
                n_features INTEGER NOT NULL,
                training_mode TEXT NOT NULL,

                -- Performance metrics
                simulated_profit_pct REAL NOT NULL,
                simulated_num_trades INTEGER NOT NULL,
                simulated_win_rate REAL NOT NULL,
                simulated_profit_factor REAL NOT NULL,
                simulated_max_drawdown REAL NOT NULL,
                simulated_sharpe REAL NOT NULL,

                -- Validation accuracy
                nn_val_acc REAL,
                xgb_val_acc REAL,
                lgb_val_acc REAL,
                ensemble_val_acc REAL,

                -- Comparison
                previous_version_id INTEGER,
                previous_profit_pct REAL,
                improvement_pct REAL,

                -- Deployment
                deployed INTEGER NOT NULL,
                deployment_reason TEXT NOT NULL,

                -- State
                expert_state TEXT NOT NULL,

                -- Extra
                category TEXT,
                training_time_seconds REAL,
                notes TEXT,

                -- Indexes
                UNIQUE(expert_id, version_id)
            )
        """)

        # Aggregate training cycles table - one row per training cycle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_cycles (
                cycle_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                training_mode TEXT NOT NULL,

                -- Aggregate metrics
                total_experts_trained INTEGER NOT NULL,
                experts_deployed INTEGER NOT NULL,
                experts_rejected INTEGER NOT NULL,
                experts_deprecated INTEGER NOT NULL,

                -- Best performer
                best_expert_id TEXT,
                best_profit_pct REAL,

                -- Worst performer
                worst_expert_id TEXT,
                worst_profit_pct REAL,

                -- Averages
                avg_profit_pct REAL,
                avg_win_rate REAL,
                avg_val_accuracy REAL,

                -- Duration
                total_training_time_seconds REAL,

                -- Notes
                notes TEXT
            )
        """)

        # Model comparisons table - detailed comparison records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                comparison_id TEXT PRIMARY KEY,
                expert_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Old model
                old_version_id INTEGER,
                old_profit_pct REAL,
                old_win_rate REAL,
                old_sharpe REAL,

                -- New model
                new_version_id INTEGER NOT NULL,
                new_profit_pct REAL NOT NULL,
                new_win_rate REAL NOT NULL,
                new_sharpe REAL NOT NULL,

                -- Comparison result
                improvement_pct REAL,
                min_improvement_threshold REAL,
                decision TEXT NOT NULL,  -- 'deploy', 'reject', 'first_model'
                reason TEXT
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_expert ON training_runs(expert_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON training_runs(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_deployed ON training_runs(deployed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycles_timestamp ON training_cycles(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_expert ON model_comparisons(expert_id)")

        conn.commit()
        conn.close()

        logger.info(f"Training history database initialized: {self.db_path}")

    def record_training_run(self, record: TrainingRunRecord) -> None:
        """Record a single training run.

        Args:
            record: Training run record to persist.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO training_runs (
                    run_id, expert_id, timestamp, version_id,
                    training_examples, n_features, training_mode,
                    simulated_profit_pct, simulated_num_trades, simulated_win_rate,
                    simulated_profit_factor, simulated_max_drawdown, simulated_sharpe,
                    nn_val_acc, xgb_val_acc, lgb_val_acc, ensemble_val_acc,
                    previous_version_id, previous_profit_pct, improvement_pct,
                    deployed, deployment_reason, expert_state,
                    category, training_time_seconds, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.run_id,
                record.expert_id,
                record.timestamp,
                record.version_id,
                record.training_examples,
                record.n_features,
                record.training_mode,
                record.simulated_profit_pct,
                record.simulated_num_trades,
                record.simulated_win_rate,
                record.simulated_profit_factor,
                record.simulated_max_drawdown,
                record.simulated_sharpe,
                record.nn_val_acc,
                record.xgb_val_acc,
                record.lgb_val_acc,
                record.ensemble_val_acc,
                record.previous_version_id,
                record.previous_profit_pct,
                record.improvement_pct,
                1 if record.deployed else 0,
                record.deployment_reason,
                record.expert_state,
                record.category,
                record.training_time_seconds,
                record.notes,
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to record training run: {e}")
        finally:
            conn.close()

    def record_model_comparison(
        self,
        expert_id: str,
        old_version_id: Optional[int],
        old_metrics: Optional[Dict[str, float]],
        new_version_id: int,
        new_metrics: Dict[str, float],
        min_improvement_threshold: float,
        decision: str,
        reason: str,
    ) -> None:
        """Record a model comparison decision.

        Args:
            expert_id: Expert identifier.
            old_version_id: Previous model version (None if first model).
            old_metrics: Previous model metrics.
            new_version_id: New model version.
            new_metrics: New model metrics.
            min_improvement_threshold: Minimum improvement required.
            decision: 'deploy', 'reject', or 'first_model'.
            reason: Human-readable explanation.
        """
        import uuid

        conn = self._get_connection()
        cursor = conn.cursor()

        old_profit = old_metrics.get("simulated_profit_pct", 0) if old_metrics else None
        new_profit = new_metrics.get("simulated_profit_pct", 0)

        improvement = None
        if old_profit is not None:
            improvement = new_profit - old_profit

        try:
            cursor.execute("""
                INSERT INTO model_comparisons (
                    comparison_id, expert_id, timestamp,
                    old_version_id, old_profit_pct, old_win_rate, old_sharpe,
                    new_version_id, new_profit_pct, new_win_rate, new_sharpe,
                    improvement_pct, min_improvement_threshold, decision, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                expert_id,
                datetime.utcnow().isoformat(),
                old_version_id,
                old_profit,
                old_metrics.get("simulated_win_rate") if old_metrics else None,
                old_metrics.get("simulated_sharpe") if old_metrics else None,
                new_version_id,
                new_profit,
                new_metrics.get("simulated_win_rate", 0),
                new_metrics.get("simulated_sharpe", 0),
                improvement,
                min_improvement_threshold,
                decision,
                reason,
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to record model comparison: {e}")
        finally:
            conn.close()

    def record_training_cycle(
        self,
        cycle_id: str,
        training_mode: str,
        expert_results: List[Dict[str, Any]],
        total_training_time: float,
        notes: Optional[str] = None,
    ) -> None:
        """Record aggregate metrics for a training cycle.

        Args:
            cycle_id: Unique cycle identifier.
            training_mode: 'batch' or 'online'.
            expert_results: List of per-expert results.
            total_training_time: Total training time in seconds.
            notes: Optional notes.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Calculate aggregates
        total = len(expert_results)
        deployed = sum(1 for r in expert_results if r.get("deployed", False))
        rejected = sum(1 for r in expert_results if not r.get("deployed", False) and r.get("state") != "deprecated")
        deprecated = sum(1 for r in expert_results if r.get("state") == "deprecated")

        profits = [r.get("profit_pct", 0) for r in expert_results]
        win_rates = [r.get("win_rate", 0) for r in expert_results]
        accuracies = [r.get("val_acc", 0) for r in expert_results if r.get("val_acc")]

        best = max(expert_results, key=lambda x: x.get("profit_pct", -999)) if expert_results else {}
        worst = min(expert_results, key=lambda x: x.get("profit_pct", 999)) if expert_results else {}

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO training_cycles (
                    cycle_id, timestamp, training_mode,
                    total_experts_trained, experts_deployed, experts_rejected, experts_deprecated,
                    best_expert_id, best_profit_pct,
                    worst_expert_id, worst_profit_pct,
                    avg_profit_pct, avg_win_rate, avg_val_accuracy,
                    total_training_time_seconds, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_id,
                datetime.utcnow().isoformat(),
                training_mode,
                total,
                deployed,
                rejected,
                deprecated,
                best.get("expert_id"),
                best.get("profit_pct"),
                worst.get("expert_id"),
                worst.get("profit_pct"),
                sum(profits) / len(profits) if profits else 0,
                sum(win_rates) / len(win_rates) if win_rates else 0,
                sum(accuracies) / len(accuracies) if accuracies else 0,
                total_training_time,
                notes,
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to record training cycle: {e}")
        finally:
            conn.close()

    def get_expert_history(
        self,
        expert_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get training history for a specific expert.

        Args:
            expert_id: Expert identifier.
            limit: Maximum records to return.

        Returns:
            List of training run records.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM training_runs
            WHERE expert_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (expert_id, limit))

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_recent_cycles(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent training cycles.

        Args:
            limit: Maximum records to return.

        Returns:
            List of training cycle records.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM training_cycles
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_performance_trends(
        self,
        expert_id: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get performance trends over time.

        Args:
            expert_id: Optional expert to filter by.
            days: Number of days to look back.

        Returns:
            Dictionary with trend data.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = datetime.utcnow().isoformat()[:10]  # Just date part

        if expert_id:
            cursor.execute("""
                SELECT
                    DATE(timestamp) as date,
                    AVG(simulated_profit_pct) as avg_profit,
                    AVG(simulated_win_rate) as avg_win_rate,
                    AVG(ensemble_val_acc) as avg_accuracy,
                    COUNT(*) as runs
                FROM training_runs
                WHERE expert_id = ?
                  AND timestamp >= DATE('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (expert_id, f"-{days} days"))
        else:
            cursor.execute("""
                SELECT
                    DATE(timestamp) as date,
                    AVG(simulated_profit_pct) as avg_profit,
                    AVG(simulated_win_rate) as avg_win_rate,
                    AVG(ensemble_val_acc) as avg_accuracy,
                    COUNT(*) as runs
                FROM training_runs
                WHERE timestamp >= DATE('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (f"-{days} days",))

        columns = [desc[0] for desc in cursor.description]
        trends = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()

        return {
            "expert_id": expert_id,
            "days": days,
            "data_points": trends,
        }

    def get_comparison_history(
        self,
        expert_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get model comparison history.

        Args:
            expert_id: Optional expert to filter by.
            limit: Maximum records to return.

        Returns:
            List of comparison records.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if expert_id:
            cursor.execute("""
                SELECT * FROM model_comparisons
                WHERE expert_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (expert_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM model_comparisons
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get overall training history statistics.

        Returns:
            Dictionary with aggregate statistics.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total runs
        cursor.execute("SELECT COUNT(*) FROM training_runs")
        total_runs = cursor.fetchone()[0]

        # Total cycles
        cursor.execute("SELECT COUNT(*) FROM training_cycles")
        total_cycles = cursor.fetchone()[0]

        # Deployment rate
        cursor.execute("SELECT COUNT(*) FROM training_runs WHERE deployed = 1")
        deployed = cursor.fetchone()[0]
        deployment_rate = deployed / total_runs if total_runs > 0 else 0

        # Average profit
        cursor.execute("SELECT AVG(simulated_profit_pct) FROM training_runs")
        avg_profit = cursor.fetchone()[0] or 0

        # Best expert ever
        cursor.execute("""
            SELECT expert_id, MAX(simulated_profit_pct) as best_profit
            FROM training_runs
            GROUP BY expert_id
            ORDER BY best_profit DESC
            LIMIT 1
        """)
        best = cursor.fetchone()

        conn.close()

        return {
            "total_training_runs": total_runs,
            "total_training_cycles": total_cycles,
            "deployment_rate": deployment_rate,
            "avg_profit_pct": avg_profit,
            "best_expert": best[0] if best else None,
            "best_profit_pct": best[1] if best else None,
        }


# Singleton instance
_tracker_instance: Optional[TrainingHistoryTracker] = None


def get_training_history_tracker() -> TrainingHistoryTracker:
    """Get or create the singleton training history tracker.

    Returns:
        TrainingHistoryTracker instance.
    """
    global _tracker_instance
    if _tracker_instance is None:
        from polyb0t.config import get_settings
        settings = get_settings()
        _tracker_instance = TrainingHistoryTracker(
            db_path=settings.ai_training_history_db
        )
    return _tracker_instance
