"""Metrics dataclasses for validation and calibration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class CalibrationMetrics:
    """Metrics for calibration quality.

    Attributes:
        expected_calibration_error: ECE - weighted average of |predicted - actual| per bin.
            Lower is better. 0.0 = perfect calibration.
        max_calibration_error: MCE - worst bin's |predicted - actual|.
            Shows the maximum miscalibration.
        brier_score: Mean squared error of probability predictions.
            Lower is better. 0.0 = perfect, 0.25 = random guessing.
        reliability_diagram: Dict mapping bin ranges to their statistics.
            Each bin has: predicted (avg confidence), actual (win rate), count.
        n_samples: Number of samples used for calibration.
    """
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    reliability_diagram: Dict[str, Dict]
    n_samples: int

    def is_well_calibrated(self, ece_threshold: float = 0.10) -> bool:
        """Check if calibration is acceptable.

        Args:
            ece_threshold: Maximum acceptable ECE (default 10%)

        Returns:
            True if ECE is below threshold
        """
        return self.expected_calibration_error < ece_threshold

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "GOOD" if self.is_well_calibrated() else "POOR"
        return (
            f"Calibration [{status}]: ECE={self.expected_calibration_error:.3f}, "
            f"MCE={self.max_calibration_error:.3f}, Brier={self.brier_score:.3f}, "
            f"n={self.n_samples}"
        )


@dataclass
class ValidationResult:
    """Result of out-of-sample validation on resolved markets.

    This captures both raw model performance and calibrated performance,
    along with detailed diagnostics.
    """
    timestamp: datetime
    n_train: int  # Samples used for calibration training
    n_test: int   # Samples used for validation

    # Raw model metrics (before calibration)
    raw_accuracy: float
    raw_precision: float
    raw_recall: float
    raw_f1: float
    raw_brier: float

    # Calibrated metrics (after calibration)
    calibrated_accuracy: float
    calibrated_brier: float

    # Calibration quality
    calibration_metrics: CalibrationMetrics

    # P&L simulation on test set
    simulated_pnl: float  # Total P&L as fraction (e.g., 0.05 = 5% gain)
    simulated_trades: int
    simulated_win_rate: float
    simulated_avg_win: float = 0.0
    simulated_avg_loss: float = 0.0

    # Per-confidence bucket breakdown
    # e.g., {"0.60-0.70": {"count": 100, "win_rate": 0.62, "expected": 0.65}}
    confidence_buckets: Dict[str, Dict] = field(default_factory=dict)

    # Optional: per-category breakdown
    category_results: Optional[Dict[str, Dict]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "raw_accuracy": self.raw_accuracy,
            "raw_precision": self.raw_precision,
            "raw_recall": self.raw_recall,
            "raw_f1": self.raw_f1,
            "raw_brier": self.raw_brier,
            "calibrated_accuracy": self.calibrated_accuracy,
            "calibrated_brier": self.calibrated_brier,
            "calibration": {
                "ece": self.calibration_metrics.expected_calibration_error,
                "mce": self.calibration_metrics.max_calibration_error,
                "brier": self.calibration_metrics.brier_score,
                "n_samples": self.calibration_metrics.n_samples,
            },
            "simulation": {
                "pnl": self.simulated_pnl,
                "trades": self.simulated_trades,
                "win_rate": self.simulated_win_rate,
                "avg_win": self.simulated_avg_win,
                "avg_loss": self.simulated_avg_loss,
            },
            "confidence_buckets": self.confidence_buckets,
            "category_results": self.category_results,
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        pnl_str = f"+{self.simulated_pnl:.1%}" if self.simulated_pnl >= 0 else f"{self.simulated_pnl:.1%}"
        return (
            f"Validation: {self.n_test} test samples\n"
            f"  Raw accuracy: {self.raw_accuracy:.1%}\n"
            f"  Calibrated accuracy: {self.calibrated_accuracy:.1%}\n"
            f"  ECE: {self.calibration_metrics.expected_calibration_error:.3f}\n"
            f"  Simulated P&L: {pnl_str} ({self.simulated_trades} trades, {self.simulated_win_rate:.1%} win rate)"
        )

    def is_model_profitable(self) -> bool:
        """Check if the model shows positive expected value."""
        return self.simulated_pnl > 0 and self.simulated_win_rate > 0.5
