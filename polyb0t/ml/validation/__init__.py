"""Validation and calibration module for ML predictions.

This module provides:
- ConfidenceCalibrator: Maps raw model confidence to empirical probability
- ResolvedMarketValidator: Validates model on resolved markets only
- Metrics dataclasses for tracking calibration quality
"""

from polyb0t.ml.validation.metrics import CalibrationMetrics, ValidationResult
from polyb0t.ml.validation.calibration import ConfidenceCalibrator
from polyb0t.ml.validation.resolved_validator import ResolvedMarketValidator

__all__ = [
    "CalibrationMetrics",
    "ValidationResult",
    "ConfidenceCalibrator",
    "ResolvedMarketValidator",
]
