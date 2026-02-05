"""Confidence calibration using isotonic regression.

Maps raw model confidence to empirical probability based on historical outcomes.
This addresses the problem where models report 67% confidence but only achieve 45% accuracy.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from polyb0t.ml.validation.metrics import CalibrationMetrics

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Maps raw model confidence to empirical probability.

    Uses isotonic regression (non-parametric) to learn the mapping from
    raw_confidence -> empirical_win_rate. Isotonic regression is ideal because:
    1. It's monotonic (higher raw confidence -> higher calibrated confidence)
    2. It's non-parametric (doesn't assume a specific functional form)
    3. It handles irregular calibration curves well

    Example:
        >>> calibrator = ConfidenceCalibrator()
        >>> calibrator.fit(raw_confidences, actual_outcomes)
        >>> calibrated = calibrator.calibrate_single(0.67)  # Returns ~0.47 if model is overconfident
    """

    def __init__(self, model_dir: str = "data/calibration"):
        """Initialize calibrator.

        Args:
            model_dir: Directory for storing calibration models
        """
        self.model_dir = model_dir
        self._isotonic = None
        self._last_fit_time: Optional[datetime] = None
        self._n_samples: int = 0
        self._load_model()

    def fit(
        self,
        raw_confidences: np.ndarray,
        actual_outcomes: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> CalibrationMetrics:
        """Fit calibration model on validation data.

        Args:
            raw_confidences: Model's raw confidence values (0-1)
            actual_outcomes: Actual binary outcomes (0 or 1)
            sample_weights: Optional weights (e.g., for recency weighting)

        Returns:
            CalibrationMetrics showing quality of fit
        """
        from sklearn.isotonic import IsotonicRegression

        if len(raw_confidences) < 50:
            logger.warning(f"Only {len(raw_confidences)} samples for calibration - need more data")
            return self._empty_metrics()

        # Ensure arrays
        raw_confidences = np.asarray(raw_confidences, dtype=np.float64)
        actual_outcomes = np.asarray(actual_outcomes, dtype=np.float64)

        # Filter out NaN/Inf
        valid_mask = np.isfinite(raw_confidences) & np.isfinite(actual_outcomes)
        raw_confidences = raw_confidences[valid_mask]
        actual_outcomes = actual_outcomes[valid_mask]

        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)[valid_mask]

        # Fit isotonic regression
        self._isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
            increasing=True,
        )
        self._isotonic.fit(raw_confidences, actual_outcomes, sample_weight=sample_weights)

        self._last_fit_time = datetime.utcnow()
        self._n_samples = len(raw_confidences)

        # Compute calibration metrics on the fit data
        calibrated = self._isotonic.predict(raw_confidences)
        metrics = self._compute_calibration_metrics(calibrated, actual_outcomes)

        # Save model
        self._save_model(metrics)

        logger.info(
            f"Calibration fit complete: {self._n_samples} samples, "
            f"ECE={metrics.expected_calibration_error:.3f}"
        )

        return metrics

    def calibrate(self, raw_confidences: np.ndarray) -> np.ndarray:
        """Apply calibration to array of raw confidences.

        Args:
            raw_confidences: Raw confidence values from model (0-1)

        Returns:
            Calibrated probabilities (0-1)
        """
        if self._isotonic is None:
            logger.debug("No calibration model loaded - returning raw confidences")
            return raw_confidences

        raw_confidences = np.asarray(raw_confidences, dtype=np.float64)
        return self._isotonic.predict(raw_confidences)

    def calibrate_single(self, raw_confidence: float) -> float:
        """Calibrate a single confidence value.

        This is the main method used at prediction time.

        Args:
            raw_confidence: Raw confidence (0-1)

        Returns:
            Calibrated confidence (0-1)
        """
        if self._isotonic is None:
            return raw_confidence

        # Clamp to valid range
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        try:
            calibrated = float(self._isotonic.predict([[raw_confidence]])[0])
            return max(0.0, min(1.0, calibrated))
        except Exception as e:
            logger.warning(f"Calibration failed for {raw_confidence}: {e}")
            return raw_confidence

    def is_loaded(self) -> bool:
        """Check if a calibration model is loaded."""
        return self._isotonic is not None

    def get_info(self) -> dict:
        """Get information about the loaded calibration model."""
        return {
            "is_loaded": self.is_loaded(),
            "last_fit_time": self._last_fit_time.isoformat() if self._last_fit_time else None,
            "n_samples": self._n_samples,
            "model_dir": self.model_dir,
        }

    def _compute_calibration_metrics(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Compute Expected Calibration Error and reliability diagram.

        Args:
            predicted: Calibrated probability predictions
            actual: Actual binary outcomes
            n_bins: Number of bins for reliability diagram

        Returns:
            CalibrationMetrics with ECE, MCE, Brier score, and reliability diagram
        """
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        reliability = {}
        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue

            bin_acc = float(actual[mask].mean())
            bin_conf = float(predicted[mask].mean())
            bin_size = int(mask.sum())

            bin_label = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
            reliability[bin_label] = {
                "predicted": bin_conf,
                "actual": bin_acc,
                "count": bin_size,
                "gap": bin_acc - bin_conf,  # Positive = underconfident, negative = overconfident
            }

            # ECE component: weighted by bin size
            ece += (bin_size / len(predicted)) * abs(bin_acc - bin_conf)
            mce = max(mce, abs(bin_acc - bin_conf))

        # Brier score: mean squared error
        brier = float(np.mean((predicted - actual) ** 2))

        return CalibrationMetrics(
            expected_calibration_error=ece,
            max_calibration_error=mce,
            brier_score=brier,
            reliability_diagram=reliability,
            n_samples=len(predicted),
        )

    def _empty_metrics(self) -> CalibrationMetrics:
        """Return empty metrics when calibration can't be performed."""
        return CalibrationMetrics(
            expected_calibration_error=1.0,
            max_calibration_error=1.0,
            brier_score=0.25,
            reliability_diagram={},
            n_samples=0,
        )

    def _save_model(self, metrics: Optional[CalibrationMetrics] = None) -> None:
        """Persist calibration model to disk."""
        os.makedirs(self.model_dir, exist_ok=True)

        # Save isotonic regression model
        isotonic_path = os.path.join(self.model_dir, "isotonic.pkl")
        with open(isotonic_path, "wb") as f:
            pickle.dump(self._isotonic, f)

        # Save metadata
        state = {
            "last_fit_time": self._last_fit_time.isoformat() if self._last_fit_time else None,
            "n_samples": self._n_samples,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if metrics:
            state["metrics"] = {
                "ece": metrics.expected_calibration_error,
                "mce": metrics.max_calibration_error,
                "brier": metrics.brier_score,
            }

        state_path = os.path.join(self.model_dir, "calibration_state.json")
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.debug(f"Saved calibration model to {self.model_dir}")

    def _load_model(self) -> None:
        """Load calibration model from disk if available."""
        isotonic_path = os.path.join(self.model_dir, "isotonic.pkl")
        state_path = os.path.join(self.model_dir, "calibration_state.json")

        if not os.path.exists(isotonic_path):
            logger.debug("No calibration model found - will use raw confidences")
            return

        try:
            with open(isotonic_path, "rb") as f:
                self._isotonic = pickle.load(f)

            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                    if state.get("last_fit_time"):
                        self._last_fit_time = datetime.fromisoformat(state["last_fit_time"])
                    self._n_samples = state.get("n_samples", 0)

            logger.info(
                f"Loaded calibration model: {self._n_samples} samples, "
                f"fit at {self._last_fit_time}"
            )
        except Exception as e:
            logger.warning(f"Failed to load calibration model: {e}")
            self._isotonic = None


# Singleton instance
_calibrator_instance: Optional[ConfidenceCalibrator] = None


def get_calibrator(model_dir: str = "data/calibration") -> ConfidenceCalibrator:
    """Get or create the singleton ConfidenceCalibrator instance.

    Args:
        model_dir: Directory for calibration models

    Returns:
        ConfidenceCalibrator instance
    """
    global _calibrator_instance
    if _calibrator_instance is None:
        _calibrator_instance = ConfidenceCalibrator(model_dir=model_dir)
    return _calibrator_instance
