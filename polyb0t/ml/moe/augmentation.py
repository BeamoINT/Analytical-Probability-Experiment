"""Data augmentation utilities for tabular data.

Implements various augmentation techniques to prevent overfitting:
- Gaussian noise injection
- Feature dropout
- Mixup interpolation
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TabularAugmenter:
    """Data augmentation for tabular market data."""

    def __init__(
        self,
        noise_std: float = 0.01,
        feature_dropout_rate: float = 0.1,
        mixup_alpha: float = 0.2,
        enable_noise: bool = True,
        enable_feature_dropout: bool = True,
        enable_mixup: bool = True,
    ):
        """Initialize augmenter.

        Args:
            noise_std: Standard deviation of Gaussian noise to add
            feature_dropout_rate: Probability of zeroing each feature
            mixup_alpha: Alpha parameter for Beta distribution in mixup
            enable_noise: Whether to apply Gaussian noise
            enable_feature_dropout: Whether to apply feature dropout
            enable_mixup: Whether to apply mixup
        """
        self.noise_std = noise_std
        self.feature_dropout_rate = feature_dropout_rate
        self.mixup_alpha = mixup_alpha
        self.enable_noise = enable_noise
        self.enable_feature_dropout = enable_feature_dropout
        self.enable_mixup = enable_mixup

    def add_gaussian_noise(
        self,
        X: np.ndarray,
        std: Optional[float] = None,
    ) -> np.ndarray:
        """Add Gaussian noise to features.

        Args:
            X: Feature matrix (n_samples, n_features)
            std: Noise standard deviation (uses self.noise_std if None)

        Returns:
            Augmented feature matrix
        """
        if std is None:
            std = self.noise_std

        noise = np.random.normal(0, std, size=X.shape)
        return X + noise

    def apply_feature_dropout(
        self,
        X: np.ndarray,
        rate: Optional[float] = None,
    ) -> np.ndarray:
        """Randomly zero out features.

        Args:
            X: Feature matrix (n_samples, n_features)
            rate: Dropout rate (uses self.feature_dropout_rate if None)

        Returns:
            Augmented feature matrix with some features zeroed
        """
        if rate is None:
            rate = self.feature_dropout_rate

        mask = np.random.binomial(1, 1 - rate, size=X.shape)
        # Scale up remaining features to maintain expected value
        scale = 1.0 / (1.0 - rate)
        return X * mask * scale

    def mixup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply mixup augmentation.

        Mixup creates virtual training examples by interpolating between
        pairs of samples and their labels.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - should be one-hot or soft labels
            alpha: Beta distribution parameter

        Returns:
            Tuple of (augmented features, interpolated labels)
        """
        if alpha is None:
            alpha = self.mixup_alpha

        n_samples = len(X)

        # Sample mixing coefficients from Beta distribution
        lam = np.random.beta(alpha, alpha, size=n_samples)
        lam = np.maximum(lam, 1 - lam)  # Ensure lam >= 0.5

        # Random permutation for pairs
        indices = np.random.permutation(n_samples)

        # Mix features
        X_mixed = lam.reshape(-1, 1) * X + (1 - lam.reshape(-1, 1)) * X[indices]

        # Mix labels (for soft labels)
        if len(y.shape) == 1:
            # Convert to soft labels for mixing
            y_mixed = lam * y + (1 - lam) * y[indices]
        else:
            y_mixed = lam.reshape(-1, 1) * y + (1 - lam.reshape(-1, 1)) * y[indices]

        return X_mixed, y_mixed

    def augment(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        apply_noise: Optional[bool] = None,
        apply_dropout: Optional[bool] = None,
        apply_mixup: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply all enabled augmentations.

        Args:
            X: Feature matrix
            y: Labels (required if mixup is enabled)
            apply_noise: Override noise setting
            apply_dropout: Override dropout setting
            apply_mixup: Override mixup setting

        Returns:
            Tuple of (augmented features, augmented labels or None)
        """
        X_aug = X.copy()
        y_aug = y.copy() if y is not None else None

        # Gaussian noise
        if (apply_noise if apply_noise is not None else self.enable_noise):
            X_aug = self.add_gaussian_noise(X_aug)

        # Feature dropout
        if (apply_dropout if apply_dropout is not None else self.enable_feature_dropout):
            X_aug = self.apply_feature_dropout(X_aug)

        # Mixup (requires labels)
        if (apply_mixup if apply_mixup is not None else self.enable_mixup) and y_aug is not None:
            X_aug, y_aug = self.mixup(X_aug, y_aug)

        return X_aug, y_aug


class TimeSeriesAugmenter:
    """Augmentation for time-ordered market data.

    Preserves temporal structure while adding noise.
    """

    def __init__(
        self,
        noise_std: float = 0.005,
        jitter_std: float = 0.01,
    ):
        self.noise_std = noise_std
        self.jitter_std = jitter_std

    def add_temporal_noise(
        self,
        X: np.ndarray,
        temporal_cols: Optional[list] = None,
    ) -> np.ndarray:
        """Add noise while respecting temporal features.

        Args:
            X: Feature matrix
            temporal_cols: Indices of temporal features (e.g., momentum, volatility)

        Returns:
            Augmented feature matrix
        """
        X_aug = X.copy()

        if temporal_cols is None:
            # Assume last few columns might be temporal
            temporal_cols = []

        # Add standard noise to non-temporal features
        non_temporal_mask = np.ones(X.shape[1], dtype=bool)
        non_temporal_mask[temporal_cols] = False

        X_aug[:, non_temporal_mask] += np.random.normal(
            0, self.noise_std, size=(X.shape[0], non_temporal_mask.sum())
        )

        # Add smaller jitter to temporal features
        X_aug[:, temporal_cols] += np.random.normal(
            0, self.jitter_std, size=(X.shape[0], len(temporal_cols))
        )

        return X_aug


def create_augmented_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    augmenter: Optional[TabularAugmenter] = None,
    n_augmented_batches: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create augmented training batches.

    Args:
        X: Feature matrix
        y: Labels
        batch_size: Size of each batch
        augmenter: Augmenter to use (creates default if None)
        n_augmented_batches: Number of augmented copies to create

    Returns:
        Tuple of (all features including augmented, all labels)
    """
    if augmenter is None:
        augmenter = TabularAugmenter()

    all_X = [X]
    all_y = [y]

    for _ in range(n_augmented_batches):
        X_aug, y_aug = augmenter.augment(X, y, apply_mixup=False)
        all_X.append(X_aug)
        all_y.append(y_aug if y_aug is not None else y)

    return np.vstack(all_X), np.concatenate(all_y)
