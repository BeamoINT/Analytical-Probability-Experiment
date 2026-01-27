"""Mixture of Experts (MoE) module for profitability-first trading.

This module implements a MoE architecture where specialized expert models
focus on different market types (categories, risk levels, time horizons),
with automatic creation/deprecation based on profitability.

Features:
- 27+ specialized experts across multiple dimensions
- Deep learning support (PyTorch neural networks + XGBoost + LightGBM)
- Smart versioning with rollback support
- State machine for expert lifecycle management
- Auto-discovery of profitable patterns
- Meta-Controller for learning optimal expert mixtures
- Attention-based neural gating network
- Optuna hyperparameter optimization
"""

from polyb0t.ml.moe.versioning import (
    ExpertState,
    ExpertVersion,
    ExpertVersionManager,
)
from polyb0t.ml.moe.expert import Expert, ExpertMetrics
from polyb0t.ml.moe.gating import GatingNetwork
from polyb0t.ml.moe.expert_pool import ExpertPool
from polyb0t.ml.moe.trainer import MoETrainer, get_moe_trainer
from polyb0t.ml.moe.auto_discovery import AutoDiscovery
from polyb0t.ml.moe.meta_controller import (
    MetaController,
    MixtureConfig,
    MixtureLearner,
    get_meta_controller,
)

# Deep learning components (optional - may not be available)
try:
    from polyb0t.ml.moe.neural_expert import (
        TabularMLP,
        NeuralExpertTrainer,
        DEVICE as NEURAL_DEVICE,
    )
    from polyb0t.ml.moe.deep_ensemble import (
        DeepExpertEnsemble,
        EnsembleMetrics,
        TORCH_AVAILABLE,
        XGBOOST_AVAILABLE,
        LIGHTGBM_AVAILABLE,
    )
    from polyb0t.ml.moe.neural_gating import (
        AttentionGatingNetwork,
        NeuralGatingTrainer,
        HybridGating,
    )
    from polyb0t.ml.moe.augmentation import (
        TabularAugmenter,
        TimeSeriesAugmenter,
        create_augmented_batches,
    )
    from polyb0t.ml.moe.hyperparameter_search import (
        HPOConfig,
        SearchSpace,
        HyperparameterOptimizer,
        CachedHPO,
        quick_hpo,
    )
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    TORCH_AVAILABLE = False
    XGBOOST_AVAILABLE = False
    LIGHTGBM_AVAILABLE = False

__all__ = [
    # Core components
    "ExpertState",
    "ExpertVersion",
    "ExpertVersionManager",
    "Expert",
    "ExpertMetrics",
    "GatingNetwork",
    "ExpertPool",
    "MoETrainer",
    "get_moe_trainer",
    "AutoDiscovery",
    "MetaController",
    "MixtureConfig",
    "MixtureLearner",
    "get_meta_controller",
    # Deep learning flags
    "DEEP_LEARNING_AVAILABLE",
    "TORCH_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
]

# Add deep learning exports if available
if DEEP_LEARNING_AVAILABLE:
    __all__.extend([
        "TabularMLP",
        "NeuralExpertTrainer",
        "NEURAL_DEVICE",
        "DeepExpertEnsemble",
        "EnsembleMetrics",
        "AttentionGatingNetwork",
        "NeuralGatingTrainer",
        "HybridGating",
        "TabularAugmenter",
        "TimeSeriesAugmenter",
        "create_augmented_batches",
        "HPOConfig",
        "SearchSpace",
        "HyperparameterOptimizer",
        "CachedHPO",
        "quick_hpo",
    ])
