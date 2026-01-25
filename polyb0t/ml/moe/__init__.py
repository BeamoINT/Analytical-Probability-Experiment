"""Mixture of Experts (MoE) module for profitability-first trading.

This module implements a MoE architecture where specialized expert models
focus on different market types (categories, risk levels, time horizons),
with automatic creation/deprecation based on profitability.

Features:
- 24 specialized experts across multiple dimensions
- Smart versioning with rollback support
- State machine for expert lifecycle management
- Auto-discovery of profitable patterns
- Meta-Controller for learning optimal expert mixtures
"""

from polyb0t.ml.moe.versioning import (
    ExpertState,
    ExpertVersion,
    ExpertVersionManager,
)
from polyb0t.ml.moe.expert import Expert, ExpertMetrics
from polyb0t.ml.moe.gating import GatingNetwork
from polyb0t.ml.moe.expert_pool import ExpertPool
from polyb0t.ml.moe.trainer import MoETrainer
from polyb0t.ml.moe.auto_discovery import AutoDiscovery
from polyb0t.ml.moe.meta_controller import (
    MetaController,
    MixtureConfig,
    MixtureLearner,
    get_meta_controller,
)

__all__ = [
    "ExpertState",
    "ExpertVersion",
    "ExpertVersionManager",
    "Expert",
    "ExpertMetrics",
    "GatingNetwork",
    "ExpertPool",
    "MoETrainer",
    "AutoDiscovery",
    "MetaController",
    "MixtureConfig",
    "MixtureLearner",
    "get_meta_controller",
]
