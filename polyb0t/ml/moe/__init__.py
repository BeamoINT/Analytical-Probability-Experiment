"""Mixture of Experts (MoE) module for profitability-first trading.

This module implements a MoE architecture where specialized expert models
focus on different market types (categories, risk levels, time horizons),
with automatic creation/deprecation based on profitability.
"""

from polyb0t.ml.moe.expert import Expert, ExpertMetrics
from polyb0t.ml.moe.gating import GatingNetwork
from polyb0t.ml.moe.expert_pool import ExpertPool
from polyb0t.ml.moe.trainer import MoETrainer
from polyb0t.ml.moe.auto_discovery import AutoDiscovery

__all__ = [
    "Expert",
    "ExpertMetrics",
    "GatingNetwork",
    "ExpertPool",
    "MoETrainer",
    "AutoDiscovery",
]
