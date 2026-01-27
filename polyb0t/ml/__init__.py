"""Machine learning module for predictive trading intelligence."""

from polyb0t.ml.features import AdvancedFeatureEngine
from polyb0t.ml.model import PricePredictor, EnsemblePredictor
from polyb0t.ml.data import DataCollector, TrainingExample
from polyb0t.ml.manager import ModelManager
from polyb0t.ml.updater import ModelUpdater
from polyb0t.ml.continuous_collector import (
    ContinuousDataCollector,
    MarketSnapshot,
    get_data_collector,
)
from polyb0t.ml.ai_trainer import AITrainer, get_ai_trainer
from polyb0t.ml.category_tracker import MarketCategoryTracker, get_category_tracker

__all__ = [
    # Core components
    "AdvancedFeatureEngine",
    "PricePredictor",
    "EnsemblePredictor",
    "DataCollector",
    "TrainingExample",
    "ModelManager",
    "ModelUpdater",
    # Continuous data collection
    "ContinuousDataCollector",
    "MarketSnapshot",
    "get_data_collector",
    # AI training
    "AITrainer",
    "get_ai_trainer",
    # Category tracking
    "MarketCategoryTracker",
    "get_category_tracker",
]

