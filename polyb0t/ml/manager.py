"""Model manager for thread-safe hot-swappable inference.

Manages ML model loading and inference for live trading with
zero-downtime model updates.
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from polyb0t.ml.model import PricePredictor, EnsemblePredictor

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe manager for ML model inference.
    
    Handles:
    - Hot-swapping of models (no downtime)
    - Fallback when no model available
    - Thread-safe predictions for concurrent access
    """
    
    def __init__(
        self,
        model_dir: Path,
        use_ensemble: bool = False,
        fallback_enabled: bool = True,
    ):
        """Initialize model manager.
        
        Args:
            model_dir: Directory containing models.
            use_ensemble: Use ensemble of models instead of single model.
            fallback_enabled: Use fallback prediction if model fails.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_ensemble = use_ensemble
        self.fallback_enabled = fallback_enabled
        
        # Current model pointer file
        self.current_model_path = self.model_dir / "current_model.txt"
        
        # Model state
        self.model: Optional[PricePredictor] = None
        self.ensemble: Optional[EnsemblePredictor] = None
        self.model_mtime: float = 0.0
        self.model_name: str = "none"
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load initial model
        self._load_current_model()
    
    def _load_current_model(self) -> bool:
        """Load model specified in current_model.txt.
        
        Returns:
            True if model loaded successfully.
        """
        try:
            # Check if pointer file exists
            if not self.current_model_path.exists():
                logger.debug("No current model pointer file found")
                return False
            
            # Read model path
            with open(self.current_model_path) as f:
                model_path_str = f.read().strip()
            
            if not model_path_str:
                return False
            
            model_path = Path(model_path_str)
            
            # Check if model file exists
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Check if model changed
            mtime = model_path.stat().st_mtime
            if mtime == self.model_mtime and self.model is not None:
                return True  # Already loaded
            
            # Load new model
            if self.use_ensemble:
                new_ensemble = EnsemblePredictor(self.model_dir)
                new_ensemble.load_models()
                
                with self.lock:
                    self.ensemble = new_ensemble
                    self.model_mtime = mtime
                    self.model_name = "ensemble"
                
                logger.info(f"✅ Loaded ensemble with {len(self.ensemble.models)} models")
                
            else:
                new_model = PricePredictor(str(model_path))
                
                with self.lock:
                    self.model = new_model
                    self.model_mtime = mtime
                    self.model_name = model_path.stem
                
                logger.info(f"✅ Loaded model: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(
        self,
        features: pd.DataFrame,
        fallback_value: float = 0.0,
    ) -> float:
        """Make thread-safe prediction.
        
        Args:
            features: Feature dataframe (single row).
            fallback_value: Value to return if prediction fails.
            
        Returns:
            Predicted price change.
        """
        # Check for model updates (cheap file stat)
        self._load_current_model()
        
        with self.lock:
            try:
                if self.use_ensemble and self.ensemble is not None:
                    return self.ensemble.predict(features)
                elif self.model is not None:
                    return self.model.predict(features)
                elif self.fallback_enabled:
                    return self._fallback_prediction(features)
                else:
                    return fallback_value
                    
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                if self.fallback_enabled:
                    return self._fallback_prediction(features)
                else:
                    return fallback_value
    
    def _fallback_prediction(self, features: pd.DataFrame) -> float:
        """Simple fallback prediction when ML model unavailable.
        
        Uses basic momentum + mean reversion heuristic.
        
        Args:
            features: Feature dataframe.
            
        Returns:
            Simple prediction.
        """
        try:
            # Use momentum signals if available
            return_1h = features.get('return_1h', pd.Series([0.0])).iloc[0]
            return_24h = features.get('return_24h', pd.Series([0.0])).iloc[0]
            
            # Blend short and long term momentum
            momentum = 0.6 * return_1h + 0.4 * return_24h
            
            # Mean reversion component
            price_vs_sma5 = features.get('price_vs_sma5', pd.Series([0.0])).iloc[0]
            mean_reversion = -0.3 * price_vs_sma5
            
            # Combine
            prediction = 0.7 * momentum + 0.3 * mean_reversion
            
            # Dampen to avoid overconfidence
            prediction *= 0.5
            
            return float(prediction)
            
        except Exception:
            return 0.0
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded.
        
        Returns:
            True if model available.
        """
        with self.lock:
            if self.use_ensemble:
                return self.ensemble is not None and len(self.ensemble.models) > 0
            else:
                return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get information about current model.
        
        Returns:
            Dictionary with model information.
        """
        with self.lock:
            if self.use_ensemble and self.ensemble is not None:
                return {
                    'type': 'ensemble',
                    'num_models': len(self.ensemble.models),
                    'model_names': list(self.ensemble.models.keys()),
                    'loaded': True,
                }
            elif self.model is not None:
                return {
                    'type': 'single',
                    'name': self.model_name,
                    'num_features': len(self.model.feature_names),
                    'metrics': self.model.training_metrics,
                    'loaded': True,
                }
            else:
                return {
                    'type': 'none',
                    'loaded': False,
                    'fallback_enabled': self.fallback_enabled,
                }
    
    def get_feature_importance(self, top_n: int = 20) -> dict:
        """Get feature importance from current model.
        
        Args:
            top_n: Return top N features.
            
        Returns:
            Dictionary of feature -> importance.
        """
        with self.lock:
            if self.model is not None:
                return self.model.get_feature_importance(top_n=top_n)
            elif self.ensemble is not None and self.ensemble.models:
                # Return importance from first model in ensemble
                first_model = list(self.ensemble.models.values())[0]
                return first_model.get_feature_importance(top_n=top_n)
            else:
                return {}
    
    def force_reload(self) -> bool:
        """Force reload of current model.
        
        Returns:
            True if reload successful.
        """
        with self.lock:
            self.model_mtime = 0.0  # Reset to force reload
            return self._load_current_model()

