"""Resolution Predictor - Predicts market outcomes (YES/NO).

Unlike the short-term price predictor, this model:
1. Trains ONLY on resolved markets
2. Predicts final outcome (0 or 1), not price changes
3. Uses market metadata, timing, and category patterns
4. Focuses on long-term accuracy
"""

import json
import logging
import os
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Tuple
import threading

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ResolutionMetrics:
    """Metrics for resolution prediction."""
    accuracy: float  # % of correct YES/NO predictions
    precision_yes: float  # When we predict YES, how often correct?
    precision_no: float  # When we predict NO, how often correct?
    calibration_error: float  # How well do probabilities match reality?
    confident_accuracy: float  # Accuracy on high-confidence predictions
    n_samples: int
    
    def score(self) -> float:
        """Overall score."""
        return self.accuracy * 0.4 + self.confident_accuracy * 0.4 - self.calibration_error * 0.2


@dataclass 
class ResolutionPrediction:
    """A prediction about market resolution."""
    market_id: str
    token_id: str
    prob_yes: float  # Probability YES wins
    prob_no: float  # Probability NO wins
    confidence: float  # How confident (0.5 = unsure, 1.0 = certain)
    predicted_outcome: str  # "YES" or "NO"
    edge_vs_price: float  # Our prob - market price
    
    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "token_id": self.token_id,
            "prob_yes": self.prob_yes,
            "prob_no": self.prob_no,
            "confidence": self.confidence,
            "predicted_outcome": self.predicted_outcome,
            "edge_vs_price": self.edge_vs_price,
        }


class ResolutionPredictor:
    """Predicts market resolution outcomes using resolved market data."""
    
    def __init__(
        self,
        model_dir: str = "data/resolution_models",
        db_path: str = "data/ai_training.db",
        min_resolved_markets: int = 100,
    ):
        self.model_dir = model_dir
        self.db_path = db_path
        self.min_resolved_markets = min_resolved_markets
        
        self._model = None
        self._model_info = None
        self._training_lock = threading.Lock()
        self._is_training = False
        
        os.makedirs(model_dir, exist_ok=True)
        self._load_model()
    
    def _load_model(self):
        """Load existing model if available."""
        model_path = os.path.join(self.model_dir, "resolution_model.pkl")
        state_path = os.path.join(self.model_dir, "resolution_state.json")
        
        if os.path.exists(model_path) and os.path.exists(state_path):
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                with open(state_path, "r") as f:
                    self._model_info = json.load(f)
                logger.info(f"Loaded resolution model v{self._model_info.get('version', 1)}")
            except Exception as e:
                logger.warning(f"Failed to load resolution model: {e}")
    
    def has_model(self) -> bool:
        """Check if a model is available."""
        return self._model is not None
    
    def get_resolved_markets_count(self) -> int:
        """Count resolved markets available for training."""
        if not os.path.exists(self.db_path):
            return 0
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Count markets that have resolution data
            cursor.execute("""
                SELECT COUNT(DISTINCT market_id) 
                FROM training_examples 
                WHERE is_fully_labeled = 1
            """)
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.warning(f"Error counting resolved markets: {e}")
            return 0
    
    def can_train(self) -> bool:
        """Check if we have enough resolved markets to train."""
        return self.get_resolved_markets_count() >= self.min_resolved_markets
    
    def _get_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], list]:
        """Get training data from resolved markets.
        
        Returns:
            Tuple of (features, labels, market_ids)
        """
        if not os.path.exists(self.db_path):
            return None, None, []
        
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            
            # Get resolved market examples
            # We want the FIRST snapshot of each market (initial conditions)
            # and whether it resolved YES or NO
            cursor.execute("""
                SELECT 
                    te.market_id,
                    te.token_id,
                    te.price,
                    te.spread_pct,
                    te.volume_24h,
                    te.liquidity,
                    te.orderbook_imbalance,
                    te.momentum_24h,
                    te.volatility_24h,
                    te.days_to_resolution,
                    te.market_age_days,
                    te.category,
                    te.final_outcome
                FROM training_examples te
                INNER JOIN (
                    SELECT market_id, MIN(created_at) as first_snapshot
                    FROM training_examples
                    WHERE is_fully_labeled = 1 AND final_outcome IS NOT NULL
                    GROUP BY market_id
                ) first ON te.market_id = first.market_id 
                       AND te.created_at = first.first_snapshot
                WHERE te.is_fully_labeled = 1 AND te.final_outcome IS NOT NULL
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None, None, []
            
            X_list = []
            y_list = []
            market_ids = []
            
            for row in rows:
                (market_id, token_id, price, spread_pct, volume_24h, liquidity,
                 orderbook_imbalance, momentum_24h, volatility_24h, 
                 days_to_resolution, market_age_days, category, final_outcome) = row
                
                # Features for resolution prediction
                features = [
                    price or 0.5,  # Initial price (market's implied probability)
                    spread_pct or 0,
                    np.log1p(volume_24h or 0),  # Log volume
                    np.log1p(liquidity or 0),  # Log liquidity
                    orderbook_imbalance or 0,
                    momentum_24h or 0,
                    volatility_24h or 0,
                    days_to_resolution or 30,
                    market_age_days or 0,
                    # Price buckets (extreme prices are informative)
                    1 if (price or 0.5) > 0.8 else 0,
                    1 if (price or 0.5) < 0.2 else 0,
                    1 if 0.4 <= (price or 0.5) <= 0.6 else 0,
                ]
                
                X_list.append(features)
                # Label: 1 if YES won, 0 if NO won
                y_list.append(1 if final_outcome == 1 else 0)
                market_ids.append(market_id)
            
            logger.info(f"Loaded {len(X_list)} resolved markets for training")
            return np.array(X_list), np.array(y_list), market_ids
            
        except Exception as e:
            logger.error(f"Error loading resolution training data: {e}")
            return None, None, []
    
    def train(self) -> Optional[ResolutionMetrics]:
        """Train the resolution predictor on resolved markets."""
        if not self.can_train():
            logger.info(f"Not enough resolved markets to train ({self.get_resolved_markets_count()} < {self.min_resolved_markets})")
            return None
        
        with self._training_lock:
            if self._is_training:
                return None
            self._is_training = True
        
        try:
            logger.info("Training resolution predictor...")
            
            X, y, market_ids = self._get_training_data()
            
            if X is None or len(X) < self.min_resolved_markets:
                logger.warning("Not enough training data")
                return None
            
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_predict, StratifiedKFold
            from sklearn.calibration import CalibratedClassifierCV
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use calibrated classifier for well-calibrated probabilities
            base_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42,
            )
            
            # Calibrate probabilities
            clf = CalibratedClassifierCV(base_clf, cv=5, method='isotonic')
            
            # Get cross-validated predictions for metrics
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_proba = cross_val_predict(clf, X_scaled, y, cv=cv, method='predict_proba')
            y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y)
            
            # Precision for YES predictions
            yes_mask = y_pred == 1
            precision_yes = np.mean(y[yes_mask] == 1) if np.sum(yes_mask) > 0 else 0
            
            # Precision for NO predictions
            no_mask = y_pred == 0
            precision_no = np.mean(y[no_mask] == 0) if np.sum(no_mask) > 0 else 0
            
            # Calibration error (Brier score component)
            calibration_error = np.mean((y_pred_proba[:, 1] - y) ** 2)
            
            # Accuracy on confident predictions (prob > 0.7 or < 0.3)
            confident_mask = (y_pred_proba[:, 1] > 0.7) | (y_pred_proba[:, 1] < 0.3)
            confident_accuracy = np.mean(y_pred[confident_mask] == y[confident_mask]) if np.sum(confident_mask) > 0 else 0
            
            metrics = ResolutionMetrics(
                accuracy=accuracy,
                precision_yes=precision_yes,
                precision_no=precision_no,
                calibration_error=calibration_error,
                confident_accuracy=confident_accuracy,
                n_samples=len(y),
            )
            
            logger.info(
                f"Resolution model trained: accuracy={accuracy:.1%}, "
                f"confident_acc={confident_accuracy:.1%}, "
                f"calibration_error={calibration_error:.3f}"
            )
            
            # Train final model on all data
            clf.fit(X_scaled, y)
            
            # Save model
            self._model = {
                "classifier": clf,
                "scaler": scaler,
            }
            
            version = (self._model_info.get("version", 0) + 1) if self._model_info else 1
            self._model_info = {
                "version": version,
                "created_at": datetime.utcnow().isoformat(),
                "n_markets": len(y),
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "precision_yes": metrics.precision_yes,
                    "precision_no": metrics.precision_no,
                    "calibration_error": metrics.calibration_error,
                    "confident_accuracy": metrics.confident_accuracy,
                },
            }
            
            # Persist
            model_path = os.path.join(self.model_dir, "resolution_model.pkl")
            state_path = os.path.join(self.model_dir, "resolution_state.json")
            
            with open(model_path, "wb") as f:
                pickle.dump(self._model, f)
            with open(state_path, "w") as f:
                json.dump(self._model_info, f, indent=2)
            
            logger.info(f"Saved resolution model v{version}")
            return metrics
            
        except Exception as e:
            logger.error(f"Resolution training failed: {e}", exc_info=True)
            return None
        finally:
            self._is_training = False
    
    def predict(
        self,
        token_id: str,
        market_id: str,
        current_price: float,
        features: dict,
    ) -> Optional[ResolutionPrediction]:
        """Predict market resolution outcome.
        
        Args:
            token_id: Token ID
            market_id: Market ID
            current_price: Current YES price
            features: Feature dictionary
            
        Returns:
            ResolutionPrediction or None
        """
        if self._model is None:
            return None
        
        try:
            # Build feature vector (same order as training)
            X = np.array([[
                current_price,
                features.get("spread_pct", 0),
                np.log1p(features.get("volume_24h", 0)),
                np.log1p(features.get("liquidity", 0)),
                features.get("orderbook_imbalance", 0),
                features.get("momentum_24h", 0),
                features.get("volatility_24h", 0),
                features.get("days_to_resolution", 30),
                features.get("market_age_days", 0),
                1 if current_price > 0.8 else 0,
                1 if current_price < 0.2 else 0,
                1 if 0.4 <= current_price <= 0.6 else 0,
            ]])
            
            # Scale
            X_scaled = self._model["scaler"].transform(X)
            
            # Predict
            proba = self._model["classifier"].predict_proba(X_scaled)[0]
            prob_no, prob_yes = proba[0], proba[1]
            
            # Confidence is how far from 0.5
            confidence = abs(prob_yes - 0.5) * 2
            
            # Predicted outcome
            predicted_outcome = "YES" if prob_yes > 0.5 else "NO"
            
            # Edge vs current price
            edge_vs_price = prob_yes - current_price
            
            return ResolutionPrediction(
                market_id=market_id,
                token_id=token_id,
                prob_yes=prob_yes,
                prob_no=prob_no,
                confidence=confidence,
                predicted_outcome=predicted_outcome,
                edge_vs_price=edge_vs_price,
            )
            
        except Exception as e:
            logger.warning(f"Resolution prediction failed: {e}")
            return None
    
    def get_model_info(self) -> Optional[dict]:
        """Get info about current model."""
        return self._model_info


# Singleton
_predictor_instance: Optional[ResolutionPredictor] = None


def get_resolution_predictor() -> ResolutionPredictor:
    """Get or create the resolution predictor singleton."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ResolutionPredictor()
    return _predictor_instance
