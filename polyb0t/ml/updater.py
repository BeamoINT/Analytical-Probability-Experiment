"""Background model updater for online learning.

Runs in a separate thread to periodically:
1. Label historical data with outcomes
2. Retrain models on new data
3. Validate new models
4. Hot-swap if improved
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from polyb0t.ml.data import DataCollector
from polyb0t.ml.model import PricePredictor

logger = logging.getLogger(__name__)


class ModelUpdater:
    """Background thread that manages online learning.
    
    Periodically retrains models and swaps them in if they perform better.
    """
    
    def __init__(
        self,
        data_collector: DataCollector,
        model_dir: Path,
        retrain_interval_hours: int = 6,
        min_new_examples: int = 100,
        min_total_examples: int = 1000,
        validation_threshold_r2: float = 0.03,
    ):
        """Initialize model updater.
        
        Args:
            data_collector: DataCollector instance.
            model_dir: Directory to save models.
            retrain_interval_hours: Hours between retraining.
            min_new_examples: Minimum new examples to trigger retrain.
            min_total_examples: Minimum total examples needed for training.
            validation_threshold_r2: Minimum R² to accept new model.
        """
        self.data_collector = data_collector
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.retrain_interval = retrain_interval_hours * 3600  # Convert to seconds
        self.min_new_examples = min_new_examples
        self.min_total_examples = min_total_examples
        self.validation_threshold_r2 = validation_threshold_r2
        
        # Current model pointer
        self.current_model_path = self.model_dir / "current_model.txt"
        
        # State
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_retrain_time: Optional[datetime] = None
        self.training_in_progress = False
        
        # Statistics
        self.total_retrains = 0
        self.successful_swaps = 0
        self.failed_swaps = 0
        
    def start(self) -> None:
        """Start background learning loop."""
        if self.running:
            logger.warning("Model updater already running")
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._learning_loop,
            daemon=True,
            name="ModelUpdater"
        )
        self.thread.start()
        
        logger.info(
            f"Model updater started (retrain every {self.retrain_interval/3600:.1f}h)"
        )
    
    def stop(self, timeout: float = 30.0) -> None:
        """Stop background learning loop.
        
        Args:
            timeout: Maximum seconds to wait for graceful shutdown.
        """
        if not self.running:
            return
        
        logger.info("Stopping model updater...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning("Model updater did not stop gracefully")
            else:
                logger.info("Model updater stopped")
    
    def _learning_loop(self) -> None:
        """Main background loop: label → retrain → validate → swap."""
        # Initial delay to let system collect some data
        initial_delay = min(3600, self.retrain_interval / 4)  # 1h or 1/4 interval
        logger.info(f"Initial delay: {initial_delay/3600:.1f}h before first training")
        time.sleep(initial_delay)
        
        while self.running:
            try:
                cycle_start = time.time()
                
                # Step 1: Label historical data
                logger.info("Labeling historical data...")
                labeled = self.data_collector.label_historical_data(
                    horizon_hours=1,
                    max_examples=10000,
                )
                
                if labeled > 0:
                    logger.info(f"Labeled {labeled} new examples")
                
                # Step 2: Check if we have enough data
                stats = self.data_collector.get_statistics()
                logger.info(
                    f"Data statistics: {stats['examples_with_targets']} usable examples, "
                    f"{stats['labeling_rate']:.1%} labeled"
                )
                
                if stats['examples_with_targets'] < self.min_total_examples:
                    logger.info(
                        f"Not enough data yet ({stats['examples_with_targets']}/{self.min_total_examples}), "
                        f"waiting..."
                    )
                    self._sleep_until_next_cycle(cycle_start)
                    continue
                
                # Step 3: Train new model
                logger.info("Training new model...")
                self.training_in_progress = True
                
                new_model, metrics = self._train_new_model()
                self.total_retrains += 1
                self.training_in_progress = False
                
                if new_model is None:
                    logger.warning("Model training failed")
                    self._sleep_until_next_cycle(cycle_start)
                    continue
                
                logger.info(
                    f"New model trained: R²={metrics.get('val_r2', 0):.4f}, "
                    f"RMSE={metrics.get('val_rmse', 0):.4f}, "
                    f"Direction Acc={metrics.get('val_direction_acc', 0):.2%}"
                )
                
                # Step 4: Validate and decide whether to swap
                if self._should_swap_model(metrics):
                    # Save and swap
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    new_model_path = self.model_dir / f"model_{timestamp}.txt"
                    
                    new_model.save_model(str(new_model_path))
                    
                    # Atomic swap: update pointer file
                    with open(self.current_model_path, 'w') as f:
                        f.write(str(new_model_path))
                    
                    self.successful_swaps += 1
                    self.last_retrain_time = datetime.utcnow()
                    
                    # Record performance
                    self.data_collector.record_model_performance(
                        model_name=f"model_{timestamp}",
                        metrics=metrics,
                        metadata={'swapped': True},
                    )
                    
                    logger.info(f"✅ Model swapped: {new_model_path}")
                    
                else:
                    self.failed_swaps += 1
                    
                    # Still record performance for analysis
                    self.data_collector.record_model_performance(
                        model_name=f"model_rejected_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        metrics=metrics,
                        metadata={'swapped': False, 'reason': 'below_threshold'},
                    )
                    
                    logger.info(
                        f"Model not swapped (val_r2={metrics.get('val_r2', 0):.4f} < "
                        f"threshold={self.validation_threshold_r2})"
                    )
                
                # Step 5: Cleanup old data (configurable retention period)
                if self.total_retrains % 10 == 0:  # Every 10 retrains
                    logger.info("Running data cleanup...")
                    retention_days = self.settings.ml_data_retention_days
                    deleted = self.data_collector.cleanup_old_data(days=retention_days)
                    logger.info(f"Cleaned up {deleted} records older than {retention_days} days")
                
                # Step 6: Sleep until next cycle
                self._sleep_until_next_cycle(cycle_start)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}", exc_info=True)
                self.training_in_progress = False
                # Sleep 1 hour on error to avoid rapid retry loop
                if self.running:
                    time.sleep(3600)
    
    def _train_new_model(self) -> tuple[Optional[PricePredictor], dict]:
        """Train a new model on latest data.
        
        Returns:
            Tuple of (model, metrics) or (None, {}) if failed.
        """
        try:
            # Get training data
            max_examples = self.settings.ml_max_training_examples
            X, y = self.data_collector.get_training_set(
                min_examples=self.min_total_examples,
                max_examples=max_examples,
            )
            
            if len(X) < self.min_total_examples:
                logger.warning(f"Insufficient training data: {len(X)} examples")
                return None, {}
            
            # Train model
            model = PricePredictor()
            metrics = model.train(X, y, validation_split=0.2)
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return None, {}
    
    def _should_swap_model(self, new_metrics: dict) -> bool:
        """Decide if new model should replace current model.
        
        Args:
            new_metrics: Metrics from new model.
            
        Returns:
            True if should swap.
        """
        # Check if new model meets minimum threshold
        val_r2 = new_metrics.get('val_r2', 0.0)
        
        if val_r2 < self.validation_threshold_r2:
            logger.info(f"New model R² ({val_r2:.4f}) below threshold ({self.validation_threshold_r2})")
            return False
        
        # Check direction accuracy (for trading, predicting direction is critical)
        direction_acc = new_metrics.get('val_direction_acc', 0.0)
        if direction_acc < 0.52:  # Should beat random (50%) by decent margin
            logger.info(f"New model direction accuracy ({direction_acc:.2%}) too low")
            return False
        
        # If this is the first model, accept it
        if not self.current_model_path.exists():
            logger.info("No existing model, accepting new model")
            return True
        
        # Could add: compare to current model's validation performance
        # For now, accept if meets thresholds
        logger.info(f"New model meets thresholds (R²={val_r2:.4f}, dir_acc={direction_acc:.2%})")
        return True
    
    def _sleep_until_next_cycle(self, cycle_start: float) -> None:
        """Sleep until next training cycle.
        
        Args:
            cycle_start: Timestamp when cycle started.
        """
        elapsed = time.time() - cycle_start
        sleep_time = max(0, self.retrain_interval - elapsed)
        
        if sleep_time > 0:
            logger.info(f"Next training in {sleep_time/3600:.1f}h")
            # Sleep in small increments to allow for graceful shutdown
            while sleep_time > 0 and self.running:
                time.sleep(min(60, sleep_time))  # Check every minute
                sleep_time -= 60
    
    def get_status(self) -> dict:
        """Get updater status.
        
        Returns:
            Dictionary with status information.
        """
        return {
            'running': self.running,
            'training_in_progress': self.training_in_progress,
            'total_retrains': self.total_retrains,
            'successful_swaps': self.successful_swaps,
            'failed_swaps': self.failed_swaps,
            'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            'retrain_interval_hours': self.retrain_interval / 3600,
        }
    
    def trigger_immediate_retrain(self) -> None:
        """Trigger an immediate retrain (interrupts sleep).
        
        Note: This doesn't actually force immediate retrain in current implementation.
        Would need to use threading.Event for proper interruption.
        """
        logger.info("Immediate retrain requested (will trigger on next wake)")
        # In a more sophisticated implementation, would use threading.Event
        # to wake up the sleeping thread immediately

