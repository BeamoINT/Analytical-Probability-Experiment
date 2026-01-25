"""Expert versioning system with rollback support.

Each expert maintains up to 3 model versions, allowing rollback
to previous versions if the current one underperforms.
"""

import logging
import os
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from polyb0t.ml.moe.expert import Expert, ExpertMetrics

logger = logging.getLogger(__name__)

# Constants
MAX_VERSIONS = 3
MIN_TRADES_FOR_VALIDATION = 50
MIN_PROFIT_FOR_ACTIVATION = -0.05  # -5%
MIN_WIN_RATE_FOR_ACTIVATION = 0.40  # 40%
MIN_PROFIT_FACTOR_FOR_ACTIVATION = 0.8

# Recovery thresholds
RECOVERY_PROFIT_THRESHOLD = 0.0  # 0%
RECOVERY_WIN_RATE_THRESHOLD = 0.45  # 45%
RECOVERY_CONSECUTIVE_CYCLES = 2


class ExpertState(Enum):
    """State machine for expert lifecycle."""
    
    UNTRAINED = "untrained"      # Never trained, no model
    PROBATION = "probation"      # New model, proving itself (no trading)
    ACTIVE = "active"            # Validated, trading enabled
    SUSPENDED = "suspended"      # Trading disabled, still training
    ROLLBACK = "rollback"        # Rolled back to previous version
    DEPRECATED = "deprecated"    # Permanently disabled


@dataclass
class ExpertVersion:
    """A specific version of an expert's model."""
    
    version_id: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    model_path: str = ""
    
    # Performance metrics for this version
    simulated_profit_pct: float = 0.0
    simulated_num_trades: int = 0
    simulated_win_rate: float = 0.0
    simulated_profit_factor: float = 0.0
    simulated_sharpe: float = 0.0
    simulated_max_drawdown: float = 0.0
    
    # Validation state
    is_validated: bool = False
    validation_passed: bool = False
    
    # Training info
    training_cycles: int = 0
    positive_cycles: int = 0  # Consecutive positive cycles
    negative_cycles: int = 0  # Consecutive negative cycles
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at.isoformat(),
            "model_path": self.model_path,
            "simulated_profit_pct": self.simulated_profit_pct,
            "simulated_num_trades": self.simulated_num_trades,
            "simulated_win_rate": self.simulated_win_rate,
            "simulated_profit_factor": self.simulated_profit_factor,
            "simulated_sharpe": self.simulated_sharpe,
            "simulated_max_drawdown": self.simulated_max_drawdown,
            "is_validated": self.is_validated,
            "validation_passed": self.validation_passed,
            "training_cycles": self.training_cycles,
            "positive_cycles": self.positive_cycles,
            "negative_cycles": self.negative_cycles,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpertVersion":
        created_at = datetime.utcnow()
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except:
                pass
        
        return cls(
            version_id=data.get("version_id", 0),
            created_at=created_at,
            model_path=data.get("model_path", ""),
            simulated_profit_pct=data.get("simulated_profit_pct", 0.0),
            simulated_num_trades=data.get("simulated_num_trades", 0),
            simulated_win_rate=data.get("simulated_win_rate", 0.0),
            simulated_profit_factor=data.get("simulated_profit_factor", 0.0),
            simulated_sharpe=data.get("simulated_sharpe", 0.0),
            simulated_max_drawdown=data.get("simulated_max_drawdown", 0.0),
            is_validated=data.get("is_validated", False),
            validation_passed=data.get("validation_passed", False),
            training_cycles=data.get("training_cycles", 0),
            positive_cycles=data.get("positive_cycles", 0),
            negative_cycles=data.get("negative_cycles", 0),
        )
    
    def score(self) -> float:
        """Calculate version score for comparison."""
        # Primary: profit
        profit_score = max(-0.5, min(0.5, self.simulated_profit_pct * 2))
        
        # Secondary: Sharpe ratio
        sharpe_bonus = max(-0.1, min(0.1, self.simulated_sharpe * 0.05))
        
        # Bonus for good profit factor
        pf_bonus = 0.05 if self.simulated_profit_factor > 1.5 else 0
        
        # Penalty for too few trades
        trade_penalty = -0.1 if self.simulated_num_trades < MIN_TRADES_FOR_VALIDATION else 0
        
        return profit_score + sharpe_bonus + pf_bonus + trade_penalty


class ExpertVersionManager:
    """Manages versions for a single expert."""
    
    def __init__(self, expert_id: str, versions_dir: str):
        self.expert_id = expert_id
        self.versions_dir = versions_dir
        self.versions: List[ExpertVersion] = []
        self.current_version_id: int = 0
        self.state: ExpertState = ExpertState.UNTRAINED
        
        # State tracking
        self._consecutive_bad_cycles: int = 0
        self._consecutive_good_cycles: int = 0
        
        os.makedirs(versions_dir, exist_ok=True)
    
    def create_new_version(self, model: Any, metrics: "ExpertMetrics") -> ExpertVersion:
        """Create a new version after training."""
        version_id = self.current_version_id + 1
        
        # Create version object
        version = ExpertVersion(
            version_id=version_id,
            created_at=datetime.utcnow(),
            model_path=os.path.join(self.versions_dir, f"v{version_id}.model.pkl"),
            simulated_profit_pct=metrics.simulated_profit_pct,
            simulated_num_trades=metrics.simulated_num_trades,
            simulated_win_rate=metrics.simulated_win_rate,
            simulated_profit_factor=metrics.simulated_profit_factor,
            simulated_sharpe=metrics.simulated_sharpe,
            simulated_max_drawdown=metrics.simulated_max_drawdown,
            training_cycles=1,
        )
        
        # Save the model
        try:
            with open(version.model_path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            logger.error(f"Failed to save version {version_id} for {self.expert_id}: {e}")
            return None
        
        # Validate the version
        version.is_validated = True
        version.validation_passed = self._validate_version(version)
        
        # Update positive/negative cycle tracking
        if metrics.simulated_profit_pct > 0:
            version.positive_cycles = 1
            version.negative_cycles = 0
        else:
            version.positive_cycles = 0
            version.negative_cycles = 1
        
        # Add to versions list
        self.versions.append(version)
        self.current_version_id = version_id
        
        # Cleanup old versions
        self._cleanup_old_versions()
        
        # Update state based on validation
        self._update_state_after_training(version)
        
        logger.info(
            f"Expert {self.expert_id}: Created v{version_id}, "
            f"profit={metrics.simulated_profit_pct:+.1%}, "
            f"validated={version.validation_passed}, state={self.state.value}"
        )
        
        return version
    
    def update_version_metrics(self, metrics: "ExpertMetrics") -> None:
        """Update the current version's metrics after retraining."""
        if not self.versions:
            return
        
        current = self.versions[-1]
        
        # Track previous profit for trend
        prev_profit = current.simulated_profit_pct
        
        # Update metrics
        current.simulated_profit_pct = metrics.simulated_profit_pct
        current.simulated_num_trades = metrics.simulated_num_trades
        current.simulated_win_rate = metrics.simulated_win_rate
        current.simulated_profit_factor = metrics.simulated_profit_factor
        current.simulated_sharpe = metrics.simulated_sharpe
        current.simulated_max_drawdown = metrics.simulated_max_drawdown
        current.training_cycles += 1
        
        # Update consecutive cycle tracking
        if metrics.simulated_profit_pct > 0:
            current.positive_cycles += 1
            current.negative_cycles = 0
            self._consecutive_good_cycles += 1
            self._consecutive_bad_cycles = 0
        else:
            current.negative_cycles += 1
            self._consecutive_bad_cycles += 1
            self._consecutive_good_cycles = 0
            if metrics.simulated_profit_pct > prev_profit:
                # Improving but still negative
                current.positive_cycles = 0
            else:
                current.positive_cycles = 0
        
        # Re-validate
        current.validation_passed = self._validate_version(current)
        
        # Update state
        self._update_state_after_training(current)
    
    def _validate_version(self, version: ExpertVersion) -> bool:
        """Check if a version passes validation criteria."""
        # Must have minimum trades
        if version.simulated_num_trades < MIN_TRADES_FOR_VALIDATION:
            return False
        
        # Must not be too unprofitable
        if version.simulated_profit_pct < MIN_PROFIT_FOR_ACTIVATION:
            return False
        
        # Must have reasonable win rate
        if version.simulated_win_rate < MIN_WIN_RATE_FOR_ACTIVATION:
            return False
        
        # Must be better than random
        if version.simulated_profit_factor < MIN_PROFIT_FACTOR_FOR_ACTIVATION:
            return False
        
        return True
    
    def _update_state_after_training(self, version: ExpertVersion) -> None:
        """Update expert state based on version performance."""
        old_state = self.state
        
        if self.state == ExpertState.UNTRAINED:
            # First training - go to probation
            self.state = ExpertState.PROBATION
            
        elif self.state == ExpertState.PROBATION:
            if version.validation_passed:
                self.state = ExpertState.ACTIVE
                logger.info(f"Expert {self.expert_id}: Promoted to ACTIVE")
            elif version.training_cycles >= 3 and not version.validation_passed:
                # Failed validation after 3 cycles
                self.state = ExpertState.SUSPENDED
                logger.info(f"Expert {self.expert_id}: Failed probation -> SUSPENDED")
                
        elif self.state == ExpertState.ACTIVE:
            if not version.validation_passed:
                # Performance dropped
                self.state = ExpertState.SUSPENDED
                logger.info(f"Expert {self.expert_id}: Performance drop -> SUSPENDED")
                
        elif self.state == ExpertState.SUSPENDED:
            if version.validation_passed and self._consecutive_good_cycles >= RECOVERY_CONSECUTIVE_CYCLES:
                # Recovery!
                self.state = ExpertState.ACTIVE
                logger.info(f"Expert {self.expert_id}: Recovered -> ACTIVE")
            elif self._consecutive_bad_cycles >= 3:
                # Try rollback
                if self._try_rollback():
                    self.state = ExpertState.ROLLBACK
                    logger.info(f"Expert {self.expert_id}: Rolling back to previous version")
                else:
                    self.state = ExpertState.DEPRECATED
                    logger.info(f"Expert {self.expert_id}: No good version -> DEPRECATED")
                    
        elif self.state == ExpertState.ROLLBACK:
            if version.validation_passed:
                self.state = ExpertState.ACTIVE
                logger.info(f"Expert {self.expert_id}: Rollback successful -> ACTIVE")
            elif self._consecutive_bad_cycles >= 5:
                self.state = ExpertState.DEPRECATED
                logger.info(f"Expert {self.expert_id}: Rollback failed -> DEPRECATED")
        
        if old_state != self.state:
            logger.info(f"Expert {self.expert_id}: State {old_state.value} -> {self.state.value}")
    
    def _try_rollback(self) -> bool:
        """Try to rollback to a previous good version."""
        # Find the best previous version
        best_version = self.get_best_version()
        
        if best_version is None or best_version.version_id == self.current_version_id:
            return False
        
        # Check if best version was validated
        if not best_version.validation_passed:
            return False
        
        logger.info(
            f"Expert {self.expert_id}: Rolling back from v{self.current_version_id} "
            f"to v{best_version.version_id}"
        )
        
        # Set current to best version
        self.current_version_id = best_version.version_id
        self._consecutive_bad_cycles = 0
        
        return True
    
    def get_best_version(self) -> Optional[ExpertVersion]:
        """Get the best performing version."""
        if not self.versions:
            return None
        
        valid_versions = [v for v in self.versions if v.validation_passed]
        if not valid_versions:
            # Return highest scoring even if not validated
            return max(self.versions, key=lambda v: v.score())
        
        return max(valid_versions, key=lambda v: v.score())
    
    def get_current_version(self) -> Optional[ExpertVersion]:
        """Get the current active version."""
        for v in self.versions:
            if v.version_id == self.current_version_id:
                return v
        return self.versions[-1] if self.versions else None
    
    def load_current_model(self) -> Optional[Any]:
        """Load the current version's model."""
        version = self.get_current_version()
        if version is None:
            return None
        
        if not os.path.exists(version.model_path):
            return None
        
        try:
            with open(version.model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model for {self.expert_id} v{version.version_id}: {e}")
            return None
    
    def _cleanup_old_versions(self) -> None:
        """Remove old versions beyond MAX_VERSIONS."""
        while len(self.versions) > MAX_VERSIONS:
            old_version = self.versions.pop(0)
            
            # Don't delete if it's the current version
            if old_version.version_id == self.current_version_id:
                continue
            
            # Delete model file
            if os.path.exists(old_version.model_path):
                try:
                    os.remove(old_version.model_path)
                    logger.debug(f"Deleted old version: {old_version.model_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {old_version.model_path}: {e}")
    
    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled for this expert."""
        return self.state in [ExpertState.ACTIVE, ExpertState.ROLLBACK]
    
    def save_state(self, path: str) -> None:
        """Save version manager state."""
        state = {
            "expert_id": self.expert_id,
            "current_version_id": self.current_version_id,
            "state": self.state.value,
            "consecutive_bad_cycles": self._consecutive_bad_cycles,
            "consecutive_good_cycles": self._consecutive_good_cycles,
            "versions": [v.to_dict() for v in self.versions],
        }
        
        with open(path, "w") as f:
            import json
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, path: str, expert_id: str, versions_dir: str) -> "ExpertVersionManager":
        """Load version manager state."""
        manager = cls(expert_id, versions_dir)
        
        if not os.path.exists(path):
            return manager
        
        try:
            import json
            with open(path, "r") as f:
                state = json.load(f)
            
            manager.current_version_id = state.get("current_version_id", 0)
            manager.state = ExpertState(state.get("state", "untrained"))
            manager._consecutive_bad_cycles = state.get("consecutive_bad_cycles", 0)
            manager._consecutive_good_cycles = state.get("consecutive_good_cycles", 0)
            manager.versions = [
                ExpertVersion.from_dict(v) for v in state.get("versions", [])
            ]
        except Exception as e:
            logger.error(f"Failed to load version state from {path}: {e}")
        
        return manager
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for status display."""
        current = self.get_current_version()
        best = self.get_best_version()
        
        return {
            "expert_id": self.expert_id,
            "state": self.state.value,
            "current_version_id": self.current_version_id,
            "total_versions": len(self.versions),
            "trading_enabled": self.is_trading_enabled(),
            "consecutive_bad_cycles": self._consecutive_bad_cycles,
            "consecutive_good_cycles": self._consecutive_good_cycles,
            "current_version": current.to_dict() if current else None,
            "best_version_id": best.version_id if best else None,
        }
