"""Auto-enable ML when sufficient training data is collected."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def auto_enable_ml_in_env(env_path: str = ".env") -> bool:
    """Automatically enable ML by updating .env file.
    
    This is called when the model updater detects sufficient training data.
    
    Args:
        env_path: Path to .env file.
        
    Returns:
        True if ML was enabled, False otherwise.
    """
    env_file = Path(env_path)
    
    if not env_file.exists():
        logger.warning(f".env file not found at {env_path}, cannot auto-enable ML")
        return False
    
    try:
        # Read current .env
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Check if ML is already enabled
        ml_enabled = False
        ml_line_idx = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith('POLYBOT_ENABLE_ML'):
                ml_line_idx = i
                if 'true' in line.lower():
                    ml_enabled = True
                break
        
        if ml_enabled:
            logger.info("ML already enabled in .env")
            return False
        
        # Enable ML
        if ml_line_idx is not None:
            # Replace existing line
            lines[ml_line_idx] = 'POLYBOT_ENABLE_ML=true  # Auto-enabled after collecting sufficient data\n'
        else:
            # Add new line
            lines.append('\n# Machine Learning Auto-Enabled\n')
            lines.append('POLYBOT_ENABLE_ML=true  # Auto-enabled after collecting sufficient data\n')
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        logger.warning(
            "✅ ML AUTO-ENABLED in .env! "
            "Restart the bot to activate ML predictions: "
            "poetry run polyb0t run --live"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to auto-enable ML in .env: {e}")
        return False


def check_and_auto_enable(
    labeled_examples: int,
    threshold: int,
    env_path: str = ".env",
) -> bool:
    """Check if auto-enable threshold is met and enable ML.
    
    Args:
        labeled_examples: Number of labeled training examples.
        threshold: Threshold to trigger auto-enable.
        env_path: Path to .env file.
        
    Returns:
        True if ML was auto-enabled, False otherwise.
    """
    if threshold <= 0:
        # Auto-enable disabled
        return False
    
    if labeled_examples < threshold:
        # Not enough data yet
        return False
    
    # Threshold met, auto-enable
    logger.info(
        f"🎓 Auto-enable threshold reached: {labeled_examples} >= {threshold} examples"
    )
    
    return auto_enable_ml_in_env(env_path)

