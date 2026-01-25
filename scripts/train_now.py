#!/usr/bin/env python3
"""Manually trigger AI/MoE training.

Run: poetry run python scripts/train_now.py
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polyb0t.config import load_env_or_exit
from polyb0t.utils.logging import setup_logging

def main():
    print("=" * 60)
    print("MANUAL TRAINING TRIGGER")
    print("=" * 60)
    
    # Load environment
    load_env_or_exit()
    setup_logging()
    
    # Import after loading env
    from polyb0t.ml.ai_orchestrator import get_ai_orchestrator
    
    print("\nInitializing AI Orchestrator...")
    orchestrator = get_ai_orchestrator()
    
    print(f"\nCurrent status:")
    print(f"  - AI Ready: {orchestrator.is_ai_ready()}")
    
    # Get MoE stats
    moe_stats = orchestrator.get_moe_stats()
    if moe_stats:
        print(f"  - Total Experts: {moe_stats.get('total_experts', 0)}")
        print(f"  - Active Experts: {moe_stats.get('active_experts', 0)}")
        print(f"  - Training Cycles: {moe_stats.get('training_cycles', 0)}")
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60)
    print("\nThis may take several minutes. Please wait...\n")
    
    # Run training
    result = orchestrator.run_training()
    
    print("\n" + "=" * 60)
    if result:
        print("TRAINING COMPLETED SUCCESSFULLY!")
    else:
        print("TRAINING COMPLETED (no new model deployed)")
    print("=" * 60)
    
    # Show updated stats
    print("\nUpdated status:")
    print(f"  - AI Ready: {orchestrator.is_ai_ready()}")
    
    moe_stats = orchestrator.get_moe_stats()
    if moe_stats:
        print(f"  - Total Experts: {moe_stats.get('total_experts', 0)}")
        print(f"  - Active Experts: {moe_stats.get('active_experts', 0)}")
        print(f"  - Training Cycles: {moe_stats.get('training_cycles', 0)}")
    
    print("\nRun 'poetry run polyb0t status --full' to see detailed status.")
    print("=" * 60)

if __name__ == "__main__":
    main()
