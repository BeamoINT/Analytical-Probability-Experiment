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
    
    # Check training data
    training_stats = orchestrator.get_training_stats()
    print(f"\nTraining data:")
    print(f"  - Total Examples: {training_stats['collector']['total_examples']}")
    print(f"  - Labeled Examples: {training_stats['collector']['labeled_examples']}")
    print(f"  - Can Train: {training_stats['can_train']}")
    
    print("\n" + "=" * 60)
    print("FORCING TRAINING (bypassing time check)...")
    print("=" * 60)
    print("\nThis may take several minutes. Please wait...\n")
    
    # Force training by resetting last training time on orchestrator
    orchestrator._last_training_time = None
    
    # Also reset on MoE trainer
    if hasattr(orchestrator, 'moe_trainer') and orchestrator.moe_trainer:
        orchestrator.moe_trainer._last_training = None
        print("Reset MoE trainer time check")
    
    # Run training directly on MoE trainer for more control
    print("\n--- Running MoE Training ---\n")
    
    moe_result = None
    if hasattr(orchestrator, 'moe_trainer') and orchestrator.moe_trainer:
        try:
            moe_result = orchestrator.moe_trainer.train()
            if moe_result:
                print(f"\nMoE Training Result:")
                print(f"  - Success: {moe_result.get('success', False)}")
                print(f"  - Experts Trained: {moe_result.get('n_experts_trained', 0)}")
                print(f"  - Training Time: {moe_result.get('training_time_seconds', 0):.1f}s")
                
                # Show per-expert results
                if 'expert_states' in moe_result:
                    print(f"\nExpert States After Training:")
                    for expert_id, state in moe_result['expert_states'].items():
                        print(f"    {expert_id}: {state}")
        except Exception as e:
            print(f"MoE training error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("MoE trainer not available!")
    
    # Also run legacy training
    print("\n--- Running Legacy Model Training ---\n")
    try:
        training_data = orchestrator.collector.get_training_data(only_labeled=True)
        print(f"Training data samples: {len(training_data)}")
        
        legacy_result = orchestrator.trainer.train_model(training_data)
        if legacy_result:
            print(f"Legacy model v{legacy_result.version} deployed")
    except Exception as e:
        print(f"Legacy training error: {e}")
    
    # Update orchestrator state
    from datetime import datetime
    orchestrator._last_training_time = datetime.utcnow()
    orchestrator._save_state()
    
    print("\n" + "=" * 60)
    if moe_result and moe_result.get('success'):
        print("TRAINING COMPLETED SUCCESSFULLY!")
    else:
        print("TRAINING COMPLETED")
    print("=" * 60)
    
    # Show updated stats
    print("\nUpdated status:")
    print(f"  - AI Ready: {orchestrator.is_ai_ready()}")
    
    moe_stats = orchestrator.get_moe_stats()
    if moe_stats:
        print(f"  - Total Experts: {moe_stats.get('total_experts', 0)}")
        print(f"  - Active Experts: {moe_stats.get('active_experts', 0)}")
        print(f"  - Probation: {moe_stats.get('probation_experts', 0)}")
        print(f"  - Suspended: {moe_stats.get('suspended_experts', 0)}")
        print(f"  - Training Cycles: {moe_stats.get('training_cycles', 0)}")
    
    print("\nRun 'poetry run polyb0t status --full' to see detailed status.")
    print("=" * 60)

if __name__ == "__main__":
    main()
