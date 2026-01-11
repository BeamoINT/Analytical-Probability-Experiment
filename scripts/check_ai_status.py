#!/usr/bin/env python3
"""Check AI model status and performance metrics."""

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print("\n" + "=" * 60)
    print("AI MODEL STATUS")
    print("=" * 60)
    
    # Check model state
    model_dir = "data/ai_models"
    state_path = os.path.join(model_dir, "trainer_state.json")
    
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        
        print(f"\n✅ TRAINED MODEL FOUND (v{state['version']})")
        print(f"   Created: {state['created_at']}")
        print(f"   Training examples: {state['training_examples']}")
        
        print("\n📊 PERFORMANCE METRICS:")
        metrics = state.get("metrics", {})
        print(f"   Directional Accuracy: {metrics.get('directional_accuracy', 0)*100:.1f}%")
        print(f"   Profitable Accuracy:  {metrics.get('profitable_accuracy', 0)*100:.1f}%")
        print(f"   R² Score:            {metrics.get('r2', 0):.3f}")
        print(f"   Mean Squared Error:  {metrics.get('mse', 0):.6f}")
        print(f"   Mean Absolute Error: {metrics.get('mae', 0):.6f}")
        
        print("\n💰 PROFITABILITY INTERPRETATION:")
        prof_acc = metrics.get('profitable_accuracy', 0)
        if prof_acc > 0.55:
            print(f"   ✅ Model shows edge ({prof_acc*100:.1f}% profitable predictions)")
            print(f"   Expected win rate: {prof_acc*100:.1f}%")
        elif prof_acc > 0.50:
            print(f"   ⚠️  Slight edge ({prof_acc*100:.1f}% profitable)")
            print(f"   Marginal profitability - may improve with more data")
        else:
            print(f"   ❌ No clear edge ({prof_acc*100:.1f}% profitable)")
            print(f"   Model needs more training data")
            
    else:
        print("\n⏳ NO TRAINED MODEL YET")
        print("   Model will train once enough labeled examples are collected.")
    
    # Check training data
    db_path = "data/ai_training.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM training_examples")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM training_examples WHERE is_fully_labeled = 1")
        labeled = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM market_snapshots")
        snapshots = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM price_history")
        prices = cursor.fetchone()[0]
        
        # Get sample of labeled examples with their labels
        cursor.execute("""
            SELECT price_change_1h, price_change_24h, direction_1h, direction_24h
            FROM training_examples 
            WHERE is_fully_labeled = 1
            LIMIT 100
        """)
        samples = cursor.fetchall()
        
        conn.close()
        
        print("\n📦 TRAINING DATA:")
        print(f"   Total examples:   {total:,}")
        print(f"   Fully labeled:    {labeled:,}")
        print(f"   Market snapshots: {snapshots:,}")
        print(f"   Price history:    {prices:,}")
        
        if samples:
            # Analyze label distribution
            up_1h = sum(1 for s in samples if s[2] == 1)
            down_1h = sum(1 for s in samples if s[2] == -1)
            flat_1h = sum(1 for s in samples if s[2] == 0)
            
            up_24h = sum(1 for s in samples if s[3] == 1)
            down_24h = sum(1 for s in samples if s[3] == -1)
            flat_24h = sum(1 for s in samples if s[3] == 0)
            
            print("\n📈 LABEL DISTRIBUTION (sample of labeled examples):")
            print(f"   1h direction:  ↑{up_1h} ↓{down_1h} →{flat_1h}")
            print(f"   24h direction: ↑{up_24h} ↓{down_24h} →{flat_24h}")
            
            # Average price changes
            changes_1h = [s[0] for s in samples if s[0] is not None]
            changes_24h = [s[1] for s in samples if s[1] is not None]
            
            if changes_1h:
                avg_1h = sum(changes_1h) / len(changes_1h) * 100
                print(f"\n   Avg 1h change:  {avg_1h:+.2f}%")
            if changes_24h:
                avg_24h = sum(changes_24h) / len(changes_24h) * 100
                print(f"   Avg 24h change: {avg_24h:+.2f}%")
        
        # Estimate time to training
        min_examples = 500  # Default threshold
        if labeled < min_examples:
            needed = min_examples - labeled
            # After 24h delay, examples become labeled at creation rate
            # ~14-16 examples per cycle, ~4 cycles per hour = ~60 per hour
            examples_per_hour = 60
            hours_needed = needed / examples_per_hour
            print(f"\n⏱️  TRAINING ESTIMATE:")
            print(f"   Need {needed} more labeled examples")
            print(f"   Rate: ~{examples_per_hour} examples/hour become labeled")
            print(f"   Estimated: ~{hours_needed:.1f} hours ({hours_needed*60:.0f} minutes)")
    else:
        print("\n❌ No training database found")
        
    # Check storage
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        size_gb = size_mb / 1024
        print(f"\n💾 STORAGE:")
        print(f"   Database size: {size_mb:.1f} MB ({size_gb:.3f} GB)")
        print(f"   Max allowed:   140 GB")
        
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
