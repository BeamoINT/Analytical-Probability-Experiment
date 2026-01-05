#!/usr/bin/env python3
"""Fix trading configuration to be more responsive to market conditions."""

import os
import sys
from pathlib import Path


def check_env_file():
    """Check and update .env file with better defaults."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create .env from env.live.example")
        return False
    
    # Read current config
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    # Find and update critical settings
    updates = {
        "POLYBOT_EDGE_THRESHOLD": "0.02",  # Lower from 0.05 to 0.02 (2% vs 5%)
        "POLYBOT_MIN_NET_EDGE": "0.01",    # Lower from 0.02 to 0.01 (1% vs 2%)
        "POLYBOT_INTENT_EXPIRY_SECONDS": "90",  # Increase from 60 to 90 seconds
        "POLYBOT_INTENT_COOLDOWN_SECONDS": "60",  # Lower from 120 to 60 seconds
    }
    
    updated_lines = []
    found_settings = set()
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            updated_lines.append(line)
            continue
        
        # Check if this line contains one of our settings
        updated = False
        for key, value in updates.items():
            if line_stripped.startswith(f"{key}="):
                # Found the setting, update it
                old_value = line_stripped.split("=", 1)[1] if "=" in line_stripped else ""
                new_line = f"{key}={value}\n"
                updated_lines.append(new_line)
                found_settings.add(key)
                if old_value != value:
                    print(f"✓ Updated {key}: {old_value} → {value}")
                updated = True
                break
        
        if not updated:
            updated_lines.append(line)
    
    # Add missing settings
    missing = set(updates.keys()) - found_settings
    if missing:
        print("\n📝 Adding missing settings:")
        for key in missing:
            new_line = f"{key}={updates[key]}\n"
            updated_lines.append(new_line)
            print(f"  + {key}={updates[key]}")
    
    # Write back
    with open(env_file, "w") as f:
        f.writelines(updated_lines)
    
    return True


def print_diagnosis():
    """Print diagnosis of the trading issues."""
    print("\n" + "="*70)
    print("🔍 TRADING SYSTEM DIAGNOSIS")
    print("="*70)
    
    print("\n❌ ISSUES FOUND:")
    print("\n1. WRONG BALANCE REPORTING")
    print("   Problem: System showing $10,000 (simulated) instead of real balance")
    print("   Cause: Portfolio object is for paper trading simulation")
    print("   Fix: ✅ Updated to use real on-chain balance in live mode")
    
    print("\n2. OLD INTENTS STUCK")
    print("   Problem: Same old pending intents shown repeatedly")
    print("   Cause: Intent cleanup not aggressive enough")
    print("   Fix: ✅ Improved intent expiration and cleanup logic")
    
    print("\n3. NO NEW SIGNALS GENERATED")
    print("   Problem: All signals rejected as 'raw_edge_below_threshold'")
    print("   Cause: Edge thresholds too conservative (5% raw edge is rare)")
    print("   Fix: ⚙️  Adjusted config to lower thresholds")
    
    print("\n" + "="*70)
    print("📊 CONFIGURATION CHANGES")
    print("="*70)
    
    print("\n• POLYBOT_EDGE_THRESHOLD: 0.05 → 0.02 (5% → 2%)")
    print("  More responsive to market opportunities")
    
    print("\n• POLYBOT_MIN_NET_EDGE: 0.02 → 0.01 (2% → 1%)")
    print("  Accept trades with smaller but positive expected value")
    
    print("\n• POLYBOT_INTENT_EXPIRY_SECONDS: 60 → 90")
    print("  Give intents more time before expiration")
    
    print("\n• POLYBOT_INTENT_COOLDOWN_SECONDS: 120 → 60")
    print("  Allow faster regeneration of similar intents")
    
    print("\n" + "="*70)
    print("⚠️  IMPORTANT NOTES")
    print("="*70)
    
    print("\n• These are STARTER thresholds for testing")
    print("• Monitor your trades and adjust based on results")
    print("• Start with small max_order_usd to limit risk")
    print("• System is still in DRY-RUN mode (no real orders)")
    
    print("\n" + "="*70)


def main():
    """Main fix script."""
    print("🔧 Polymarket Trading System - Configuration Fix")
    print("="*70)
    
    # Print diagnosis
    print_diagnosis()
    
    # Update config
    print("\n" + "="*70)
    print("📝 UPDATING CONFIGURATION")
    print("="*70 + "\n")
    
    if check_env_file():
        print("\n✅ Configuration updated successfully!")
        print("\n📋 NEXT STEPS:")
        print("   1. Review the changes in .env")
        print("   2. Restart the trading bot: poetry run polyb0t run")
        print("   3. Monitor logs for new signals being generated")
        print("   4. Check intents with: poetry run polyb0t intents list")
        print("   5. Approve good intents with: poetry run polyb0t intents approve <id>")
        
        print("\n💡 TIP: Start monitoring with:")
        print("   tail -f live_run.log | grep -E '(signals|intents|edge)'")
    else:
        print("\n❌ Configuration update failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

