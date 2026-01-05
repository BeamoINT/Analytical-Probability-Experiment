#!/usr/bin/env python3
"""Clean up old stuck intents from the database."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from polyb0t.data.storage import get_session, TradeIntentDB, init_db
from polyb0t.execution.intents import IntentStatus


def cleanup_old_intents():
    """Mark old pending intents as expired."""
    session = get_session()
    
    try:
        # Find all pending intents older than 2 minutes
        cutoff = datetime.utcnow() - timedelta(minutes=2)
        
        old_pending = (
            session.query(TradeIntentDB)
            .filter(TradeIntentDB.status == IntentStatus.PENDING.value)
            .filter(TradeIntentDB.created_at < cutoff)
            .all()
        )
        
        if not old_pending:
            print("✅ No old pending intents found. Database is clean!")
            return 0
        
        print(f"\n🧹 Found {len(old_pending)} old pending intents:")
        for intent in old_pending:
            age_minutes = (datetime.utcnow() - intent.created_at).total_seconds() / 60
            print(f"  - {intent.intent_id[:8]}: created {age_minutes:.1f} minutes ago")
        
        print(f"\n❓ Mark these {len(old_pending)} intents as EXPIRED? (y/n): ", end="")
        response = input().strip().lower()
        
        if response != 'y':
            print("❌ Cleanup cancelled.")
            return 1
        
        # Mark as expired
        for intent in old_pending:
            intent.status = IntentStatus.EXPIRED.value
        
        session.commit()
        print(f"\n✅ Marked {len(old_pending)} old intents as EXPIRED")
        print("\n💡 Now restart the bot to see fresh intents:")
        print("   poetry run polyb0t run")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        session.rollback()
        return 1
    finally:
        session.close()


def list_current_intents():
    """List current pending intents."""
    session = get_session()
    
    try:
        pending = (
            session.query(TradeIntentDB)
            .filter(TradeIntentDB.status == IntentStatus.PENDING.value)
            .order_by(TradeIntentDB.created_at.desc())
            .all()
        )
        
        if not pending:
            print("\n✅ No pending intents in database")
            return
        
        print(f"\n📋 Current Pending Intents ({len(pending)}):")
        print("-" * 70)
        
        for intent in pending:
            age_seconds = (datetime.utcnow() - intent.created_at).total_seconds()
            expires_in = (intent.expires_at - datetime.utcnow()).total_seconds() if intent.expires_at else 0
            
            print(f"ID: {intent.intent_id[:8]}")
            print(f"  Type: {intent.intent_type}")
            print(f"  Side: {intent.side}")
            print(f"  Price: {intent.price:.3f} USD" if intent.price else "  Price: N/A")
            print(f"  Edge: {intent.edge:+.3f}" if intent.edge else "  Edge: N/A")
            print(f"  Age: {age_seconds:.0f}s")
            print(f"  Expires in: {expires_in:.0f}s")
            print()
        
    finally:
        session.close()


def main():
    """Main cleanup script."""
    print("🔧 Polymarket Trading System - Intent Cleanup")
    print("=" * 70)
    
    # Initialize database
    init_db()
    
    # Show current intents
    list_current_intents()
    
    # Cleanup old ones
    return cleanup_old_intents()


if __name__ == "__main__":
    sys.exit(main())

