#!/usr/bin/env python3
"""Set CONDITIONAL token allowance for Polymarket CLOB (enables selling).

This approves the CLOB to transfer your outcome tokens, which is required for selling positions.
Run this ONCE to enable auto-selling.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Set conditional token allowance."""
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType
    
    # Load .env
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print("ERROR: .env file not found")
        sys.exit(1)
    
    # Parse .env manually
    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value.strip('"').strip("'")
    
    private_key = env_vars.get('POLYBOT_POLYGON_PRIVATE_KEY')
    api_key = env_vars.get('POLYBOT_CLOB_API_KEY')
    api_secret = env_vars.get('POLYBOT_CLOB_API_SECRET')
    passphrase = env_vars.get('POLYBOT_CLOB_PASSPHRASE') or env_vars.get('POLYBOT_CLOB_API_PASSPHRASE')
    funder = env_vars.get('POLYBOT_FUNDER_ADDRESS')
    signature_type = int(env_vars.get('POLYBOT_SIGNATURE_TYPE', '1'))
    
    if not all([private_key, api_key, api_secret, passphrase]):
        print("ERROR: Missing required credentials in .env")
        print(f"  POLYBOT_POLYGON_PRIVATE_KEY: {'✓' if private_key else '✗'}")
        print(f"  POLYBOT_CLOB_API_KEY: {'✓' if api_key else '✗'}")
        print(f"  POLYBOT_CLOB_API_SECRET: {'✓' if api_secret else '✗'}")
        print(f"  POLYBOT_CLOB_PASSPHRASE: {'✓' if passphrase else '✗'}")
        sys.exit(1)
    
    print("=" * 60)
    print("POLYMARKET CONDITIONAL TOKEN ALLOWANCE")
    print("=" * 60)
    print()
    print("This will approve the Polymarket exchange to transfer your")
    print("outcome tokens, enabling the bot to SELL positions.")
    print()
    
    # Initialize client
    print("Connecting to Polymarket CLOB...")
    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=private_key,
        creds=ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=passphrase,
        ),
        signature_type=signature_type,
        funder=funder,
    )
    
    # Check current allowance
    print("\nChecking current CONDITIONAL token allowance...")
    try:
        current = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
        )
        print(f"Current allowance: {current}")
    except Exception as e:
        print(f"Could not check current allowance: {e}")
    
    # Set unlimited allowance
    print("\nSetting UNLIMITED allowance for CONDITIONAL tokens...")
    print("(This enables selling of outcome tokens)")
    
    try:
        result = client.update_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
        )
        print(f"\n✓ SUCCESS! Conditional token allowance set.")
        print(f"  Result: {result}")
        print()
        print("The bot can now SELL positions automatically!")
        print()
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        print()
        print("You may need to do this manually on polymarket.com")
        sys.exit(1)


if __name__ == "__main__":
    main()

