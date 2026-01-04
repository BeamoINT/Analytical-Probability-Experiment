#!/usr/bin/env python3
"""
Generate Polymarket L2 API Credentials (One-Time Setup)

This script generates L2 credentials (API Key, Secret, Passphrase) for Polymarket CLOB
using the official py-clob-client library. 

⚠️  SECURITY REQUIREMENTS:
1. Use a DEDICATED hot wallet with minimal funds
2. Never use your main wallet
3. Delete the private key after generating credentials
4. Only the L2 credentials should be stored in .env

Usage:
    # Set environment variables
    export POLY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY
    export POLY_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS
    
    # Run script
    python scripts/generate_l2_creds.py
    
    # CRITICAL: Unset the private key immediately
    unset POLY_PRIVATE_KEY
"""

import os
import sys


def main() -> None:
    """Generate L2 credentials using official Polymarket CLOB client."""
    
    print("\n" + "=" * 70)
    print("POLYMARKET L2 CREDENTIAL GENERATOR")
    print("=" * 70)
    print("\n⚠️  SECURITY WARNING:")
    print("This script requires your wallet PRIVATE KEY to generate L2 credentials.")
    print("- Use a DEDICATED hot wallet with minimal funds")
    print("- NEVER use your main wallet")
    print("- The private key is used ONLY to sign the L1 auth message")
    print("- After generation, DELETE the private key from your environment")
    print("=" * 70 + "\n")
    
    # Check if py-clob-client is installed
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        print("❌ ERROR: py-clob-client not installed\n")
        print("Install it with:")
        print("  pip install py-clob-client\n")
        print("Or if using Poetry in your main project:")
        print("  poetry add py-clob-client\n")
        sys.exit(1)
    
    # Get environment variables
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    funder = os.environ.get("POLY_FUNDER_ADDRESS")
    
    if not private_key:
        print("❌ ERROR: POLY_PRIVATE_KEY environment variable not set\n")
        print("Set it with:")
        print("  export POLY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY\n")
        sys.exit(1)
    
    if not funder:
        print("❌ ERROR: POLY_FUNDER_ADDRESS environment variable not set\n")
        print("Set it with:")
        print("  export POLY_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS\n")
        sys.exit(1)
    
    # Validate inputs
    if not private_key.startswith("0x"):
        print("⚠️  Warning: Private key should start with '0x', adding it...")
        private_key = "0x" + private_key
    
    if not funder.startswith("0x"):
        print("⚠️  Warning: Funder address should start with '0x', adding it...")
        funder = "0x" + funder
    
    if len(private_key) != 66:  # 0x + 64 hex chars
        print(f"❌ ERROR: Invalid private key length: {len(private_key)} (expected 66)\n")
        sys.exit(1)
    
    if len(funder) != 42:  # 0x + 40 hex chars
        print(f"❌ ERROR: Invalid address length: {len(funder)} (expected 42)\n")
        sys.exit(1)
    
    # Configuration
    HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137  # Polygon mainnet
    SIGNATURE_TYPE = 0  # 0 = EOA (standard wallet)
    
    print(f"Configuration:")
    print(f"  CLOB Host:      {HOST}")
    print(f"  Chain ID:       {CHAIN_ID} (Polygon)")
    print(f"  Funder Address: {funder}")
    print(f"  Signature Type: {SIGNATURE_TYPE} (EOA)\n")
    
    # Confirm before proceeding
    response = input("Continue with credential generation? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("Aborted.")
        sys.exit(0)
    
    print("\n" + "=" * 70)
    print("GENERATING L2 CREDENTIALS...")
    print("=" * 70)
    print("This will sign an L1 authentication message with your wallet.")
    print("The CLOB will return your L2 API credentials.\n")
    
    try:
        # Initialize CLOB client
        print("🔑 Initializing CLOB client...")
        client = ClobClient(
            host=HOST,
            chain_id=CHAIN_ID,
            key=private_key,
            funder=funder,
            signature_type=SIGNATURE_TYPE
        )
        
        # Generate/derive API credentials
        print("🔐 Calling create_or_derive_api_key()...")
        print("    (This signs a message with your wallet)\n")
        
        creds = client.create_or_derive_api_key()
        
        print("=" * 70)
        print("✅ SUCCESS - L2 CREDENTIALS GENERATED")
        print("=" * 70)
        print("\n⚠️  SAVE THESE CREDENTIALS IMMEDIATELY ⚠️\n")
        print(f"POLYBOT_CLOB_API_KEY={creds['apiKey']}")
        print(f"POLYBOT_CLOB_API_SECRET={creds['secret']}")
        print(f"POLYBOT_CLOB_API_PASSPHRASE={creds['passphrase']}")
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Copy the credentials above")
        print("2. Add them to your .env file:")
        print("   ")
        print("   POLYBOT_CLOB_API_KEY=<value>")
        print("   POLYBOT_CLOB_API_SECRET=<value>")
        print("   POLYBOT_CLOB_API_PASSPHRASE=<value>")
        print("   ")
        print("3. CRITICAL: Delete the private key from your environment:")
        print("   ")
        print("   unset POLY_PRIVATE_KEY")
        print("   ")
        print("4. Verify authentication:")
        print("   ")
        print("   poetry run polyb0t auth check")
        print("   ")
        print("=" * 70)
        print("\n✅ Your machine should NEVER store the private key again.")
        print("   Only the L2 credentials above should be in your .env file.\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR GENERATING CREDENTIALS")
        print("=" * 70)
        print(f"\nError: {e}\n")
        print("Common issues:")
        print("- Invalid private key format")
        print("- Network connectivity problems")
        print("- CLOB API endpoint unavailable")
        print("- Wrong chain ID or signature type")
        print("\nPlease verify your inputs and try again.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

