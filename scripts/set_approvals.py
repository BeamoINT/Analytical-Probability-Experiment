#!/usr/bin/env python3
"""Set token approvals for the CLOB exchange contract."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polyb0t.config import get_settings


def main():
    settings = get_settings()
    
    print("=== Polymarket Token Approval Setup ===\n")
    
    # Check required settings
    private_key = settings.polygon_private_key
    funder = settings.funder_address or settings.user_address
    
    if not private_key:
        print("ERROR: POLYBOT_POLYGON_PRIVATE_KEY not set")
        return 1
    
    print(f"Funder/Wallet Address: {funder}")
    print(f"Private Key: {private_key[:10]}...{private_key[-6:]}")
    print(f"Chain ID: {settings.chain_id}")
    print(f"CLOB Base URL: {settings.clob_base_url}")
    print(f"Signature Type: {settings.signature_type}")
    print()
    
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        
        # Create client
        creds = ApiCreds(
            api_key=settings.clob_api_key or "",
            api_secret=settings.clob_api_secret or "",
            api_passphrase=settings.clob_passphrase or "",
        )
        
        client = ClobClient(
            host=settings.clob_base_url,
            chain_id=int(settings.chain_id),
            key=private_key,
            creds=creds,
            signature_type=int(settings.signature_type),
            funder=funder,
        )
        
        print("Attempting to set allowances...")
        print("This will set approval for:")
        print("  - USDC token -> CLOB Exchange")
        print("  - CTF tokens -> CLOB Exchange (for selling)")
        print()
        
        # Try to set allowances
        try:
            result = client.set_allowances()
            print(f"set_allowances() result: {result}")
            print("\n✅ Allowances set successfully!")
        except Exception as e:
            print(f"\n❌ set_allowances() failed: {e}")
            print("\nThis likely means the private key cannot authorize approvals")
            print("for the funder wallet (common with proxy/embedded wallets).")
            
            # Try alternative: check what we can do
            print("\n--- Diagnosing the issue ---")
            
            # Get the signer address from private key
            from eth_account import Account
            signer = Account.from_key(private_key)
            print(f"Signer address (from private key): {signer.address}")
            print(f"Funder address (where tokens are): {funder}")
            
            if signer.address.lower() != funder.lower():
                print("\n⚠️  MISMATCH DETECTED!")
                print("The private key corresponds to a DIFFERENT address than the funder.")
                print("This is expected for proxy wallet setups (signature_type=1).")
                print("\nThe signer can sign orders, but CANNOT set on-chain approvals")
                print("for the funder wallet.")
                print("\nTo fix this, you need to:")
                print("1. Export your Polymarket wallet's actual private key")
                print("   (look for 'Export Private Key' in Polymarket settings)")
                print("2. Update POLYBOT_POLYGON_PRIVATE_KEY with that key")
                print("3. Set POLYBOT_FUNDER_ADDRESS to your wallet address")
                print("4. Or: set signature_type=0 if using EOA directly")
            else:
                print("\nAddresses match. The approval transaction might have failed")
                print("due to insufficient gas (POL) or other on-chain issues.")
            
            return 1
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure py-clob-client is installed: pip install py-clob-client")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

