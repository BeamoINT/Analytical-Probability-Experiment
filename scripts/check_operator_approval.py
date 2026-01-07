#!/usr/bin/env python3
"""Check if the signer is approved as operator for the funder wallet."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polyb0t.config import get_settings

# Polymarket Exchange contracts that need operator approval
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Operator approval ABI (for the Exchange contracts)
OPERATOR_ABI = [
    {
        "inputs": [
            {"name": "", "type": "address"},  # owner
            {"name": "", "type": "address"}   # operator
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "", "type": "address"},  # owner
            {"name": "", "type": "address"}   # operator
        ],
        "name": "operators",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]


def main():
    settings = get_settings()
    
    print("=== Operator Approval Check ===\n")
    
    private_key = settings.polygon_private_key
    funder = settings.funder_address or settings.user_address
    rpc_url = settings.polygon_rpc_url or "https://polygon-rpc.com"
    
    if not private_key:
        print("ERROR: No private key configured")
        return 1
    
    from eth_account import Account
    signer = Account.from_key(private_key)
    
    print(f"Signer (from private key): {signer.address}")
    print(f"Funder (token holder):     {funder}")
    print(f"RPC URL: {rpc_url}")
    print()
    
    try:
        from web3 import Web3
        
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            print("ERROR: Cannot connect to Polygon RPC")
            return 1
        
        print(f"Connected to Polygon (chain_id={w3.eth.chain_id})\n")
        
        funder_cs = Web3.to_checksum_address(funder)
        signer_cs = Web3.to_checksum_address(signer.address)
        
        # Check operator approval on both exchange contracts
        for name, addr in [
            ("NEG_RISK_CTF_EXCHANGE", NEG_RISK_CTF_EXCHANGE),
            ("CTF_EXCHANGE", CTF_EXCHANGE),
        ]:
            print(f"--- {name} ---")
            print(f"Contract: {addr}")
            
            contract = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=OPERATOR_ABI)
            
            # Try isApprovedForAll first
            try:
                approved = contract.functions.isApprovedForAll(funder_cs, signer_cs).call()
                print(f"isApprovedForAll(funder, signer): {approved}")
            except Exception as e:
                print(f"isApprovedForAll: Not available or error: {e}")
            
            # Try operators mapping (some contracts use this)
            try:
                operator_value = contract.functions.operators(funder_cs, signer_cs).call()
                print(f"operators(funder, signer): {operator_value} (1 = approved)")
            except Exception as e:
                print(f"operators: Not available")
            
            print()
        
        # Also check the API credentials are valid
        print("--- API Credentials Check ---")
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            
            client = ClobClient(
                host=settings.clob_base_url,
                chain_id=int(settings.chain_id),
                key=private_key,
                creds=ApiCreds(
                    api_key=settings.clob_api_key or "",
                    api_secret=settings.clob_api_secret or "",
                    api_passphrase=settings.clob_passphrase or "",
                ),
                signature_type=int(settings.signature_type),
                funder=funder,
            )
            
            # Try to get the API key info
            try:
                # Get derived address from API
                api_keys = client.derive_api_key()
                print(f"API key can be derived: True")
                print(f"Derived key: {api_keys.get('apiKey', 'N/A')[:20]}...")
            except Exception as e:
                print(f"API key derivation: {e}")
            
            # Try to get balance to verify API works
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                balance = client.get_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.USDC)
                )
                print(f"USDC Balance check via API: {balance}")
            except Exception as e:
                print(f"USDC balance check failed: {e}")
                
        except Exception as e:
            print(f"API check error: {e}")
        
        print("\n--- Summary ---")
        print("If operator approvals show False/0, you need to approve your signer")
        print("as an operator for your funder wallet on the Polymarket exchange.")
        print("\nThis is typically done when you first enable API trading on Polymarket.")
        print("Try going to polymarket.com -> Settings -> API Keys and regenerate.")
        
        return 0
        
    except ImportError as e:
        print(f"Import error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

