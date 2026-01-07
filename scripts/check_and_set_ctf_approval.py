#!/usr/bin/env python3
"""Check and attempt to set CTF token approval on-chain."""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polyb0t.config import get_settings

# Polymarket contract addresses on Polygon
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # Conditional Token Framework
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # Exchange for neg risk markets
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Regular CTF exchange

# ERC1155 approval check ABI
APPROVAL_ABI = [
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "operator", "type": "address"}
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"}
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]


def main():
    settings = get_settings()
    
    print("=== CTF Token Approval Check ===\n")
    
    funder = settings.funder_address or settings.user_address
    rpc_url = settings.polygon_rpc_url or "https://polygon-rpc.com"
    
    print(f"Wallet (funder): {funder}")
    print(f"RPC URL: {rpc_url}")
    print(f"CTF Contract: {CTF_ADDRESS}")
    print(f"NEG_RISK Exchange: {NEG_RISK_CTF_EXCHANGE}")
    print(f"CTF Exchange: {CTF_EXCHANGE}")
    print()
    
    try:
        from web3 import Web3
        
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            print("ERROR: Cannot connect to Polygon RPC")
            return 1
        
        print(f"Connected to Polygon (chain_id={w3.eth.chain_id})")
        
        # Create CTF contract instance
        ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=APPROVAL_ABI)
        
        # Check approval for both exchange contracts
        print("\n--- Checking Approvals ---")
        
        funder_checksum = Web3.to_checksum_address(funder)
        
        # Check NEG_RISK exchange approval
        neg_risk_approved = ctf.functions.isApprovedForAll(
            funder_checksum,
            Web3.to_checksum_address(NEG_RISK_CTF_EXCHANGE)
        ).call()
        print(f"NEG_RISK_CTF_EXCHANGE approved: {neg_risk_approved}")
        
        # Check regular CTF exchange approval
        ctf_approved = ctf.functions.isApprovedForAll(
            funder_checksum,
            Web3.to_checksum_address(CTF_EXCHANGE)
        ).call()
        print(f"CTF_EXCHANGE approved: {ctf_approved}")
        
        if neg_risk_approved and ctf_approved:
            print("\n✅ Both exchanges are approved! The bot should be able to sell.")
            print("If sells are still failing, there might be a different issue.")
            return 0
        
        print("\n❌ Approval missing! Need to set approval for the exchange(s).")
        
        # Try to set approval
        print("\n--- Attempting to Set Approval ---")
        
        private_key = settings.polygon_private_key
        if not private_key:
            print("ERROR: No private key configured")
            return 1
        
        from eth_account import Account
        signer = Account.from_key(private_key)
        print(f"Signer address: {signer.address}")
        
        if signer.address.lower() == funder.lower():
            # Signer IS the funder - we can set approval directly
            print("Signer matches funder - can set approval directly!")
            
            for exchange_name, exchange_addr, is_approved in [
                ("NEG_RISK_CTF_EXCHANGE", NEG_RISK_CTF_EXCHANGE, neg_risk_approved),
                ("CTF_EXCHANGE", CTF_EXCHANGE, ctf_approved),
            ]:
                if is_approved:
                    print(f"{exchange_name}: Already approved ✓")
                    continue
                
                print(f"\nSetting approval for {exchange_name}...")
                
                # Build the transaction
                tx = ctf.functions.setApprovalForAll(
                    Web3.to_checksum_address(exchange_addr),
                    True
                ).build_transaction({
                    'from': signer.address,
                    'nonce': w3.eth.get_transaction_count(signer.address),
                    'gas': 100000,
                    'gasPrice': w3.eth.gas_price,
                    'chainId': 137,
                })
                
                # Sign and send
                signed = w3.eth.account.sign_transaction(tx, private_key)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                print(f"Transaction sent: {tx_hash.hex()}")
                
                # Wait for confirmation
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                if receipt['status'] == 1:
                    print(f"✅ {exchange_name} approval set successfully!")
                else:
                    print(f"❌ Transaction failed!")
                    return 1
            
            print("\n✅ All approvals set! Restart the bot and try again.")
            return 0
        else:
            # Signer is different from funder (proxy wallet setup)
            print(f"\n⚠️  Signer ({signer.address}) is different from funder ({funder})")
            print("This is a proxy wallet setup. Cannot set approval directly.")
            print("\n--- Manual Approval Required ---")
            print("You need to approve the exchange contracts from your Polymarket wallet.")
            print("\nOption 1: Use Polymarket UI")
            print("  - Go to polymarket.com -> Settings -> look for Token Approvals")
            print("\nOption 2: Use Polygonscan (if wallet supports WalletConnect)")
            print(f"  1. Go to: https://polygonscan.com/address/{CTF_ADDRESS}#writeContract")
            print("  2. Connect your Polymarket wallet")
            print("  3. Find 'setApprovalForAll' function")
            print(f"  4. Set operator: {NEG_RISK_CTF_EXCHANGE}")
            print("  5. Set approved: true")
            print("  6. Click 'Write' and confirm")
            print(f"\n  Repeat for CTF_EXCHANGE: {CTF_EXCHANGE}")
            
            return 1
            
    except ImportError:
        print("ERROR: web3 not installed. Run: pip install web3")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

