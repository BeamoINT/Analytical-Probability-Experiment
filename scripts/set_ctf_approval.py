#!/usr/bin/env python3
"""Set ERC-1155 approval for ALL Polymarket outcome tokens (enables selling).

This calls setApprovalForAll on the CTF contract to approve the CLOB exchange
to transfer ANY outcome token you hold. This is a one-time operation.
"""

import sys
from pathlib import Path
from web3 import Web3
from eth_account import Account

# Polygon mainnet RPC
POLYGON_RPC = "https://polygon-rpc.com"

# Contract addresses on Polygon (Polymarket)
# CTF = Conditional Tokens Framework (ERC-1155 that holds outcome tokens)
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# The CLOB exchange needs approval to transfer tokens
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# For Neg Risk markets (multi-outcome), there's a separate adapter
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# ERC-1155 setApprovalForAll ABI
ERC1155_ABI = [
    {
        "inputs": [
            {"name": "operator", "type": "address"},
            {"name": "approved", "type": "bool"}
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "operator", "type": "address"}
        ],
        "name": "isApprovedForAll",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]


def approve_operator(w3, ctf_contract, wallet_address, private_key, operator_address, operator_name):
    """Approve an operator if not already approved."""
    print(f"\nChecking approval for {operator_name}...")
    print(f"  Operator: {operator_address}")
    
    is_approved = ctf_contract.functions.isApprovedForAll(
        Web3.to_checksum_address(wallet_address),
        Web3.to_checksum_address(operator_address)
    ).call()
    
    if is_approved:
        print(f"  ✓ Already approved!")
        return True
    
    print(f"  ✗ Not approved - sending approval transaction...")
    
    # Get nonce
    nonce = w3.eth.get_transaction_count(wallet_address)
    
    # Build transaction
    base_gas_price = w3.eth.gas_price
    
    txn = ctf_contract.functions.setApprovalForAll(
        Web3.to_checksum_address(operator_address),
        True
    ).build_transaction({
        'from': wallet_address,
        'nonce': nonce,
        'gas': 100000,
        'maxFeePerGas': min(base_gas_price * 2, w3.to_wei(100, 'gwei')),
        'maxPriorityFeePerGas': w3.to_wei(35, 'gwei'),
        'chainId': 137
    })
    
    # Sign and send
    signed_txn = w3.eth.account.sign_transaction(txn, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    
    print(f"  Transaction sent: {tx_hash.hex()}")
    print(f"  Waiting for confirmation...")
    
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    
    if receipt['status'] == 1:
        print(f"  ✓ SUCCESS! Approved {operator_name}")
        print(f"    Block: {receipt['blockNumber']}, Gas used: {receipt['gasUsed']}")
        return True
    else:
        print(f"  ✗ FAILED! Transaction reverted")
        return False


def main():
    """Set approval for all Polymarket exchange contracts."""
    
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
    if not private_key:
        print("ERROR: POLYBOT_POLYGON_PRIVATE_KEY not found in .env")
        sys.exit(1)
    
    # Remove 0x prefix if present for account derivation
    pk_for_account = private_key[2:] if private_key.startswith('0x') else private_key
    
    print("=" * 60)
    print("POLYMARKET CTF TOKEN APPROVAL (setApprovalForAll)")
    print("=" * 60)
    print()
    print("This will approve the Polymarket exchange contracts to transfer")
    print("ANY outcome tokens you hold, enabling the bot to SELL positions.")
    print()
    
    # Connect to Polygon
    print("Connecting to Polygon...")
    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    
    if not w3.is_connected():
        print("ERROR: Could not connect to Polygon RPC")
        sys.exit(1)
    
    print(f"Connected! Chain ID: {w3.eth.chain_id}")
    
    # Get account
    account = Account.from_key(pk_for_account)
    wallet_address = account.address
    
    print(f"Wallet address: {wallet_address}")
    
    # Check POL balance for gas
    balance = w3.eth.get_balance(wallet_address)
    print(f"POL balance: {w3.from_wei(balance, 'ether'):.4f} POL")
    
    if balance < w3.to_wei(0.01, 'ether'):
        print("\n⚠️  WARNING: Low POL balance! You need POL for gas fees.")
        print("   Deposit some POL (MATIC) to your wallet.")
    
    # Get CTF contract
    ctf = w3.eth.contract(
        address=Web3.to_checksum_address(CTF_ADDRESS),
        abi=ERC1155_ABI
    )
    
    print(f"\nCTF Contract: {CTF_ADDRESS}")
    
    # Approve all necessary operators
    success_count = 0
    operators = [
        (CTF_EXCHANGE, "CTF Exchange (main)"),
        (NEG_RISK_CTF_EXCHANGE, "Neg Risk CTF Exchange"),
        (NEG_RISK_ADAPTER, "Neg Risk Adapter"),
    ]
    
    for operator_addr, operator_name in operators:
        try:
            if approve_operator(w3, ctf, wallet_address, private_key, operator_addr, operator_name):
                success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print()
    print("=" * 60)
    if success_count == len(operators):
        print("✓ ALL APPROVALS SET! The bot can now SELL any position.")
    else:
        print(f"⚠️  {success_count}/{len(operators)} approvals set.")
        print("   Some sells may still fail. Try again or approve manually.")
    print("=" * 60)


if __name__ == "__main__":
    main()

