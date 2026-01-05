#!/usr/bin/env python3
"""Set USDC allowance for Polymarket CLOB contract.

This approves the CLOB to spend unlimited USDC from your wallet.
"""

import sys
from web3 import Web3
from eth_account import Account

# Polygon mainnet RPC
POLYGON_RPC = "https://polygon-rpc.com"

# Contract addresses on Polygon
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC on Polygon
CLOB_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Polymarket CLOB Exchange

# Max uint256 (unlimited allowance)
MAX_UINT256 = 2**256 - 1

# ERC20 approve function ABI
ERC20_ABI = [
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function"
    }
]


def main():
    """Set unlimited USDC allowance for Polymarket CLOB."""
    
    # Load config
    import os
    from pathlib import Path
    
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
                env_vars[key] = value
    
    private_key = env_vars.get('POLYBOT_POLYGON_PRIVATE_KEY')
    if not private_key:
        print("ERROR: POLYBOT_POLYGON_PRIVATE_KEY not found in .env")
        sys.exit(1)
    
    # Remove 0x prefix if present
    if private_key.startswith('0x'):
        private_key = private_key[2:]
    
    # Connect to Polygon
    print("Connecting to Polygon...")
    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    
    if not w3.is_connected():
        print("ERROR: Could not connect to Polygon RPC")
        sys.exit(1)
    
    print(f"Connected! Chain ID: {w3.eth.chain_id}")
    
    # Get account
    account = Account.from_key(private_key)
    wallet_address = account.address
    
    print(f"Wallet address: {wallet_address}")
    
    # Get USDC contract
    usdc = w3.eth.contract(
        address=Web3.to_checksum_address(USDC_ADDRESS),
        abi=ERC20_ABI
    )
    
    # Check current allowance
    print(f"\nChecking current USDC allowance for CLOB...")
    current_allowance = usdc.functions.allowance(
        Web3.to_checksum_address(wallet_address),
        Web3.to_checksum_address(CLOB_ADDRESS)
    ).call()
    
    print(f"Current allowance: {current_allowance / 1e6:.2f} USDC")
    
    if current_allowance >= MAX_UINT256 // 2:
        print("✓ Allowance is already set to unlimited!")
        return
    
    # Build approve transaction
    print(f"\nSetting unlimited allowance...")
    print(f"Approving CLOB contract: {CLOB_ADDRESS}")
    print(f"Amount: UNLIMITED (2^256 - 1)")
    
    # Get nonce
    nonce = w3.eth.get_transaction_count(wallet_address)
    
    # Build transaction
    txn = usdc.functions.approve(
        Web3.to_checksum_address(CLOB_ADDRESS),
        MAX_UINT256
    ).build_transaction({
        'from': wallet_address,
        'nonce': nonce,
        'gas': 100000,
        'maxFeePerGas': w3.eth.gas_price * 2,
        'maxPriorityFeePerGas': w3.to_wei(50, 'gwei'),
        'chainId': 137
    })
    
    print(f"\nTransaction details:")
    print(f"  Gas limit: {txn['gas']}")
    print(f"  Gas price: {w3.from_wei(txn['maxFeePerGas'], 'gwei'):.2f} gwei")
    
    # Sign transaction
    signed_txn = w3.eth.account.sign_transaction(txn, private_key)
    
    # Send transaction
    print(f"\nSending transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    
    print(f"Transaction hash: {tx_hash.hex()}")
    print(f"Waiting for confirmation...")
    
    # Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    
    if receipt['status'] == 1:
        print(f"\n✓ SUCCESS! Allowance set to unlimited.")
        print(f"  Block: {receipt['blockNumber']}")
        print(f"  Gas used: {receipt['gasUsed']}")
        print(f"\nYou can now trade unlimited USDC on Polymarket!")
    else:
        print(f"\n✗ FAILED! Transaction reverted.")
        print(f"  Receipt: {receipt}")
        sys.exit(1)


if __name__ == "__main__":
    main()

