#!/usr/bin/env python3
"""Speed up the stuck USDC allowance transaction by replacing it with higher gas.

This sends a replacement transaction with the same nonce but higher gas price,
effectively canceling the old one and making the new one confirm faster.
"""

import sys
from web3 import Web3
from eth_account import Account

# Polygon mainnet RPC
POLYGON_RPC = "https://polygon-rpc.com"

# Contract addresses on Polygon
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CLOB_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

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
    }
]


def main():
    """Replace stuck transaction with higher gas price."""
    
    # Load config
    import os
    from pathlib import Path
    
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print("ERROR: .env file not found")
        sys.exit(1)
    
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
    
    # Get current nonce (this will be the same as the stuck transaction)
    current_nonce = w3.eth.get_transaction_count(wallet_address)
    pending_nonce = w3.eth.get_transaction_count(wallet_address, 'pending')
    
    print(f"\nCurrent nonce: {current_nonce}")
    print(f"Pending nonce: {pending_nonce}")
    
    if pending_nonce > current_nonce:
        print(f"Found {pending_nonce - current_nonce} pending transaction(s)")
        print("Sending replacement transaction with MUCH higher gas...")
        nonce = current_nonce  # Use the same nonce to replace
    else:
        print("No pending transactions found. Sending with current nonce...")
        nonce = current_nonce
    
    # Build replacement transaction with MUCH higher gas
    print(f"\nSetting unlimited USDC allowance...")
    print(f"Using aggressive gas price for fast confirmation")
    
    # Get current gas price and multiply by 3 for fast confirmation
    current_gas = w3.eth.gas_price
    max_fee = min(current_gas * 3, w3.to_wei(500, 'gwei'))  # Cap at 500 gwei
    priority_fee = w3.to_wei(100, 'gwei')  # High priority
    
    txn = usdc.functions.approve(
        Web3.to_checksum_address(CLOB_ADDRESS),
        MAX_UINT256
    ).build_transaction({
        'from': wallet_address,
        'nonce': nonce,
        'gas': 100000,
        'maxFeePerGas': max_fee,
        'maxPriorityFeePerGas': priority_fee,
        'chainId': 137
    })
    
    print(f"\nTransaction details:")
    print(f"  Nonce: {nonce}")
    print(f"  Gas limit: {txn['gas']}")
    print(f"  Max fee: {w3.from_wei(txn['maxFeePerGas'], 'gwei'):.2f} gwei")
    print(f"  Priority fee: {w3.from_wei(txn['maxPriorityFeePerGas'], 'gwei'):.2f} gwei")
    
    # Estimate cost
    max_cost = txn['gas'] * txn['maxFeePerGas']
    print(f"  Max cost: {w3.from_wei(max_cost, 'ether'):.4f} POL")
    
    # Check balance
    balance = w3.eth.get_balance(wallet_address)
    print(f"  Your balance: {w3.from_wei(balance, 'ether'):.4f} POL")
    
    if balance < max_cost:
        print(f"\n✗ ERROR: Insufficient POL for gas!")
        print(f"  Need: {w3.from_wei(max_cost, 'ether'):.4f} POL")
        print(f"  Have: {w3.from_wei(balance, 'ether'):.4f} POL")
        sys.exit(1)
    
    # Sign and send
    signed_txn = w3.eth.account.sign_transaction(txn, private_key)
    
    print(f"\nSending replacement transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    
    print(f"Transaction hash: {tx_hash.hex()}")
    print(f"View on PolygonScan: https://polygonscan.com/tx/{tx_hash.hex()}")
    print(f"\nWaiting for confirmation (should be ~5-10 seconds)...")
    
    # Wait for receipt
    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        
        if receipt['status'] == 1:
            print(f"\n✓ SUCCESS! Transaction confirmed!")
            print(f"  Block: {receipt['blockNumber']}")
            print(f"  Gas used: {receipt['gasUsed']}")
            print(f"  Actual cost: {w3.from_wei(receipt['gasUsed'] * receipt['effectiveGasPrice'], 'ether'):.6f} POL")
            print(f"\n✓✓ USDC allowance is now set to UNLIMITED!")
            print(f"✓✓ The bot can now trade on Polymarket!")
        else:
            print(f"\n✗ FAILED! Transaction reverted.")
            sys.exit(1)
    except Exception as e:
        print(f"\nTransaction sent but confirmation timed out.")
        print(f"Check status at: https://polygonscan.com/tx/{tx_hash.hex()}")
        print(f"It should still confirm within a few minutes.")


if __name__ == "__main__":
    main()

