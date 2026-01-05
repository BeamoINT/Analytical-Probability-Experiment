#!/usr/bin/env python3
"""Simple credential test - check if the issue is timing."""

import os
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from web3 import Web3
from dotenv import load_dotenv, set_key
import time

# Load current env
load_dotenv()

PRIVATE_KEY = os.getenv("POLYBOT_POLYGON_PRIVATE_KEY")
FUNDER_ADDRESS = os.getenv("POLYBOT_FUNDER_ADDRESS") 
CHAIN_ID = int(os.getenv("POLYBOT_CHAIN_ID", "137"))
SIGNATURE_TYPE = int(os.getenv("POLYBOT_SIGNATURE_TYPE", "1"))

# Derive the signer address
w3 = Web3()
account = w3.eth.account.from_key(PRIVATE_KEY)
signer_address = account.address

print(f"🔍 Wallet Info:")
print(f"   Signer: {signer_address}")
print(f"   Funder: {FUNDER_ADDRESS}")

# Create client
print(f"\n📝 Creating CLOB client...")
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=CHAIN_ID,
    key=PRIVATE_KEY,
    signature_type=SIGNATURE_TYPE,
    funder=FUNDER_ADDRESS,
)

# Try public endpoint first (no auth needed)
print(f"\n🧪 Test 1: Public endpoint (no auth)...")
try:
    markets = client.get_markets()
    print(f"   ✅ Can fetch markets ({len(markets)})")
except Exception as e:
    print(f"   ❌ Public endpoint failed: {e}")
    exit(1)

# Derive credentials
print(f"\n🔐 Deriving credentials...")
creds = client.create_or_derive_api_creds()
print(f"   API Key: {creds.api_key[:30]}...")
print(f"   Secret:  {creds.api_secret[:30]}...")

# Save to env
env_path = ".env"
set_key(env_path, "POLYBOT_CLOB_API_KEY", creds.api_key)
set_key(env_path, "POLYBOT_CLOB_API_SECRET", creds.api_secret)
set_key(env_path, "POLYBOT_CLOB_PASSPHRASE", creds.api_passphrase)
print(f"   💾 Saved to {env_path}")

# Wait a moment for credentials to activate
print(f"\n⏳ Waiting 3 seconds for credentials to activate...")
time.sleep(3)

# Test authenticated endpoint
print(f"\n🧪 Test 2: Authenticated endpoint...")
attempt = 1
max_attempts = 3

while attempt <= max_attempts:
    try:
        print(f"   Attempt {attempt}/{max_attempts}...")
        test_client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=CHAIN_ID,
            key=PRIVATE_KEY,
            creds=creds,
            signature_type=SIGNATURE_TYPE,
            funder=FUNDER_ADDRESS,
        )
        
        trades = test_client.get_trades()
        print(f"   ✅ SUCCESS! Fetched {len(trades)} trades")
        print(f"\n🚀 Credentials are working! Bot can be restarted.")
        break
        
    except Exception as e:
        print(f"   ❌ Attempt {attempt} failed: {e}")
        if attempt < max_attempts:
            wait_time = 5 * attempt
            print(f"   ⏳ Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        attempt += 1

if attempt > max_attempts:
    print(f"\n⚠️  All attempts failed.")
    print(f"💡 Possible fixes:")
    print(f"   1. The bot was working earlier - this might be temporary")
    print(f"   2. Try restarting bot anyway - credentials may work once loaded fresh")
    print(f"   3. Check if you can login to polymarket.com in browser")

