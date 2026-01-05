#!/usr/bin/env python3
"""Delete old credentials and create fresh new ones."""

import os
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from dotenv import load_dotenv, set_key

# Load current env
load_dotenv()

PRIVATE_KEY = os.getenv("POLYBOT_POLYGON_PRIVATE_KEY")
FUNDER_ADDRESS = os.getenv("POLYBOT_FUNDER_ADDRESS") 
CHAIN_ID = int(os.getenv("POLYBOT_CHAIN_ID", "137"))
SIGNATURE_TYPE = int(os.getenv("POLYBOT_SIGNATURE_TYPE", "1"))

print(f"🗑️  Deleting old credentials and creating new ones...")
print(f"   Funder: {FUNDER_ADDRESS[:10]}...")
print(f"   Chain: {CHAIN_ID}")
print(f"   Sig Type: {SIGNATURE_TYPE}")

# Create client
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=CHAIN_ID,
    key=PRIVATE_KEY,
    signature_type=SIGNATURE_TYPE,
    funder=FUNDER_ADDRESS,
)

# Try to delete old credentials first
try:
    print("\n🗑️  Attempting to delete old credentials...")
    client.delete_api_key()
    print("✅ Old credentials deleted")
except Exception as e:
    print(f"ℹ️  No old credentials to delete (or delete failed): {e}")

# Create brand new credentials
print("\n📝 Creating brand new credentials...")
creds = client.create_or_derive_api_creds()

print(f"\n✅ New credentials created!")
print(f"   API Key: {creds.api_key[:20]}...")
print(f"   Secret:  {creds.api_secret[:20]}...")
print(f"   Pass:    {creds.api_passphrase[:20]}...")

# Update .env file
env_path = ".env"
set_key(env_path, "POLYBOT_CLOB_API_KEY", creds.api_key)
set_key(env_path, "POLYBOT_CLOB_API_SECRET", creds.api_secret)
set_key(env_path, "POLYBOT_CLOB_PASSPHRASE", creds.api_passphrase)

print(f"\n💾 Updated {env_path}")

# Test the new credentials
print("\n🧪 Testing new credentials...")
try:
    test_client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=CHAIN_ID,
        key=PRIVATE_KEY,
        creds=creds,
        signature_type=SIGNATURE_TYPE,
        funder=FUNDER_ADDRESS,
    )
    trades = test_client.get_trades()
    print(f"✅ SUCCESS! Fetched {len(trades)} trades")
    print("\n🚀 Ready to restart bot!")
except Exception as e:
    print(f"❌ TEST FAILED: {e}")
    print("⚠️  Credentials created but may not be working yet")

