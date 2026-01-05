#!/usr/bin/env python3
"""Refresh expired CLOB API credentials."""

import os
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from dotenv import load_dotenv, set_key

# Load current env
load_dotenv()

USER_ADDRESS = os.getenv("POLYBOT_USER_ADDRESS")
FUNDER_ADDRESS = os.getenv("POLYBOT_FUNDER_ADDRESS") 
PRIVATE_KEY = os.getenv("POLYBOT_POLYGON_PRIVATE_KEY")
CHAIN_ID = int(os.getenv("POLYBOT_CHAIN_ID", "137"))
SIGNATURE_TYPE = int(os.getenv("POLYBOT_SIGNATURE_TYPE", "1"))

print(f"🔑 Generating new CLOB credentials...")
print(f"   User: {USER_ADDRESS[:10]}...")
print(f"   Funder: {FUNDER_ADDRESS[:10]}...")
print(f"   Chain: {CHAIN_ID}")
print(f"   Sig Type: {SIGNATURE_TYPE}")

# Create client to derive credentials
client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=CHAIN_ID,
    key=PRIVATE_KEY,
    signature_type=SIGNATURE_TYPE,
    funder=FUNDER_ADDRESS,
)

# Derive L2 credentials
print("\n📝 Deriving new credentials...")
creds = client.create_or_derive_api_creds()

print(f"\n✅ New credentials generated!")
print(f"   API Key: {creds.api_key[:20]}...")
print(f"   Secret:  {creds.api_secret[:20]}...")
print(f"   Pass:    {creds.api_passphrase[:20]}...")

# Update .env file
env_path = ".env"
set_key(env_path, "POLYBOT_CLOB_API_KEY", creds.api_key)
set_key(env_path, "POLYBOT_CLOB_API_SECRET", creds.api_secret)
set_key(env_path, "POLYBOT_CLOB_PASSPHRASE", creds.api_passphrase)

print(f"\n💾 Updated {env_path}")
print("\n🚀 Ready to restart bot!")

