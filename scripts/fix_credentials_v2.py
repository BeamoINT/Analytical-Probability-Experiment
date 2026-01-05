#!/usr/bin/env python3
"""Fix credential derivation - try multiple methods."""

import os
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from py_clob_client.headers import create_level_1_headers
from web3 import Web3
from dotenv import load_dotenv, set_key
import time

# Load current env
load_dotenv()

PRIVATE_KEY = os.getenv("POLYBOT_POLYGON_PRIVATE_KEY")
FUNDER_ADDRESS = os.getenv("POLYBOT_FUNDER_ADDRESS") 
CHAIN_ID = int(os.getenv("POLYBOT_CHAIN_ID", "137"))
SIGNATURE_TYPE = int(os.getenv("POLYBOT_SIGNATURE_TYPE", "1"))

# Derive the signer address from private key
w3 = Web3()
account = w3.eth.account.from_key(PRIVATE_KEY)
signer_address = account.address

print(f"🔍 Configuration:")
print(f"   Signer:    {signer_address}")
print(f"   Funder:    {FUNDER_ADDRESS}")
print(f"   Sig Type:  {SIGNATURE_TYPE}")

print(f"\n📝 Method 1: Standard credential derivation...")
try:
    client1 = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=CHAIN_ID,
        key=PRIVATE_KEY,
        signature_type=SIGNATURE_TYPE,
        funder=FUNDER_ADDRESS,
    )
    
    creds1 = client1.create_or_derive_api_creds()
    print(f"✅ Generated credentials")
    print(f"   Key: {creds1.api_key[:30]}...")
    
    # Test immediately
    print(f"🧪 Testing method 1 credentials...")
    test1 = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=CHAIN_ID,
        key=PRIVATE_KEY,
        creds=creds1,
        signature_type=SIGNATURE_TYPE,
        funder=FUNDER_ADDRESS,
    )
    
    markets = test1.get_markets()
    print(f"   ✅ Public call works ({len(markets)} markets)")
    
    # Wait a moment
    time.sleep(2)
    
    trades = test1.get_trades()
    print(f"   ✅ AUTH WORKS! ({len(trades)} trades)")
    
    # Save these credentials
    env_path = ".env"
    set_key(env_path, "POLYBOT_CLOB_API_KEY", creds1.api_key)
    set_key(env_path, "POLYBOT_CLOB_API_SECRET", creds1.api_secret)
    set_key(env_path, "POLYBOT_CLOB_PASSPHRASE", creds1.api_passphrase)
    
    print(f"\n💾 Saved working credentials to {env_path}")
    print(f"🚀 Ready to restart bot!")
    
except Exception as e:
    print(f"❌ Method 1 failed: {e}")
    
    print(f"\n📝 Method 2: Try with explicit headers...")
    try:
        # Generate L1 signature headers
        timestamp = str(int(time.time() * 1000))
        headers = create_level_1_headers(
            private_key=PRIVATE_KEY,
            timestamp=timestamp,
            signature_type=SIGNATURE_TYPE,
            funder=FUNDER_ADDRESS,
        )
        print(f"Generated headers: {headers}")
        
    except Exception as e2:
        print(f"❌ Method 2 also failed: {e2}")
        
        print(f"\n💡 Possible issues:")
        print(f"   1. Credentials may need time to activate on Polymarket's end")
        print(f"   2. The API server might be having issues")
        print(f"   3. Try waiting 1-2 minutes and running bot again")

