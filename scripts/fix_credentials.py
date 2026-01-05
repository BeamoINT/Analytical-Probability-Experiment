#!/usr/bin/env python3
"""Fix credential derivation for proxy mode by using correct signer address."""

import os
from py_clob_client.client import ClobClient
from web3 import Web3
from dotenv import load_dotenv, set_key

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

print(f"🔍 Wallet Configuration:")
print(f"   Signer (EOA):     {signer_address}")
print(f"   Funder (Proxy):   {FUNDER_ADDRESS}")
print(f"   Signature Type:   {SIGNATURE_TYPE}")
print(f"   Chain ID:         {CHAIN_ID}")

if SIGNATURE_TYPE == 1:
    print(f"\n✅ Proxy mode detected - using signer address for credential derivation")
    
    # For proxy mode, we need to pass the signer as a separate parameter
    # The ClobClient in proxy mode expects:
    # - key: private key of the signer
    # - funder: the proxy wallet address
    # - creds_remember_me: True (to derive/cache credentials)
    
    print(f"\n📝 Creating credentials with correct configuration...")
    
    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=CHAIN_ID,
        key=PRIVATE_KEY,
        signature_type=SIGNATURE_TYPE,
        funder=FUNDER_ADDRESS,
        creds_remember_me=True,  # This will derive and cache credentials
    )
    
    # The credentials are now stored internally in the client
    # We need to derive them explicitly
    try:
        print(f"🔐 Deriving API credentials...")
        creds = client.create_or_derive_api_creds()
        
        print(f"\n✅ Credentials derived successfully!")
        print(f"   API Key: {creds.api_key[:20]}...")
        print(f"   Secret:  {creds.api_secret[:20]}...")
        print(f"   Pass:    {creds.api_passphrase[:20]}...")
        
        # Update .env file
        env_path = ".env"
        set_key(env_path, "POLYBOT_CLOB_API_KEY", creds.api_key)
        set_key(env_path, "POLYBOT_CLOB_API_SECRET", creds.api_secret)
        set_key(env_path, "POLYBOT_CLOB_PASSPHRASE", creds.api_passphrase)
        
        print(f"\n💾 Updated {env_path}")
        
        # Test the credentials
        print(f"\n🧪 Testing credentials...")
        from py_clob_client.clob_types import ApiCreds
        
        test_client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=CHAIN_ID,
            key=PRIVATE_KEY,
            creds=ApiCreds(
                api_key=creds.api_key,
                api_secret=creds.api_secret,
                api_passphrase=creds.api_passphrase,
            ),
            signature_type=SIGNATURE_TYPE,
            funder=FUNDER_ADDRESS,
        )
        
        # Try fetching trades
        trades = test_client.get_trades()
        print(f"✅ SUCCESS! Authenticated and fetched {len(trades)} trades")
        
        # Try fetching markets to double-check
        markets = test_client.get_markets()
        print(f"✅ Can also fetch {len(markets)} markets")
        
        print(f"\n🚀 Credentials are working! Ready to restart bot.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"\n⚠️  Not in proxy mode - using standard credential derivation")

