#!/bin/bash
# ==============================================================================
# POLYMARKET L2 CREDENTIAL GENERATOR
# ==============================================================================
# This script helps you generate L2 credentials for Polymarket CLOB API
#
# ⚠️  SECURITY REQUIREMENTS:
# - Use a DEDICATED hot wallet with minimal funds
# - Never use your main wallet
# - The private key is used ONLY to generate credentials, then deleted
# ==============================================================================

set -e  # Exit on error

cd "$(dirname "$0")"

echo ""
echo "=========================================================================="
echo "POLYMARKET L2 CREDENTIAL GENERATOR"
echo "=========================================================================="
echo ""
echo "⚠️  SECURITY WARNING:"
echo "This script will ask for your wallet PRIVATE KEY to generate L2 credentials."
echo ""
echo "- Use a DEDICATED hot wallet with minimal funds"
echo "- NEVER use your main wallet"
echo "- The private key is used ONLY to sign the L1 auth message"
echo "- After generation, the private key is automatically deleted"
echo "=========================================================================="
echo ""

# Check if user wants to continue
read -p "Do you want to continue? (yes/no): " CONTINUE
if [[ ! "$CONTINUE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=========================================================================="
echo "STEP 1: Enter Your Credentials"
echo "=========================================================================="
echo ""

# Get private key
echo "Enter your wallet PRIVATE KEY (it will be hidden):"
echo "Format: 0x... (66 characters)"
read -s POLY_PRIVATE_KEY
echo ""

# Get wallet address
echo "Enter your wallet ADDRESS:"
echo "Format: 0x... (42 characters)"
read POLY_FUNDER_ADDRESS
echo ""

# Validate inputs
if [[ ! "$POLY_PRIVATE_KEY" =~ ^0x[0-9a-fA-F]{64}$ ]]; then
    echo "❌ ERROR: Invalid private key format"
    echo "   Expected: 0x followed by 64 hexadecimal characters"
    exit 1
fi

if [[ ! "$POLY_FUNDER_ADDRESS" =~ ^0x[0-9a-fA-F]{40}$ ]]; then
    echo "❌ ERROR: Invalid address format"
    echo "   Expected: 0x followed by 40 hexadecimal characters"
    exit 1
fi

echo "=========================================================================="
echo "STEP 2: Generating L2 Credentials"
echo "=========================================================================="
echo ""
echo "🔐 Calling Polymarket CLOB to generate your API credentials..."
echo "    (This signs a message with your wallet)"
echo ""

# Export for the Python script
export POLY_PRIVATE_KEY
export POLY_FUNDER_ADDRESS

# Run the generation script
if python3 scripts/generate_l2_creds.py; then
    echo ""
    echo "=========================================================================="
    echo "✅ SUCCESS!"
    echo "=========================================================================="
    echo ""
    echo "Your credentials have been displayed above."
    echo ""
    echo "NEXT STEPS:"
    echo "1. Copy the three credentials (API_KEY, API_SECRET, API_PASSPHRASE)"
    echo "2. Open your .env file"
    echo "3. Add the credentials to .env (see example below)"
    echo "4. Save .env"
    echo "5. Run: python3 -m polyb0t.cli.main auth check"
    echo ""
    echo "Example .env entries:"
    echo "  POLYBOT_CLOB_API_KEY=pk_..."
    echo "  POLYBOT_CLOB_API_SECRET=sk_..."
    echo "  POLYBOT_CLOB_API_PASSPHRASE=..."
    echo "  POLYBOT_FUNDER_ADDRESS=$POLY_FUNDER_ADDRESS"
    echo "  POLYBOT_SIGNATURE_TYPE=0"
    echo ""
else
    echo ""
    echo "=========================================================================="
    echo "❌ GENERATION FAILED"
    echo "=========================================================================="
    echo ""
    echo "Please check the error message above."
    echo "Common issues:"
    echo "- Invalid private key or address"
    echo "- Network connectivity problems"
    echo "- CLOB API unavailable"
    echo ""
fi

# CRITICAL: Delete the private key from environment
unset POLY_PRIVATE_KEY
unset POLY_FUNDER_ADDRESS

echo "=========================================================================="
echo "🔒 Private key has been deleted from memory"
echo "=========================================================================="
echo ""

