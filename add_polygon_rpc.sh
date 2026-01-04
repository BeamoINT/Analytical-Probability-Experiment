#!/bin/bash

# Quick script to add Polygon RPC URL to .env

echo ""
echo "============================================================"
echo "  Add Polygon RPC URL to .env"
echo "============================================================"
echo ""
echo "Choose an option:"
echo ""
echo "1) Quick Setup (Public RPC)"
echo "   - Uses: https://polygon-rpc.com"
echo "   - Pros: Works immediately, no signup"
echo "   - Cons: Public endpoint, may be slower"
echo ""
echo "2) Better Setup (Free Alchemy)"
echo "   - Get free key from: https://www.alchemy.com/"
echo "   - Pros: Fast, reliable, 300M req/month"
echo "   - Cons: Requires 5min signup"
echo ""
echo "3) Custom RPC URL"
echo "   - Enter your own RPC endpoint"
echo ""
read -p "Enter choice (1/2/3): " choice

ENV_FILE=".env"

case $choice in
  1)
    RPC_URL="https://polygon-rpc.com"
    echo ""
    echo "Using public RPC: $RPC_URL"
    ;;
  2)
    echo ""
    echo "Go to: https://www.alchemy.com/"
    echo "1. Sign up (free)"
    echo "2. Create app: Network = Polygon Mainnet"
    echo "3. Copy HTTP URL"
    echo ""
    read -p "Paste Alchemy HTTP URL: " RPC_URL
    ;;
  3)
    echo ""
    read -p "Enter your Polygon RPC URL: " RPC_URL
    ;;
  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

# Validate URL format
if [[ ! "$RPC_URL" =~ ^https?:// ]]; then
    echo "❌ Error: URL must start with http:// or https://"
    exit 1
fi

# Add to .env
if grep -q "^POLYBOT_POLYGON_RPC_URL=" "$ENV_FILE" 2>/dev/null; then
    # Update existing
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|^POLYBOT_POLYGON_RPC_URL=.*|POLYBOT_POLYGON_RPC_URL=$RPC_URL|" "$ENV_FILE"
    else
        sed -i "s|^POLYBOT_POLYGON_RPC_URL=.*|POLYBOT_POLYGON_RPC_URL=$RPC_URL|" "$ENV_FILE"
    fi
    echo ""
    echo "✅ Updated POLYBOT_POLYGON_RPC_URL in $ENV_FILE"
else
    # Add new
    echo "" >> "$ENV_FILE"
    echo "# Polygon RPC for on-chain balance" >> "$ENV_FILE"
    echo "POLYBOT_POLYGON_RPC_URL=$RPC_URL" >> "$ENV_FILE"
    echo ""
    echo "✅ Added POLYBOT_POLYGON_RPC_URL to $ENV_FILE"
fi

echo ""
echo "============================================================"
echo "  Verifying Setup"
echo "============================================================"
echo ""
echo "Running: python3 -m polyb0t.cli.main doctor"
echo ""

python3 -m polyb0t.cli.main doctor

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "============================================================"
    echo "🎉 SUCCESS! All checks passed."
    echo "============================================================"
    echo ""
    echo "Your bot is ready! Try these commands:"
    echo ""
    echo "  polyb0t status     # Check current status"
    echo "  polyb0t run --live # Run in dry-run mode"
    echo ""
else
    echo "============================================================"
    echo "⚠️  Some checks failed. Review output above."
    echo "============================================================"
    echo ""
    echo "If Polygon RPC check still fails:"
    echo "  - Check RPC URL is correct"
    echo "  - Try a different RPC provider"
    echo "  - Check network connectivity"
    echo ""
fi

