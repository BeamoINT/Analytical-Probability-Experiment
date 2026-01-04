# Quick Start: L2 Credentials Setup

**Goal:** Generate Polymarket L2 credentials in 5 minutes and verify authentication.

---

## Prerequisites

- ✅ A Polymarket account with a funded wallet
- ✅ Your wallet's **private key** (temporarily, for credential generation only)
- ✅ Your wallet **address** (shown in Polymarket UI)

---

## Step 1: Install Dependencies

```bash
cd /path/to/polyb0t
poetry install
```

This installs `py-clob-client` and `web3` needed for credential generation.

---

## Step 2: Generate L2 Credentials (One-Time)

### Set Environment Variables (Temporary)

```bash
export POLY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
export POLY_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS
```

⚠️ **These will be deleted after generation**

### Run Generation Script

```bash
poetry run python scripts/generate_l2_creds.py
```

### Expected Output

```
======================================================================
✅ SUCCESS - L2 CREDENTIALS GENERATED
======================================================================

⚠️  SAVE THESE CREDENTIALS IMMEDIATELY ⚠️

POLYBOT_CLOB_API_KEY=pk_123abc...
POLYBOT_CLOB_API_SECRET=sk_456def...
POLYBOT_CLOB_API_PASSPHRASE=xyz789...
```

**Copy these three values immediately!**

### Delete Private Key (CRITICAL)

```bash
unset POLY_PRIVATE_KEY
unset POLY_FUNDER_ADDRESS
```

✅ **Your machine should never store the private key again**

---

## Step 3: Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Mode
POLYBOT_MODE=live
POLYBOT_DRY_RUN=true
POLYBOT_LOOP_INTERVAL_SECONDS=10

# Your Wallet
POLYBOT_USER_ADDRESS=0xYOUR_WALLET_ADDRESS
POLYBOT_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS

# L2 Credentials (paste from Step 2)
POLYBOT_CLOB_API_KEY=pk_123abc...
POLYBOT_CLOB_API_SECRET=sk_456def...
POLYBOT_CLOB_API_PASSPHRASE=xyz789...

# Signature Type (0=EOA for MetaMask)
POLYBOT_SIGNATURE_TYPE=0

# CLOB Endpoint
POLYBOT_CLOB_BASE_URL=https://clob.polymarket.com
POLYBOT_CHAIN_ID=137
```

---

## Step 4: Verify Authentication

```bash
poetry run polyb0t auth check
```

**Expected:**

```
Auth OK (read-only).
Open orders: 0, positions: 0
```

✅ **Success!** Your L2 credentials are working.

---

## Step 5: Full System Check

```bash
poetry run polyb0t doctor
```

This verifies:
- ✅ Gamma API connectivity
- ✅ CLOB orderbook access
- ✅ CLOB authentication

---

## Next Steps

### Test in Dry-Run Mode

```bash
poetry run polyb0t run --live
```

The bot will:
- Generate trade intents
- **NOT execute** (dry-run mode)
- Log all recommendations

### View Pending Intents

```bash
poetry run polyb0t intents list
```

### When Ready for Real Trading

1. **Update `.env`:**
   ```env
   POLYBOT_DRY_RUN=false
   POLYBOT_MAX_ORDER_USD=1.0
   ```

2. **Approve one intent:**
   ```bash
   poetry run polyb0t intents approve <intent_id>
   ```

3. **Verify in Polymarket UI** that the order appears

---

## Troubleshooting

### ❌ "Auth check FAILED: 401 Unauthorized"

- Verify credentials are correct in `.env`
- Check no extra spaces or quotes
- Regenerate credentials if needed

### ❌ "Cannot find py-clob-client"

```bash
poetry add py-clob-client web3
```

### ❌ "Invalid private key length"

- Ensure key starts with `0x`
- Should be 66 characters total
- No spaces or quotes

---

## Safety Checklist

Before going live:

- ✅ L2 credentials generated
- ✅ Private key deleted from environment
- ✅ `.env` contains **only L2 creds** (no private key)
- ✅ `.env` is gitignored
- ✅ `auth check` passes
- ✅ `POLYBOT_DRY_RUN=true`
- ✅ Using dedicated hot wallet with small funds
- ✅ `POLYBOT_MAX_ORDER_USD` set to safe value

---

## Full Documentation

For detailed explanations, troubleshooting, and advanced configuration:

📖 **[docs/L2_CREDENTIALS_SETUP.md](./L2_CREDENTIALS_SETUP.md)**

---

## Summary

You've successfully:
1. ✅ Generated L2 credentials
2. ✅ Configured `.env`
3. ✅ Verified authentication
4. ✅ Deleted private key

**Your bot is now ready for dry-run testing!**

