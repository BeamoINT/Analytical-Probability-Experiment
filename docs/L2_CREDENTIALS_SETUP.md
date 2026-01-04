# Polymarket L2 Credentials Setup Guide

Complete guide for generating and configuring Polymarket L2 CLOB credentials for the PolyB0T trading bot.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Option A: Generate via Polymarket UI](#option-a-generate-via-polymarket-ui-safest)
4. [Option B: Generate via CLI](#option-b-generate-via-cli-one-time)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Overview

To place orders and access authenticated endpoints on Polymarket, you need **L2 credentials**:
- `CLOB_API_KEY` (starts with `pk_`)
- `CLOB_API_SECRET` (starts with `sk_`)
- `CLOB_API_PASSPHRASE`

These credentials are created **once** and derived from your wallet signature.

### Important Security Notes

⚠️ **NEVER commit credentials to git** - they're automatically gitignored in `.env`  
⚠️ **Use a dedicated hot wallet** with minimal funds for automated trading  
⚠️ **Never share your API secret or passphrase**  
⚠️ **The bot NEVER needs your wallet private key permanently** - only for one-time L2 credential generation

---

## Prerequisites

1. **A Polymarket account** with a funded wallet
2. **Know your account type:**
   - **EOA** (MetaMask, standard wallet) → `SIGNATURE_TYPE=0`
   - **Proxy** (Polymarket proxy wallet) → `SIGNATURE_TYPE=1`
   - **Safe** (Gnosis Safe multi-sig) → `SIGNATURE_TYPE=2`
3. **Your wallet address** (the one shown in Polymarket UI)

---

## Option A: Generate via Polymarket UI (Safest)

If you have access to the **Builder/Developer** area in Polymarket:

### Steps:

1. Go to your **Polymarket profile**
2. Navigate to **Builder Settings → Keys**
3. Click **Create New**
4. Sign the message with your wallet
5. Copy the generated credentials:
   - `apiKey`
   - `secret`
   - `passphrase`

### Add to `.env`:

```bash
cp .env.example .env
```

Edit `.env`:

```env
POLYBOT_CLOB_API_KEY=pk_...
POLYBOT_CLOB_API_SECRET=sk_...
POLYBOT_CLOB_API_PASSPHRASE=...
```

✅ **Skip to [Verification](#verification)**

---

## Option B: Generate via CLI (One-Time)

If you **don't see Builder settings** or prefer CLI, use the official Python CLOB client.

### Step 1: Create Isolated Environment

Do **NOT** do this in your main bot repo:

```bash
mkdir ~/polymarket-l2-setup
cd ~/polymarket-l2-setup
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install py-clob-client

```bash
pip install py-clob-client web3
```

Verify:

```bash
python -c "from py_clob_client.client import ClobClient; print('OK')"
```

### Step 3: Export Wallet Key (Temporarily)

⚠️ This key is used **ONLY** to generate credentials, then deleted:

```bash
export POLY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
export POLY_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS
```

- `POLY_FUNDER_ADDRESS` is the wallet address shown in Polymarket
- For **EOA wallets**, this is the same as your wallet address
- For **proxy/Safe**, this might differ (check Polymarket docs)

### Step 4: Run Generation Script

Copy our provided script:

```bash
# From your bot repo directory
cp scripts/generate_l2_creds.py ~/polymarket-l2-setup/
cd ~/polymarket-l2-setup
python generate_l2_creds.py
```

The script will:
1. Ask for confirmation
2. Sign an L1 auth message
3. Display your L2 credentials

### Example Output:

```
======================================================================
✅ SUCCESS - L2 CREDENTIALS GENERATED
======================================================================

⚠️  SAVE THESE CREDENTIALS IMMEDIATELY ⚠️

POLYBOT_CLOB_API_KEY=pk_123abc...
POLYBOT_CLOB_API_SECRET=sk_456def...
POLYBOT_CLOB_API_PASSPHRASE=xyz789...

======================================================================
```

### Step 5: Delete Private Key (CRITICAL)

```bash
unset POLY_PRIVATE_KEY
unset POLY_FUNDER_ADDRESS
deactivate
cd ~
rm -rf ~/polymarket-l2-setup
```

✅ **Your machine should NEVER store the private key again**

---

## Configuration

### 1. Copy `.env.example`

```bash
cd /path/to/polyb0t
cp .env.example .env
```

### 2. Edit `.env`

Open `.env` and set:

```env
# Core Mode
POLYBOT_MODE=live
POLYBOT_DRY_RUN=true
POLYBOT_LOOP_INTERVAL_SECONDS=10

# Your Polymarket Wallet
POLYBOT_USER_ADDRESS=0xYOUR_WALLET_ADDRESS
POLYBOT_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS

# L2 Credentials (REQUIRED)
POLYBOT_CLOB_API_KEY=pk_...
POLYBOT_CLOB_API_SECRET=sk_...
POLYBOT_CLOB_API_PASSPHRASE=...

# Signature Type (MUST match your account)
POLYBOT_SIGNATURE_TYPE=0  # 0=EOA, 1=PROXY, 2=SAFE

# CLOB Endpoint
POLYBOT_CLOB_BASE_URL=https://clob.polymarket.com
POLYBOT_CHAIN_ID=137
```

### Signature Types Reference:

| Account Type | `SIGNATURE_TYPE` | `FUNDER_ADDRESS` |
|--------------|------------------|------------------|
| MetaMask EOA | `0` | Same as `USER_ADDRESS` |
| Magic/Email  | `1` | Check Polymarket docs |
| Gnosis Safe  | `2` | Safe contract address |

### Optional: Polygon RPC (for balance checks)

```env
POLYBOT_POLYGON_RPC_URL=https://polygon-rpc.com
# Or get a free one: https://www.alchemy.com/
```

---

## Verification

### 1. Test Authentication (No Trading)

```bash
poetry run polyb0t auth check
```

**Expected output:**

```
Auth OK (read-only).
Open orders: 0, positions: 0
```

**If it fails:**

```
Auth check FAILED: missing required CLOB credentials
```

→ Check your `.env` has all three credentials

### 2. Full System Check

```bash
poetry run polyb0t doctor
```

This tests:
- ✅ Gamma API connectivity
- ✅ CLOB orderbook access
- ✅ Polygon RPC balance (if configured)
- ✅ CLOB auth (read-only)

---

## Troubleshooting

### ❌ "Auth check FAILED: 401 Unauthorized"

**Causes:**
- Wrong API key/secret/passphrase
- Credentials expired or revoked
- Wrong CLOB endpoint URL

**Fix:**
1. Verify credentials in `.env`
2. Regenerate credentials if needed
3. Check `POLYBOT_CLOB_BASE_URL=https://clob.polymarket.com`

---

### ❌ "Invalid private key length"

**Cause:** Private key format wrong when generating

**Fix:**
- Ensure key starts with `0x`
- Should be 66 characters (0x + 64 hex chars)
- No spaces or quotes

```bash
# Correct format:
export POLY_PRIVATE_KEY=0x1234567890abcdef...
```

---

### ❌ "Signature type mismatch"

**Cause:** Wrong `SIGNATURE_TYPE` for your account

**Fix:**
1. Check your Polymarket account type:
   - **MetaMask?** → `SIGNATURE_TYPE=0`
   - **Magic/Email?** → `SIGNATURE_TYPE=1`
   - **Safe?** → `SIGNATURE_TYPE=2`
2. Update `.env`
3. Retry `polyb0t auth check`

---

### ❌ "Cannot find py-clob-client"

**Cause:** Library not installed

**Fix:**

```bash
pip install py-clob-client web3
```

Or add to your project:

```bash
poetry add py-clob-client
```

---

### ❌ Orders fail with "insufficient permissions"

**Cause:** L2 creds might be read-only or wrong account

**Fix:**
1. Verify credentials match the wallet in `USER_ADDRESS`
2. Check `FUNDER_ADDRESS` is correct
3. Regenerate credentials if needed

---

## Safety Checklist

Before you run live (with `DRY_RUN=false`), verify:

- ✅ L2 credentials generated once
- ✅ Private key deleted from environment
- ✅ `.env` contains **only L2 creds** (no private key)
- ✅ `.env` is gitignored
- ✅ `auth check` passes
- ✅ `POLYBOT_DRY_RUN=true`
- ✅ Using a dedicated hot wallet with small funds
- ✅ `POLYBOT_MAX_ORDER_USD` set to safe value (e.g. `1.0`)

---

## Next Steps

Once credentials are verified:

1. **Test in dry-run mode:**
   ```bash
   poetry run polyb0t run --live
   ```

2. **Check pending intents:**
   ```bash
   poetry run polyb0t intents list
   ```

3. **When ready for real execution:**
   ```env
   POLYBOT_DRY_RUN=false
   POLYBOT_MAX_ORDER_USD=1.0
   ```

4. **Approve one intent:**
   ```bash
   poetry run polyb0t intents approve <intent_id>
   ```

5. **Verify order appears in Polymarket UI**

---

## Additional Resources

- [Polymarket CLOB Authentication Docs](https://docs.polymarket.com/developers/CLOB/authentication)
- [Builder Profile & Keys](https://docs.polymarket.com/developers/builders/builder-profile)
- [Placing Your First Order](https://docs.polymarket.com/quickstart/first-order)
- [py-clob-client L1 Methods](https://docs.polymarket.com/developers/CLOB/clients/methods-l1)

---

## Support

If you have issues:

1. Run `poetry run polyb0t doctor` for diagnostics
2. Check logs: `tail -f live_run.log`
3. Verify all environment variables: `poetry run polyb0t status --json-output`
4. Consult Polymarket documentation

**Remember:** The bot is designed to be safe-by-default. All trading requires explicit approval unless `AUTO_APPROVE_INTENTS=true` (never use in production).

