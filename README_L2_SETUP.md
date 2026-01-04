# Polymarket L2 Credentials Setup - Complete Guide

This guide implements the **exact, minimal, one-time CLI flow** to generate Polymarket L2 credentials for the PolyB0T trading bot.

---

## 🚀 Quick Start (5 Minutes)

**Goal:** Generate L2 credentials and verify authentication.

### 1. Install Dependencies

```bash
poetry install
```

### 2. Generate L2 Credentials

```bash
# Set temporary environment variables
export POLY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
export POLY_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS

# Run generation script
poetry run python scripts/generate_l2_creds.py

# CRITICAL: Delete private key immediately
unset POLY_PRIVATE_KEY
unset POLY_FUNDER_ADDRESS
```

### 3. Configure `.env`

```bash
cp .env.example .env
```

Add your credentials to `.env`:

```env
POLYBOT_CLOB_API_KEY=pk_...
POLYBOT_CLOB_API_SECRET=sk_...
POLYBOT_CLOB_API_PASSPHRASE=...
```

### 4. Verify

```bash
poetry run polyb0t auth check
```

✅ **Done!** See [docs/QUICKSTART_L2_SETUP.md](docs/QUICKSTART_L2_SETUP.md) for details.

---

## 📚 Documentation

### Essential Guides

1. **[QUICKSTART_L2_SETUP.md](docs/QUICKSTART_L2_SETUP.md)** - 5-minute setup guide
2. **[L2_CREDENTIALS_SETUP.md](docs/L2_CREDENTIALS_SETUP.md)** - Complete reference with troubleshooting
3. **[SIGNATURE_TYPES.md](docs/SIGNATURE_TYPES.md)** - Understanding signature types (EOA/Proxy/Safe)

### Quick Reference

| File | Purpose |
|------|---------|
| `.env.example` | Template configuration with all options |
| `scripts/generate_l2_creds.py` | One-time credential generation script |
| `polyb0t auth check` | Verify L2 credentials work |
| `polyb0t doctor` | Full system diagnostics |

---

## 🔐 Security Requirements

### ⚠️ Critical Safety Rules

1. **Use a dedicated hot wallet** with minimal funds for automated trading
2. **Never use your main wallet** for bot trading
3. **The bot NEVER needs your private key permanently** - only for one-time credential generation
4. **Delete the private key** immediately after generating L2 credentials
5. **Never commit `.env`** to version control (it's gitignored)

### What Gets Stored Where

| Credential | Storage | Purpose |
|------------|---------|---------|
| Private Key | **NOWHERE** (deleted after generation) | One-time L2 credential generation |
| L2 API Key | `.env` (gitignored) | Authenticate CLOB requests |
| L2 API Secret | `.env` (gitignored) | Sign CLOB requests (HMAC) |
| L2 Passphrase | `.env` (gitignored) | Additional auth factor |

---

## 🎯 Two Ways to Get L2 Credentials

### Option A: Polymarket UI (Safest)

If you have access to **Builder/Developer** settings:

1. Go to Polymarket → Profile → **Builder Settings → Keys**
2. Click **Create New**
3. Sign with your wallet
4. Copy: `apiKey`, `secret`, `passphrase`
5. Add to `.env`

✅ **No private key needed on your machine**

### Option B: CLI Generation (This Guide)

Use the official `py-clob-client` library:

1. Run `scripts/generate_l2_creds.py` (requires private key temporarily)
2. Save credentials to `.env`
3. Delete private key

✅ **Works for all account types**

---

## 📋 Configuration Checklist

Before running the bot:

- [ ] L2 credentials generated
- [ ] Private key deleted from environment
- [ ] `.env` file created and configured
- [ ] `POLYBOT_SIGNATURE_TYPE` matches your wallet type
- [ ] `POLYBOT_FUNDER_ADDRESS` set correctly
- [ ] `auth check` passes
- [ ] `POLYBOT_DRY_RUN=true` (for initial testing)
- [ ] `POLYBOT_MAX_ORDER_USD` set to safe value (e.g., 1.0)

---

## 🔧 Configuration Reference

### Required Environment Variables

```env
# Mode
POLYBOT_MODE=live
POLYBOT_DRY_RUN=true
POLYBOT_LOOP_INTERVAL_SECONDS=10

# Account
POLYBOT_USER_ADDRESS=0xYOUR_WALLET_ADDRESS
POLYBOT_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS

# L2 Credentials (REQUIRED)
POLYBOT_CLOB_API_KEY=pk_...
POLYBOT_CLOB_API_SECRET=sk_...
POLYBOT_CLOB_API_PASSPHRASE=...

# Signature Type (REQUIRED)
POLYBOT_SIGNATURE_TYPE=0  # 0=EOA, 1=PROXY, 2=SAFE

# CLOB Endpoint
POLYBOT_CLOB_BASE_URL=https://clob.polymarket.com
POLYBOT_CHAIN_ID=137
```

### Signature Types

| Type | Value | Wallet Type |
|------|-------|-------------|
| EOA | `0` | MetaMask, Ledger, standard wallets |
| POLY_PROXY | `1` | Magic.link, email login |
| POLY_GNOSIS_SAFE | `2` | Gnosis Safe multi-sig |

See [docs/SIGNATURE_TYPES.md](docs/SIGNATURE_TYPES.md) for details.

---

## 🧪 Testing & Verification

### 1. Authentication Check

```bash
poetry run polyb0t auth check
```

Expected:
```
Auth OK (read-only).
Open orders: 0, positions: 0
```

### 2. Full System Check

```bash
poetry run polyb0t doctor
```

Verifies:
- ✅ Gamma API connectivity
- ✅ CLOB orderbook access
- ✅ Polygon RPC (if configured)
- ✅ CLOB authentication

### 3. Dry-Run Test

```bash
poetry run polyb0t run --live
```

The bot will:
- Generate trade intents
- **NOT execute** (dry-run mode)
- Log all recommendations

### 4. View Intents

```bash
poetry run polyb0t intents list
```

---

## 🚦 Going Live

### Step 1: Update Configuration

```env
POLYBOT_DRY_RUN=false
POLYBOT_MAX_ORDER_USD=1.0  # Start small!
```

### Step 2: Approve One Intent

```bash
# List pending intents
poetry run polyb0t intents list

# Approve specific intent
poetry run polyb0t intents approve <intent_id>
```

### Step 3: Verify in Polymarket UI

Check that the order appears in your Polymarket account.

### Step 4: Monitor

```bash
# Watch logs
tail -f live_run.log

# Check status
poetry run polyb0t status
```

---

## 🐛 Troubleshooting

### ❌ "Auth check FAILED: 401 Unauthorized"

**Fix:**
1. Verify credentials in `.env` are correct
2. Check no extra spaces or quotes
3. Regenerate credentials if needed

### ❌ "Cannot find py-clob-client"

**Fix:**
```bash
poetry add py-clob-client web3
```

### ❌ "Invalid private key length"

**Fix:**
- Ensure key starts with `0x`
- Should be 66 characters (0x + 64 hex)
- No spaces or quotes

### ❌ "Signature verification failed"

**Fix:**
1. Check `POLYBOT_SIGNATURE_TYPE` matches your wallet type
2. See [docs/SIGNATURE_TYPES.md](docs/SIGNATURE_TYPES.md)
3. Verify `POLYBOT_FUNDER_ADDRESS` is correct

### ❌ Orders fail but auth passes

**Possible causes:**
1. Wrong `FUNDER_ADDRESS`
2. Insufficient balance
3. Wrong `SIGNATURE_TYPE`

**Debug:**
```bash
poetry run polyb0t status
poetry run polyb0t doctor
tail -f live_run.log
```

---

## 📖 Additional Resources

### Official Polymarket Documentation

- [CLOB Authentication](https://docs.polymarket.com/developers/CLOB/authentication)
- [Builder Profile & Keys](https://docs.polymarket.com/developers/builders/builder-profile)
- [L1 Methods (create_or_derive_api_key)](https://docs.polymarket.com/developers/CLOB/clients/methods-l1)
- [Placing Your First Order](https://docs.polymarket.com/quickstart/first-order)

### PolyB0T Documentation

- [Main README](README.md) - Bot overview and features
- [Live Mode README](LIVE_MODE_README.md) - Live trading guide
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details

---

## 🆘 Support

If you encounter issues:

1. **Run diagnostics:**
   ```bash
   poetry run polyb0t doctor
   ```

2. **Check logs:**
   ```bash
   tail -f live_run.log
   ```

3. **Verify configuration:**
   ```bash
   poetry run polyb0t status --json-output
   ```

4. **Consult documentation:**
   - [QUICKSTART_L2_SETUP.md](docs/QUICKSTART_L2_SETUP.md)
   - [L2_CREDENTIALS_SETUP.md](docs/L2_CREDENTIALS_SETUP.md)
   - [SIGNATURE_TYPES.md](docs/SIGNATURE_TYPES.md)

---

## ✅ Final Checklist

Before live trading:

- [ ] L2 credentials generated and tested
- [ ] Private key permanently deleted
- [ ] `.env` configured correctly
- [ ] `auth check` passes
- [ ] `doctor` shows all green
- [ ] Tested in dry-run mode
- [ ] `POLYBOT_MAX_ORDER_USD` set conservatively
- [ ] Using dedicated hot wallet
- [ ] Understand signature type configuration
- [ ] Know how to approve/reject intents
- [ ] Monitoring setup (logs, status checks)

---

## 🎉 You're Ready!

Your bot is now configured with L2 credentials and ready for trading.

**Remember:**
- Start with `DRY_RUN=true`
- Use small `MAX_ORDER_USD` values initially
- Monitor closely for the first few trades
- All trading requires explicit approval (unless `AUTO_APPROVE_INTENTS=true`, which should NEVER be used in production)

**Happy trading! 🚀**

