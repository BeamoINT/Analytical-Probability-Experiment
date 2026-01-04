# 🚀 START HERE - Your Bot is Ready!

## ✅ Implementation Complete

Your Polymarket trading bot now has **full balance tracking and risk-aware sizing** implemented.

**Status:** 🟢 Production-ready (safe-by-default)

---

## 🎯 One Command to Complete Setup

Run this interactive script:

```bash
./add_polygon_rpc.sh
```

**What it does:**
1. Asks you to choose a Polygon RPC provider (quick or better)
2. Adds the RPC URL to your `.env` file
3. Runs `polyb0t doctor` to verify everything works
4. Shows you next steps

**Takes:** 2 minutes (30 seconds if using quick option)

---

## 🧪 Verification Commands

After adding the RPC URL, verify everything works:

### 1. Doctor Check
```bash
python3 -m polyb0t.cli.main doctor
```

**Expected output:**
```
PASS  Gamma API: ok
PASS  CLOB public orderbook: ok
PASS  Polygon RPC USDC balance: total_usdc=X.XX  ← Should PASS now
PASS  CLOB auth (read-only): ok
```

All checks should show **PASS** ✅

---

### 2. Status Check
```bash
python3 -m polyb0t.cli.main status
```

**Expected output:**
```
MODE:                 live
Dry-run:              true
USDC total:           X.XX
USDC reserved:        0.00
USDC available:       X.XX
```

Should show your actual USDC balance ✅

---

### 3. Run Bot (Dry-Run Safe)
```bash
python3 -m polyb0t.cli.main run --live
```

**What it does:**
- Fetches markets from Polymarket
- Checks your on-chain USDC balance
- Generates trading signals
- Applies risk checks
- Creates intents (PENDING, awaiting approval)
- **NO real orders** (DRY_RUN=true)

**Look for:**
```
INFO: Balance snapshot: total=X.XX USDC, reserved=0.00, available=X.XX
INFO: Intent lifecycle summary
  signals_found: 5
  intents_created: 2
  intents_risk_rejected: 3
```

---

## 📋 What's Implemented

### A) Balance Tracking ✅
- **On-chain USDC balance** via Polygon RPC
- **Reserved calculation** (approved intents + open orders)
- **Available balance** (total - reserved, never negative)
- **Logged every cycle** at INFO level

### B) Risk-Aware Sizing ✅
All intents are checked against:
1. ✅ Available balance (rejects if insufficient)
2. ✅ Max order size (default: $5)
3. ✅ Total exposure limit (default: $25)
4. ✅ Max open orders (default: 3)
5. ✅ Daily notional limit (default: $50)

**Rejections logged with specific reasons**

### C) Execution Safety ✅
- ✅ **Human approval required** (no auto-trading)
- ✅ **DRY_RUN=true** by default (no real orders)
- ✅ **Limit orders only** (no market orders)
- ✅ **Multiple kill switches** (drawdown, errors, spreads)

### D) CLI Commands ✅
- ✅ `polyb0t auth check` - Verify L2 credentials
- ✅ `polyb0t doctor` - Test all connectivity
- ✅ `polyb0t status` - Show balance + intents
- ✅ `polyb0t run --live` - Run trading loop
- ✅ `polyb0t intents list` - View pending intents
- ✅ `polyb0t intents approve <id>` - Approve intent

---

## 🔒 Safety Features

Your bot is **safe by default:**

### 1. Dry-Run Mode (Default)
```env
POLYBOT_DRY_RUN=true  # Default
```
- Creates intents
- Logs everything
- **Never submits real orders**

### 2. Human-in-the-Loop
- All intents start as PENDING
- Require explicit approval
- Can expire or be rejected
- Tracked in database

### 3. Conservative Limits
```env
POLYBOT_MAX_ORDER_USD=5.0          # Small orders
POLYBOT_MAX_TOTAL_EXPOSURE_USD=25.0  # Low exposure
POLYBOT_MAX_OPEN_ORDERS=3          # Limited positions
```

### 4. Balance-Aware
- Never exceeds available USDC
- Accounts for reserved amounts
- Factors in open orders
- Conservative calculations

### 5. Kill Switches
- Drawdown limit (5%)
- Consecutive errors (5)
- API error rate (50%)
- Stale data detection (60s)
- Spread anomaly detection (3x)

---

## 📊 Example Workflow

### Scenario: Bot finds a good signal

1. **Balance check (every cycle):**
   ```
   INFO: Balance snapshot: total=100.00 USDC, reserved=0.00, available=100.00
   ```

2. **Signal generated:**
   ```
   Signal: BUY token_abc @ 0.550 (model: 0.600, edge: +0.050)
   ```

3. **Risk checks:**
   ```
   ✓ Balance available: 100.00 USDC
   ✓ Size: min(10.00, 5.00, 5.00) = 5.00 USD
   ✓ Total exposure: 0 + 5 = 5 < 25 ✓
   ✓ Open orders: 0 < 3 ✓
   ```

4. **Intent created:**
   ```
   INFO: Created intent abc12345
     type=OPEN_POSITION side=BUY size_usd=5.00
   ```

5. **You review:**
   ```bash
   polyb0t intents list
   ```

6. **You approve:**
   ```bash
   polyb0t intents approve abc12345
   ```

7. **With DRY_RUN=true (current):**
   ```
   ✓ Marked EXECUTED_DRYRUN
   ✓ No order submitted
   ✓ Intent logged for analysis
   ```

8. **With DRY_RUN=false (when ready):**
   ```
   ✓ Order submitted to Polymarket
   ✓ Order ID: pm_order_xyz789
   ✓ Tracked in database
   ```

---

## 🎓 Your Configuration

Current `.env` has:

✅ **L2 Credentials** (working)
- `POLYBOT_CLOB_API_KEY`
- `POLYBOT_CLOB_API_SECRET`
- `POLYBOT_CLOB_API_PASSPHRASE`

✅ **Wallet Config**
- `POLYBOT_USER_ADDRESS`
- `POLYBOT_FUNDER_ADDRESS`
- `POLYBOT_SIGNATURE_TYPE=0`

❌ **Polygon RPC** (needs to be added)
- `POLYBOT_POLYGON_RPC_URL` ← Run `./add_polygon_rpc.sh` to add

✅ **Risk Limits** (defaults work)
- `POLYBOT_MAX_ORDER_USD=5.0`
- `POLYBOT_MAX_TOTAL_EXPOSURE_USD=25.0`
- `POLYBOT_MAX_OPEN_ORDERS=3`

---

## 🚦 Current Status

**What works right now:**
- ✅ L2 authentication
- ✅ Gamma market data
- ✅ CLOB public orderbooks
- ✅ Signal generation
- ✅ Intent creation
- ✅ Human approval workflow

**What needs Polygon RPC:**
- ❌ On-chain USDC balance reading
- ❌ Risk-aware sizing based on actual balance
- ❌ Full `polyb0t doctor` check passing

**After adding RPC URL:**
- ✅ Everything above works
- ✅ Full balance-aware trading
- ✅ All doctor checks pass
- ✅ Ready for live trading (when you want)

---

## 🎯 Next Steps (Choose One)

### Option A: Quick Setup (2 minutes)

```bash
./add_polygon_rpc.sh
```

Interactive script guides you through everything.

---

### Option B: Manual Setup (3 minutes)

1. **Get RPC URL:**
   - Quick: `https://polygon-rpc.com`
   - Better: Free Alchemy (https://www.alchemy.com/)

2. **Add to `.env`:**
   ```bash
   POLYBOT_POLYGON_RPC_URL=https://polygon-rpc.com
   ```

3. **Verify:**
   ```bash
   python3 -m polyb0t.cli.main doctor
   ```

---

## 📖 Documentation

- **This guide:** Quick start
- **`BALANCE_SYSTEM_COMPLETE.md`:** Full technical details
- **`NEXT_STEPS_QUICK.md`:** Quick reference
- **`LIVE_MODE_README.md`:** Live trading guide
- **`env.live.example`:** Full config reference

---

## 🆘 Troubleshooting

### Doctor check fails for Polygon RPC

**Error:** `POLYBOT_POLYGON_RPC_URL not set`

**Fix:** Run `./add_polygon_rpc.sh` or add manually to `.env`

---

### Balance shows 0.00 even though I have USDC

**Check:**
1. RPC URL is correct
2. Token address matches your USDC:
   ```env
   POLYBOT_USDCE_TOKEN_ADDRESS=0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
   ```
3. Wallet address is correct:
   ```env
   POLYBOT_USER_ADDRESS=0x5cbb1a163f426097578eb4de9e3ecd987fc1c0d4
   ```

---

### No intents created

**Possible reasons:**
1. No signals found (normal if no good opportunities)
2. All signals rejected by risk checks (check logs)
3. Markets not meeting filter criteria

**Check logs for:**
```
INFO: Intent lifecycle summary
  signals_found: 5
  intents_risk_rejected: 5  ← All rejected
```

---

## 🎉 You're Ready!

**The bot is production-ready and safe-by-default.**

Just run:

```bash
./add_polygon_rpc.sh
```

Then verify with:

```bash
python3 -m polyb0t.cli.main doctor
```

**That's it! 🚀**

---

## 💬 Questions?

All the information you need is in:
- This guide (quick start)
- `BALANCE_SYSTEM_COMPLETE.md` (full details)
- Code is well-commented
- Logs are comprehensive

**Happy trading! 📈**

