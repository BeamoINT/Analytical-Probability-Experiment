# L2 Setup Verification Checklist

Use this checklist to verify your L2 credentials setup is complete and working.

---

## ✅ Pre-Setup Checklist

Before generating credentials:

- [ ] Have a Polymarket account
- [ ] Know your wallet address
- [ ] Know your account type (EOA/Proxy/Safe)
- [ ] Have access to wallet private key (temporarily)
- [ ] Using a dedicated hot wallet (recommended)
- [ ] Have small amount of funds for testing

---

## ✅ Installation Checklist

- [ ] Repository cloned
- [ ] `poetry install` completed successfully
- [ ] `py-clob-client` installed (check: `poetry show py-clob-client`)
- [ ] `web3` installed (check: `poetry show web3`)
- [ ] `.env.example` file exists
- [ ] `scripts/generate_l2_creds.py` exists and is executable

---

## ✅ Credential Generation Checklist

### Option A: UI Generation

- [ ] Accessed Polymarket Builder Profile
- [ ] Created new API key
- [ ] Copied `apiKey`
- [ ] Copied `secret`
- [ ] Copied `passphrase`
- [ ] Saved credentials securely

### Option B: CLI Generation

- [ ] Set `POLY_PRIVATE_KEY` environment variable
- [ ] Set `POLY_FUNDER_ADDRESS` environment variable
- [ ] Ran `poetry run python scripts/generate_l2_creds.py`
- [ ] Script completed successfully
- [ ] Copied all three credentials
- [ ] **CRITICAL:** Ran `unset POLY_PRIVATE_KEY`
- [ ] **CRITICAL:** Ran `unset POLY_FUNDER_ADDRESS`
- [ ] Private key permanently deleted from environment

---

## ✅ Configuration Checklist

- [ ] Copied `.env.example` to `.env`
- [ ] Set `POLYBOT_MODE=live`
- [ ] Set `POLYBOT_DRY_RUN=true` (for initial testing)
- [ ] Set `POLYBOT_USER_ADDRESS` (your wallet address)
- [ ] Set `POLYBOT_FUNDER_ADDRESS` (usually same as USER_ADDRESS)
- [ ] Set `POLYBOT_SIGNATURE_TYPE` (0=EOA, 1=PROXY, 2=SAFE)
- [ ] Set `POLYBOT_CLOB_API_KEY`
- [ ] Set `POLYBOT_CLOB_API_SECRET`
- [ ] Set `POLYBOT_CLOB_API_PASSPHRASE`
- [ ] Set `POLYBOT_CLOB_BASE_URL=https://clob.polymarket.com`
- [ ] Set `POLYBOT_CHAIN_ID=137`
- [ ] Set `POLYBOT_LOOP_INTERVAL_SECONDS=10`
- [ ] Verified no extra spaces or quotes in credentials
- [ ] Verified `.env` is in `.gitignore`

---

## ✅ Verification Checklist

### Auth Check

Run: `poetry run polyb0t auth check`

- [ ] Command runs without errors
- [ ] Output shows: "Auth OK (read-only)"
- [ ] Shows open orders count
- [ ] Shows positions count
- [ ] No "401 Unauthorized" errors
- [ ] No "missing credentials" errors

### Doctor Check

Run: `poetry run polyb0t doctor`

- [ ] Gamma API: PASS
- [ ] CLOB public orderbook: PASS
- [ ] CLOB auth (read-only): PASS
- [ ] Polygon RPC (optional): PASS or skipped
- [ ] Exit code: 0

### Status Check

Run: `poetry run polyb0t status`

- [ ] Shows mode: live
- [ ] Shows dry_run: true
- [ ] Shows user address
- [ ] Shows intent counts
- [ ] No error messages
- [ ] Account state section appears (if configured)

---

## ✅ Database Checklist

- [ ] Ran `poetry run polyb0t db init`
- [ ] Database file created (`polybot.db`)
- [ ] No errors during initialization

---

## ✅ Dry-Run Testing Checklist

Run: `poetry run polyb0t run --live`

- [ ] Bot starts without errors
- [ ] Connects to Gamma API
- [ ] Fetches markets successfully
- [ ] Generates signals (if markets available)
- [ ] Creates trade intents
- [ ] Logs show "DRY_RUN mode"
- [ ] No actual orders submitted
- [ ] Can stop with Ctrl+C

---

## ✅ Intent Management Checklist

Run: `poetry run polyb0t intents list`

- [ ] Command runs without errors
- [ ] Shows pending intents (if any)
- [ ] Shows intent details (ID, type, side, price, size)
- [ ] Can approve with: `polyb0t intents approve <id>`
- [ ] Can reject with: `polyb0t intents reject <id>`

---

## ✅ Security Checklist

- [ ] Private key NOT in `.env`
- [ ] Private key NOT in any file
- [ ] Private key NOT in environment variables
- [ ] `.env` in `.gitignore`
- [ ] `.env` NOT committed to git
- [ ] Only L2 credentials in `.env`
- [ ] Using dedicated hot wallet (not main wallet)
- [ ] `POLYBOT_DRY_RUN=true` for initial testing

---

## ✅ Documentation Checklist

- [ ] Read `README_L2_SETUP.md`
- [ ] Read `docs/QUICKSTART_L2_SETUP.md`
- [ ] Understand signature types (if needed)
- [ ] Know how to approve/reject intents
- [ ] Know how to check status
- [ ] Know how to view logs

---

## ✅ Going Live Checklist

**Only after all above checks pass:**

- [ ] Tested in dry-run mode for sufficient time
- [ ] Understand intent approval flow
- [ ] Set `POLYBOT_MAX_ORDER_USD` to safe value (e.g., 1.0)
- [ ] Set `POLYBOT_MAX_TOTAL_EXPOSURE_USD` conservatively
- [ ] Set `POLYBOT_MAX_OPEN_ORDERS` to low number (e.g., 3)
- [ ] Updated `.env`: `POLYBOT_DRY_RUN=false`
- [ ] Confirmed change with user
- [ ] Monitoring setup ready (logs, status checks)
- [ ] Know how to stop bot (Ctrl+C)
- [ ] Know how to cancel orders

---

## ✅ First Live Trade Checklist

- [ ] Bot running with `DRY_RUN=false`
- [ ] Intent generated and appears in `intents list`
- [ ] Reviewed intent details carefully
- [ ] Approved ONE intent: `polyb0t intents approve <id>`
- [ ] Checked Polymarket UI for order
- [ ] Order appears in Polymarket account
- [ ] Monitored order status
- [ ] Verified in `polyb0t status`

---

## ✅ Ongoing Operations Checklist

Daily:
- [ ] Check status: `polyb0t status`
- [ ] Review pending intents: `polyb0t intents list`
- [ ] Approve/reject as needed
- [ ] Check logs: `tail -f live_run.log`
- [ ] Verify orders in Polymarket UI

Weekly:
- [ ] Generate report: `polyb0t report --today`
- [ ] Review performance
- [ ] Adjust risk limits if needed
- [ ] Clean up old intents: `polyb0t intents cleanup`

---

## 🐛 Troubleshooting Checklist

If `auth check` fails:
- [ ] Verified all three credentials in `.env`
- [ ] Checked for extra spaces or quotes
- [ ] Verified credentials are correct
- [ ] Tried regenerating credentials
- [ ] Checked `POLYBOT_CLOB_BASE_URL`

If signature errors:
- [ ] Verified `SIGNATURE_TYPE` matches account
- [ ] Checked `FUNDER_ADDRESS` is correct
- [ ] Consulted `docs/SIGNATURE_TYPES.md`

If orders fail:
- [ ] Verified sufficient balance
- [ ] Checked `FUNDER_ADDRESS`
- [ ] Verified `SIGNATURE_TYPE`
- [ ] Reviewed logs for specific errors

---

## 📊 Final Verification

**All systems go if:**

✅ `polyb0t auth check` → "Auth OK"  
✅ `polyb0t doctor` → All PASS  
✅ `polyb0t status` → Shows correct configuration  
✅ `polyb0t run --live` → Starts without errors  
✅ `polyb0t intents list` → Works correctly  
✅ Private key permanently deleted  
✅ `.env` not committed to git  
✅ Documentation reviewed  

---

## 🎉 Success Criteria

You're ready for live trading when:

1. ✅ All checklist items above are complete
2. ✅ Tested in dry-run mode successfully
3. ✅ Understand the approval workflow
4. ✅ Know how to monitor and control the bot
5. ✅ Have safety limits configured
6. ✅ Using dedicated hot wallet with small funds

---

## 📞 Getting Help

If stuck:

1. Run: `poetry run polyb0t doctor`
2. Check: `docs/L2_CREDENTIALS_SETUP.md` (Troubleshooting section)
3. Review: `docs/SIGNATURE_TYPES.md` (if signature issues)
4. Consult: `docs/CLI_REFERENCE.md` (for command help)
5. Check logs: `tail -f live_run.log`

---

**Remember:** Start small, monitor closely, and scale up gradually!

