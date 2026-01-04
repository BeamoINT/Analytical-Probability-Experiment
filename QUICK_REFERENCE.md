# L2 Credentials Setup - Quick Reference Card

## 🚀 5-Minute Setup

```bash
# 1. Install
poetry install

# 2. Generate credentials
export POLY_PRIVATE_KEY=0xYOUR_KEY
export POLY_FUNDER_ADDRESS=0xYOUR_ADDRESS
poetry run python scripts/generate_l2_creds.py
unset POLY_PRIVATE_KEY  # CRITICAL!

# 3. Configure
cp .env.example .env
# Add: POLYBOT_CLOB_API_KEY, POLYBOT_CLOB_API_SECRET, POLYBOT_CLOB_API_PASSPHRASE

# 4. Verify
poetry run polyb0t auth check
poetry run polyb0t doctor
```

---

## 📖 Documentation Quick Links

| Need | Document |
|------|----------|
| Quick start | `docs/QUICKSTART_L2_SETUP.md` |
| Detailed guide | `docs/L2_CREDENTIALS_SETUP.md` |
| Commands | `docs/CLI_REFERENCE.md` |
| Signature types | `docs/SIGNATURE_TYPES.md` |
| Navigation | `docs/INDEX.md` |

---

## 🔑 Essential Commands

```bash
# Verify credentials
polyb0t auth check

# Full diagnostics
polyb0t doctor

# Check status
polyb0t status

# Run bot (dry-run)
polyb0t run --live

# List intents
polyb0t intents list

# Approve intent
polyb0t intents approve <id>
```

---

## ⚙️ Configuration (.env)

```env
# Required
POLYBOT_MODE=live
POLYBOT_DRY_RUN=true
POLYBOT_USER_ADDRESS=0xYOUR_ADDRESS
POLYBOT_FUNDER_ADDRESS=0xYOUR_ADDRESS
POLYBOT_SIGNATURE_TYPE=0  # 0=EOA, 1=PROXY, 2=SAFE

# L2 Credentials (from generation script)
POLYBOT_CLOB_API_KEY=pk_...
POLYBOT_CLOB_API_SECRET=sk_...
POLYBOT_CLOB_API_PASSPHRASE=...

# Endpoint
POLYBOT_CLOB_BASE_URL=https://clob.polymarket.com
POLYBOT_CHAIN_ID=137
```

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| Auth fails | Check all 3 credentials in `.env` |
| Signature error | Verify `SIGNATURE_TYPE` matches wallet |
| Orders fail | Check `FUNDER_ADDRESS` is correct |
| Can't find script | Run from repo root directory |

**Detailed help:** `docs/L2_CREDENTIALS_SETUP.md` (Troubleshooting section)

---

## 🔐 Security Checklist

- [ ] Private key deleted after generation
- [ ] Only L2 credentials in `.env`
- [ ] `.env` not committed to git
- [ ] `DRY_RUN=true` for testing
- [ ] Using dedicated hot wallet

---

## 📊 Verification

```bash
# Should all pass:
polyb0t auth check     # → "Auth OK"
polyb0t doctor         # → All PASS
polyb0t status         # → Shows config
```

---

## 🎯 Going Live

```bash
# 1. Update .env
POLYBOT_DRY_RUN=false
POLYBOT_MAX_ORDER_USD=1.0

# 2. Start bot
polyb0t run --live

# 3. Approve intents
polyb0t intents list
polyb0t intents approve <id>

# 4. Monitor
polyb0t status
tail -f live_run.log
```

---

**Full docs:** `README_L2_SETUP.md`
