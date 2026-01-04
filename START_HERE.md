# 🚀 START HERE - Generate L2 Credentials

## ✅ Dependencies Installed

Good news! All required dependencies (`py-clob-client`, `web3`) are already installed and ready.

---

## 🔑 Generate Your L2 Credentials (5 minutes)

### **Option 1: Use the Interactive Script (Recommended)**

Run this command:

```bash
./GENERATE_CREDENTIALS_NOW.sh
```

The script will:
1. Ask for your wallet private key (hidden input)
2. Ask for your wallet address
3. Generate L2 credentials
4. Display them for you to copy
5. Automatically delete your private key

---

### **Option 2: Manual Generation**

If you prefer manual control:

```bash
# 1. Set environment variables
export POLY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
export POLY_FUNDER_ADDRESS=0x5cbb1a163f426097578eb4de9e3ecd987fc1c0d4

# 2. Generate credentials
python3 scripts/generate_l2_creds.py

# 3. DELETE private key immediately
unset POLY_PRIVATE_KEY
unset POLY_FUNDER_ADDRESS
```

---

## 📝 Add Credentials to .env

After generation, you'll see output like:

```
POLYBOT_CLOB_API_KEY=pk_123abc...
POLYBOT_CLOB_API_SECRET=sk_456def...
POLYBOT_CLOB_API_PASSPHRASE=xyz789...
```

**Open your `.env` file and add these lines:**

```bash
# Open .env in your editor
nano .env
# or
code .env
```

**Add/update these lines:**

```env
# L2 CLOB Credentials
POLYBOT_CLOB_API_KEY=pk_...     # Paste your key
POLYBOT_CLOB_API_SECRET=sk_...  # Paste your secret
POLYBOT_CLOB_API_PASSPHRASE=... # Paste your passphrase

# Signature Configuration
POLYBOT_FUNDER_ADDRESS=0x5cbb1a163f426097578eb4de9e3ecd987fc1c0d4
POLYBOT_SIGNATURE_TYPE=0
```

**Save the file.**

---

## ✅ Verify Authentication

Run this command:

```bash
python3 -m polyb0t.cli.main auth check
```

**Success looks like:**
```
Auth OK (read-only).
Open orders: 0, positions: 0
```

**If it fails**, run diagnostics:
```bash
python3 -m polyb0t.cli.main doctor
```

---

## 🎉 You're Ready!

Once `auth check` passes, you can:

```bash
# Run the bot in dry-run mode
python3 -m polyb0t.cli.main run --live

# Check status
python3 -m polyb0t.cli.main status

# List trade intents
python3 -m polyb0t.cli.main intents list

# Approve an intent
python3 -m polyb0t.cli.main intents approve <intent_id>
```

---

## 🔐 Security Checklist

Before you start:

- [ ] Using a **dedicated hot wallet** with small funds
- [ ] NOT using your main wallet
- [ ] Will delete private key after generation
- [ ] `.env` is gitignored (already done)
- [ ] `POLYBOT_DRY_RUN=true` in `.env` (for testing)

---

## 🐛 Troubleshooting

### "Invalid private key format"
- Must start with `0x`
- Must be 66 characters total (0x + 64 hex)
- No spaces or quotes

### "Auth check fails"
- Check all 3 credentials are in `.env`
- No extra spaces or quotes around values
- Credentials match exactly what was generated

### "Cannot connect to CLOB"
- Check internet connection
- Verify you're not in a restricted region
- Try again in a few minutes

---

## 📚 Full Documentation

For detailed guides:
- **Quick Start**: `docs/QUICKSTART_L2_SETUP.md`
- **Complete Guide**: `README_L2_SETUP.md`
- **CLI Reference**: `docs/CLI_REFERENCE.md`
- **Troubleshooting**: `docs/L2_CREDENTIALS_SETUP.md`

---

**Ready? Run:** `./GENERATE_CREDENTIALS_NOW.sh`

