# Polymarket Signature Types Reference

Understanding signature types is critical for proper order routing and authentication on Polymarket.

---

## Overview

Polymarket supports three signature types for different wallet configurations:

| Type | Value | Description | Use Case |
|------|-------|-------------|----------|
| **EOA** | `0` | Externally Owned Account | MetaMask, hardware wallets, standard private keys |
| **POLY_PROXY** | `1` | Polymarket Proxy Wallet | Magic.link, email-based accounts |
| **POLY_GNOSIS_SAFE** | `2` | Gnosis Safe Multi-sig | Multi-signature wallets, DAOs |

---

## How to Determine Your Signature Type

### Method 1: Check Your Wallet Type

**If you use:**
- ✅ **MetaMask** → `SIGNATURE_TYPE=0`
- ✅ **Hardware wallet** (Ledger, Trezor) → `SIGNATURE_TYPE=0`
- ✅ **WalletConnect** → `SIGNATURE_TYPE=0`
- ✅ **Magic.link / Email login** → `SIGNATURE_TYPE=1`
- ✅ **Gnosis Safe** → `SIGNATURE_TYPE=2`

### Method 2: Check Polymarket Account Settings

1. Go to Polymarket
2. Open your profile/settings
3. Look for "Account Type" or "Wallet Type"
4. Match to the table above

---

## Configuration by Type

### Type 0: EOA (Externally Owned Account)

**Most common** - standard Ethereum wallet with a private key.

```env
POLYBOT_SIGNATURE_TYPE=0
POLYBOT_USER_ADDRESS=0xYOUR_WALLET_ADDRESS
POLYBOT_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS  # Same as USER_ADDRESS
```

**Characteristics:**
- You control the private key
- Direct signing of transactions
- `FUNDER_ADDRESS` = `USER_ADDRESS`

---

### Type 1: POLY_PROXY (Polymarket Proxy)

Used by **Magic.link** and some email-based login systems.

```env
POLYBOT_SIGNATURE_TYPE=1
POLYBOT_USER_ADDRESS=0xYOUR_DISPLAYED_ADDRESS
POLYBOT_FUNDER_ADDRESS=0xPROXY_CONTRACT_ADDRESS
```

**Characteristics:**
- Polymarket manages a proxy contract for you
- Signing happens through the proxy
- `FUNDER_ADDRESS` may differ from `USER_ADDRESS`
- Check Polymarket docs or support for your specific proxy address

**⚠️ Important:** If using this type, verify your `FUNDER_ADDRESS` with Polymarket support or documentation.

---

### Type 2: POLY_GNOSIS_SAFE (Multi-sig)

For **Gnosis Safe** multi-signature wallets.

```env
POLYBOT_SIGNATURE_TYPE=2
POLYBOT_USER_ADDRESS=0xYOUR_SAFE_ADDRESS
POLYBOT_FUNDER_ADDRESS=0xYOUR_SAFE_ADDRESS  # Usually same as Safe address
```

**Characteristics:**
- Multiple signers required for transactions
- Higher security for large funds
- More complex signing flow
- May require additional configuration

**⚠️ Note:** Automated trading with Safe wallets may require additional setup for multi-sig approval flows.

---

## Common Issues

### ❌ "Signature verification failed"

**Cause:** Wrong `SIGNATURE_TYPE` for your account

**Fix:**
1. Verify your wallet type (see "How to Determine" above)
2. Update `POLYBOT_SIGNATURE_TYPE` in `.env`
3. Retry: `poetry run polyb0t auth check`

---

### ❌ "Invalid funder address"

**Cause:** `FUNDER_ADDRESS` doesn't match your account type

**Fix for EOA (Type 0):**
```env
POLYBOT_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS  # Same as USER_ADDRESS
```

**Fix for Proxy (Type 1):**
- Contact Polymarket support for your proxy contract address
- Or check Polymarket UI under account settings

**Fix for Safe (Type 2):**
```env
POLYBOT_FUNDER_ADDRESS=0xYOUR_SAFE_ADDRESS  # Your Safe contract address
```

---

### ❌ Orders fail but auth passes

**Possible causes:**
1. Wrong `FUNDER_ADDRESS` (orders route to wrong account)
2. Insufficient balance in funder account
3. Allowances not set correctly

**Debug steps:**
```bash
# Check account state
poetry run polyb0t status

# Verify balance
poetry run polyb0t doctor

# Check logs
tail -f live_run.log
```

---

## Testing Your Configuration

### Step 1: Verify Auth

```bash
poetry run polyb0t auth check
```

Expected:
```
Auth OK (read-only).
```

### Step 2: Check Account State

```bash
poetry run polyb0t status
```

Should show:
- Your wallet address
- Current balances (if RPC configured)
- Open orders/positions

### Step 3: Test in Dry-Run

```bash
# Ensure dry-run is enabled
POLYBOT_DRY_RUN=true poetry run polyb0t run --live
```

Monitor logs for any signature-related errors.

---

## Advanced: Custom Configurations

### Using Different Funder (Rare)

Some advanced setups use a separate funder address:

```env
POLYBOT_USER_ADDRESS=0xYOUR_TRADING_ADDRESS
POLYBOT_FUNDER_ADDRESS=0xYOUR_FUNDING_ADDRESS
POLYBOT_SIGNATURE_TYPE=1
```

**Only do this if:**
- You know exactly why you need it
- Polymarket documentation explicitly describes your setup
- You've verified with Polymarket support

---

## Reference Links

- [Polymarket CLOB Authentication](https://docs.polymarket.com/developers/CLOB/authentication)
- [Signature Types Documentation](https://docs.polymarket.com/developers/CLOB/signature-types)
- [Account Types Guide](https://docs.polymarket.com/developers/builders/account-types)

---

## Quick Reference Table

| Wallet Type | `SIGNATURE_TYPE` | `FUNDER_ADDRESS` |
|-------------|------------------|------------------|
| MetaMask | `0` | Same as `USER_ADDRESS` |
| Ledger/Trezor | `0` | Same as `USER_ADDRESS` |
| WalletConnect | `0` | Same as `USER_ADDRESS` |
| Magic.link | `1` | Check Polymarket docs |
| Email Login | `1` | Check Polymarket docs |
| Gnosis Safe | `2` | Safe contract address |

---

## Support

If you're still unsure:

1. **Check Polymarket UI** - account settings usually indicate wallet type
2. **Contact Polymarket Support** - they can confirm your account configuration
3. **Test with small amounts** - use `POLYBOT_MAX_ORDER_USD=0.10` initially
4. **Monitor logs** - signature errors will show specific issues

**Remember:** When in doubt, start with `SIGNATURE_TYPE=0` (EOA) if you use MetaMask or a standard wallet.

