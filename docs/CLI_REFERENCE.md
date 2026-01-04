# PolyB0T CLI Reference

Complete command-line interface reference for PolyB0T.

---

## Table of Contents

1. [Core Commands](#core-commands)
2. [L2 Credentials & Auth](#l2-credentials--auth)
3. [Intent Management](#intent-management)
4. [Order Management](#order-management)
5. [Database Commands](#database-commands)
6. [Monitoring & Reports](#monitoring--reports)

---

## Core Commands

### `polyb0t run`

Run the trading bot continuously.

**Modes:**
- `--paper` - Paper trading (simulated)
- `--live` - Live mode with human-in-the-loop approval

**Examples:**

```bash
# Paper trading
polyb0t run --paper

# Live mode (dry-run by default)
polyb0t run --live

# Live mode with real execution (after configuring DRY_RUN=false)
POLYBOT_DRY_RUN=false polyb0t run --live
```

**What it does:**
1. Fetches markets from Gamma API
2. Filters tradable universe
3. Generates trading signals
4. Creates trade intents (live mode) or executes paper trades
5. Tracks portfolio and PnL
6. Logs all decisions

---

### `polyb0t universe`

Show current tradable universe.

**Example:**

```bash
polyb0t universe
```

**Output:**
- Total markets fetched
- Tradable markets (after filters)
- First 20 markets with details

---

### `polyb0t status`

Show current system status.

**Options:**
- `--json-output` - Output as JSON

**Example:**

```bash
polyb0t status
polyb0t status --json-output
```

**Shows:**
- Mode and dry-run status
- Last cycle time
- Exposure and drawdown
- Intent counts by status
- Account state (if configured)

---

### `polyb0t api`

Start the FastAPI server.

**Options:**
- `--host HOST` - API host (default: 0.0.0.0)
- `--port PORT` - API port (default: 8000)

**Example:**

```bash
polyb0t api
polyb0t api --host 127.0.0.1 --port 9000
```

**Endpoints:**
- `GET /health` - Health check
- `GET /status` - Current status
- `GET /report` - Trading report
- `GET /metrics` - Key metrics

---

## L2 Credentials & Auth

### `polyb0t auth check`

Verify L2 CLOB credentials (read-only).

**Example:**

```bash
polyb0t auth check
```

**Success output:**
```
Auth OK (read-only).
Open orders: 0, positions: 0
```

**Failure output:**
```
Auth check FAILED: missing required CLOB credentials
  - POLYBOT_CLOB_API_KEY
  - POLYBOT_CLOB_API_SECRET
  - POLYBOT_CLOB_API_PASSPHRASE
```

**What it checks:**
- All three L2 credentials present
- Can authenticate with CLOB API
- Can fetch account state (read-only)

---

### `polyb0t doctor`

Full system diagnostics.

**Example:**

```bash
polyb0t doctor
```

**Tests:**
- ✅ Gamma API connectivity
- ✅ CLOB public orderbook access
- ✅ Polygon RPC USDC balance (if configured)
- ✅ CLOB authentication (if credentials set)

**Exit codes:**
- `0` - All checks passed
- `2` - One or more checks failed

---

### `python scripts/generate_l2_creds.py`

Generate L2 credentials (one-time setup).

**Prerequisites:**
```bash
export POLY_PRIVATE_KEY=0xYOUR_PRIVATE_KEY
export POLY_FUNDER_ADDRESS=0xYOUR_WALLET_ADDRESS
```

**Example:**

```bash
poetry run python scripts/generate_l2_creds.py
```

**Output:**
```
POLYBOT_CLOB_API_KEY=pk_...
POLYBOT_CLOB_API_SECRET=sk_...
POLYBOT_CLOB_API_PASSPHRASE=...
```

**After generation:**
```bash
unset POLY_PRIVATE_KEY  # CRITICAL!
```

See: [L2 Credentials Setup Guide](../README_L2_SETUP.md)

---

## Intent Management

Intents are proposed trading actions awaiting approval (live mode only).

### `polyb0t intents list`

List pending trade intents.

**Options:**
- `--all` - Show all intents (not just pending)
- `--json-output` - Output as JSON

**Example:**

```bash
polyb0t intents list
polyb0t intents list --all
polyb0t intents list --json-output
```

**Output:**
- Intent ID (short)
- Status (PENDING, APPROVED, EXECUTED, etc.)
- Type (OPEN_POSITION, CLOSE_POSITION, etc.)
- Side, price, size
- Edge, expiry time

---

### `polyb0t intents approve`

Approve a trade intent for execution.

**Arguments:**
- `INTENT_ID` - Intent identifier

**Options:**
- `--yes` - Skip confirmation prompt

**Example:**

```bash
polyb0t intents approve abc12345
polyb0t intents approve abc12345 --yes
```

**Interactive flow:**
1. Shows intent details
2. Asks for confirmation
3. Marks as APPROVED
4. Will execute in next cycle (if not dry-run)

---

### `polyb0t intents reject`

Reject a trade intent.

**Arguments:**
- `INTENT_ID` - Intent identifier

**Options:**
- `--yes` - Skip confirmation prompt

**Example:**

```bash
polyb0t intents reject abc12345
polyb0t intents reject abc12345 --yes
```

---

### `polyb0t intents expire`

Manually expire old pending intents.

**Example:**

```bash
polyb0t intents expire
```

**What it does:**
- Finds intents past expiry time
- Marks as EXPIRED
- Reports count

---

### `polyb0t intents cleanup`

Cleanup duplicate pending intents.

**Options:**
- `--mode MODE` - Cleanup mode: `supersede` or `expire` (default: supersede)
- `--yes` - Skip confirmation

**Example:**

```bash
polyb0t intents cleanup
polyb0t intents cleanup --mode expire --yes
```

**Modes:**
- `supersede` - Keep newest, mark others as SUPERSEDED
- `expire` - Mark all duplicates as EXPIRED

---

## Order Management

### `polyb0t orders list`

List open orders (live mode).

**Example:**

```bash
polyb0t orders list
```

**Output:**
- Order ID
- Token ID
- Side (BUY/SELL)
- Price, size
- Filled size

---

### `polyb0t orders cancel`

Request cancellation of an order (creates approval-gated intent).

**Arguments:**
- `ORDER_ID` - Order identifier

**Options:**
- `--token-id TOKEN_ID` - Token ID (required)
- `--market-id MARKET_ID` - Market ID (optional)
- `--yes` - Skip confirmation

**Example:**

```bash
polyb0t orders cancel abc123 --token-id 0x456def
polyb0t orders cancel abc123 --token-id 0x456def --yes
```

**What it does:**
1. Creates CANCEL_ORDER intent
2. Requires approval: `polyb0t intents approve <id>`
3. Executes cancellation after approval

---

## Database Commands

### `polyb0t db init`

Initialize database tables.

**Example:**

```bash
polyb0t db init
```

**Safe to run multiple times** (creates tables if not exist).

---

### `polyb0t db reset`

Drop and recreate all tables.

**⚠️ WARNING: Deletes ALL data**

**Example:**

```bash
polyb0t db reset
```

**Confirmation required.**

---

## Monitoring & Reports

### `polyb0t report`

Generate trading report.

**Options:**
- `--today` - Show today's report
- `--json-output` - Output as JSON

**Example:**

```bash
polyb0t report --today
polyb0t report --json-output
```

**Shows:**
- Portfolio summary (cash, exposure, PnL)
- Today's activity (signals, orders, fills)
- Open positions
- Top signals
- Account state (if configured)

---

## Environment Variables

All commands respect environment variables from `.env`:

**Core:**
- `POLYBOT_MODE` - Trading mode: `paper` or `live`
- `POLYBOT_DRY_RUN` - Dry-run mode: `true` or `false`
- `POLYBOT_LOOP_INTERVAL_SECONDS` - Main loop interval

**L2 Credentials:**
- `POLYBOT_CLOB_API_KEY` - CLOB API key
- `POLYBOT_CLOB_API_SECRET` - CLOB API secret
- `POLYBOT_CLOB_API_PASSPHRASE` - CLOB passphrase

**Account:**
- `POLYBOT_USER_ADDRESS` - Your wallet address
- `POLYBOT_FUNDER_ADDRESS` - Funder address
- `POLYBOT_SIGNATURE_TYPE` - Signature type (0=EOA, 1=PROXY, 2=SAFE)

See [.env.example](../.env.example) for all options.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Configuration error (missing .env, invalid credentials) |

---

## Examples

### Complete Setup Flow

```bash
# 1. Install
poetry install

# 2. Generate L2 credentials
export POLY_PRIVATE_KEY=0xYOUR_KEY
export POLY_FUNDER_ADDRESS=0xYOUR_ADDRESS
poetry run python scripts/generate_l2_creds.py
unset POLY_PRIVATE_KEY

# 3. Configure
cp .env.example .env
# Edit .env with credentials

# 4. Verify
poetry run polyb0t auth check
poetry run polyb0t doctor

# 5. Initialize DB
poetry run polyb0t db init

# 6. Test dry-run
poetry run polyb0t run --live
```

### Daily Operations

```bash
# Morning: Check status
polyb0t status

# Review pending intents
polyb0t intents list

# Approve good intents
polyb0t intents approve <id>

# Reject bad intents
polyb0t intents reject <id>

# Evening: Generate report
polyb0t report --today
```

### Troubleshooting

```bash
# Full diagnostics
polyb0t doctor

# Check auth
polyb0t auth check

# View logs
tail -f live_run.log

# Check open orders
polyb0t orders list

# Check account state
polyb0t status
```

---

## Getting Help

```bash
# General help
polyb0t --help

# Command-specific help
polyb0t run --help
polyb0t intents --help
polyb0t auth --help
```

---

## See Also

- [L2 Credentials Setup](../README_L2_SETUP.md)
- [Quick Start Guide](QUICKSTART_L2_SETUP.md)
- [Signature Types](SIGNATURE_TYPES.md)
- [Main README](../README.md)

