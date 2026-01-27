# GCP VM Reference - PolyB0T Production Server

## SSH Connection Details

```bash
# Quick connect command
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194

# With specific options
ssh -i /Users/HP/.ssh/gcp_vm -o StrictHostKeyChecking=accept-new beamo_beamosupport_com@34.2.57.194
```

| Field | Value |
|-------|-------|
| Host IP | `34.2.57.194` |
| Username | `beamo_beamosupport_com` |
| SSH Key | `/Users/HP/.ssh/gcp_vm` |
| Project Path | `~/Analytical-Probability-Experiment` |
| Python Venv | `~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12` |

---

## Running Processes

The bot runs two main processes:

1. **Trading Bot**: `polyb0t run --live`
   - PID stored in `live_run.pid`
   - Logs to `live_run.log` (can be very large, ~4GB+)
   - Uses ~90% CPU when active

2. **API Server**: `polyb0t api --host 0.0.0.0 --port 8000`
   - FastAPI REST server
   - WebSocket support for real-time updates

---

## Key Directories & Files

```
~/Analytical-Probability-Experiment/
├── .env                      # Active configuration (SENSITIVE - has API keys)
├── .env.example              # Template for configuration
├── polybot.db                # Main SQLite database (~10GB)
├── live_run.log              # Bot logs (can be huge)
├── polyb0t/                  # Main Python package
│   ├── api/                  # FastAPI REST endpoints
│   ├── cli/                  # CLI commands (click-based)
│   ├── config/               # Pydantic settings (POLYBOT_ prefix)
│   ├── data/                 # API clients (Gamma, CLOB), storage
│   ├── execution/            # Portfolio, orders, intents, live executor
│   ├── ml/                   # AI orchestrator, trainer, MoE
│   │   └── moe/              # Mixture of Experts system
│   ├── models/               # Strategy, features, filters, risk
│   ├── services/             # Scheduler, reporter, Discord, health
│   └── utils/                # Logging, rate limiter
├── data/
│   ├── ai_models/            # Trained AI models (~327MB)
│   │   └── current_model.pkl # Current deployed model
│   ├── ai_training.db        # Training examples (~972MB)
│   ├── moe_models/           # MoE expert models (~20MB)
│   │   ├── experts/          # Individual expert models
│   │   ├── gating.model.pkl  # Gating network
│   │   └── pool_state.json   # Expert pool state
│   └── category_stats.db     # Category performance tracking
└── tests/                    # Test suite
```

---

## CLI Commands Reference

```bash
# Activate virtualenv first
source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate

# Core Commands
polyb0t run --paper           # Paper trading mode
polyb0t run --live            # Live mode (human-in-the-loop)
polyb0t status                # Portfolio status, balance, positions
polyb0t doctor                # Connectivity diagnostics

# Intent Management (Live Mode)
polyb0t intents list          # View pending trade intents
polyb0t intents approve <id>  # Approve a trade
polyb0t intents reject <id>   # Reject a trade

# Analysis
polyb0t universe              # Show tradable markets
polyb0t report --today        # Daily trading report
polyb0t diagnose-filters      # Debug market filtering

# Database
polyb0t db init               # Initialize schema
polyb0t db reset              # Reset database (DESTRUCTIVE)

# AI/ML
polyb0t reset-experts         # Reset all expert states

# API Server
polyb0t api                   # Start REST API on port 8000
```

---

## Configuration (.env key variables)

```bash
# Mode
POLYBOT_MODE=live             # paper or live
POLYBOT_DRY_RUN=true          # true = no real orders (SAFE)
POLYBOT_PLACING_ORDERS=true   # Master trading switch

# Wallet
POLYBOT_USER_ADDRESS=0x5cbB1a163F426097578EB4de9e3ECD987Fc1c0d4

# CLOB API (for order execution)
POLYBOT_CLOB_API_KEY=...
POLYBOT_CLOB_API_SECRET=...
POLYBOT_CLOB_PASSPHRASE=...

# Trading Parameters
POLYBOT_EDGE_THRESHOLD=0.01   # Minimum edge to trade (1%)
POLYBOT_MIN_NET_EDGE=0.005    # Min edge after fees (0.5%)
POLYBOT_MAX_POSITION_PCT=2.0  # Max 2% per position
POLYBOT_MAX_TOTAL_EXPOSURE_PCT=20.0

# Market Filtering
POLYBOT_RESOLVE_MIN_DAYS=7    # Min days until resolution
POLYBOT_RESOLVE_MAX_DAYS=180  # Max days until resolution
POLYBOT_MIN_LIQUIDITY=100
POLYBOT_MAX_SPREAD=0.10       # 10% max spread

# Risk Management
POLYBOT_DRAWDOWN_LIMIT_PCT=25.0
POLYBOT_MAX_DAILY_LOSS_PCT=15.0
POLYBOT_GLOBAL_STOP_LOSS_PCT=15

# Intent System
POLYBOT_INTENT_EXPIRY_SECONDS=90
POLYBOT_AUTO_APPROVE_INTENTS=false

# External APIs
POLYBOT_OPENAI_API_KEY=...    # For GPT-5.2 news analysis
POLYBOT_DISCORD_WEBHOOK_URL=... # Discord notifications
```

---

## Database Schema (Key Tables)

| Table | Purpose |
|-------|---------|
| `markets` | Market metadata (554 records) |
| `market_outcomes` | Outcome tokens for each market |
| `signals` | Generated trading signals (36,086 records) |
| `trade_intents` | Live mode approval queue (10,778 records) |
| `simulated_orders` | Paper trading orders |
| `simulated_fills` | Paper trading fills |
| `balance_snapshots` | Historical balance tracking (46,515 records) |
| `kill_switch_events` | Risk management triggers |
| `account_states` | Account state snapshots |

Query example:
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('polybot.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM signals')
print(cursor.fetchone()[0])
"
```

---

## MoE Expert System

**24 Experts by Category:**

| Type | Experts |
|------|---------|
| Market Category | sports, politics_us, politics_intl, crypto, economics, entertainment, tech, weather, science, legal |
| Risk Level | low_risk, medium_risk, high_risk |
| Time Horizon | short_term, medium_term, long_term |
| Market Dynamics | high_volume, low_volume, high_liquidity, high_volatility, low_volatility, momentum_strong, weekend_trader, market_close |

**Trained Experts (have .model.pkl):**
- high_risk, high_volume, long_term, low_risk, low_volatility, medium_risk, momentum_strong

**Best Performer:** `momentum_strong` (+3.67% simulated profit)

---

## Useful SSH Commands

```bash
# Check running processes
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "ps aux | grep polyb0t"

# View recent logs (last 50 lines)
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "tail -50 ~/Analytical-Probability-Experiment/live_run.log"

# Check bot status
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t status"

# Run diagnostics
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t doctor"

# Check disk space
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "df -h"

# Check memory usage
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "free -h"
```

---

## System Resources (as of last check)

| Resource | Value |
|----------|-------|
| CPU Cores | 2 |
| CPU Freq | 2250 MHz |
| Memory | 7.75 GB total |
| Disk | 144.26 GB total, ~14.4% used |
| Bot Memory | ~346 MB |
| Uptime | 357+ hours |

---

## API Endpoints

Base URL: `http://34.2.57.194:8000`

Key endpoints (see `polyb0t/api/` for full list):
- `GET /health` - Health check
- `GET /status` - Bot status
- `GET /markets` - Tradable markets
- `GET /positions` - Current positions
- `GET /intents` - Pending intents
- `WebSocket /ws` - Real-time updates

---

## Important Notes

1. **DRY_RUN=true** means the bot computes everything but doesn't submit real orders - this is the SAFE default

2. **Human-in-the-loop**: In live mode, trades require manual approval via `polyb0t intents approve <id>`

3. **AUTO_APPROVE_INTENTS=false**: Currently disabled, meaning all trades need manual approval

4. **The log file (`live_run.log`) can grow very large** - consider rotating or truncating periodically

5. **sqlite3 is NOT installed** on the VM - use Python to query databases

6. **The main database is ~10GB** - be careful with queries that scan full tables

---

## Troubleshooting

**Bot not running?**
```bash
cd ~/Analytical-Probability-Experiment
source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate
nohup polyb0t run --live > live_run.log 2>&1 &
```

**Check for errors:**
```bash
grep -i error live_run.log | tail -20
```

**Restart the bot:**
```bash
pkill -f "polyb0t run"
# Wait a moment, then start again
nohup polyb0t run --live > live_run.log 2>&1 &
```
