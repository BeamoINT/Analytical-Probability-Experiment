# PolyB0T - Autonomous Polymarket Paper Trading Bot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A production-quality MVP for autonomous paper trading on Polymarket with comprehensive risk management, market analysis, and performance tracking.**

## ⚠️ Important Safety Notice

**LIVE MODE NOW AVAILABLE WITH L2 CREDENTIALS**

This bot supports two modes:
- **Paper Trading Mode**: Simulated trading with no real money (safe for testing)
- **Live Mode**: Real trading with human-in-the-loop approval (requires L2 credentials)

**For Live Trading:**
- ✅ Requires Polymarket L2 CLOB credentials (API Key, Secret, Passphrase)
- ✅ All trades require explicit approval (human-in-the-loop)
- ✅ Dry-run mode available for testing without execution
- ✅ See **[L2 Credentials Setup Guide](README_L2_SETUP.md)** for complete instructions

**Geographic and Regulatory Compliance:**
- Users must comply with Polymarket's Terms of Service
- Respect all geographic restrictions and access controls
- This software does not bypass or circumvent any platform restrictions
- Users are responsible for ensuring lawful use in their jurisdiction

## 🎯 What It Does

PolyB0T is an **institutional-grade autonomous trading bot** with **deep learning** capabilities:

1. **Discovers Markets**: Fetches active Polymarket markets via Gamma API
2. **Filters Universe**: Identifies tradable markets resolving in 30-60 days with sufficient liquidity
3. **Analyzes Opportunities**: Computes features, implied probabilities, and edge calculations
4. **Generates Signals**: Uses a baseline value-based strategy with explainable logic
5. **🧠 Machine Learning**: Learns from market data and adapts predictions over time
6. **Manages Risk**: Enforces strict position sizing, exposure limits, and drawdown controls
7. **Simulates Trading**: Paper trades with realistic fill simulation including slippage and fees
8. **Tracks Performance**: Comprehensive logging, metrics, and daily reports

### 🧠 Deep Learning Features

- **75GB database capacity** - stores 3 years of market history
- **25M training examples** - institutional-scale learning
- **Dense 15-min price snapshots** - captures intraday movements
- **Automatic backfill** - no data loss when stopped/restarted
- **Auto-enable ML** - activates automatically when ready (2,000+ examples)
- **Continuous online learning** - retrains every 6 hours, hot-swaps models
- **Broad market tracking** - learns from 50+ markets simultaneously

See **[DEEP_LEARNING_UPGRADE.md](DEEP_LEARNING_UPGRADE.md)** for complete details.

## 🏗️ Architecture

```
polyb0t/
├── config/          # Configuration management (pydantic)
├── data/            # Data layer
│   ├── gamma_client.py     # Gamma Markets API client
│   ├── clob_client.py      # CLOB API client (read-only)
│   ├── models.py           # Pydantic data models
│   └── storage.py          # SQLAlchemy database models
├── models/          # Trading logic
│   ├── filters.py          # Market filtering pipeline
│   ├── features.py         # Feature engineering
│   ├── strategy_baseline.py # Baseline trading strategy
│   └── risk.py             # Risk management
├── execution/       # Execution layer
│   ├── orders.py           # Order models
│   ├── portfolio.py        # Portfolio management
│   └── simulator.py        # Paper trading simulator
├── services/        # Services layer
│   ├── scheduler.py        # Main trading loop
│   ├── reporter.py         # Reporting and metrics
│   └── health.py           # Health monitoring
├── api/             # FastAPI endpoints
│   └── app.py              # REST API
├── cli/             # Command-line interface
│   └── main.py             # CLI commands
└── utils/           # Utilities
    └── logging.py          # Structured logging
```

## 🚀 Quick Start

### Ubuntu/Debian Users - One-Command Setup ⚡

```bash
git clone https://github.com/BeamoINT/Analytical-Probability-Experiment.git
cd Analytical-Probability-Experiment
./ubuntu_setup.sh
```

**That's it!** The script auto-installs Python 3.11+, Poetry, and all dependencies.  
See **[UBUNTU_QUICK_START.md](UBUNTU_QUICK_START.md)** for details.

---

### Manual Setup (All Platforms)

```bash
git clone https://github.com/BeamoINT/Analytical-Probability-Experiment.git
cd Analytical-Probability-Experiment

# Create runtime env file (required)
cp .env.example .env
# Edit .env (required keys: POLYBOT_MODE, POLYBOT_USER_ADDRESS, POLYBOT_LOOP_INTERVAL_SECONDS, POLYBOT_DRY_RUN)

# Install dependencies (Poetry-only workflow)
poetry install

# Smoke test CLI
poetry run polyb0t --help

# Initialize DB
poetry run polyb0t db init

# Universe command (reads Gamma)
poetry run polyb0t universe

# Live mode loop (10s, dry-run by default)
poetry run polyb0t run --live
```

### Prerequisites

- Python 3.11 or higher
- Poetry (recommended) or pip
- PostgreSQL (optional, SQLite works for dev)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd "Polymarket Auto Trading API"
```

2. **Install dependencies**

Using Poetry (recommended):
```bash
poetry install
poetry shell
```

Using pip:
```bash
pip install -r requirements.txt  # Generate from pyproject.toml if needed
```

3. **Configure environment**

Recommended: use the repo’s committed defaults + a machine-level secrets file.

- Read: `docs/SECRETS_AND_AUTOSTART.md`
- Run once per machine:

```bash
bash scripts/bootstrap_env.sh
```

```bash
cp .env.example .env
# Edit .env with your settings
```

**Live mode with L2 credentials**

For live trading, you need Polymarket L2 CLOB credentials. See the complete setup guide:

📖 **[L2 Credentials Setup Guide](README_L2_SETUP.md)** - Complete instructions for generating and configuring L2 credentials

Quick setup:
```bash
# 1. Generate L2 credentials (one-time)
export POLY_PRIVATE_KEY=0xYOUR_KEY
export POLY_FUNDER_ADDRESS=0xYOUR_ADDRESS
poetry run python scripts/generate_l2_creds.py
unset POLY_PRIVATE_KEY  # Delete immediately!

# 2. Configure .env
cp .env.example .env
# Add: POLYBOT_CLOB_API_KEY, POLYBOT_CLOB_API_SECRET, POLYBOT_CLOB_API_PASSPHRASE

# 3. Verify authentication
poetry run polyb0t auth check
```

> ⚠️ The bot NEVER needs your private key permanently - only for one-time L2 credential generation

Key configuration options:
- `POLYBOT_DB_URL`: Database connection string
- `POLYBOT_PAPER_BANKROLL`: Initial paper trading capital (default: $10,000)
- `POLYBOT_RESOLVE_MIN_DAYS`: Minimum days until market resolution (default: 30)
- `POLYBOT_RESOLVE_MAX_DAYS`: Maximum days until market resolution (default: 60)
- `POLYBOT_EDGE_THRESHOLD`: Minimum edge to trade (default: 0.05 or 5%)
- `POLYBOT_MAX_POSITION_PCT`: Max position size % of bankroll (default: 2%)
- `POLYBOT_DRAWDOWN_LIMIT_PCT`: Stop trading if drawdown exceeds (default: 15%)

4. **Initialize database**

```bash
polyb0t db init
```

### Running the Bot

**Start paper trading:**
```bash
polyb0t run --paper
```

The bot will:
- Fetch markets every 5 minutes (configurable)
- Filter tradable universe
- Generate trading signals
- Execute paper trades
- Track portfolio and PnL
- Log all decisions

**View tradable universe:**
```bash
polyb0t universe
```

**Generate report:**
```bash
polyb0t report --today
```

**Start API server:**
```bash
polyb0t api
# Access at http://localhost:8000
# Endpoints: /health, /status, /report, /metrics
```

**Live monitoring (read-only, approval required):**
```bash
polyb0t run --live   # defaults to DRY_RUN=true for safety
```

**Check status with account state:**
```bash
polyb0t status
```

## 📊 How It Works

### 1. Market Discovery & Filtering

The bot fetches markets from Polymarket's Gamma API and applies filters:

- **Resolution Window**: Only markets resolving in 30-60 days (configurable)
- **Liquidity**: Minimum $1,000 liquidity/volume (configurable)
- **Spread**: Maximum 5% bid-ask spread (configurable)
- **Status**: Active markets only (not closed or inactive)
- **Blacklist**: Manual exclusion list for ambiguous markets

### 2. Feature Engineering

For each market outcome, the bot computes:

- **Price Features**: Best bid/ask, mid price, spread
- **Depth Features**: Order book depth, depth imbalance
- **Trade Features**: Recent trade activity, buy/sell pressure, momentum
- **Market Features**: Volume, liquidity, time to resolution

### 3. Baseline Strategy

**Simple, explainable value-based approach:**

1. **Market Probability**: Implied from orderbook mid price
2. **Model Probability**: Adjusted using:
   - Shrinkage toward 0.5 (reduces overconfidence)
   - Momentum adjustment (recent price trends)
   - Mean reversion component (extreme prices revert)
3. **Edge Calculation**: `edge = p_model - p_market`
4. **Signal Generation**: Trade if `|edge| >= threshold`

**Position Sizing:**
- Base size: 2% of bankroll (configurable)
- Adjusted by signal confidence
- Never exceeds available cash

### 4. Risk Management

**Hard Limits:**
- Max position size: 2% of bankroll
- Max total exposure: 20% of bankroll
- Max per-category exposure: 10% of bankroll
- Drawdown limit: 15% (halts trading if exceeded)

**Risk Checks Before Every Trade:**
- Available cash sufficient?
- Would total exposure exceed limit?
- Already have position in this token?
- Drawdown limit breached?

**No Martingale**: Fixed fractional sizing, no doubling after losses

### 5. Execution Simulation

**Conservative Fill Logic:**

Limit orders only fill if:
- A subsequent trade crosses the order price, OR
- Order book depth suggests likely fill

**Realism:**
- Configurable slippage (default: 10 bps)
- Trading fees (default: 20 bps)
- Order timeouts (default: 5 minutes)
- Partial fills supported

### 6. Logging & Metrics

**Structured JSON Logging:**
- Every market fetch
- Every signal generated
- Every order placed
- Every fill executed
- Every PnL snapshot

**Database Tracking:**
- Markets and metadata
- Order book snapshots
- Recent trades
- Signals and features
- Simulated orders and fills
- Portfolio positions
- PnL history

## 📈 Understanding Reports

```bash
polyb0t report --today
```

**Portfolio Section:**
- Cash Balance: Available cash
- Total Exposure: Dollar value of open positions
- Unrealized PnL: Mark-to-market profit/loss
- Realized PnL: Closed position PnL
- Total Equity: Cash + Unrealized PnL
- Return %: Performance vs initial bankroll

**Daily Activity:**
- Signals Generated: Trading opportunities identified
- Orders Placed: Number of paper orders
- Fills Executed: Number of simulated fills
- Fees Paid: Simulated trading costs

**Positions:**
- Current holdings with entry prices and PnL

**Top Signals:**
- Recent signals with highest edge

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=polyb0t --cov-report=html

# Specific test file
pytest tests/test_strategy.py -v
```

**Test Coverage:**
- Market filtering logic
- Strategy signal generation
- Risk management checks
- Position sizing calculations
- Portfolio tracking
- Order simulation
- Fill logic

## 🔧 Configuration Reference

### Environment Variables

All settings can be configured via environment variables with `POLYBOT_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `POLYBOT_DB_URL` | `sqlite:///./polybot.db` | Database connection |
| `POLYBOT_PAPER_BANKROLL` | `10000.0` | Initial capital |
| `POLYBOT_RESOLVE_MIN_DAYS` | `30` | Min days to resolution |
| `POLYBOT_RESOLVE_MAX_DAYS` | `60` | Max days to resolution |
| `POLYBOT_MIN_LIQUIDITY` | `1000` | Min market liquidity |
| `POLYBOT_MAX_SPREAD` | `0.05` | Max bid-ask spread |
| `POLYBOT_EDGE_THRESHOLD` | `0.05` | Min edge to trade |
| `POLYBOT_MAX_POSITION_PCT` | `2.0` | Max position % |
| `POLYBOT_MAX_TOTAL_EXPOSURE_PCT` | `20.0` | Max total exposure % |
| `POLYBOT_DRAWDOWN_LIMIT_PCT` | `15.0` | Drawdown halt trigger |
| `POLYBOT_LOOP_INTERVAL_SECONDS` | `300` | Main loop interval |
| `POLYBOT_INTENT_EXPIRY_SECONDS` | `60` | Intent expiry window (seconds) |
| `POLYBOT_INTENT_COOLDOWN_SECONDS` | `120` | Dedup cooldown for equivalent intents (seconds) |
| `POLYBOT_INTENT_PRICE_ROUND` | `0.001` | Price rounding used for intent fingerprinting |
| `POLYBOT_INTENT_SIZE_BUCKET_USD` | `10` | USD bucket size used for intent fingerprinting |
| `POLYBOT_DEDUP_EDGE_DELTA` | `0.01` | Allow new intent within cooldown if edge changes by more than this |
| `POLYBOT_DEDUP_PRICE_DELTA` | `0.01` | Allow new intent within cooldown if price changes by more than this |
| `POLYBOT_LOG_LEVEL` | `INFO` | Logging level |

### Database

**SQLite (Development):**
```bash
POLYBOT_DB_URL=sqlite:///./polybot.db
```

**PostgreSQL (Production):**
```bash
POLYBOT_DB_URL=postgresql://user:password@localhost:5432/polybot
```

## 🧑‍⚖️ Live Mode (Human-in-the-Loop)

Live mode is **approval-gated**: the bot can propose actions as **trade intents**, but **only explicit user approval** can progress them.

### Workflow

```bash
# Run live loop (recommendations only; never submits orders)
POLYBOT_MODE=live POLYBOT_DRY_RUN=true polyb0t run --live

# View pending intents
polyb0t intents list

# Approve an intent (dry-run: marks EXECUTED_DRYRUN; no order submitted)
polyb0t intents approve <intent_id> --yes

# Reject an intent
polyb0t intents reject <intent_id> --yes
```

### Balance & Sizing (USDC on Polygon)

- Configure `POLYBOT_POLYGON_RPC_URL` and `POLYBOT_USDCE_TOKEN_ADDRESS` (see `env.live.example`).
- Live intent sizing uses **available USDC** with conservative reservations.

### Connectivity & Auth Checks

```bash
# Quick smoke test (Gamma/CLOB/RPC/auth)
polyb0t doctor

# Authenticated read-only check (never submits orders)
polyb0t auth check
```

### Safety Notes

- **Dry-run** (`POLYBOT_DRY_RUN=true`) means approvals **will not place orders**. Approved intents are finalized as `EXECUTED_DRYRUN` so they don’t reappear.
- **Live execution** requires `POLYBOT_DRY_RUN=false` *and* configured CLOB credentials. If credentials are missing, execution is refused and the intent is marked failed (no crash).
- The system includes **dedup + cooldown** to prevent repeated intents every cycle.

## 🐛 Troubleshooting

**Database errors:**
```bash
# Reset database
polyb0t db reset

# Re-initialize
polyb0t db init
```

**API connection issues:**
- Check network connectivity
- Verify Gamma/CLOB API URLs in `.env`
- Review logs for specific error messages

**No tradable markets:**
- Adjust filtering parameters in `.env`
- Check market availability on Polymarket
- Review logs to see which filters are excluding markets

**Bot not trading:**
- Check edge threshold (might be too high)
- Verify sufficient paper bankroll
- Check if drawdown limit was hit
- Review risk checks in logs

## 📚 Additional Resources

- [Polymarket](https://polymarket.com/)
- [Gamma Markets API](https://gamma-api.polymarket.com/)
- [CLOB API Documentation](https://docs.polymarket.com/)

## 🤝 Contributing

This is an MVP for educational and strategy development purposes. Contributions welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## ⚖️ License

MIT License - see LICENSE file for details

## ⚠️ Disclaimer

**THIS SOFTWARE IS PROVIDED "AS IS" FOR EDUCATIONAL PURPOSES ONLY.**

- No warranties or guarantees of any kind
- Not financial advice
- Not investment advice
- Use at your own risk
- Past performance does not indicate future results
- Paper trading results may differ significantly from live trading
- Users are responsible for compliance with all applicable laws and regulations
- The authors assume no liability for any losses incurred

**Trading prediction markets involves significant risk, including the loss of capital.**

---

## 🔍 CLI Command Reference

### Basic Commands

```bash
# Initialize database
polyb0t db init

# Reset database (WARNING: deletes all data)
polyb0t db reset

# Run paper trading bot
polyb0t run --paper

# Run live mode (dry-run by default)
polyb0t run --live

# View tradable markets
polyb0t universe

# Generate trading report
polyb0t report --today
polyb0t report --json-output  # JSON format

# Check system status
polyb0t status
polyb0t status --json-output

# Start API server
polyb0t api --host 0.0.0.0 --port 8000

# Get help
polyb0t --help
polyb0t run --help
```

### L2 Credentials & Authentication

```bash
# Verify L2 credentials (read-only check)
polyb0t auth check

# Full system diagnostics
polyb0t doctor

# Generate L2 credentials (one-time setup)
python scripts/generate_l2_creds.py
```

### Intent Management (Live Mode)

```bash
# List pending intents
polyb0t intents list
polyb0t intents list --all  # Show all statuses

# Approve an intent
polyb0t intents approve <intent_id>

# Reject an intent
polyb0t intents reject <intent_id>

# Expire old intents
polyb0t intents expire

# Cleanup duplicate intents
polyb0t intents cleanup
```

### Order Management (Live Mode)

```bash
# List open orders
polyb0t orders list

# Cancel an order (creates approval-gated intent)
polyb0t orders cancel <order_id> --token-id <token_id>
```

## 📊 API Endpoints

When running `polyb0t api`:

- `GET /` - API information
- `GET /health` - Health check with uptime metrics
- `GET /status` - Current positions and exposure
- `GET /report` - Full trading report
- `GET /metrics` - Recent fills and top signals

Example:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

## 🎓 How to Run (Summary)

1. **Setup**:
   ```bash
   poetry install
   poetry shell
   cp .env.example .env
   polyb0t db init
   ```

2. **Start Trading**:
   ```bash
   polyb0t run --paper
   ```

3. **Monitor** (in another terminal):
   ```bash
   # Watch logs in real-time
   tail -f logs/polybot.log  # if configured

   # Or check status via API
   polyb0t api  # in another terminal
   curl http://localhost:8000/status
   ```

4. **Review Performance**:
   ```bash
   polyb0t report --today
   ```

That's it! The bot will run continuously, making paper trades based on real market data.

---

**Built with ❤️ for responsible algorithmic trading research and education.**

