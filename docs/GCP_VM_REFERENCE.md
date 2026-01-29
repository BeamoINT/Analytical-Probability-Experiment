# GCP VM Reference - PolyB0T Production Server

> **IMPORTANT**: The bot runs as a systemd service. Always use `systemctl` commands to manage it, NOT manual process management (pkill/nohup).

---

## Quick Reference Card

| Item | Value |
|------|-------|
| **SSH Command** | `ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194` |
| **Host IP** | `34.2.57.194` |
| **Username** | `beamo_beamosupport_com` |
| **SSH Key** | `/Users/HP/.ssh/gcp_vm` |
| **Project Path** | `~/Analytical-Probability-Experiment` |
| **Service Name** | `polybot.service` |
| **OS** | Ubuntu 24.04.3 LTS (Noble Numbat) |
| **Kernel** | 6.14.0-1021-gcp |
| **Python** | 3.12.3 |
| **Poetry** | 2.2.1 |
| **Hostname** | `analyticalprobabilityexperiment` |

---

## System Hardware

| Resource | Value |
|----------|-------|
| **CPU** | AMD EPYC 7B12 (2 cores) |
| **Memory** | 7.8 GB total (~1.3 GB used typical) |
| **Swap** | None configured |
| **Disk** | 145 GB total, ~19 GB used (13%) |
| **Virtualenv** | `~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12` |

---

## Running System Services

Key services on the VM:
- `polybot.service` - Our trading bot + API
- `ssh.service` - OpenBSD Secure Shell
- `google-guest-agent.service` - GCP Guest Agent
- `google-osconfig-agent.service` - GCP OSConfig Agent
- `chrony.service` - NTP time sync
- `unattended-upgrades.service` - Auto security updates

### Listening Ports

| Port | Service |
|------|---------|
| 22 | SSH |
| 8000 | PolyB0T API (Python/uvicorn) |
| 53 | systemd-resolved (localhost only) |

---

## SYSTEMD SERVICE MANAGEMENT (USE THIS!)

The bot runs as a systemd service that manages BOTH the trading bot and API server together.

### Essential Commands

```bash
# RESTART THE SERVICE (after code changes)
sudo systemctl restart polybot.service

# Check service status
sudo systemctl status polybot.service

# Stop the service
sudo systemctl stop polybot.service

# Start the service
sudo systemctl start polybot.service

# View service logs (systemd journal)
sudo journalctl -u polybot.service -f              # Follow live
sudo journalctl -u polybot.service --no-pager -n 50  # Last 50 lines
sudo journalctl -u polybot.service --since "1 hour ago"

# Enable service to start on boot (already enabled)
sudo systemctl enable polybot.service
```

### Service Configuration

**Location**: `/etc/systemd/system/polybot.service`

```ini
[Unit]
Description=PolyB0T Trading Bot with Web Dashboard
After=network.target

[Service]
Type=simple
User=beamo_beamosupport_com
WorkingDirectory=/home/beamo_beamosupport_com/Analytical-Probability-Experiment
ExecStart=/bin/bash /home/beamo_beamosupport_com/Analytical-Probability-Experiment/scripts/start_all.sh
Restart=always
RestartSec=10
Environment="PATH=/home/beamo_beamosupport_com/.local/bin:/usr/local/bin:/usr/bin:/bin"
StandardOutput=journal
StandardError=journal
SyslogIdentifier=polybot

[Install]
WantedBy=multi-user.target
```

### What the Service Runs

The service executes `scripts/start_all.sh` which:
1. Changes to project directory
2. Starts **Trading Bot**: `poetry run polyb0t run --live` (background)
3. Waits 5 seconds for bot initialization
4. Starts **API Server**: `poetry run polyb0t api --host 0.0.0.0 --port 8000` (background)
5. Waits for either process to exit, then stops both

Both processes run together - if one exits, the script stops both and systemd restarts everything (after 10 seconds).

---

## SSH Connection

```bash
# Quick connect
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194

# With host key acceptance
ssh -i /Users/HP/.ssh/gcp_vm -o StrictHostKeyChecking=accept-new beamo_beamosupport_com@34.2.57.194

# Execute single command
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "command here"
```

---

## Deploying Code Changes

**Standard deployment workflow:**

```bash
# 1. On local machine: commit and push changes
git add . && git commit -m "Your changes" && git push

# 2. SSH to server and pull
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "cd ~/Analytical-Probability-Experiment && git pull"

# 3. Restart the service to apply changes
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "sudo systemctl restart polybot.service"

# 4. Verify it's running
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "sudo systemctl status polybot.service"
```

**One-liner for deploy:**
```bash
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "cd ~/Analytical-Probability-Experiment && git pull && sudo systemctl restart polybot.service && sleep 3 && sudo systemctl status polybot.service"
```

---

## Directory Structure & File Sizes

```
~/Analytical-Probability-Experiment/
├── .env                          # Configuration (SENSITIVE - has API keys)
├── polybot.db                    # Main SQLite database (~12GB)
├── live_run.log                  # Bot logs (~52MB, grows continuously)
├── poetry.lock                   # ~488KB
├── README.md                     # ~18KB
├── scripts/
│   ├── start_all.sh              # Startup script (called by systemd)
│   ├── check_ai_status.py        # ~58KB - AI diagnostics script
│   └── polybot.service           # Service file template
├── data/
│   ├── ai_training.db            # Training examples (~762MB + 100MB WAL)
│   ├── training_data.db          # Price history (~62MB)
│   ├── category_stats.db         # Category performance (~56KB)
│   ├── market_memory.db          # Market memory (~28KB)
│   ├── rule_weights.db           # Rule weights (~20KB)
│   ├── ai_orchestrator_state.json # ~4KB
│   ├── system_stats.json         # ~4KB
│   ├── ai_models/                # ~327MB total
│   │   ├── current_model.pkl     # ~12MB - deployed model
│   │   ├── orchestrator_state.json
│   │   ├── trainer_state.json
│   │   └── versions/             # Model version history
│   └── moe_models/               # ~36MB total
│       ├── experts/              # Individual expert models
│       ├── gating.model.pkl      # ~39KB - gating network
│       ├── gating.meta.pkl       # ~1.2KB
│       ├── pool_state.json       # ~825B - expert pool state
│       └── versions/             # Expert version history
├── docs/                         # Documentation (~9KB index)
├── tests/                        # Test files
└── polyb0t/                      # Main Python package
    ├── api/                      # FastAPI REST endpoints
    ├── cli/                      # CLI commands
    ├── config/                   # Settings (POLYBOT_ env prefix)
    ├── data/                     # API clients, storage
    ├── execution/                # Portfolio, orders, intents
    ├── ml/                       # AI orchestrator, MoE
    ├── models/                   # Strategy, filters, risk
    └── services/                 # Scheduler, Discord, health
```

---

## Database Reference

### Main Database (`polybot.db`) - ~12GB

| Table | Rows | Purpose |
|-------|------|---------|
| `orderbook_snapshots` | 3,950,136 | Historical orderbook data (LARGEST) |
| `signals` | 145,102 | Generated trading signals |
| `balance_snapshots` | 47,605 | Historical balance tracking |
| `account_states` | 45,939 | Account state history |
| `trade_intents` | 18,589 | Live mode approval queue |
| `markets` | 5,350 | Market metadata |
| `market_outcomes` | 822 | Outcome tokens |
| `pnl_snapshots` | 32,747 | P&L history |
| `closed_trades` | 0 | Trade P&L tracking (NEW) |
| `kill_switch_events` | 0 | Kill switch triggers |
| `portfolio_positions` | 0 | Position tracking |
| `simulated_orders` | 0 | Paper trading orders |
| `simulated_fills` | 0 | Paper trading fills |
| `trades` | 0 | Executed trades |

#### Key Table Schemas

**trade_intents** - Live mode approval queue
```
intent_uuid, intent_id, cycle_id, intent_type, fingerprint
token_id, market_id, side, price, size, size_usd, edge
p_market, p_model, reason, risk_checks (JSON), signal_data (JSON)
status, created_at, expires_at, approved_at, approved_by
executed_at, execution_result (JSON), submitted_order_id
error_message, superseded_by_intent_id, superseded_at
```

**signals** - Generated trading signals
```
cycle_id, token_id, market_id, timestamp
p_market, p_model, edge, features (JSON)
signal_type, confidence, created_at
```

**closed_trades** - Trade P&L tracking (NEW)
```
trade_id, open_intent_id, close_intent_id
token_id, market_id, side
entry_price, entry_size_usd, entry_time
exit_price, exit_size_usd, exit_time, exit_reason
realized_pnl_usd, realized_pnl_pct, is_winner
hold_time_hours, status, created_at, updated_at
```

### AI Training Database (`data/ai_training.db`) - ~762MB

| Table | Rows | Purpose |
|-------|------|---------|
| `market_snapshots` | 131,248 | Point-in-time market data |
| `price_history` | 125,118 | Price tracking for labeling |
| `training_examples` | 42,941 | Labeled ML training data |
| `tracked_markets` | 784 | Markets being tracked |
| `collector_state` | 1 | Data collector state |
| `storage_stats` | 1 | Storage statistics |

#### Training Examples Schema
```
example_id, token_id, market_id, created_at
features (JSON), category, market_title
price_change_15m, price_change_1h, price_change_4h
price_change_24h, price_change_7d, price_change_to_resolution
direction_1h, direction_24h, predicted_change
resolved_outcome, labeled_at, is_fully_labeled
schema_version, available_features, prediction_evaluated
```

### Training Data Database (`data/training_data.db`) - ~62MB

| Table | Rows | Purpose |
|-------|------|---------|
| `price_history` | 295,564 | Extended price history |
| `training_data` | 0 | (unused) |
| `model_performance` | 0 | (unused) |

### Database Access (sqlite3 not installed)

Use Python to query databases:
```bash
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && python3 -c \"
import sqlite3
conn = sqlite3.connect('polybot.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM trade_intents')
print('Trade intents:', cursor.fetchone()[0])
\""
```

---

## AI/ML System

### Mixture of Experts (MoE) Architecture

The system uses a gating network to route predictions to specialized expert models.

**Pool State** (`data/moe_models/pool_state.json`):
- Total training cycles: 10
- Total trades tracked: 62
- Best expert: `high_volatility`
- Total profit: ~0.39%

**26 Expert Models** (in `data/moe_models/experts/`):

| Category | Experts |
|----------|---------|
| **Topic-based** | sports, politics_us, politics_intl, crypto, economics, entertainment, tech, weather, science, legal |
| **Risk-based** | low_risk, medium_risk, high_risk |
| **Time-based** | short_term, medium_term, long_term |
| **Volume-based** | high_volume, low_volume |
| **Market-based** | high_liquidity, high_volatility, low_volatility |
| **Behavioral** | momentum_strong, weekend_trader, market_close |
| **Dynamic** | dynamic_combo_* (auto-created based on patterns) |

Expert files include:
- `{expert}.model.pkl` - Trained model (~200KB-3MB)
- `{expert}.meta.pkl` - Model metadata (~1-2KB)
- `{expert}.versions.json` - Version history

### AI Orchestrator State (`data/ai_models/orchestrator_state.json`)

```json
{
  "last_training_time": "2026-01-27T20:28:40.825925",
  "last_example_time": "2026-01-27T20:39:08.863928"
}
```

Training examples are created every 5 minutes during operation.

### Model Training Flow

1. **Data Collection**: Scheduler collects market snapshots every cycle
2. **Example Creation**: AIOrchestrator creates training examples every 5 minutes
3. **Labeling**: Price changes are tracked and examples labeled when prices move
4. **Training**: MoE experts retrain periodically with new labeled data
5. **Prediction**: Gating network routes inference to best expert(s)

### Automatic Retraining System (Added 2026-01-28)

The system now automatically retrains every 6 hours with model comparison:

1. **Scheduled Training**: Every 6 hours (`ai_retrain_interval_hours` in settings)
2. **Model Comparison**: New model must show >= 1% improvement in simulated profit
3. **Training History**: All training runs recorded to `data/training_history.db`
4. **Versioning**: Each model version tracked with full metrics

Training history database tracks:
- `training_runs` - Individual training runs with metrics
- `training_cycles` - Batch training cycles
- `model_comparisons` - A/B comparison decisions

---

## Configuration Reference

### .env File (Key Non-Sensitive Settings)

```bash
# Mode
POLYBOT_MODE=live
POLYBOT_DRY_RUN=true              # Safety: prevents real orders
POLYBOT_LOOP_INTERVAL_SECONDS=10  # Trading cycle interval

# Wallet
POLYBOT_USER_ADDRESS=0x5cbB1a163F426097578EB4de9e3ECD987Fc1c0d4

# Database
POLYBOT_DB_URL=sqlite:///./polybot.db

# Market Filtering
POLYBOT_RESOLVE_MIN_DAYS=7        # Min days until resolution
POLYBOT_RESOLVE_MAX_DAYS=180      # Max days until resolution
POLYBOT_MIN_LIQUIDITY=100         # Min market liquidity USD
POLYBOT_MAX_SPREAD=0.10           # Max bid-ask spread (10%)

# Strategy
POLYBOT_EDGE_THRESHOLD=0.01       # Min edge to trade (1%)
POLYBOT_MIN_NET_EDGE=0.005        # Min net edge after costs
POLYBOT_MIN_ORDER_USD=1.0         # Min order size

# Risk Management
POLYBOT_DRAWDOWN_LIMIT_PCT=25.0   # Max drawdown before kill switch
POLYBOT_MAX_DAILY_LOSS_PCT=15.0   # Max daily loss

# Execution
POLYBOT_ORDER_TIMEOUT_SECONDS=300
POLYBOT_SLIPPAGE_BPS=10
POLYBOT_INTENT_EXPIRY_SECONDS=90
POLYBOT_INTENT_COOLDOWN_SECONDS=60
POLYBOT_MAX_OPEN_ORDERS=5

# Behavior
POLYBOT_AUTO_APPROVE_INTENTS=false
POLYBOT_LIVE_ALLOW_OPEN_SELL_INTENTS=false

# AI Training
POLYBOT_AI_TRAINING_MODE=online   # 'online' for continuous training
POLYBOT_AI_RETRAIN_INTERVAL_HOURS=6  # Retrain every 6 hours
POLYBOT_AI_MIN_IMPROVEMENT_PCT=1.0   # Min improvement to deploy new model

# Logging
POLYBOT_LOG_LEVEL=INFO
```

---

## API Endpoints

Base URL: `http://34.2.57.194:8000`

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /status` | Bot status |
| `GET /markets` | Tradable markets |
| `GET /positions` | Current positions |
| `GET /intents` | Pending intents |
| `GET /report` | Trading report |
| `GET /metrics` | Performance metrics |
| `WebSocket /ws` | Real-time updates |

---

## Useful SSH One-Liners

```bash
# Check service status
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "sudo systemctl status polybot.service"

# View recent application logs
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "tail -50 ~/Analytical-Probability-Experiment/live_run.log"

# View systemd journal logs
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "sudo journalctl -u polybot.service --no-pager -n 30"

# Check disk space
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "df -h"

# Check memory
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "free -h"

# Check running processes
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "ps aux | grep polyb0t | grep -v grep"

# Run polyb0t CLI commands
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t status"

# Git pull latest changes
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "cd ~/Analytical-Probability-Experiment && git pull"

# Check database row counts
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && python3 -c \"
import sqlite3
conn = sqlite3.connect('polybot.db')
cursor = conn.cursor()
for table in ['trade_intents', 'signals', 'markets']:
    cursor.execute(f'SELECT COUNT(*) FROM {table}')
    print(f'{table}: {cursor.fetchone()[0]}')
\""

# Check AI training stats
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && python3 -c \"
import sqlite3
conn = sqlite3.connect('data/ai_training.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM training_examples')
print(f'Training examples: {cursor.fetchone()[0]}')
\""

# View MoE pool state
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cat ~/Analytical-Probability-Experiment/data/moe_models/pool_state.json"

# Check training history (after first training cycle)
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && python3 -c \"
import sqlite3
conn = sqlite3.connect('data/training_history.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM training_runs')
print(f'Training runs: {cursor.fetchone()[0]}')
cursor.execute('SELECT expert_name, validation_accuracy, simulated_profit_pct, deployed FROM training_runs ORDER BY trained_at DESC LIMIT 5')
for row in cursor.fetchall():
    print(f'  {row[0]}: acc={row[1]:.1%}, profit={row[2]:.2%}, deployed={row[3]}')
\""
```

---

## Troubleshooting

### Service won't start?
```bash
# Check detailed error
sudo journalctl -u polybot.service --no-pager -n 100

# Check if port 8000 is in use
sudo ss -tlnp | grep 8000

# Check for Python syntax errors
cd ~/Analytical-Probability-Experiment
source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate
python -m py_compile polyb0t/cli/main.py
```

### Database locked errors?
This can happen when multiple processes access SQLite concurrently. The system now uses WAL mode and busy_timeout to mitigate this.
```bash
# Check for multiple bot processes
ps aux | grep polyb0t | grep -v grep

# If duplicates exist, restart cleanly
sudo systemctl stop polybot.service
sleep 5
sudo systemctl start polybot.service
```

### WebSocket reconnection loops?
Normal behavior - the WebSocket client reconnects automatically after timeouts (340s default).

### Need to manually test?
```bash
# Only for testing - normally use systemd!
cd ~/Analytical-Probability-Experiment
source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate
polyb0t run --live  # Run in foreground to see output
```

### Large log file?
```bash
# Check log size
ls -lh live_run.log

# Truncate log file (keeps last 1000 lines)
tail -1000 live_run.log > live_run.log.tmp && mv live_run.log.tmp live_run.log
```

### Check for Python errors:
```bash
grep -i "error\|exception\|traceback" live_run.log | tail -30
```

### Check AI system health:
```bash
cd ~/Analytical-Probability-Experiment
source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate
python scripts/check_ai_status.py
```

### Model showing 100% accuracy (data leakage)?
This was fixed on 2026-01-28 by:
1. Removing leaky features (`price_change_1h/4h/24h`) from training
2. Changing from time-based to stratified train/validation split

Current expected accuracy: 65-75%

---

## Typical Runtime Behavior

During normal operation, each trading cycle (~10 seconds):
1. Fetches balance snapshot (~$289 USDC currently)
2. Fetches account positions (1 active: Super Bowl)
3. Fetches 5000+ markets from Polymarket API
4. Filters to ~2900 tradable (by liquidity, resolution date)
5. Enriches top 250 markets with orderbook data
6. Filters to ~67 after spread/depth checks
7. Generates AI signals (confidence ~67-69%, edge ~55-63%)
8. Creates training examples every 5 minutes
9. Expires old intents (90s expiry)
10. In DRY_RUN mode: logs signals but doesn't execute

---

## Important Reminders

1. **ALWAYS use systemctl** to manage the bot, not pkill/nohup
2. **DRY_RUN=true** is the safe default - bot doesn't submit real orders
3. **Restart service after code changes** - `sudo systemctl restart polybot.service`
4. **Check both logs**: `live_run.log` AND `journalctl -u polybot.service`
5. **The main database is ~12GB** - be careful with full table scans
6. **No crontab** - all scheduling handled by the bot itself
7. **Poetry not in PATH** - use `~/.local/bin/poetry` or activate virtualenv first
8. **sqlite3 CLI not installed** - use Python for database queries
9. **Training history** - stored in `data/training_history.db` after first training cycle

---

## Change Log

- **2026-01-28**: Fixed 100% accuracy bug (data leakage + stratified split)
- **2026-01-28**: Added automatic 6-hour retraining with model comparison
- **2026-01-28**: Added training history database for diagnostics
- **2026-01-28**: Fixed SQLite database locking (WAL mode + busy_timeout)
- **2026-01-28**: Fixed OpenAI API compatibility (`max_completion_tokens`)
