# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PolyB0T is an autonomous trading bot for Polymarket prediction markets on Polygon. It uses a Mixture of Experts (MoE) ensemble with continuous learning for signal generation, supports both paper and live trading (with human-in-the-loop approval), and includes comprehensive risk management.

## Commands

```bash
# Setup
poetry install                    # Install dependencies
poetry install --with dev         # Include dev tools
poetry shell                      # Enter virtualenv

# Database
polyb0t db init                   # Initialize schema
polyb0t db reset                  # Reset (deletes all data)

# Testing
pytest tests/ -v                  # Run all tests
pytest tests/test_strategy.py -v  # Single file
pytest tests/test_strategy.py::test_function -v  # Single test
pytest --cov=polyb0t              # With coverage

# Code Quality
make lint                         # Ruff linting
make format                       # Black formatting
make type-check                   # Mypy type checking
make check-all                    # All checks

# Running
polyb0t run --paper               # Paper trading (default)
polyb0t run --live                # Live mode (approval-gated)
polyb0t api                       # REST API server

# Live Mode Intent Management
polyb0t intents list              # View pending intents
polyb0t intents approve <id>      # Approve trade
polyb0t intents reject <id>       # Reject trade

# Monitoring & Diagnostics
polyb0t status                    # Portfolio status
polyb0t report --today            # Daily report
polyb0t doctor                    # System diagnostics
polyb0t diagnose-filters          # Analyze why markets are being filtered out
```

## Architecture

```
TradingScheduler (Main Loop - polyb0t/services/scheduler.py)
    │
    ├── GammaClient → fetches markets from Polymarket API
    ├── MarketFilter → filters by liquidity, spread, resolution date
    ├── BaselineStrategy → value-based signals (p_market vs p_model)
    ├── AIOrchestrator/MoE → ensemble predictions
    └── RiskManager → position sizing, exposure limits, kill switches
            │
            ├── Paper: PaperTradingSimulator (simulates fills)
            └── Live: TradeIntent → user approval → LiveExecutor → CLOB
```

### Key Directories

- `polyb0t/api/` - FastAPI REST endpoints & WebSocket server
- `polyb0t/cli/` - Command-line interface (Click-based)
- `polyb0t/config/` - Pydantic settings with `POLYBOT_` env prefix
- `polyb0t/data/` - API clients (Gamma, CLOB, WebSocket), Pydantic models, SQLAlchemy storage
- `polyb0t/execution/` - Portfolio, orders, paper simulator, live executor, intents
- `polyb0t/ml/` - AI orchestrator, trainer, MoE experts, continuous learning
- `polyb0t/models/` - Strategy, features, filters, risk management
- `polyb0t/services/` - Scheduler, reporter, health checks, arbitrage scanner

### ML/MoE Architecture

The `polyb0t/ml/` directory contains the machine learning system:

- **AIOrchestrator** (`ai_orchestrator.py`) - Central coordinator for data collection, training, and prediction
- **MoE System** (`moe/`) - Mixture of Experts ensemble:
  - `expert_pool.py` - Manages collection of trained experts
  - `trainer.py` - Trains experts and gating network (optimizes for profitability, not accuracy)
  - `expert.py` - Individual expert models (sklearn classifiers or neural networks)
  - `deep_ensemble.py` - Deep learning backends (PyTorch, XGBoost, LightGBM)
  - `auto_discovery.py` - Automatically discovers new expert opportunities
- **Data Collection** (`continuous_collector.py`, `historical_fetcher.py`) - Collects training data with automatic backfill
- **Category Tracker** (`category_tracker.py`) - Learns which market types to avoid

Training runs every 6 hours and hot-swaps models without downtime.

### Data Flow

1. **Market Discovery**: GammaClient → MarketFilter → filtered universe
2. **Signal Generation**: FeatureEngine → BaselineStrategy + AIOrchestrator → TradingSignal
3. **Risk Check**: RiskManager validates position sizing and exposure
4. **Execution**:
   - Paper: PaperTradingSimulator with slippage/fees
   - Live: Creates TradeIntent → user approves via CLI → LiveExecutor submits to CLOB
5. **Learning**: ContinuousDataCollector → MoETrainer → ExpertPool updates

## Configuration

All settings via `.env` file with `POLYBOT_` prefix. Key variables:

- `POLYBOT_MODE` - `paper` or `live`
- `POLYBOT_DRY_RUN` - Safety switch for live mode (prevents actual orders)
- `POLYBOT_PLACING_ORDERS` - Master trading switch
- `POLYBOT_CLOB_API_KEY/SECRET/PASSPHRASE` - L2 credentials for live trading
- `POLYBOT_EDGE_THRESHOLD` - Minimum edge to trade (default 5%)
- `POLYBOT_MAX_POSITION_PCT` - Max per-position % of bankroll
- `POLYBOT_MAX_TOTAL_EXPOSURE_PCT` - Max total exposure %

See `.env.example` for full reference.

## Key Patterns

- **Async-first**: All I/O uses `async/await`
- **Structured logging**: `structlog` with JSON format
- **Database persistence**: All events stored (orders, fills, signals, intents)
- **Human-in-the-loop**: Live mode requires explicit intent approval via CLI
- **Risk by default**: Hard limits on positions, exposure, drawdown with kill switches
- **AI primary, baseline fallback**: MoE ensemble predictions with explainable baseline strategy
- **Singleton pattern**: Core components use `get_*()` factory functions with caching (e.g., `get_settings()`, `get_ai_orchestrator()`, `get_expert_pool()`)

## Testing

Tests use pytest with `asyncio_mode = "auto"`. Key fixtures in `tests/conftest.py`:
- `db_session` - In-memory SQLite session
- `sample_market`, `sample_orderbook`, `sample_trades` - Test data fixtures
- `_set_required_env` (autouse) - Sets required env vars and clears settings cache

Run a single test function:
```bash
pytest tests/test_strategy.py::test_generate_signal_buy -v
```

## Paper vs Live Mode

- **Paper**: `PaperTradingSimulator` simulates execution with realistic slippage/fees
- **Live**:
  - Bot creates `TradeIntent` (pending approval)
  - User runs `polyb0t intents approve <id>` to execute
  - Intents expire after 60s by default
  - `DRY_RUN=true` prevents order submission for safety testing

## Database

SQLite (dev) or PostgreSQL (prod). Key tables:
- `markets`, `market_outcomes` - Market metadata
- `signals` - Generated trading signals
- `simulated_orders/fills` - Paper trading records
- `trade_intents` - Live mode approval queue
- `ai_training_examples` - ML training data

ML training data is stored separately in `data/ai_training.db` and `data/historical_prices.db`.

## GCP Production VM

The bot runs on a Google Cloud VM as a systemd service.

### Quick Reference

| Item | Value |
|------|-------|
| **SSH Command** | `ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194` |
| **Host IP** | `34.2.57.194` |
| **Username** | `beamo_beamosupport_com` |
| **SSH Key** | `/Users/HP/.ssh/gcp_vm` |
| **Project Path** | `~/Analytical-Probability-Experiment` |
| **Service Name** | `polybot.service` |

### VM Specs

| Resource | Value |
|----------|-------|
| **OS** | Ubuntu 24.04.3 LTS |
| **CPU** | AMD EPYC 7B12 (2 cores) |
| **Memory** | 7.8 GB |
| **Disk** | 145 GB total |
| **Python** | 3.12.3 |
| **Virtualenv** | `~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12` |

### Systemd Service Management

**IMPORTANT**: Always use `systemctl` commands, NOT manual process management (pkill/nohup).

```bash
# Restart (use after code changes)
sudo systemctl restart polybot.service

# Check status
sudo systemctl status polybot.service

# Stop/Start
sudo systemctl stop polybot.service
sudo systemctl start polybot.service

# View logs
sudo journalctl -u polybot.service -f              # Follow live
sudo journalctl -u polybot.service --no-pager -n 50  # Last 50 lines
```

The service runs `scripts/start_all.sh` which starts both the trading bot (`polyb0t run --live`) and API server (`polyb0t api --host 0.0.0.0 --port 8000`).

### Deployment Workflow

```bash
# 1. Local: commit and push
git add . && git commit -m "Your changes" && git push

# 2. Deploy to VM (one-liner)
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "cd ~/Analytical-Probability-Experiment && git pull && sudo systemctl restart polybot.service"

# 3. Verify
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "sudo systemctl status polybot.service"
```

### Common SSH Commands

```bash
# SSH into VM
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194

# View application logs
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "tail -100 ~/Analytical-Probability-Experiment/live_run.log"

# Check for errors
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "grep -i 'error\|exception\|traceback' ~/Analytical-Probability-Experiment/live_run.log | tail -30"

# Run polyb0t CLI command
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t status"

# Check disk/memory
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "df -h && free -h"
```

### Key Paths on VM

- **Project**: `~/Analytical-Probability-Experiment`
- **Config**: `~/Analytical-Probability-Experiment/.env`
- **Main DB**: `~/Analytical-Probability-Experiment/polybot.db` (~12GB)
- **Logs**: `~/Analytical-Probability-Experiment/live_run.log`
- **AI Models**: `~/Analytical-Probability-Experiment/data/ai_models/`
- **MoE Models**: `~/Analytical-Probability-Experiment/data/moe_models/`
- **Service File**: `/etc/systemd/system/polybot.service`

### API Access

The API runs on port 8000: `http://34.2.57.194:8000`

Endpoints: `/health`, `/status`, `/markets`, `/positions`, `/intents`, `/report`, `/metrics`, WebSocket `/ws`

See `docs/GCP_VM_REFERENCE.md` for comprehensive VM documentation including database schemas, troubleshooting, and detailed configuration.
