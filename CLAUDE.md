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
make check-all                    # All checks (lint + format + type + test)

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

# Useful Scripts
python scripts/train_now.py       # Manually trigger ML training
python scripts/check_ai_status.py # Inspect AI/MoE system state
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
- `polyb0t/ml/validation/` - Resolved market validation, confidence calibration, Brier scores
- `polyb0t/models/` - Strategy, features, filters, risk management
- `polyb0t/services/` - Scheduler, reporter, health checks, arbitrage scanner
- `polyb0t/backtest/` - Backtesting engine, strategy interfaces, analytics (separate from main loop)

### Data Flow

1. **Market Discovery**: GammaClient → MarketFilter → filtered universe
2. **Signal Generation**: FeatureEngine → BaselineStrategy + AIOrchestrator → TradingSignal
3. **Risk Check**: RiskManager validates position sizing and exposure
4. **Execution**:
   - Paper: PaperTradingSimulator with slippage/fees
   - Live: Creates TradeIntent → user approves via CLI → LiveExecutor submits to CLOB
5. **Learning**: ContinuousDataCollector → MoETrainer → ExpertPool updates

### ML/MoE Architecture

- **AIOrchestrator** (`ml/ai_orchestrator.py`) - Central coordinator for data collection, training, and prediction
- **MoE System** (`ml/moe/`) - Mixture of Experts ensemble:
  - `expert_pool.py` - Manages collection of trained experts (32 experts: categories, risk, time, volume, volatility, timing, horizon, news, smart money)
  - `trainer.py` - Trains experts and gating network (optimizes for profitability, not accuracy)
  - `expert.py` - Individual expert models (sklearn classifiers or neural networks)
  - `deep_ensemble.py` - Deep learning backends (PyTorch, XGBoost, LightGBM)
  - `versioning.py` - Expert lifecycle state machine: ACTIVE → PROBATION → SUSPENDED → DEPRECATED
- **Data Collection** (`continuous_collector.py`, `historical_fetcher.py`) - Collects training data with automatic backfill

Training runs every 6 hours (configurable via `POLYBOT_AI_RETRAIN_INTERVAL_HOURS`) and hot-swaps models without downtime.

**ML Data Storage** (separate from main `polybot.db`):
- `data/ai_training.db` - Continuous training snapshots
- `data/historical_prices.db` - Price timeseries
- `data/historical_training.db` - Resolved market outcomes
- `data/moe_models/{expert_id}/expert_v{version}.pkl` - Expert models (pickle)
- `data/moe_models/gating/gating_v{version}.pkl` - Gating network
- `.pt` files for PyTorch deep learning models

## Critical Patterns & Gotchas

### Singleton Pattern (used everywhere)

All core components use `get_*()` factory functions. Never instantiate directly.

Two patterns exist:
1. **`@lru_cache`** - Used only for `get_settings()`. Clear with `get_settings.cache_clear()`.
2. **Global variable** - Used for everything else (`get_ai_orchestrator()`, `get_expert_pool()`, `get_moe_trainer()`, `get_session()`, etc.). Uses `global _instance` with None check.

### Two "models" Packages (don't confuse them)

- `polyb0t.data.models` - **Pydantic** models for API response validation (Market, OrderBook, Trade)
- `polyb0t.data.storage` - **SQLAlchemy** ORM models for DB persistence (MarketDB, SignalDB, etc., inherit from `Base`)

### Database Sessions

- `init_db()` must be called before `get_session()` or it raises `RuntimeError`
- Sessions are NOT auto-closed — always call `session.close()` explicitly
- SQLite has best-effort additive schema migrations in `init_db()` via `_ensure_sqlite_*` functions

### Async/Event Loop

- Most I/O is `async/await`. CLI entry points use `asyncio.run()` at the top level.
- Never call `asyncio.run()` inside an already-running event loop — use the sync/async bridge pattern from `services/outcome_recorder.py` (`asyncio.get_event_loop()` + `ensure_future` or `run_until_complete`).
- Tests use `asyncio_mode = "auto"` — just write `async def test_*()` functions.

### Threading

The codebase mixes async (I/O) and threads (CPU-bound). Background threads exist in:
- `InsiderTracker` (thread-safe reputation updates with `threading.Lock`)
- `SystemMonitor` (resource monitoring)
- `ContinuousDataCollector` (optional background collection)
- `AITrainer` (`threading.Lock` to prevent concurrent training)

### ML Data Leakage Prevention

- `price_change_1h/4h/24h` are **labels, not features** — never include as input features
- Post-resolution `outcome.price` is ground truth — never use as a feature
- Validation uses `is_fully_labeled=1` and stored `predicted_change` to avoid look-ahead bias

### Environment Variables

All settings require `POLYBOT_` prefix. Required vars (validated at startup by `config/env_loader.py`):
- `POLYBOT_MODE` - `paper` or `live`
- `POLYBOT_DRY_RUN` - Safety switch
- `POLYBOT_USER_ADDRESS` - Wallet address
- `POLYBOT_LOOP_INTERVAL_SECONDS` - Main loop interval

See `.env.example` for full reference.

## Testing

Tests use pytest with `asyncio_mode = "auto"`. Key fixtures in `tests/conftest.py`:
- `_set_required_env` (autouse) - Sets required env vars AND calls `get_settings.cache_clear()`. This is critical — without it, settings from previous tests leak via the `@lru_cache`.
- `db_session` - In-memory SQLite session (creates/drops all tables per test)
- `sample_market`, `sample_orderbook`, `sample_trades` - Test data fixtures

When writing new tests, the autouse fixture handles env setup. If you need custom settings, use `monkeypatch.setenv()` before any code that calls `get_settings()`.

## Configuration

Key variables beyond the required ones:

- `POLYBOT_PLACING_ORDERS` - Master trading switch (False = monitor/collect only)
- `POLYBOT_CLOB_API_KEY/SECRET/PASSPHRASE` - L2 credentials for live trading
- `POLYBOT_EDGE_THRESHOLD` - Minimum edge to trade (default 5%)
- `POLYBOT_MAX_POSITION_PCT` - Max per-position % of bankroll
- `POLYBOT_MAX_TOTAL_EXPOSURE_PCT` - Max total exposure %
- `POLYBOT_AI_RETRAIN_INTERVAL_HOURS` - Training frequency (default 6)
- `POLYBOT_AI_USE_DEEP_LEARNING` - Enable neural networks (default True)

## Paper vs Live Mode

- **Paper**: `PaperTradingSimulator` simulates execution with realistic slippage/fees
- **Live**:
  - Bot creates `TradeIntent` (pending approval)
  - User runs `polyb0t intents approve <id>` to execute
  - Intents expire after 60s by default (dedup + cooldown prevents repeated intents)
  - `DRY_RUN=true` prevents order submission (approvals finalize as `EXECUTED_DRYRUN`)
  - Polymarket does NOT support shorting (selling tokens you don't own)

## GCP Production VM

The bot runs on a Google Cloud VM as a systemd service.

### Quick Reference

| Item | Value |
|------|-------|
| **SSH** | `ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194` |
| **Project Path** | `~/Analytical-Probability-Experiment` |
| **Service** | `polybot.service` |
| **Virtualenv** | `~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12` |

### Systemd Service Management

**IMPORTANT**: Always use `systemctl` commands, NOT manual process management (pkill/nohup).

```bash
sudo systemctl restart polybot.service           # Restart (use after code changes)
sudo systemctl status polybot.service             # Check status
sudo journalctl -u polybot.service -f             # Follow live logs
sudo journalctl -u polybot.service --no-pager -n 50  # Last 50 lines
```

The service runs `scripts/start_all.sh` which starts both the trading bot (`polyb0t run --live`) and API server (`polyb0t api --host 0.0.0.0 --port 8000`).

### Deployment

```bash
# Deploy from local (one-liner)
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "cd ~/Analytical-Probability-Experiment && git pull && sudo systemctl restart polybot.service"
```

### Key Paths on VM

- **Config**: `~/Analytical-Probability-Experiment/.env`
- **Main DB**: `~/Analytical-Probability-Experiment/polybot.db` (~12GB)
- **Logs**: `~/Analytical-Probability-Experiment/live_run.log`
- **AI/MoE Models**: `~/Analytical-Probability-Experiment/data/ai_models/` and `data/moe_models/`

### Common SSH Commands

```bash
# View logs
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "tail -100 ~/Analytical-Probability-Experiment/live_run.log"

# Check for errors
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "grep -i 'error\|exception\|traceback' ~/Analytical-Probability-Experiment/live_run.log | tail -30"

# Run CLI command on VM
ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 \
  "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t status"
```

### API Access

The API runs on port 8000: `http://34.2.57.194:8000`

Endpoints: `/health`, `/status`, `/markets`, `/positions`, `/intents`, `/report`, `/metrics`, WebSocket `/ws`

See `docs/GCP_VM_REFERENCE.md` for comprehensive VM documentation including database schemas, troubleshooting, and detailed configuration.
