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
Scheduler (Main Loop)
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
- `polyb0t/cli/` - Command-line interface
- `polyb0t/config/` - Pydantic settings with `POLYBOT_` env prefix
- `polyb0t/data/` - API clients (Gamma, CLOB, WebSocket), models, storage
- `polyb0t/execution/` - Portfolio, orders, paper simulator, live executor, intents
- `polyb0t/ml/` - AI orchestrator, trainer, MoE experts, continuous learning
- `polyb0t/models/` - Strategy, features, filters, risk management
- `polyb0t/services/` - Scheduler, reporter, Discord notifier, health checks

### Data Flow

1. **Market Discovery**: GammaClient → MarketFilter → filtered universe
2. **Signal Generation**: FeatureEngine → BaselineStrategy + AIOrchestrator → TradingSignal
3. **Risk Check**: RiskManager validates position sizing and exposure
4. **Execution**:
   - Paper: PaperTradingSimulator with slippage/fees
   - Live: Creates TradeIntent → user approves via CLI → LiveExecutor submits to CLOB
5. **Learning**: ContinuousDataCollector → AITrainer → MoE ensemble updates

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
