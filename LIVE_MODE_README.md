# PolyB0T Live Mode - Human-in-the-Loop Trading

## 🎯 Overview

PolyB0T now supports **live mode** with real-time data and human-in-the-loop execution. This mode connects to live Polymarket data feeds and proposes trades, but **ALL actions require explicit user approval** before execution.

### Key Features

✅ **Real-time data** - 10-second refresh cycles  
✅ **Human-in-the-loop** - Every trade requires approval  
✅ **Trade intents** - Proposed actions with expiry timers  
✅ **Account tracking** - Read-only position/balance monitoring  
✅ **Exit management** - Auto-proposed take-profit/stop-loss  
✅ **Kill switches** - Automated safety halts  
✅ **Dry-run mode** - Test workflow without execution  
✅ **Full audit trail** - Every decision logged  

## 🔒 Safety First

**NO AUTONOMOUS TRADING**: Every order, exit, and claim requires explicit user confirmation. The bot proposes, you decide.

### Default Safety Settings

- `DRY_RUN=true` by default in live mode
- Intents expire after 60 seconds
- Multiple kill switches (drawdown, API errors, stale data)
- Daily loss limits
- Position size caps
- Rate limiting

## 🚀 Quick Start

### 1. Install & Configure

```bash
# Install dependencies
poetry install

# Create configuration
cp .env.example .env

# Edit .env
POLYBOT_MODE=live
POLYBOT_DRY_RUN=true  # Start with dry-run!
POLYBOT_LOOP_INTERVAL_SECONDS=10
```

### 2. Initialize Database

```bash
polyb0t db init
```

### 3. Start Live Monitoring

```bash
polyb0t run --live
```

The bot will:
- Fetch live market data every 10 seconds
- Analyze opportunities
- Create trade intents
- Wait for your approval

### 4. Approve Trades (Separate Terminal)

```bash
# List pending intents
polyb0t intents list

# Approve specific intent
polyb0t intents approve <intent_id>

# Or reject
polyb0t intents reject <intent_id>
```

### 5. Monitor via API (Optional)

```bash
# Start API server
polyb0t api

# View intents
curl http://localhost:8000/intents

# Approve via API
curl -X POST http://localhost:8000/intents/{id}/approve
```

## 📋 Trade Intent Workflow

### What is a Trade Intent?

A **trade intent** is a proposed action awaiting your approval. It contains:

- **Type**: OPEN_POSITION, CLOSE_POSITION, CLAIM_SETTLEMENT
- **Details**: Market, token, side, price, size
- **Reason**: Human-readable explanation
- **Edge**: Expected advantage
- **Risk Checks**: All passed constraints
- **Expiry**: Time limit (default 60s)

### Intent Lifecycle

```
PENDING → (user approves) → APPROVED → (executor runs) → EXECUTED
       → (user rejects) → REJECTED
       → (timer expires) → EXPIRED
```

### Example Intent

```
Intent ID:    abc123...
Type:         OPEN_POSITION
Token:        token_yes_rain_tomorrow
Side:         BUY
Price:        0.598
Size:         $150.00
Edge:         +0.065 (model: 0.663, market: 0.598)
Reason:       BUY signal with 6.5% edge, confidence 0.82
Risk Checks:  ✓ All passed
Created:      2026-01-04T20:30:15Z
Expires:      2026-01-04T20:31:15Z
Time Left:    45s
```

**Your options**:
- `polyb0t intents approve abc123` → Execute trade (if dry-run=false)
- `polyb0t intents reject abc123` → Cancel
- Wait 45s → Intent expires automatically

## 🎮 Command Reference

### Running

```bash
# Paper trading (simulated)
polyb0t run --paper

# Live monitoring with approval workflow
polyb0t run --live
```

### Intent Management

```bash
# List pending intents
polyb0t intents list

# List all intents (including historical)
polyb0t intents list --all

# JSON output
polyb0t intents list --json-output

# Approve intent
polyb0t intents approve <intent_id>

# Reject intent
polyb0t intents reject <intent_id>

# Manually expire old intents
polyb0t intents expire
```

### Status & Reporting

```bash
# Current status
polyb0t status

# Daily report
polyb0t report --today

# View tradable universe
polyb0t universe
```

## 🛡️ Risk Management

### Position Limits

- **Max per position**: 2% of bankroll
- **Max total exposure**: 20% of bankroll
- **Max per category**: 10% of bankroll
- **Max per market**: $1,000 notional

### Rate Limits

- **Max orders per hour**: 20
- **Loop interval**: 10 seconds (configurable)
- **API retry**: Exponential backoff

### Kill Switches

Automated halts trigger when:

| Switch | Condition | Default Threshold |
|--------|-----------|-------------------|
| Drawdown | Portfolio loss | 15% |
| Daily Loss | Loss in 24h | 10% |
| API Errors | Failed API calls | 50% error rate |
| Stale Data | Data age | 60 seconds |
| Spread Anomaly | Spread vs normal | 3x multiplier |

When triggered:
- Trading halts immediately
- Pending intents rejected
- Logged to database
- Manual clear required

```bash
# View active kill switches
polyb0t status

# Clear specific switch (manual override)
polyb0t kill-switches clear DRAWDOWN
```

## 📊 Exit Management

Automatic exit proposals for positions:

### Take-Profit

- **Triggers**: Position up 10% (configurable)
- **Action**: Creates exit intent
- **Example**: "Take-profit: PnL 12.5% >= target 10%"

### Stop-Loss

- **Triggers**: Position down 5% (configurable)
- **Action**: Creates exit intent
- **Example**: "Stop-loss: Loss 6.2% >= limit 5%"

### Time-Based

- **Triggers**: 2 days before market resolution
- **Action**: Creates exit intent
- **Reason**: Avoid resolution risk

### Liquidity-Based

- **Triggers**: Spread widens significantly
- **Action**: Creates exit intent
- **Reason**: Deteriorating liquidity

**All exits require approval** - no automatic execution.

## 🔌 API Endpoints

When running `polyb0t api`:

### Intents

```bash
GET  /intents                    # List intents
GET  /intents?status=PENDING     # Filter by status
GET  /intents/{id}               # Get specific intent
POST /intents/{id}/approve       # Approve intent
POST /intents/{id}/reject        # Reject intent
```

### Status & Metrics

```bash
GET  /health                     # Health check
GET  /status                     # Positions & exposure
GET  /report                     # Trading report
GET  /metrics                    # Recent activity
```

### Example Usage

```bash
# Get pending intents
curl http://localhost:8000/intents?status=PENDING

# Approve intent
curl -X POST http://localhost:8000/intents/abc123/approve

# Check status
curl http://localhost:8000/status | jq
```

## 💻 Account State Tracking

If wallet address is configured:

```bash
POLYBOT_POLYMARKET_WALLET_ADDRESS=0x...
```

Bot attempts to fetch:
- Current positions
- Open orders
- Account balances

**Note**: Endpoint availability depends on Polymarket API. Some may require authentication or may not be publicly available. Implementation includes graceful fallbacks.

## ⚙️ Configuration

### Key Settings

```bash
# Mode
POLYBOT_MODE=live
POLYBOT_DRY_RUN=true              # IMPORTANT: Start with dry-run

# Timing
POLYBOT_LOOP_INTERVAL_SECONDS=10  # Live mode refresh rate

# Intents
POLYBOT_INTENT_EXPIRY_SECONDS=60  # How long intents remain valid

# Risk Limits
POLYBOT_MAX_DAILY_LOSS_PCT=10.0
POLYBOT_MAX_ORDERS_PER_HOUR=20
POLYBOT_MAX_NOTIONAL_PER_MARKET=1000.0

# Exit Triggers
POLYBOT_TAKE_PROFIT_PCT=10.0
POLYBOT_STOP_LOSS_PCT=5.0
POLYBOT_TIME_EXIT_DAYS_BEFORE=2

# Kill Switches
POLYBOT_MAX_API_ERROR_RATE_PCT=50.0
POLYBOT_MAX_STALE_DATA_SECONDS=60
POLYBOT_MAX_SPREAD_MULTIPLIER=3.0
```

See `CONFIG_LIVE_MODE.md` for complete reference.

## 🧪 Dry-Run vs Live Execution

### Dry-Run Mode (Recommended)

```bash
POLYBOT_DRY_RUN=true
```

- Connects to live data ✅
- Creates trade intents ✅
- Approval workflow ✅
- **Logs but doesn't execute** ✅
- Perfect for testing ✅
- Zero risk ✅

**Use case**: Test the entire workflow, validate strategy, train on approval process.

### Live Execution Mode

```bash
POLYBOT_DRY_RUN=false
```

- Approved intents **place real orders** ⚠️
- Uses real funds ⚠️
- Subject to fees ⚠️
- Requires startup confirmation ⚠️

**Use case**: After extensive dry-run testing and validation.

## 📝 Logging & Auditing

Every action is logged in structured JSON format:

```json
{
  "timestamp": "2026-01-04T20:30:15Z",
  "level": "INFO",
  "message": "Created trade intent",
  "intent_id": "abc123...",
  "intent_type": "OPEN_POSITION",
  "side": "BUY",
  "edge": 0.065,
  "cycle_id": "xyz789..."
}
```

Database tables track:
- All intents (created, approved, executed)
- Account state snapshots
- Kill switch events
- Risk check results
- Execution outcomes

## 🔐 Security & Credentials

### Required for Live Execution

```bash
# L2 private key (for signing orders)
POLYBOT_POLYGON_PRIVATE_KEY=0x...

# CLOB API credentials (if required)
POLYBOT_CLOB_API_KEY=...
POLYBOT_CLOB_API_SECRET=...
POLYBOT_CLOB_PASSPHRASE=...
```

### Security Rules

1. **NEVER** commit credentials to git
2. **NEVER** print or log private keys
3. **NEVER** share credentials
4. Store in `.env` only (gitignored)
5. Use official Polymarket methods only
6. Rotate credentials regularly
7. Separate test vs production keys

## ⚠️ Important Limitations

### MVP Implementation

The live executor includes **placeholder implementations** for:

- **Order submission** to CLOB API (needs actual signing logic)
- **Fill monitoring** (needs websocket or polling)
- **Settlement claiming** (needs on-chain transaction logic)

**These must be implemented based on official Polymarket documentation before enabling live execution.**

### Account State

Account state fetching makes assumptions about endpoint availability:

- Endpoint paths are assumed
- Authentication may be required
- Some endpoints may not exist
- Graceful fallbacks included

**Verify against actual Polymarket API documentation.**

## 🎓 Recommended Workflow

### Phase 1: Dry-Run Validation (1-2 weeks)

1. Configure with `DRY_RUN=true`
2. Run continuously
3. Approve intents as they come
4. Observe behavior
5. Validate risk checks
6. Check exit proposals
7. Monitor kill switches
8. Review logs and database

### Phase 2: Strategy Tuning

1. Adjust `EDGE_THRESHOLD` based on false positives
2. Tune exit thresholds
3. Refine risk limits
4. Optimize refresh interval
5. Test different market conditions

### Phase 3: Live Preparation

1. Implement actual CLOB API integration
2. Test order signing offline
3. Verify API credentials work
4. Start with minimal position sizes
5. Test with small bankroll first
6. Have emergency stop plan

### Phase 4: Live Execution (If Proceeding)

1. Set `DRY_RUN=false`
2. Confirm understanding of risks
3. Start with conservative limits
4. Monitor closely
5. Scale gradually
6. Keep manual override ready

## 🐛 Troubleshooting

### No Intents Being Created

- Lower `EDGE_THRESHOLD` temporarily
- Check market filters aren't too restrictive
- Verify live data is arriving
- Review signal generation logs

### Intents Expiring Before Approval

- Increase `INTENT_EXPIRY_SECONDS`
- Set up API monitoring for alerts
- Consider auto-approval for testing (dangerous!)

### Kill Switches Triggering

- Review trigger reason in logs
- Check if thresholds are too tight
- Investigate root cause (API issues, spread widening, etc.)
- Clear manually if false positive

### API Connection Issues

- Check network connectivity
- Verify API endpoints are correct
- Review rate limiting settings
- Check Polymarket API status

## 📚 Additional Resources

- `CONFIG_LIVE_MODE.md` - Detailed configuration guide
- `README.md` - Main project documentation
- `CONTRIBUTING.md` - Development guidelines
- Logs - Check structured JSON logs for details
- Database - Query tables for historical data

## ⚖️ Compliance & Disclaimers

**CRITICAL REMINDERS**:

- ✅ Comply with Polymarket Terms of Service
- ✅ Respect geographic restrictions
- ✅ Use official credential generation methods only
- ✅ No automated trading without approval
- ✅ Understand financial risks
- ✅ Start with dry-run mode
- ✅ Test extensively before live execution
- ✅ Trading involves risk of loss

**This software is provided "AS IS" for educational purposes. No warranties. Use at your own risk.**

## 🤝 Support

For issues:

1. Check logs (structured JSON format)
2. Review database tables (intents, kill_switch_events)
3. Validate configuration
4. Test in dry-run mode
5. Check API connectivity
6. Review documentation

---

**Built with safety as the top priority. Trade responsibly.** 🛡️

