# Live Mode Configuration Guide

This document explains how to configure PolyB0T for live mode with human-in-the-loop trading.

## Overview

Live mode enables real-time data monitoring and trade execution **with mandatory human approval**. All trading actions require explicit user confirmation before execution.

## Configuration File

Copy your existing `.env` file or create a new one with these settings:

```bash
# Mode Configuration
POLYBOT_MODE=live                    # Enable live mode
POLYBOT_DRY_RUN=true                # CRITICAL: Start with dry-run enabled

# Timing (10-second refresh for live mode)
POLYBOT_LOOP_INTERVAL_SECONDS=10

# Intent System
POLYBOT_INTENT_EXPIRY_SECONDS=60    # Intents expire after 60 seconds

# Risk Management (Enhanced for live mode)
POLYBOT_MAX_DAILY_LOSS_PCT=10.0
POLYBOT_MAX_ORDERS_PER_HOUR=20
POLYBOT_MAX_NOTIONAL_PER_MARKET=1000.0

# Exit Management
POLYBOT_ENABLE_TAKE_PROFIT=true
POLYBOT_TAKE_PROFIT_PCT=10.0
POLYBOT_ENABLE_STOP_LOSS=true
POLYBOT_STOP_LOSS_PCT=5.0
POLYBOT_ENABLE_TIME_EXIT=true
POLYBOT_TIME_EXIT_DAYS_BEFORE=2

# Kill Switches
POLYBOT_MAX_API_ERROR_RATE_PCT=50.0
POLYBOT_MAX_STALE_DATA_SECONDS=60
POLYBOT_MAX_SPREAD_MULTIPLIER=3.0

# Account Access (if available)
# POLYBOT_POLYMARKET_WALLET_ADDRESS=0x...
# POLYBOT_CLOB_API_KEY=...          # Only if required for read access
```

## Live Mode Credentials

**IMPORTANT**: Only use official Polymarket methods to generate credentials.

### For Order Execution (when DRY_RUN=false):

```bash
# L2 Private Key (for signing orders)
POLYBOT_POLYGON_PRIVATE_KEY=0x...

# CLOB API Credentials (if required)
POLYBOT_CLOB_API_KEY=...
POLYBOT_CLOB_API_SECRET=...
POLYBOT_CLOB_PASSPHRASE=...
```

### Security Rules:

1. **NEVER** commit credentials to git
2. **NEVER** print or log private keys
3. **NEVER** share credentials
4. Store in `.env` file only (gitignored)
5. Use separate credentials for testing vs production
6. Rotate credentials regularly

## Dry-Run vs Live Execution

### Dry-Run Mode (Recommended to Start)

```bash
POLYBOT_DRY_RUN=true
```

- Connects to live data feeds
- Creates trade intents
- Allows approval workflow testing
- **Logs** approved actions but does NOT execute them
- Safe for testing and validation
- No real orders placed

### Live Execution Mode

```bash
POLYBOT_DRY_RUN=false
```

- Approved intents **WILL** place real orders
- Uses real funds
- Subject to trading fees
- **Requires explicit confirmation** at startup
- Only enable after extensive dry-run testing

## Allowing SELL intents (including manual positions)

SELL can either:
- **Reduce/close an existing LONG** position (including positions you opened manually), or
- **Open/increase a SHORT** position.

By default, PolyB0T will only allow SELL intents when you currently hold that token **LONG** (so it can reduce/close manual positions without opening new shorts).

If you want to also allow SELL intents that could open shorts, set:

```bash
POLYBOT_LIVE_ALLOW_OPEN_SELL_INTENTS=true
```

⚠️ Enabling this means the bot may propose/execute SELL orders that reduce or fully close positions you opened yourself.

## Kill Switches

Automated safety halts trigger when:

1. **Drawdown Limit**: Portfolio drops >15%
2. **Daily Loss Limit**: Daily loss exceeds 10%
3. **API Error Rate**: >50% API calls failing
4. **Stale Data**: Data age >60 seconds
5. **Spread Anomaly**: Spreads 3x normal levels

When triggered:
- Trading halts immediately
- All pending intents rejected
- Manual intervention required to resume

## Exit Management

Automatic exit proposals generated for:

### Take-Profit
- Triggers at +10% PnL (configurable)
- Creates exit intent for user approval
- Locks in gains

### Stop-Loss
- Triggers at -5% PnL (configurable)
- Creates exit intent for user approval
- Limits losses

### Time-Based
- Triggers 2 days before market resolution
- Reduces resolution risk
- Ensures liquidity availability

### Liquidity-Based
- Triggers when spreads deteriorate
- Protects against illiquid markets
- Automatic monitoring

All exits require user approval - no automatic execution.

## Account State Tracking

If wallet address is configured:

```bash
POLYBOT_POLYMARKET_WALLET_ADDRESS=0x...
```

The bot will attempt to fetch:
- Current positions
- Open orders
- Account balances

**Note**: Endpoint availability depends on Polymarket API. Some endpoints may require authentication or may not be publicly available.

## Rate Limiting

To avoid rate limits:

1. Set `POLYBOT_LOOP_INTERVAL_SECONDS=10` minimum
2. Monitor `POLYBOT_MAX_ORDERS_PER_HOUR` (default 20)
3. Watch API error logs
4. Implement exponential backoff (built-in)

## Monitoring

### Via CLI

```bash
# List pending intents
polyb0t intents list

# Approve specific intent
polyb0t intents approve <intent_id>

# Reject intent
polyb0t intents reject <intent_id>

# Check status
polyb0t status
```

### Via API

```bash
# Get pending intents
curl http://localhost:8000/intents?status=PENDING

# Approve intent
curl -X POST http://localhost:8000/intents/{intent_id}/approve

# Reject intent
curl -X POST http://localhost:8000/intents/{intent_id}/reject
```

## Step-by-Step Setup

### 1. Initial Testing (Dry-Run)

```bash
# Configure for dry-run
export POLYBOT_MODE=live
export POLYBOT_DRY_RUN=true

# Initialize database
polyb0t db init

# Start bot
polyb0t run --live
```

### 2. Monitor & Approve

In another terminal:

```bash
# Watch for intents
polyb0t intents list

# Approve good ones
polyb0t intents approve <id>
```

### 3. Validate Behavior

- Confirm intents are created correctly
- Verify risk checks are working
- Check exit proposals are reasonable
- Validate kill switches trigger appropriately
- Review logs for any issues

### 4. Enable Live Execution (After Extensive Testing)

```bash
# ONLY after thorough dry-run validation
export POLYBOT_DRY_RUN=false

# Confirm you understand risks
polyb0t run --live
# Bot will ask for confirmation
```

## Troubleshooting

### Intents Not Being Created

- Check `POLYBOT_EDGE_THRESHOLD` isn't too high
- Verify markets pass filters
- Check logs for risk check failures

### Kill Switches Triggering

- Review specific trigger in logs
- Adjust thresholds if appropriate
- Clear with: `polyb0t kill-switches clear <type>`

### API Errors

- Check credentials are correct
- Verify rate limits not exceeded
- Ensure network connectivity
- Check Polymarket API status

### No Account Data

- Verify wallet address is correct
- Check if auth is required for endpoints
- Review API endpoint assumptions in code
- Some endpoints may not be publicly available

## Safety Checklist

Before enabling live execution:

- [ ] Tested extensively in dry-run mode
- [ ] Understand all risk parameters
- [ ] Reviewed and adjusted kill switch thresholds
- [ ] Tested intent approval workflow
- [ ] Verified exit proposals are appropriate
- [ ] Have monitoring system in place
- [ ] Understand financial risks
- [ ] Complied with Polymarket ToS
- [ ] Started with small position limits
- [ ] Have emergency stop plan

## Compliance

- ✅ Respect geographic restrictions
- ✅ Follow Polymarket Terms of Service
- ✅ Use official credential generation methods only
- ✅ No automated trading without approval
- ✅ Understand regulatory implications
- ✅ Keep accurate records

## Support

For issues:
1. Check logs in structured JSON format
2. Review kill switch events table
3. Examine intent approval workflow
4. Validate configuration settings
5. Test with paper mode first

**Remember**: Live mode with DRY_RUN=false uses real funds. Start conservatively and scale gradually.

