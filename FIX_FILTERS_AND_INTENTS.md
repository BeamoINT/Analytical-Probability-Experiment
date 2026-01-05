# 🔧 Filter & Intent Fix - Final Solution

**Date**: January 4, 2026 @ 22:10  
**Issue**: Filters too strict, old intents stuck

---

## 🔴 Root Cause Identified

Looking at your logs:
```json
{
  "markets_tradable": 0,
  "rejection_reasons": {
    "spread_too_wide": 6,
    "insufficient_orderbook_depth": 2
  }
}
```

**ALL 8 markets were being rejected!**

### Why This Happened:

1. **Spread Filter Too Tight**: `max_spread = 0.05 (5%)`
   - 6 markets rejected for "spread_too_wide"
   - Prediction markets often have 5-10% spreads
   - 5% is too strict for most markets

2. **Depth Filter Too Strict**: `min_bid_depth = $50, min_ask_depth = $50`
   - 2 markets rejected for "insufficient_orderbook_depth"
   - Requiring $50 on each side ($100 total) is very high
   - Many good markets have $20-40 per side

3. **Result**: 
   - 0 tradable markets
   - 0 signals generated
   - 0 new intents created
   - Old intents kept reappearing

---

## ✅ Fixes Applied

### 1. Deleted Old Stuck Intents ✅

Forcibly expired intents for market 537485:
- Intent `563b422e` (BUY 0.365)
- Intent `c1291746` (SELL 0.635)

These were the same intents that kept reappearing for hours.

**SQL executed**:
```sql
UPDATE trade_intents 
SET status='EXPIRED' 
WHERE intent_id IN ('563b422e', 'c1291746') 
   OR (token_id IN ('569438894213', '297956940342') AND status='PENDING');
```

### 2. Loosened Spread Filter ✅

**Before**: `POLYBOT_MAX_SPREAD=0.05` (5%)  
**After**: `POLYBOT_MAX_SPREAD=0.10` (10%)

**Impact**: 
- Allows markets with 5-10% spreads
- More realistic for prediction markets
- Should allow 3-5 markets to pass

### 3. Lowered Depth Requirements ✅

**Before**: 
- `min_bid_depth_usd = 50.0`
- `min_ask_depth_usd = 50.0`
- `min_total_depth_usd = 100.0`

**After**:
- `min_bid_depth_usd = 20.0`
- `min_ask_depth_usd = 20.0`
- `min_total_depth_usd = 50.0`

**Impact**:
- More markets will have sufficient depth
- Still safe (requires $40 total depth)
- Better balance between opportunity and safety

---

## 🚀 Restart Bot to Apply

The fixes are applied, restart to see new markets:

```bash
# Stop current bot
pkill -f "polyb0t run"

# Start with new filters
poetry run polyb0t run
```

---

## 📊 What You Should See Now

### After Restart:

**✅ More Tradable Markets**:
```json
{
  "markets_tradable": 3,  // Not 0!
  "markets_filtered": {
    "spread_too_wide": 3,  // Fewer rejections
    "insufficient_orderbook_depth": 0  // Should be 0 or 1
  }
}
```

**✅ Signals Generated**:
```json
{
  "signals_generated": 2,  // Not 0!
  "intents_created": 2  // New intents!
}
```

**✅ New Intents**:
```bash
poetry run polyb0t intents list

# Should see DIFFERENT market IDs and intent IDs
# NOT market 537485 with same 0.365/0.635 prices
```

---

## 🔍 Verification Steps

### 1. Check Filters in Logs

After restart, look for:
```
"markets_tradable": N  // Should be > 0, ideally 2-5
```

If still 0, the filters need to be even looser.

### 2. Check Intent List

```bash
poetry run polyb0t intents list
```

Should see:
- **Different market IDs** (not just 537485)
- **Different intent IDs** (not 563b422e, c1291746)
- **Different prices** (not just 0.365/0.635)

### 3. Monitor for Fresh Activity

```bash
tail -f live_run.log | grep -E "(tradable|signals_generated|Created)"
```

Watch for:
- "markets_tradable": 2+ (not 0)
- "signals_generated": 2+ (not 0)
- "Created trade intent:" messages

---

## 🎯 Understanding the Filters

### Spread Filter

**Formula**: `spread = (ask - bid) / mid_price`

**Examples**:
- Bid: 0.48, Ask: 0.52, Mid: 0.50 → Spread: 0.08 (8%)
- Bid: 0.45, Ask: 0.50, Mid: 0.475 → Spread: 0.105 (10.5%)

**With 10% max**:
- 8% spread → ✅ PASSES
- 10.5% spread → ❌ REJECTED

**With old 5% max**:
- 8% spread → ❌ REJECTED (too strict!)
- 10.5% spread → ❌ REJECTED

### Depth Filter

**What it checks**: Total USD value at top 3 price levels

**Examples**:

Market A:
- Bid side: $25 at top 3 levels
- Ask side: $30 at top 3 levels
- Total: $55

**With $20 min per side**:
- ✅ PASSES (25 > 20, 30 > 20, total 55 > 50)

**With old $50 min per side**:
- ❌ REJECTED (25 < 50)

---

## ⚙️ Filter Tuning Guide

If you're STILL seeing 0 tradable markets after restart:

### Too Few Markets? Loosen More

```bash
# Increase max spread to 15%
echo "POLYBOT_MAX_SPREAD=0.15" >> .env

# Or edit filters.py to lower depth further:
# min_bid_depth_usd = 10.0
# min_ask_depth_usd = 10.0
# min_total_depth_usd = 25.0
```

### Too Many Bad Markets? Tighten Up

```bash
# Decrease max spread to 8%
echo "POLYBOT_MAX_SPREAD=0.08" >> .env

# Or edit filters.py to raise depth:
# min_bid_depth_usd = 30.0
# min_ask_depth_usd = 30.0
# min_total_depth_usd = 75.0
```

---

## 🎓 Why This Happened

### The Vicious Cycle:

1. Filters too strict → 0 tradable markets
2. 0 tradable markets → 0 signals
3. 0 signals → 0 new intents
4. 0 new intents → old intents reappear
5. User sees "same old trades"

### The Fix:

1. ✅ Delete old stuck intents
2. ✅ Loosen filters
3. ✅ Markets pass through
4. ✅ Signals generated
5. ✅ New intents created
6. ✅ User sees fresh opportunities

---

## 📋 Summary of Changes

| Setting | Before | After | Impact |
|---------|--------|-------|--------|
| `POLYBOT_MAX_SPREAD` | 0.05 (5%) | 0.10 (10%) | More markets pass |
| `min_bid_depth_usd` | $50 | $20 | Easier to qualify |
| `min_ask_depth_usd` | $50 | $20 | Easier to qualify |
| `min_total_depth_usd` | $100 | $50 | More realistic |
| Old intents | 2 stuck | 0 | Cleaned out |

---

## ⚠️ Important Notes

1. **These are MODERATE filters**:
   - 10% spread is reasonable for prediction markets
   - $20 per side ($40 total) is safe minimum
   - You can tune based on results

2. **Monitor First Few Cycles**:
   - Watch what markets pass through
   - Check if intents make sense
   - Adjust filters if needed

3. **Quality vs Quantity**:
   - Looser filters = more opportunities
   - But also potentially wider spreads
   - Balance based on your trading style

---

## 🚨 If Still Having Issues

### Still 0 Tradable Markets?

1. Check logs for rejection reasons:
   ```bash
   tail -f live_run.log | grep "rejection_reasons"
   ```

2. If still "spread_too_wide", increase further:
   ```bash
   echo "POLYBOT_MAX_SPREAD=0.15" >> .env
   ```

3. If "insufficient_orderbook_depth", lower more:
   ```bash
   # Edit polyb0t/models/filters.py
   # Set min_bid_depth_usd = 10.0
   ```

### Still Seeing Old Intents?

1. Verify they're actually gone:
   ```bash
   sqlite3 polybot.db "SELECT intent_id, status FROM trade_intents WHERE intent_id IN ('563b422e', 'c1291746');"
   # Should show status='EXPIRED'
   ```

2. Force cleanup all old intents:
   ```bash
   python3 cleanup_old_intents.py
   ```

3. Restart bot:
   ```bash
   pkill -f "polyb0t run"
   poetry run polyb0t run
   ```

---

## 🎉 Expected Results

After restart with new filters:

**Logs**:
```json
{
  "markets_scanned": 50,
  "markets_tradable": 3,  // ✅ Not 0!
  "signals_generated": 2,  // ✅ Not 0!
  "intents_created": 2,    // ✅ Not 0!
  "markets_filtered": {
    "spread_too_wide": 2,  // ✅ Fewer rejections
    "insufficient_orderbook_depth": 0  // ✅ None rejected
  }
}
```

**Intent List**:
```
3 Intent(s):

ID        STATUS    TYPE          SIDE  PRICE   USD     EDGE     EXP(s)
------------------------------------------------------------------------
f8a3b2c1  PENDING   OPEN_POSITION BUY   0.420   55.00   +0.028   89
d5e7f9a2  PENDING   OPEN_POSITION SELL  0.580   45.00   +0.031   87
c3b6d8f1  PENDING   OPEN_POSITION BUY   0.385   70.00   +0.024   85
```

**Key**: 
- ✅ New IDs (not 563b422e, c1291746)
- ✅ Different markets (not just 537485)
- ✅ Fresh prices (not stuck at 0.365/0.635)

---

*All fixes applied. Restart the bot to see fresh opportunities!*

