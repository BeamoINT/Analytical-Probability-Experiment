# 📊 Percentage-Based Position Sizing Update

## ✅ Changes Made

The bot now uses **aggressive, balance-aware position sizing**:

### New Settings (Applied)

| Setting | Old Value | New Value | Description |
|---------|-----------|-----------|-------------|
| **Max bet per trade** | 15% of portfolio | **45% of available cash** | Each trade can use up to 45% of your available USDC |
| **Max total exposure** | 40% of portfolio | **90% of total portfolio** | All open positions combined can use 90% of your total value |

---

## 🎯 How It Works Now

### Example With Your $219 Balance

#### Scenario 1: No Open Positions
```
Available Cash: $219
Reserved (in open trades): $0
Total Portfolio: $219

Max bet size per trade: $219 × 45% = $98.55
Max total exposure: $219 × 90% = $197.10

Bot can create:
- Trade 1: Up to $98.55
- Trade 2: Up to $98.55 (but total exposure capped at $197.10)
```

#### Scenario 2: One Position Already Open
```
Available Cash: $120
Reserved (Trade 1): $99
Total Portfolio: $219

Max bet size per trade: $120 × 45% = $54
Max total exposure: $219 × 90% = $197.10
Current exposure: $99
Remaining capacity: $197.10 - $99 = $98.10

Bot can create:
- Trade 2: Up to $54 (limited by 45% of available)
```

#### Scenario 3: After Winning
```
Available Cash: $280 (after profits)
Reserved: $0
Total Portfolio: $280

Max bet size per trade: $280 × 45% = $126
Max total exposure: $280 × 90% = $252

Bot can create larger trades as your balance grows!
```

---

## 📈 How It Scales

As your balance grows, bet sizes grow proportionally:

| Balance | Max Per Trade (45%) | Max Total Exposure (90%) |
|---------|---------------------|--------------------------|
| $100 | $45 | $90 |
| $219 | $98.55 | $197.10 |
| $500 | $225 | $450 |
| $1,000 | $450 | $900 |

---

## 🛡️ Safety Features Still Active

Even with aggressive sizing, the bot still:

✅ **Risk checks** - Validates every trade  
✅ **Edge threshold** - Only trades with 5%+ edge  
✅ **Liquidity filters** - Only liquid markets  
✅ **Spread limits** - Avoids wide spreads  
✅ **Approval required** - You must approve every trade  
✅ **Dry-run default** - Safe testing mode  

---

## ⚠️ Important Notes

### 1. This Is Aggressive
- **45% per trade** is aggressive for trading
- Professional traders typically use 10-25% max
- You're risking nearly half your cash per position

### 2. Total Exposure 90%
- Almost all your money can be in play at once
- Only 10% stays liquid as reserve
- Higher potential gains, but also higher risk

### 3. Recommended For
- ✅ Experienced traders
- ✅ High-conviction strategies
- ✅ Small account sizes (<$500)
- ❌ Risk-averse users
- ❌ Large capital

---

## 🔄 To Change Back to Conservative

If this feels too aggressive, edit the code:

```python
# In polyb0t/models/position_sizing.py

# Conservative (safer):
self.max_pct_per_trade = 0.15  # 15% per trade
self.max_pct_total_exposure = 0.40  # 40% total

# Moderate (balanced):
self.max_pct_per_trade = 0.25  # 25% per trade
self.max_pct_total_exposure = 0.60  # 60% total

# Aggressive (current):
self.max_pct_per_trade = 0.45  # 45% per trade
self.max_pct_total_exposure = 0.90  # 90% total
```

---

## 📝 Technical Changes

### Files Modified:

1. **`polyb0t/models/position_sizing.py`**
   - Changed `max_pct_per_trade`: 0.15 → 0.45
   - Changed `max_pct_total_exposure`: 0.40 → 0.90
   - Updated calculation to use available cash (not total bankroll) for per-trade limit

2. **`polyb0t/config/settings.py`**
   - Increased `max_order_usd`: $5 → $10,000 (allows percentage-based sizing)
   - Increased `max_total_exposure_usd`: $25 → $100,000 (allows percentage-based sizing)

---

## ✅ Example Trade Intents (Before vs After)

### Before (Conservative)
```
Balance: $219
Max per trade: $219 × 15% = $32.85

Intent 1: BUY $32.85
Intent 2: BUY $32.85
Total Exposure: $65.70 (30% of balance)
```

### After (Aggressive)
```
Balance: $219
Max per trade: $219 × 45% = $98.55

Intent 1: BUY $98.55
Intent 2: BUY $98.55
Total Exposure: $197.10 (90% of balance)
```

---

## 🎯 Bottom Line

**Your bot now trades much more aggressively:**
- Each trade can be **3x larger** (45% vs 15%)
- Total exposure is **2.25x higher** (90% vs 40%)
- Risk and reward both significantly increased

**This matches your request:** Maximum leverage while staying within percentage-based limits.

---

**Updated:** January 2026  
**System:** PolyB0t v2.1 - Percentage-Based Aggressive Sizing

