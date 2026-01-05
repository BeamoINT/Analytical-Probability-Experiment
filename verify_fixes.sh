#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🔍 Polymarket Trading System - Fix Verification${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Test 1: Check configuration
echo -e "${YELLOW}📝 Test 1: Checking Configuration${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    exit 1
fi

# Check edge threshold
EDGE_THRESHOLD=$(grep "POLYBOT_EDGE_THRESHOLD" .env | cut -d'=' -f2)
if [ -z "$EDGE_THRESHOLD" ]; then
    echo -e "${RED}❌ POLYBOT_EDGE_THRESHOLD not set${NC}"
else
    if (( $(echo "$EDGE_THRESHOLD <= 0.02" | bc -l) )); then
        echo -e "${GREEN}✅ EDGE_THRESHOLD = $EDGE_THRESHOLD (good)${NC}"
    else
        echo -e "${YELLOW}⚠️  EDGE_THRESHOLD = $EDGE_THRESHOLD (may be too high)${NC}"
    fi
fi

# Check mode
MODE=$(grep "^POLYBOT_MODE" .env | cut -d'=' -f2)
if [ "$MODE" == "live" ]; then
    echo -e "${GREEN}✅ MODE = live${NC}"
else
    echo -e "${YELLOW}⚠️  MODE = $MODE (expected 'live')${NC}"
fi

# Check RPC
RPC=$(grep "POLYBOT_POLYGON_RPC_URL" .env | cut -d'=' -f2)
if [ -z "$RPC" ]; then
    echo -e "${RED}❌ POLYGON_RPC_URL not set (needed for balance)${NC}"
else
    echo -e "${GREEN}✅ POLYGON_RPC_URL set${NC}"
fi

echo ""

# Test 2: Check database
echo -e "${YELLOW}📊 Test 2: Checking Database${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "polybot.db" ]; then
    SIZE=$(du -h polybot.db | cut -f1)
    echo -e "${GREEN}✅ Database exists (size: $SIZE)${NC}"
    
    # Count pending intents
    PENDING_COUNT=$(sqlite3 polybot.db "SELECT COUNT(*) FROM trade_intents WHERE status='PENDING';" 2>/dev/null || echo "0")
    echo -e "   ${BLUE}ℹ️  Pending intents: $PENDING_COUNT${NC}"
else
    echo -e "${YELLOW}⚠️  Database not found (will be created on first run)${NC}"
fi

echo ""

# Test 3: Check log files
echo -e "${YELLOW}📝 Test 3: Checking Recent Logs${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "live_run.log" ]; then
    echo -e "${GREEN}✅ Log file exists${NC}"
    
    # Check for recent activity
    RECENT_LINES=$(tail -20 live_run.log | wc -l)
    if [ $RECENT_LINES -gt 0 ]; then
        echo -e "   ${BLUE}ℹ️  Recent log entries: $RECENT_LINES${NC}"
        
        # Check for balance reporting
        if tail -50 live_run.log | grep -q "Account: balance"; then
            BALANCE=$(tail -50 live_run.log | grep "Account: balance" | tail -1 | sed 's/.*balance=\([0-9.]*\).*/\1/')
            echo -e "   ${GREEN}✅ Real balance found in logs: ${BALANCE} USDC${NC}"
        elif tail -50 live_run.log | grep -q "Portfolio: equity"; then
            echo -e "   ${RED}❌ Still using simulated portfolio equity${NC}"
        else
            echo -e "   ${YELLOW}⚠️  No balance info in recent logs${NC}"
        fi
        
        # Check for signal generation
        if tail -50 live_run.log | grep -q "Generated.*signals"; then
            SIGNALS=$(tail -50 live_run.log | grep "Generated.*signals" | tail -1 | grep -o 'Generated [0-9]*' | grep -o '[0-9]*')
            if [ "$SIGNALS" -gt 0 ]; then
                echo -e "   ${GREEN}✅ Signals being generated: $SIGNALS in last cycle${NC}"
            else
                echo -e "   ${YELLOW}⚠️  No signals generated in recent cycles${NC}"
            fi
        fi
    fi
else
    echo -e "${YELLOW}⚠️  No log file yet (normal if not started)${NC}"
fi

echo ""

# Test 4: Check if bot is running
echo -e "${YELLOW}🤖 Test 4: Checking Bot Status${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if pgrep -f "polyb0t run" > /dev/null; then
    PID=$(pgrep -f "polyb0t run")
    echo -e "${GREEN}✅ Bot is running (PID: $PID)${NC}"
    
    # Check how long it's been running
    if [ -f "live_run.pid" ]; then
        STORED_PID=$(cat live_run.pid)
        if [ "$PID" == "$STORED_PID" ]; then
            echo -e "   ${BLUE}ℹ️  PID matches stored PID${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Bot is not running${NC}"
    echo -e "   ${BLUE}ℹ️  Start with: poetry run polyb0t run${NC}"
fi

echo ""

# Test 5: Quick intent check
echo -e "${YELLOW}📋 Test 5: Quick Intent Check${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v poetry &> /dev/null; then
    echo -e "${GREEN}✅ Poetry is available${NC}"
    
    # Try to list intents (timeout after 5 seconds)
    INTENT_OUTPUT=$(timeout 5 poetry run polyb0t intents list 2>&1 || echo "timeout")
    
    if echo "$INTENT_OUTPUT" | grep -q "Intent(s)"; then
        INTENT_COUNT=$(echo "$INTENT_OUTPUT" | head -1 | grep -o '[0-9]*')
        echo -e "   ${GREEN}✅ Intent system working: $INTENT_COUNT intent(s)${NC}"
        
        # Check if they're fresh (not the old stuck ones)
        if echo "$INTENT_OUTPUT" | grep -q "85c72448"; then
            echo -e "   ${YELLOW}⚠️  Old intent ID detected (may need cleanup)${NC}"
        else
            echo -e "   ${GREEN}✅ No old stuck intents found${NC}"
        fi
    elif echo "$INTENT_OUTPUT" | grep -q "timeout"; then
        echo -e "   ${YELLOW}⚠️  Intent check timed out${NC}"
    else
        echo -e "   ${BLUE}ℹ️  No intents currently (normal if just started)${NC}"
    fi
else
    echo -e "${RED}❌ Poetry not found${NC}"
fi

echo ""

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 Verification Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create a simple pass/fail summary
PASS_COUNT=0
TOTAL_COUNT=5

if [ ! -z "$EDGE_THRESHOLD" ] && (( $(echo "$EDGE_THRESHOLD <= 0.02" | bc -l) )); then
    PASS_COUNT=$((PASS_COUNT + 1))
fi

if [ "$MODE" == "live" ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
fi

if [ -f "polybot.db" ]; then
    PASS_COUNT=$((PASS_COUNT + 1))
fi

if tail -50 live_run.log 2>/dev/null | grep -q "Account: balance"; then
    PASS_COUNT=$((PASS_COUNT + 1))
fi

if pgrep -f "polyb0t run" > /dev/null; then
    PASS_COUNT=$((PASS_COUNT + 1))
fi

echo -e "${GREEN}✅ Passed: $PASS_COUNT/$TOTAL_COUNT checks${NC}"

if [ $PASS_COUNT -eq $TOTAL_COUNT ]; then
    echo ""
    echo -e "${GREEN}🎉 All systems operational!${NC}"
    echo ""
    echo -e "${BLUE}📋 Next steps:${NC}"
    echo "   1. Monitor logs: tail -f live_run.log"
    echo "   2. Check intents: poetry run polyb0t intents list"
    echo "   3. Approve good opportunities: poetry run polyb0t intents approve <id>"
elif [ $PASS_COUNT -ge 3 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  System mostly ready, but some issues detected${NC}"
    echo ""
    echo -e "${BLUE}📋 Suggested actions:${NC}"
    
    if [ -z "$RPC" ]; then
        echo "   • Set POLYBOT_POLYGON_RPC_URL in .env"
    fi
    
    if ! pgrep -f "polyb0t run" > /dev/null; then
        echo "   • Start the bot: poetry run polyb0t run"
    fi
    
    if ! tail -50 live_run.log 2>/dev/null | grep -q "Account: balance"; then
        echo "   • Restart bot to apply fixes: pkill -f 'polyb0t run' && poetry run polyb0t run"
    fi
else
    echo ""
    echo -e "${RED}❌ Several issues detected${NC}"
    echo ""
    echo -e "${BLUE}📋 Suggested actions:${NC}"
    echo "   1. Run: python3 fix_trading_config.py"
    echo "   2. Check .env configuration"
    echo "   3. Restart bot: poetry run polyb0t run"
    echo "   4. Review: QUICK_START_AFTER_FIX.md"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

