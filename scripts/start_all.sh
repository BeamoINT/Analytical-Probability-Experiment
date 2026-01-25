#!/bin/bash
# PolyB0T Start Script - Runs both trading bot and API server
# This script is designed to be run by systemd

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Function to cleanup on exit
cleanup() {
    echo "Stopping PolyB0T services..."
    kill $BOT_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

echo "Starting PolyB0T from: $PROJECT_DIR"

# Start the trading bot in background
echo "Starting trading bot..."
poetry run polyb0t run --live &
BOT_PID=$!
echo "Trading bot started with PID: $BOT_PID"

# Give the bot a moment to initialize
sleep 5

# Start the API server in background
echo "Starting API server on 0.0.0.0:8000..."
poetry run polyb0t api --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "API server started with PID: $API_PID"

echo "PolyB0T fully started. Bot PID: $BOT_PID, API PID: $API_PID"

# Wait for either process to exit
wait -n $BOT_PID $API_PID

# If one exits, stop the other
echo "One process exited, stopping all..."
cleanup
