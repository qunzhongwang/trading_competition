#!/usr/bin/env bash
# Start the trading bot in competition (Roostoo) mode.
# Uses Binance WS for data, Roostoo REST for execution.
#
# Required env vars: ROOSTOO_API_KEY, ROOSTOO_API_SECRET
# Optional: BINANCE_API_KEY, BINANCE_API_SECRET (for authenticated WS)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Validate required env vars (either COMP or regular keys)
if [ -z "${ROOSTOO_COMP_API_KEY:-}${ROOSTOO_API_KEY:-}" ] || [ -z "${ROOSTOO_COMP_API_SECRET:-}${ROOSTOO_API_SECRET:-}" ]; then
    echo "ERROR: Roostoo API credentials must be set (either ROOSTOO_COMP_* or ROOSTOO_*)"
    echo "  export ROOSTOO_COMP_API_KEY=your_key"
    echo "  export ROOSTOO_COMP_API_SECRET=your_secret"
    echo "  Or create a .env file in $PROJECT_DIR"
    exit 1
fi

# Ensure logs directory exists
mkdir -p logs

echo "=== Starting Trading Competition Bot ==="
echo "Mode: roostoo"
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================="

# Run with auto-restart on crash (max 5 restarts with backoff)
MAX_RESTARTS=50
RESTART_COUNT=0
BACKOFF=30

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    if uv run python main.py --mode roostoo; then
        echo "Bot exited cleanly"
        break
    else
        EXIT_CODE=$?
        RESTART_COUNT=$((RESTART_COUNT + 1))
        WAIT=$((BACKOFF * RESTART_COUNT))
        echo "Bot crashed (exit=$EXIT_CODE), restart $RESTART_COUNT/$MAX_RESTARTS in ${WAIT}s..."
        sleep $WAIT
    fi
done

if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
    echo "ERROR: Bot crashed $MAX_RESTARTS times, giving up"
    exit 1
fi
