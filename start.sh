#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "No .venv found. Run: uv venv .venv && uv sync"
    exit 1
fi

MAX_RESTARTS=50
RESTART_DELAY=5
count=0

echo "=== Trading bot starting ($(date)) ==="
echo "=== Logs: $LOG_DIR ==="
echo "=== Ctrl+C to stop, or detach tmux with Ctrl+B then D ==="

while [ $count -lt $MAX_RESTARTS ]; do
    LOGFILE="$LOG_DIR/bot_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date)] Starting run #$((count+1))... logging to $LOGFILE"

    if python main.py "$@" 2>&1 | tee -a "$LOGFILE"; then
        echo "[$(date)] Bot exited cleanly."
        break
    else
        EXIT_CODE=$?
        count=$((count + 1))
        echo "[$(date)] Bot crashed (exit $EXIT_CODE). Restart $count/$MAX_RESTARTS in ${RESTART_DELAY}s..."
        sleep "$RESTART_DELAY"
    fi
done

if [ $count -ge $MAX_RESTARTS ]; then
    echo "[$(date)] Hit max restarts ($MAX_RESTARTS). Giving up."
    exit 1
fi
