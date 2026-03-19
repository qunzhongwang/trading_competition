#!/usr/bin/env bash
# Usage: ./scripts/run.sh [paper|roostoo|live]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate
exec python main.py --mode "${1:-paper}" "${@:2}"
