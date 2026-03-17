#!/usr/bin/env bash
# Deploy trading competition framework to AWS EC2.
# Usage: ./scripts/deploy_aws.sh <EC2_HOST> [SSH_KEY_PATH]
#
# Prerequisites:
#   - EC2 instance running Ubuntu 22.04+ or Amazon Linux 2023
#   - SSH access configured
#   - Env vars ROOSTOO_API_KEY and ROOSTOO_API_SECRET set locally or on EC2

set -euo pipefail

EC2_HOST="${1:?Usage: deploy_aws.sh <EC2_HOST> [SSH_KEY_PATH]}"
SSH_KEY="${2:-~/.ssh/id_rsa}"
SSH_OPTS="-o StrictHostKeyChecking=no -i $SSH_KEY"
REMOTE_DIR="/home/ubuntu/trading_competition"
REPO_URL="${REPO_URL:-$(git remote get-url origin 2>/dev/null || echo '')}"

echo "=== Deploying to $EC2_HOST ==="

# 1. Install system dependencies on EC2
ssh $SSH_OPTS ubuntu@"$EC2_HOST" << 'REMOTE_SETUP'
set -euo pipefail

# Install Python 3.11+ and git
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv git curl

# Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "System setup complete: $(python3.11 --version), uv $(uv --version)"
REMOTE_SETUP

# 2. Sync code to EC2
if [ -n "$REPO_URL" ]; then
    echo "Cloning/pulling from $REPO_URL..."
    ssh $SSH_OPTS ubuntu@"$EC2_HOST" << REMOTE_CLONE
    set -euo pipefail
    export PATH="\$HOME/.local/bin:\$PATH"
    if [ -d "$REMOTE_DIR" ]; then
        cd "$REMOTE_DIR" && git pull --rebase
    else
        git clone "$REPO_URL" "$REMOTE_DIR"
    fi
REMOTE_CLONE
else
    echo "No git remote; rsyncing local files..."
    rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'logs' \
        --exclude '.git' --exclude 'artifacts' \
        -e "ssh $SSH_OPTS" \
        . ubuntu@"$EC2_HOST":"$REMOTE_DIR"/
fi

# 3. Install dependencies on EC2
ssh $SSH_OPTS ubuntu@"$EC2_HOST" << REMOTE_INSTALL
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
cd "$REMOTE_DIR"
uv sync
echo "Dependencies installed"
REMOTE_INSTALL

# 4. Copy start script and set up systemd service
ssh $SSH_OPTS ubuntu@"$EC2_HOST" << 'REMOTE_SERVICE'
set -euo pipefail

cat > /tmp/trading-bot.service << 'EOF'
[Unit]
Description=Trading Competition Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading_competition
ExecStart=/home/ubuntu/trading_competition/scripts/start_competition.sh
Restart=on-failure
RestartSec=30
Environment=PATH=/home/ubuntu/.local/bin:/usr/bin:/bin
EnvironmentFile=-/home/ubuntu/trading_competition/.env

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/trading-bot.service /etc/systemd/system/trading-bot.service
sudo systemctl daemon-reload
sudo systemctl enable trading-bot

echo "Systemd service installed. Start with: sudo systemctl start trading-bot"
echo "Logs: journalctl -u trading-bot -f"
REMOTE_SERVICE

echo ""
echo "=== Deployment complete ==="
echo "Next steps:"
echo "  1. SSH into EC2: ssh $SSH_OPTS ubuntu@$EC2_HOST"
echo "  2. Set API keys: echo 'ROOSTOO_API_KEY=...\nROOSTOO_API_SECRET=...' > $REMOTE_DIR/.env"
echo "  3. Start the bot: sudo systemctl start trading-bot"
echo "  4. Check logs: journalctl -u trading-bot -f"
