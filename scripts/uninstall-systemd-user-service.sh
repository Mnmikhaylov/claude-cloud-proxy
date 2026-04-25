#!/usr/bin/env bash

set -euo pipefail

SERVICE_NAME="claude-cloud-proxy.service"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
SERVICE_PATH="$CONFIG_HOME/systemd/user/$SERVICE_NAME"

systemctl --user disable --now "$SERVICE_NAME" >/dev/null 2>&1 || true
rm -f "$SERVICE_PATH"
systemctl --user daemon-reload

echo "Removed $SERVICE_NAME"
echo "Configuration file was left intact: $CONFIG_HOME/claude-cloud-proxy/env"
