#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
SERVICE_NAME="claude-cloud-proxy.service"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
CONFIG_DIR="$CONFIG_HOME/claude-cloud-proxy"
ENV_FILE="$CONFIG_DIR/env"
SYSTEMD_USER_DIR="$CONFIG_HOME/systemd/user"
SERVICE_PATH="$SYSTEMD_USER_DIR/$SERVICE_NAME"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python binary not found or not executable: $PYTHON_BIN" >&2
    echo "Create a venv first: python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

mkdir -p "$CONFIG_DIR" "$SYSTEMD_USER_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
    cat >"$ENV_FILE" <<'EOF'
CLOUD_RU_BASE_URL=https://foundation-models.api.cloud.ru/v1
# CLOUD_RU_API_KEY=
# PROXY_API_KEY=
PROXY_HOST=127.0.0.1
PROXY_PORT=2222
PROXY_MAX_REQUEST_BYTES=33554432
PROXY_TIMEOUT_SECONDS=120
PROXY_LOG_LEVEL=INFO
EOF
    chmod 600 "$ENV_FILE"
fi

export ROOT_DIR PYTHON_BIN SERVICE_PATH ENV_FILE
python3 - <<'PY'
from __future__ import annotations

import os
from pathlib import Path


def systemd_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


root_dir = os.environ["ROOT_DIR"]
python_bin = os.environ["PYTHON_BIN"]
service_path = Path(os.environ["SERVICE_PATH"])
env_file = os.environ["ENV_FILE"]

unit = f"""[Unit]
Description=Claude Code Cloud.ru Foundation Models proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={systemd_quote(root_dir)}
Environment={systemd_quote(f"PYTHONPATH={root_dir}/src")}
EnvironmentFile={systemd_quote(env_file)}
ExecStart={systemd_quote(python_bin)} -m claude_cloud_proxy
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""

service_path.write_text(unit, encoding="utf-8")
PY

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"

echo "Installed and started $SERVICE_NAME"
echo "unit: $SERVICE_PATH"
echo "env:  $ENV_FILE"
echo "logs: journalctl --user -u $SERVICE_NAME -f"
