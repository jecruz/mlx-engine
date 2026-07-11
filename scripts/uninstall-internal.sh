#!/usr/bin/env bash
set -euo pipefail

LABEL="ai.lmstudio.mlx-engine-adapter"
RUNTIME_DIR="${MLX_ENGINE_RUNTIME_DIR:-$HOME/.local/share/mlx-engine}"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
rm -f "$PLIST_PATH"
rm -rf "$RUNTIME_DIR"
echo "Removed mlx-engine internal runtime and launchd service"
