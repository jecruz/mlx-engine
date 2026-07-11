#!/usr/bin/env bash
set -euo pipefail

LABEL="ai.lmstudio.mlx-engine-adapter"
RUNTIME_DIR="${MLX_ENGINE_RUNTIME_DIR:-$HOME/.local/share/mlx-engine}"
REVISION="${1:-}"

if [[ -z "$REVISION" || ! -d "$RUNTIME_DIR/releases/$REVISION" ]]; then
  echo "Usage: $0 REVISION" >&2
  echo "Available revisions:" >&2
  find "$RUNTIME_DIR/releases" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; 2>/dev/null || true
  exit 2
fi
launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
rsync -a --delete --exclude '.venv' --exclude 'releases' "$RUNTIME_DIR/releases/$REVISION/" "$RUNTIME_DIR/"
"$RUNTIME_DIR/.venv/bin/python" -m pip install --no-deps --force-reinstall "$RUNTIME_DIR"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl kickstart -k "gui/$(id -u)/$LABEL"
echo "Rolled back runtime to $REVISION"
