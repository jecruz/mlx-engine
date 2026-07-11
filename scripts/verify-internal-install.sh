#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${MLX_ENGINE_RUNTIME_DIR:-$HOME/.local/share/mlx-engine}"
PORT="${MLX_ENGINE_PORT:-3180}"
EXPECTED_REVISION="${1:-}"

if [[ ! -x "$RUNTIME_DIR/.venv/bin/mlx-engine-version" ]]; then
  echo "Installed runtime is missing at $RUNTIME_DIR" >&2
  exit 1
fi
INSTALLED_REVISION="$(cat "$RUNTIME_DIR/REVISION")"
if [[ -n "$EXPECTED_REVISION" && "$INSTALLED_REVISION" != "$EXPECTED_REVISION" ]]; then
  echo "Revision mismatch: installed=$INSTALLED_REVISION expected=$EXPECTED_REVISION" >&2
  exit 1
fi
"$RUNTIME_DIR/.venv/bin/mlx-engine-version"
curl --fail --silent --show-error "http://127.0.0.1:$PORT/health"
echo
