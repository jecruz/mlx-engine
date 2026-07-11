#!/usr/bin/env bash
set -euo pipefail

LABEL="ai.lmstudio.mlx-engine-adapter"
RUNTIME_DIR="${MLX_ENGINE_RUNTIME_DIR:-$HOME/.local/share/mlx-engine}"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$HOME/Library/Logs/mlx-engine"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH=""
MODEL_NAME=""
PORT="3180"
START_SERVICE=1

usage() {
  echo "Usage: $0 --model PATH [--served-model-name NAME] [--port PORT] [--no-start]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --served-model-name) MODEL_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --no-start) START_SERVICE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH" ]]; then
  echo "--model must name an existing model directory" >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "--port must be an integer from 1024 through 65535" >&2
  exit 2
fi
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}" 

mkdir -p "$RUNTIME_DIR" "$RUNTIME_DIR/releases" "$LOG_DIR" "$(dirname "$PLIST_PATH")"
if [[ -f "$RUNTIME_DIR/REVISION" ]]; then
  PREVIOUS_REVISION="$(cat "$RUNTIME_DIR/REVISION")"
  SNAPSHOT="$RUNTIME_DIR/releases/$PREVIOUS_REVISION"
  mkdir -p "$SNAPSHOT"
  rsync -a --delete --exclude '.venv' --exclude 'releases' "$RUNTIME_DIR/" "$SNAPSHOT/"
fi

rsync -a --delete \
  --exclude '.git' --exclude '.venv*' --exclude '.planning' --exclude '__pycache__' \
  --exclude 'models' --exclude 'reports' --exclude 'releases' "$SOURCE_DIR/" "$RUNTIME_DIR/"
git -C "$SOURCE_DIR" rev-parse HEAD > "$RUNTIME_DIR/REVISION"

python3.11 -m venv "$RUNTIME_DIR/.venv"
"$RUNTIME_DIR/.venv/bin/python" -m pip install --upgrade pip
"$RUNTIME_DIR/.venv/bin/python" -m pip install -r "$RUNTIME_DIR/requirements.txt"
"$RUNTIME_DIR/.venv/bin/python" -m pip install --no-deps "$RUNTIME_DIR"

python3 - "$RUNTIME_DIR/install/$LABEL.plist.in" "$PLIST_PATH" \
  "$RUNTIME_DIR" "$MODEL_PATH" "$MODEL_NAME" "$PORT" "$LOG_DIR" <<'PY'
from pathlib import Path
import sys

template, output, runtime, model, model_name, port, log_dir = sys.argv[1:]
text = Path(template).read_text()
for key, value in {
    "@RUNTIME_DIR@": runtime,
    "@MODEL_PATH@": model,
    "@MODEL_NAME@": model_name,
    "@PORT@": port,
    "@LOG_DIR@": log_dir,
}.items():
    text = text.replace(key, value)
Path(output).write_text(text)
PY
plutil -lint "$PLIST_PATH"

launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
if (( START_SERVICE )); then
  launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
fi

"$RUNTIME_DIR/.venv/bin/mlx-engine-version"
echo "Installed runtime: $RUNTIME_DIR"
echo "Service definition: $PLIST_PATH"
